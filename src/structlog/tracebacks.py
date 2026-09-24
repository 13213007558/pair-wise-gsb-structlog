# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Extract a structured traceback from an exception.

Based on work by Will McGugan
<https://github.com/hynek/structlog/pull/407#issuecomment-1150926246>`_ from
`rich.traceback
<https://github.com/Textualize/rich/blob/972dedff/rich/traceback.py>`_.
"""

from __future__ import annotations

import os
import os.path
import sys

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from traceback import walk_tb
from types import ModuleType, TracebackType
from typing import Any, TypeAlias


try:
    import rich
    import rich.pretty
except ImportError:
    rich = None  # type: ignore[assignment]

from .typing import ExcInfo


__all__ = [
    "ExceptionDictTransformer",
    "Frame",
    "Stack",
    "SyntaxError_",
    "Trace",
    "extract",
    "safe_str",
    "to_repr",
]


SHOW_LOCALS = True
LOCALS_MAX_LENGTH = 10
LOCALS_MAX_STRING = 80
MAX_FRAMES = 50

OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]


@dataclass
class Frame:
    """
    Represents a single stack frame.

    .. versionchanged:: 26.2.0
       Added the *locals_omitted* field.
    """

    filename: str
    lineno: int
    name: str
    locals: dict[str, str] | None = None
    locals_omitted: int = 0


@dataclass
class SyntaxError_:  # noqa: N801
    """
    Contains detailed information about :exc:`SyntaxError` exceptions.
    """

    offset: int
    filename: str
    line: str
    lineno: int
    msg: str


@dataclass
class Stack:
    """
    Represents an exception and a list of stack frames.

    .. versionchanged:: 25.2.0
       Added the *exc_notes* field.

    .. versionchanged:: 25.4.0
       Added the *is_group* and *exceptions* fields.

    .. versionchanged:: 26.2.0
       Added the *skipped_exceptions* and *skipped_causes* fields.
    """

    exc_type: str
    exc_value: str
    exc_notes: list[str] = field(default_factory=list)
    syntax_error: SyntaxError_ | None = None
    is_cause: bool = False
    frames: list[Frame] = field(default_factory=list)
    is_group: bool = False
    exceptions: list[Trace] = field(default_factory=list)
    skipped_exceptions: int = 0
    skipped_causes: int = 0


@dataclass
class Trace:
    """
    Container for a list of stack traces.
    """

    stacks: list[Stack]


def safe_str(_object: Any) -> str:
    """Don't allow exceptions from __str__ to propagate."""
    try:
        return str(_object)
    except Exception as error:  # noqa: BLE001
        return f"<str-error {str(error)!r}>"


def to_repr(
    obj: Any,
    max_length: int | None = None,
    max_string: int | None = None,
    use_rich: bool = True,
) -> str:
    """
    Get repr string for an object, but catch errors.

    :func:`repr()` is used for strings, too, so that secret wrappers that
    inherit from :func:`str` and overwrite ``__repr__()`` are handled correctly
    (i.e. secrets are not logged in plain text).

    Args:
        obj: Object to get a string representation for.

        max_length: Maximum length of containers before abbreviating, or
            ``None`` for no abbreviation.

        max_string: Maximum length of string before truncating, or ``None`` to
            disable truncating.

        use_rich: If ``True`` (the default), use rich_ to compute the repr.
            If ``False`` or if rich_ is not installed, fall back to a simpler
            algorithm.

    Returns:
        The string representation of *obj*.

    .. versionchanged:: 24.3.0
       Added *max_length* argument.  Use :program:`rich` to render locals if it
       is available.  Call :func:`repr()` on strings in fallback
       implementation.
    """
    if use_rich and rich is not None:
        # Let rich render the repr if it is available.
        # It produces much better results for containers and dataclasses/attrs.
        obj_repr = rich.pretty.traverse(
            obj, max_length=max_length, max_string=max_string
        ).render()
    else:
        # Generate a (truncated) repr if rich is not available.
        # Handle str/bytes differently to get better results for truncated
        # representations.  Also catch all errors, similarly to "safe_str()".
        try:
            if isinstance(obj, (str, bytes)):
                if max_string is not None and len(obj) > max_string:
                    truncated = len(obj) - max_string
                    obj_repr = f"{obj[:max_string]!r}+{truncated}"
                else:
                    obj_repr = repr(obj)
            else:
                obj_repr = repr(obj)
                if max_string is not None and len(obj_repr) > max_string:
                    truncated = len(obj_repr) - max_string
                    obj_repr = f"{obj_repr[:max_string]!r}+{truncated}"
        except Exception as error:  # noqa: BLE001
            obj_repr = f"<repr-error {str(error)!r}>"

    return obj_repr


@dataclass
class _Budget:
    """
    Remaining tree-wide extraction budgets.  ``None`` means no limit.
    """

    frames: int | None = None
    branches: int | None = None
    locals_bytes: int | None = None


def _count_remaining(exc_value: BaseException, seen: set[int]) -> int:
    """
    Count how many exception nodes would still be extracted from *exc_value*,
    including its exception group members and its cause/context chain.

    *seen* is updated with the visited exceptions.
    """

    def count(exc: BaseException) -> int:
        """Count the exception nodes of one chain, group members included."""
        n = 0
        while True:
            if id(exc) in seen:
                break
            seen.add(id(exc))
            n += 1
            if sys.version_info >= (3, 11) and isinstance(
                exc, BaseExceptionGroup  # noqa: F821
            ):
                for sub in exc.exceptions:
                    n += count(sub)

            cause = getattr(exc, "__cause__", None)
            if cause and cause.__traceback__:
                exc = cause
                continue

            cause = exc.__context__
            if (
                cause
                and cause.__traceback__
                and not getattr(exc, "__suppress_context__", False)
            ):
                exc = cause
                continue

            break

        return n

    return count(exc_value)


def extract(
    exc_type: type[BaseException],
    exc_value: BaseException,
    traceback: TracebackType | None,
    *,
    show_locals: bool = False,
    locals_max_length: int = LOCALS_MAX_LENGTH,
    locals_max_string: int = LOCALS_MAX_STRING,
    locals_hide_dunder: bool = True,
    locals_hide_sunder: bool = False,
    use_rich: bool = True,
    _seen: set[int] | None = None,
    _budget: _Budget | None = None,
) -> Trace:
    """
    Extract traceback information.

    Args:
        exc_type: Exception type.

        exc_value: Exception value.

        traceback: Python Traceback object.

        show_locals: Enable display of local variables. Defaults to False.

        locals_max_length:
            Maximum length of containers before abbreviating, or ``None`` for
            no abbreviation.

        locals_max_string:
            Maximum length of string before truncating, or ``None`` to disable
            truncating.

        locals_hide_dunder:
            Hide locals prefixed with double underscore.
            Defaults to True.

        locals_hide_sunder:
            Hide locals prefixed with single underscore.
            This implies hiding *locals_hide_dunder*.
            Defaults to False.

        use_rich: If ``True`` (the default), use rich_ to compute the repr.
            If ``False`` or if rich_ is not installed, fall back to a simpler
            algorithm.

    Returns:
        A Trace instance with structured information about all exceptions.

    .. versionadded:: 22.1.0

    .. versionchanged:: 24.3.0
       Added *locals_max_length*, *locals_hide_sunder*, *locals_hide_dunder*
       and *use_rich* arguments.

    .. versionchanged:: 25.4.0
       Handle exception groups.

    .. versionchanged:: 25.5.0
       Handle loops in exception cause chain.

    .. versionchanged:: 26.2.0
       Added private *_budget* argument for tree-wide size limits.
    """

    stacks: list[Stack] = []
    is_cause = False

    if _seen is None:
        _seen = set()

    while True:
        exc_id = id(exc_value)
        if exc_id in _seen:
            break

        if (
            _budget is not None
            and _budget.branches is not None
            and _budget.branches <= 0
        ):
            # The branch budget is exhausted: count the rest of the chain
            # and attach it to the last extracted stack.
            stacks[-1].skipped_causes = _count_remaining(
                exc_value, set(_seen)
            )
            break

        _seen.add(exc_id)
        if _budget is not None and _budget.branches is not None:
            _budget.branches -= 1

        stack = Stack(
            exc_type=safe_str(exc_type.__name__),
            exc_value=safe_str(exc_value),
            exc_notes=[
                safe_str(note) for note in getattr(exc_value, "__notes__", ())
            ],
            is_cause=is_cause,
        )

        if sys.version_info >= (3, 11):
            if isinstance(exc_value, (BaseExceptionGroup, ExceptionGroup)):  # noqa: F821
                stack.is_group = True
                children = exc_value.exceptions
                for index, exception in enumerate(children):
                    if (
                        _budget is not None
                        and _budget.branches is not None
                        and _budget.branches <= 0
                    ):
                        # Count all omitted members (and their subtrees).
                        remaining_seen = set(_seen)
                        stack.skipped_exceptions = sum(
                            _count_remaining(rest, remaining_seen)
                            for rest in children[index:]
                        )
                        break
                    stack.exceptions.append(
                        extract(
                            type(exception),
                            exception,
                            exception.__traceback__,
                            show_locals=show_locals,
                            locals_max_length=locals_max_length,
                            locals_max_string=locals_max_string,
                            locals_hide_dunder=locals_hide_dunder,
                            locals_hide_sunder=locals_hide_sunder,
                            use_rich=use_rich,
                            _seen=_seen,
                            _budget=_budget,
                        )
                    )

        if isinstance(exc_value, SyntaxError):
            stack.syntax_error = SyntaxError_(
                offset=exc_value.offset or 0,
                filename=exc_value.filename or "?",
                lineno=exc_value.lineno or 0,
                line=exc_value.text or "",
                msg=exc_value.msg,
            )

        stacks.append(stack)
        append = stack.frames.append  # pylint: disable=no-member

        def get_locals(
            iter_locals: Iterable[tuple[str, object]],
        ) -> Iterable[tuple[str, object]]:
            """Extract locals from an iterator of key pairs."""
            if not (locals_hide_dunder or locals_hide_sunder):
                yield from iter_locals
                return
            for key, value in iter_locals:
                if locals_hide_dunder and key.startswith("__"):
                    continue
                if locals_hide_sunder and key.startswith("_"):
                    continue
                yield key, value

        # Collect the raw frames first, so that frames that are dropped by
        # a frame budget never get their locals rendered.
        tb_frames: list[tuple[Any, int] | None] = list(walk_tb(traceback))
        n_frames = len(tb_frames)
        keep = n_frames
        if _budget is not None and _budget.frames is not None:
            keep = min(n_frames, max(_budget.frames, 0))

        skipped = n_frames - keep
        if skipped:
            # Keep the first frames (your own code) and the last frames
            # (where the error actually happened).
            if keep >= 2:
                head = keep - keep // 2
                tail = keep // 2
            elif keep == 1:
                head, tail = 0, 1
            else:
                head = tail = 0
            selected = [
                *tb_frames[:head],
                None,  # Placeholder for the skipped-frames marker frame.
                *tb_frames[n_frames - tail :],
            ]
        else:
            selected = tb_frames

        for tb_frame in selected:
            if tb_frame is None:
                append(
                    Frame(
                        filename="",
                        lineno=-1,
                        name=f"Skipped frames: {skipped}",
                    )
                )
                continue

            frame_summary, line_no = tb_frame
            if _budget is not None and _budget.frames is not None:
                _budget.frames -= 1

            filename = frame_summary.f_code.co_filename
            if filename and not filename.startswith("<"):
                filename = os.path.abspath(filename)
            # Rich has this, but we are not rich and like to keep all frames:
            # if frame_summary.f_locals.get("_rich_traceback_omit", False):
            #     continue  # noqa: ERA001

            frame_locals: dict[str, str] | None = None
            locals_omitted = 0
            if show_locals:
                frame_locals = {}
                for key, value in get_locals(frame_summary.f_locals.items()):
                    if (
                        _budget is not None
                        and _budget.locals_bytes is not None
                        and _budget.locals_bytes <= 0
                    ):
                        # The locals budget is exhausted: count the local,
                        # but don't render it.
                        locals_omitted += 1
                        continue
                    rendered = to_repr(
                        value,
                        max_length=locals_max_length,
                        max_string=locals_max_string,
                        use_rich=use_rich,
                    )
                    if _budget is not None and _budget.locals_bytes is not None:
                        _budget.locals_bytes -= len(key) + len(rendered)
                    frame_locals[key] = rendered

            append(
                Frame(
                    filename=filename or "?",
                    lineno=line_no,
                    name=frame_summary.f_code.co_name,
                    locals=frame_locals,
                    locals_omitted=locals_omitted,
                )
            )

        cause = getattr(exc_value, "__cause__", None)
        if cause and cause.__traceback__:
            exc_type = cause.__class__
            exc_value = cause
            traceback = cause.__traceback__
            is_cause = True
            continue

        cause = exc_value.__context__
        if (
            cause
            and cause.__traceback__
            and not getattr(exc_value, "__suppress_context__", False)
        ):
            exc_type = cause.__class__
            exc_value = cause
            traceback = cause.__traceback__
            is_cause = False
            continue

        # No cover, code is reached but coverage doesn't recognize it.
        break  # pragma: no cover

    return Trace(stacks=stacks)


class ExceptionDictTransformer:
    """
    Return a list of exception stack dictionaries for an exception.

    These dictionaries are based on :class:`Stack` instances generated by
    :func:`extract()` and can be dumped to JSON.

    Args:
        show_locals:
            Whether or not to include the values of a stack frame's local
            variables.

        locals_max_length:
            Maximum length of containers before abbreviating, or ``None`` for
            no abbreviation.

        locals_max_string:
            Maximum length of string before truncating, or ``None`` to disable
            truncating.

        locals_hide_dunder:
            Hide locals prefixed with double underscore.
            Defaults to True.

        locals_hide_sunder:
            Hide locals prefixed with single underscore.
            This implies hiding *locals_hide_dunder*.
            Defaults to False.

        suppress:
            Optional sequence of modules or paths for which to suppress the
            display of locals even if *show_locals* is ``True``.

        max_frames:
            Maximum number of frames in each stack.  Frames are removed from
            the inside out.  The idea is, that the first frames represent your
            code responsible for the exception and last frames the code where
            the exception actually happened.  With larger web frameworks, this
            does not always work, so you should stick with the default.

        max_total_frames:
            Maximum number of frames across the *whole* exception tree,
            including all stacks of exception group members and of the
            cause/context chain.  ``None`` (the default) disables the
            limit.

            Frames are allocated in extraction order: the outermost
            exception first, then exception group members in order (each
            with its own cause/context chain), then the cause/context
            chain.  If a stack's frames don't fit into the remaining
            budget, its first and last frames are kept (only the last one
            if just one fits) and a marker frame with the number of
            skipped frames is inserted in between.

        max_branches:
            Maximum number of exceptions (stacks) extracted from the whole
            exception tree, including exception group members and the
            cause/context chain.  ``None`` (the default) disables the
            limit.

            Exceptions are allocated in the same order as for
            *max_total_frames*.  Omitted exception group members are
            reported in the parent stack's ``skipped_exceptions`` count
            and an omitted cause/context chain in the last extracted
            stack's ``skipped_causes`` count.  The exception type,
            value, and notes of every extracted exception are always kept,
            so the hierarchy of the tree stays visible.

        max_locals_bytes:
            Approximate maximum number of bytes used for rendering local
            variables across the whole exception tree.  ``None`` (the
            default) disables the limit.

            Locals are rendered in extraction order and rendering stops
            once the budget is exhausted; the number of locals that were
            not rendered is reported in each frame's ``locals_omitted``
            count.  The budget is *not* enforced by rendering everything
            and truncating the result afterwards: rendering stops between
            locals, so the last rendered local may exceed the budget.

        use_rich: If ``True`` (the default), use rich_ to compute the repr of
            locals.  If ``False`` or if rich_ is not installed, fall back to
            a simpler algorithm.

    .. seealso::
        :doc:`exceptions` for a broader explanation of *structlog*'s exception
        features.

    .. versionchanged:: 24.3.0
       Added *locals_max_length*, *locals_hide_sunder*, *locals_hide_dunder*,
       *suppress* and *use_rich* arguments.

    .. versionchanged:: 25.1.0
       *locals_max_length* and *locals_max_string* may be None to disable
       truncation.

    .. versionchanged:: 25.4.0
       Handle exception groups.

    .. versionchanged:: 26.2.0
       Added *max_total_frames*, *max_branches*, and *max_locals_bytes*
       arguments that limit the size of the whole exception tree.
    """

    show_locals: bool
    locals_max_length: int
    locals_max_string: int
    locals_hide_dunder: bool
    locals_hide_sunder: bool
    suppress: Sequence[str]
    max_frames: int
    max_total_frames: int | None
    max_branches: int | None
    max_locals_bytes: int | None
    use_rich: bool

    def __init__(
        self,
        *,
        show_locals: bool = SHOW_LOCALS,
        locals_max_length: int = LOCALS_MAX_LENGTH,
        locals_max_string: int = LOCALS_MAX_STRING,
        locals_hide_dunder: bool = True,
        locals_hide_sunder: bool = False,
        suppress: Iterable[str | ModuleType] = (),
        max_frames: int = MAX_FRAMES,
        max_total_frames: int | None = None,
        max_branches: int | None = None,
        max_locals_bytes: int | None = None,
        use_rich: bool = True,
    ) -> None:
        if locals_max_length is not None and locals_max_length < 0:
            msg = f'"locals_max_length" must be >= 0: {locals_max_length}'
            raise ValueError(msg)
        if locals_max_string is not None and locals_max_string < 0:
            msg = f'"locals_max_string" must be >= 0: {locals_max_string}'
            raise ValueError(msg)
        if max_frames < 2:
            msg = f'"max_frames" must be >= 2: {max_frames}'
            raise ValueError(msg)
        if max_total_frames is not None and max_total_frames < 1:
            msg = f'"max_total_frames" must be >= 1: {max_total_frames}'
            raise ValueError(msg)
        if max_branches is not None and max_branches < 1:
            msg = f'"max_branches" must be >= 1: {max_branches}'
            raise ValueError(msg)
        if max_locals_bytes is not None and max_locals_bytes < 0:
            msg = f'"max_locals_bytes" must be >= 0: {max_locals_bytes}'
            raise ValueError(msg)
        self.show_locals = show_locals
        self.locals_max_length = locals_max_length
        self.locals_max_string = locals_max_string
        self.locals_hide_dunder = locals_hide_dunder
        self.locals_hide_sunder = locals_hide_sunder
        self.suppress = []
        for suppress_entity in suppress:
            if not isinstance(suppress_entity, str):
                if suppress_entity.__file__ is None:
                    msg = (
                        f'"suppress" item {suppress_entity!r} must be a '
                        f"module with '__file__' attribute"
                    )
                    raise ValueError(msg)
                path = os.path.dirname(suppress_entity.__file__)
            else:
                path = suppress_entity
            path = os.path.normpath(os.path.abspath(path))
            self.suppress.append(path)
        self.max_frames = max_frames
        self.max_total_frames = max_total_frames
        self.max_branches = max_branches
        self.max_locals_bytes = max_locals_bytes
        self.use_rich = use_rich

    def __call__(self, exc_info: ExcInfo) -> list[dict[str, Any]]:
        budget: _Budget | None = None
        if (
            self.max_total_frames is not None
            or self.max_branches is not None
            or self.max_locals_bytes is not None
        ):
            budget = _Budget(
                frames=self.max_total_frames,
                branches=self.max_branches,
                locals_bytes=self.max_locals_bytes,
            )

        trace = extract(
            *exc_info,
            show_locals=self.show_locals,
            locals_max_length=self.locals_max_length,
            locals_max_string=self.locals_max_string,
            locals_hide_dunder=self.locals_hide_dunder,
            locals_hide_sunder=self.locals_hide_sunder,
            use_rich=self.use_rich,
            _budget=budget,
        )

        for stack in trace.stacks:
            if len(stack.frames) <= self.max_frames:
                continue

            half = (
                self.max_frames // 2
            )  # Force int division to handle odd numbers correctly
            fake_frame = Frame(
                filename="",
                lineno=-1,
                name=f"Skipped frames: {len(stack.frames) - (2 * half)}",
            )
            stack.frames[:] = [
                *stack.frames[:half],
                fake_frame,
                *stack.frames[-half:],
            ]

        return self._as_dict(trace)

    def _as_dict(self, trace: Trace) -> list[dict[str, Any]]:
        stack_dicts = []
        for stack in trace.stacks:
            stack_dict = asdict(stack)
            for frame_dict in stack_dict["frames"]:
                if frame_dict["locals"] is None or any(
                    frame_dict["filename"].startswith(path)
                    for path in self.suppress
                ):
                    del frame_dict["locals"]
                if frame_dict["locals_omitted"] == 0:
                    del frame_dict["locals_omitted"]
            if stack_dict["skipped_exceptions"] == 0:
                del stack_dict["skipped_exceptions"]
            if stack_dict["skipped_causes"] == 0:
                del stack_dict["skipped_causes"]
            if stack.is_group:
                stack_dict["exceptions"] = [
                    self._as_dict(t) for t in stack.exceptions
                ]
            stack_dicts.append(stack_dict)
        return stack_dicts
