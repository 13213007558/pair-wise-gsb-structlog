# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Fan-out: send one log event through multiple independent processor chains.

A common use case is emitting the same event to an audit log (structured
fields preserved) and to a console (redacted and rendered as text) without
the branches leaking mutations into each other.

`FanOut` is a regular processor: put it at the end of your configured
processor chain. Shared pre-processing runs once, then the event is
dispatched to each `Branch` -- its own processor chain plus a sink.

.. versionadded:: 26.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from .exceptions import DropEvent
from .typing import EventDict, Processor, WrappedLogger


__all__ = [
    "Branch",
    "BranchResult",
    "FanOut",
    "FanOutError",
    "FanOutReentrancyError",
]


class FanOutReentrancyError(Exception):
    """
    A sink re-entered the same `FanOut` logger while it was dispatching.

    This would recurse forever, so it's detected and raised instead.

    .. versionadded:: 26.2.0
    """


@dataclass
class BranchResult:
    """
    The outcome of dispatching one event to one `Branch`.

    .. versionadded:: 26.2.0
    """

    name: str
    index: int
    status: Literal["ok", "dropped", "failed"]
    value: Any = None
    exception: BaseException | None = None


class FanOutError(Exception):
    """
    One or more fan-out branches failed under the ``"collect"`` strategy.

    Raised only after all branches ran. Carries the per-branch outcomes in
    `results`. Failed branches are never replayed automatically.

    .. versionadded:: 26.2.0
    """

    def __init__(self, results: Sequence[BranchResult]) -> None:
        self.results = list(results)
        failed = [r for r in self.results if r.status == "failed"]
        msg = (
            f"{len(failed)} of {len(self.results)} fan-out branches failed: "
            + ", ".join(f"{r.name} ({r.exception!r})" for r in failed)
        )
        super().__init__(msg)


def _identity(value: Any) -> Any:
    return value


def _snapshot(value: Any, adapter: Callable[[Any], Any]) -> Any:
    """
    Recursively copy built-in containers (dict, list); pass everything else
    through *adapter* (identity by default).

    User objects are treated as read-only: they're shared by reference and
    their ``__deepcopy__`` hooks are never called.
    """
    if isinstance(value, dict):
        return {k: _snapshot(v, adapter) for k, v in value.items()}

    if isinstance(value, list):
        return [_snapshot(v, adapter) for v in value]

    return adapter(value)


def _to_args_kw(result: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Normalize a final processor's return value like
    `BoundLoggerBase._process_event` does.
    """
    if isinstance(result, (str, bytes, bytearray)):
        return (result,), {}

    if isinstance(result, tuple):
        return result

    if isinstance(result, Mapping):
        return (), dict(result)

    msg = (
        "Last processor of a fan-out branch didn't return an appropriate "
        "value. Valid return values are a dict, a tuple of (args, kwargs), "
        "bytes, or a str."
    )
    raise ValueError(msg)


def _call_sink(
    sink: Any, method_name: str, args: tuple[Any, ...], kw: dict[str, Any]
) -> Any:
    """
    Call *sink* like a wrapped logger method if it has one, otherwise as a
    plain callable.
    """
    target = getattr(sink, method_name, None)
    if target is None:
        if not callable(sink):
            msg = (
                f"Sink {sink!r} is neither callable nor has a "
                f"{method_name!r} method."
            )
            raise TypeError(msg)
        target = sink

    return target(*args, **kw)


@dataclass
class Branch:
    """
    One fan-out destination: its own processor chain and its own sink.

    Args:
        processors:
            The branch's processor chain. Receives a private snapshot of the
            event dict.

        sink:
            The final destination. If it has a method matching the log method
            name (like a wrapped logger, e.g. `logging.Logger`), that method
            is called; otherwise the sink itself is called. Either way it
            receives ``(*args, **kwargs)`` as normalized from the last
            processor's return value.

        name:
            Used in error reporting. Defaults to ``"branch-<index>"``.

    .. versionadded:: 26.2.0
    """

    processors: Sequence[Processor]
    sink: Any
    name: str = ""


@dataclass
class FanOut:
    """
    A terminal processor that dispatches the event to multiple branches.

    Each branch gets an independent snapshot of the event dict: built-in
    containers (``dict`` / ``list``) are copied recursively, so a branch can
    mutate nested structures without affecting other branches or the bound
    context. All other objects are treated as read-only and shared by
    reference -- their ``__deepcopy__`` hooks are never called. Use
    *snapshot_adapter* to customize how such values are snapshotted.

    Must be the last processor in the configured chain: after dispatching,
    it raises `structlog.DropEvent` internally so the outer wrapped logger
    doesn't also emit the event.

    `structlog.DropEvent` raised by a branch drops only that branch's event.
    Other exceptions are handled according to *on_error*:

    - ``"fail_fast"``: the first exception propagates immediately and
      remaining branches don't run.
    - ``"collect"``: all branches run; afterwards a `FanOutError` carrying
      every `BranchResult` is raised if any branch failed. Failed branches
      are not replayed.

    If a sink logs again through the same fan-out logger, a
    `FanOutReentrancyError` is raised instead of recursing forever.

    Args:
        branches: The `Branch` instances to dispatch to, in order.

        on_error: ``"fail_fast"`` (default) or ``"collect"``.

        snapshot_adapter:
            Called with each non-container value of the event dict to
            produce its per-branch snapshot. Defaults to identity (share by
            reference).

    .. versionadded:: 26.2.0
    """

    branches: Sequence[Branch]
    on_error: Literal["fail_fast", "collect"] = "fail_fast"
    snapshot_adapter: Callable[[Any], Any] = _identity
    _active: ContextVar[bool] = field(
        init=False,
        repr=False,
        compare=False,
        default=None,  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        if self.on_error not in ("fail_fast", "collect"):
            msg = (
                f"on_error must be 'fail_fast' or 'collect', "
                f"not {self.on_error!r}."
            )
            raise ValueError(msg)

        self._active = ContextVar(
            f"structlog_fanout_active_{id(self)}", default=False
        )

    def __call__(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        if self._active.get():
            msg = (
                "Re-entrant use of a fan-out logger detected: a sink logged "
                "through the same FanOut processor while it was "
                "dispatching. Use a different logger inside sinks."
            )
            raise FanOutReentrancyError(msg)

        token = self._active.set(True)
        try:
            results = [
                self._dispatch(logger, method_name, event_dict, branch, index)
                for index, branch in enumerate(self.branches)
            ]
        finally:
            self._active.reset(token)

        if self.on_error == "collect" and any(
            r.status == "failed" for r in results
        ):
            raise FanOutError(results)

        # Terminal node: stop the outer chain so the outer wrapped logger
        # doesn't emit the event too.
        raise DropEvent

    def _dispatch(
        self,
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
        branch: Branch,
        index: int,
    ) -> BranchResult:
        name = branch.name or f"branch-{index}"
        try:
            branch_event: Any = _snapshot(event_dict, self.snapshot_adapter)
            for proc in branch.processors:
                branch_event = proc(logger, method_name, branch_event)

            args, kw = _to_args_kw(branch_event)
            value = _call_sink(branch.sink, method_name, args, kw)
        except DropEvent:
            return BranchResult(name=name, index=index, status="dropped")
        except FanOutReentrancyError:
            raise
        except Exception as e:
            if self.on_error == "fail_fast":
                raise

            return BranchResult(
                name=name, index=index, status="failed", exception=e
            )

        return BranchResult(name=name, index=index, status="ok", value=value)
