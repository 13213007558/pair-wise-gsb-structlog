# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

from __future__ import annotations

import asyncio
import inspect
import json
import sys

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

import structlog

from structlog import tracebacks


try:
    import rich
except ImportError:
    rich = None


class SecretStr(str):  # noqa: SLOT000
    """
    Secrets representation as used in Typed Settings or Pydantic.
    """

    def __repr__(self) -> str:
        return "*******"


@pytest.fixture(autouse=True)
def _unimport_rich() -> Iterator[None]:
    with patch.object(tracebacks, "rich", None):
        yield


def get_next_lineno() -> int:
    return inspect.currentframe().f_back.f_lineno + 1


@pytest.mark.parametrize(("data", "expected"), [(3, "3"), ("spam", "spam")])
def test_save_str(data: Any, expected: str):
    """
    "safe_str()" returns the str repr of an object.
    """
    assert expected == tracebacks.safe_str(data)


def test_safe_str_error():
    """
    "safe_str()" does not fail if __str__() raises an exception.
    """

    class Baam:
        def __str__(self) -> str:
            raise ValueError("BAAM!")

    with pytest.raises(ValueError, match="BAAM!"):
        str(Baam())

    assert "<str-error 'BAAM!'>" == tracebacks.safe_str(Baam())


@pytest.mark.parametrize(
    ("data", "max_len", "expected"),
    [
        (3, None, "3"),
        ("spam", None, "'spam'"),
        (b"spam", None, "b'spam'"),
        ("bacon", 3, "'bac'+2"),
        ("bacon", 4, "'baco'+1"),
        ("bacon", 5, "'bacon'"),
        (SecretStr("password"), None, "*******"),
        (["spam", "eggs", "bacon"], 10, "\"['spam', '\"+15"),
    ],
)
def test_to_repr(data: Any, max_len: int | None, expected: str) -> None:
    """
    "to_repr()" returns the repr of an object, trimmed to max_len.
    """
    assert expected == tracebacks.to_repr(data, max_string=max_len)


@pytest.mark.skipif(rich is None, reason="rich not installed")
@pytest.mark.parametrize(
    ("use_rich", "data", "max_len", "expected"),
    [
        (True, 3, None, "3"),
        (True, "spam", None, "'spam'"),
        (True, b"spam", None, "b'spam'"),
        (True, "bacon", 3, "'bac'+2"),
        (True, "bacon", 5, "'bacon'"),
        (True, SecretStr("password"), None, "*******"),
        (True, ["spam", "eggs", "bacon"], 4, "['spam', 'eggs', 'baco'+1]"),
        (False, "bacon", 3, "'bac'+2"),
        (False, ["spam", "eggs", "bacon"], 4, '"[\'sp"+21'),
    ],
)
def test_to_repr_rich(
    use_rich: bool,
    data: Any,
    max_len: int | None,
    expected: str,
) -> None:
    """
    "to_repr()" uses Rich to get a nice repr if it is installed and if
    "use_rich" is True.
    """
    with patch.object(tracebacks, "rich", rich):
        actual = tracebacks.to_repr(
            data, max_string=max_len, use_rich=use_rich
        )

    assert expected == actual


def test_to_repr_error() -> None:
    """
    "to_repr()" does not fail if __repr__() raises an exception.
    """

    class Baam:
        def __repr__(self) -> str:
            raise ValueError("BAAM!")

    with pytest.raises(ValueError, match="BAAM!"):
        repr(Baam())

    assert "<repr-error 'BAAM!'>" == tracebacks.to_repr(Baam())


def test_simple_exception():
    """
    Tracebacks are parsed for simple, single exceptions.
    """
    try:
        lineno = get_next_lineno()
        1 / 0
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ZeroDivisionError",
            exc_value="division by zero",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_simple_exception",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_simple_exception_with_notes():
    """
    Notes are included in the traceback.
    """
    try:
        lineno = get_next_lineno()
        1 / 0
    except Exception as e:
        e.add_note("This is a note.")
        e.add_note("This is another note.")
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ZeroDivisionError",
            exc_value="division by zero",
            exc_notes=["This is a note.", "This is another note."],
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_simple_exception_with_notes",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_raise_hide_cause():
    """
    If "raise ... from None" is used, the trace looks like from a simple
    exception.
    """
    try:
        try:
            1 / 0
        except ArithmeticError:
            lineno = get_next_lineno()
            raise ValueError("onoes") from None
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ValueError",
            exc_value="onoes",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_raise_hide_cause",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_raise_with_cause():
    """
    If "raise ... from orig" is used, the orig trace is included and marked as
    cause.
    """
    try:
        try:
            lineno_1 = get_next_lineno()
            1 / 0
        except ArithmeticError as orig_exc:
            lineno_2 = get_next_lineno()
            raise ValueError("onoes") from orig_exc
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ValueError",
            exc_value="onoes",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno_2,
                    name="test_raise_with_cause",
                    locals=None,
                ),
            ],
        ),
        tracebacks.Stack(
            exc_type="ZeroDivisionError",
            exc_value="division by zero",
            syntax_error=None,
            is_cause=True,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno_1,
                    name="test_raise_with_cause",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_raise_with_cause_no_tb():
    """
    If an exception's cause has no traceback, that cause is ignored.
    """
    try:
        lineno = get_next_lineno()
        raise ValueError("onoes") from RuntimeError("I am fake")
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ValueError",
            exc_value="onoes",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_raise_with_cause_no_tb",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_raise_nested():
    """
    If an exc is raised during handling another one, the orig trace is
    included.
    """
    try:
        try:
            lineno_1 = get_next_lineno()
            1 / 0
        except ArithmeticError:
            lineno_2 = get_next_lineno()
            raise ValueError("onoes")  # noqa: B904
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ValueError",
            exc_value="onoes",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno_2,
                    name="test_raise_nested",
                    locals=None,
                ),
            ],
        ),
        tracebacks.Stack(
            exc_type="ZeroDivisionError",
            exc_value="division by zero",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno_1,
                    name="test_raise_nested",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_raise_no_msg():
    """
    If exception classes (not instances) are raised, "exc_value" is an empty
    string.
    """
    try:
        lineno = get_next_lineno()
        raise RuntimeError
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="RuntimeError",
            exc_value="",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_raise_no_msg",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_syntax_error():
    """
    For SyntaxError, extra info about that error is added to the trace.
    """
    try:
        # raises SyntaxError: invalid syntax
        lineno = get_next_lineno()
        eval("2 +* 2")
    except SyntaxError as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="SyntaxError",
            exc_value="invalid syntax (<string>, line 1)",
            syntax_error=tracebacks.SyntaxError_(
                offset=4,
                filename="<string>",
                line="2 +* 2",
                lineno=1,
                msg="invalid syntax",
            ),
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_syntax_error",
                ),
            ],
        ),
    ] == trace.stacks


def test_filename_with_bracket():
    """
    Filenames with brackets (e.g., "<string>") are handled properly.
    """
    try:
        lineno = get_next_lineno()
        exec(compile("1/0", filename="<string>", mode="exec"))
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ZeroDivisionError",
            exc_value="division by zero",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_filename_with_bracket",
                    locals=None,
                ),
                tracebacks.Frame(
                    filename="<string>",
                    lineno=1,
                    name="<module>",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_filename_not_a_file():
    """
    "Invalid" filenames are appended to CWD as if they were actual files.
    """
    try:
        lineno = get_next_lineno()
        exec(compile("1/0", filename="string", mode="exec"))
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert [
        tracebacks.Stack(
            exc_type="ZeroDivisionError",
            exc_value="division by zero",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_filename_not_a_file",
                    locals=None,
                ),
                tracebacks.Frame(
                    filename=str(Path.cwd().joinpath("string")),
                    lineno=1,
                    name="<module>",
                    locals=None,
                ),
            ],
        ),
    ] == trace.stacks


def test_show_locals():
    """
    Local variables in each frame can optionally be captured.
    """

    def bar(a):
        print(1 / a)

    def foo(a):
        bar(a)

    try:
        foo(0)
    except Exception as e:
        trace = tracebacks.extract(
            type(e), e, e.__traceback__, show_locals=True
        )

    stack_locals = [f.locals for f in trace.stacks[0].frames]
    # The first frames contain functions with "random" memory addresses,
    # so we only check the variable names:
    assert stack_locals[0].keys() == {"foo", "e", "bar"}
    assert stack_locals[1].keys() == {"a", "bar"}
    assert stack_locals[2] == {"a": "0"}


def test_recursive():
    """
    Recursion errors give a lot of frames but don't break stuff.
    """

    def foo(n):
        return bar(n)

    def bar(n):
        return foo(n)

    try:
        lineno = get_next_lineno()
        foo(1)
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    frames = trace.stacks[0].frames
    trace.stacks[0].frames = []

    assert [
        tracebacks.Stack(
            exc_type="RecursionError",
            exc_value="maximum recursion depth exceeded",
            syntax_error=None,
            is_cause=False,
            frames=[],
        ),
    ] == trace.stacks
    assert (
        len(frames) > sys.getrecursionlimit() - 50
    )  # Buffer for frames from pytest
    assert (
        tracebacks.Frame(
            filename=__file__,
            lineno=lineno,
            name="test_recursive",
        )
        == frames[0]
    )

    # If we run the tests under Python 3.12 with sysmon enabled, it inserts
    # frames at the end.
    if sys.version_info >= (3, 12):
        frames = [f for f in frames if "coverage" not in f.filename]

    # Depending on whether we invoke pytest directly or run tox, either "foo()"
    # or "bar()" is at the end of the stack.
    assert frames[-1] in [
        tracebacks.Frame(
            filename=__file__,
            lineno=lineno - 7,
            name="foo",
        ),
        tracebacks.Frame(
            filename=__file__,
            lineno=lineno - 4,
            name="bar",
        ),
    ]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_exception_groups() -> None:
    """
    Exception groups are detected and a list of Trace instances is added to
    the exception group's Trace.
    """
    lineno = get_next_lineno()

    async def t1() -> None:
        1 / 0

    async def t2() -> None:
        raise ValueError("Blam!")

    async def main():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(t1())
            tg.create_task(t2())

    try:
        asyncio.run(main())
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert "ExceptionGroup" == trace.stacks[0].exc_type
    assert (
        "unhandled errors in a TaskGroup (2 sub-exceptions)"
        == trace.stacks[0].exc_value
    )
    exceptions = trace.stacks[0].exceptions
    assert [
        tracebacks.Trace(
            stacks=[
                tracebacks.Stack(
                    exc_type="ZeroDivisionError",
                    exc_value="division by zero",
                    exc_notes=[],
                    syntax_error=None,
                    is_cause=False,
                    frames=[
                        tracebacks.Frame(
                            filename=__file__,
                            lineno=lineno + 2,
                            name="t1",
                            locals=None,
                        )
                    ],
                    is_group=False,
                    exceptions=[],
                )
            ]
        ),
        tracebacks.Trace(
            stacks=[
                tracebacks.Stack(
                    exc_type="ValueError",
                    exc_value="Blam!",
                    exc_notes=[],
                    syntax_error=None,
                    is_cause=False,
                    frames=[
                        tracebacks.Frame(
                            filename=__file__,
                            lineno=lineno + 5,
                            name="t2",
                            locals=None,
                        )
                    ],
                    is_group=False,
                    exceptions=[],
                )
            ]
        ),
    ] == exceptions


@pytest.mark.parametrize(
    ("kwargs", "local_variable"),
    [
        (
            {"locals_max_string": None},
            "x" * (tracebacks.LOCALS_MAX_STRING + 10),
        ),
        (
            {"locals_max_length": None},
            list(range(tracebacks.LOCALS_MAX_LENGTH + 10)),
        ),
    ],
)
def test_exception_dict_transformer_locals_max_accept_none_argument(
    kwargs, local_variable
):
    """
    ExceptionDictTransformer works when either locals_max_string or
    locals_max_length is set to None.
    """

    try:
        my_local = local_variable
        1 / 0
    except Exception as e:
        format_json = tracebacks.ExceptionDictTransformer(
            show_locals=True, **kwargs
        )
        result = format_json((type(e), e, e.__traceback__))

    _ = my_local

    assert len(result) == 1
    assert len(result[0]["frames"]) == 1

    frame = result[0]["frames"][0]

    assert frame["locals"]["my_local"].strip("'") == str(local_variable)


def test_json_traceback():
    """
    Tracebacks are formatted to JSON with all information.
    """
    try:
        lineno = get_next_lineno()
        1 / 0
    except Exception as e:
        format_json = tracebacks.ExceptionDictTransformer(show_locals=False)
        result = format_json((type(e), e, e.__traceback__))

    assert [
        {
            "exc_type": "ZeroDivisionError",
            "exc_value": "division by zero",
            "exc_notes": [],
            "exceptions": [],
            "frames": [
                {
                    "filename": __file__,
                    "lineno": lineno,
                    "name": "test_json_traceback",
                }
            ],
            "is_cause": False,
            "is_group": False,
            "syntax_error": None,
        },
    ] == result


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_json_traceback_with_notes():
    """
    Tracebacks are formatted to JSON with all information.
    """
    try:
        lineno = get_next_lineno()
        1 / 0
    except Exception as e:
        e.add_note("This is a note.")
        e.add_note("This is another note.")
        format_json = tracebacks.ExceptionDictTransformer(show_locals=False)
        result = format_json((type(e), e, e.__traceback__))

    assert [
        {
            "exc_type": "ZeroDivisionError",
            "exc_value": "division by zero",
            "exc_notes": ["This is a note.", "This is another note."],
            "exceptions": [],
            "frames": [
                {
                    "filename": __file__,
                    "lineno": lineno,
                    "name": "test_json_traceback_with_notes",
                }
            ],
            "is_cause": False,
            "is_group": False,
            "syntax_error": None,
        },
    ] == result


def test_json_traceback_locals_max_string():
    """
    Local variables in each frame are trimmed to locals_max_string.
    """
    try:
        _var = "spamspamspam"
        lineno = get_next_lineno()
        1 / 0
    except Exception as e:
        result = tracebacks.ExceptionDictTransformer(locals_max_string=4)(
            (type(e), e, e.__traceback__)
        )
    assert [
        {
            "exc_type": "ZeroDivisionError",
            "exc_value": "division by zero",
            "exc_notes": [],
            "exceptions": [],
            "frames": [
                {
                    "filename": __file__,
                    "lineno": lineno,
                    "locals": {
                        "_var": "'spam'+8",
                        "e": "'Zero'+33",
                        "lineno": str(lineno),
                    },
                    "name": "test_json_traceback_locals_max_string",
                }
            ],
            "is_cause": False,
            "is_group": False,
            "syntax_error": None,
        },
    ] == result


@pytest.mark.parametrize(
    ("max_frames", "expected_frames", "skipped_idx", "skipped_count"),
    [
        (2, 3, 1, 2),
        (3, 3, 1, 2),
        (4, 4, -1, 0),
        (5, 4, -1, 0),
    ],
)
def test_json_traceback_max_frames(
    max_frames: int, expected_frames: int, skipped_idx: int, skipped_count: int
):
    """
    Only max_frames frames are included in the traceback and the skipped frames
    are reported.
    """

    def spam():
        return 1 / 0

    def eggs():
        spam()

    def bacon():
        eggs()

    try:
        bacon()
    except Exception as e:
        format_json = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_frames=max_frames
        )
        result = format_json((type(e), e, e.__traceback__))
        trace = result[0]
        assert len(trace["frames"]) == expected_frames, trace["frames"]
        if skipped_count:
            assert trace["frames"][skipped_idx] == {
                "filename": "",
                "lineno": -1,
                "name": f"Skipped frames: {skipped_count}",
            }
        else:
            assert not any(f["lineno"] == -1 for f in trace["frames"])


@pytest.mark.parametrize(
    ("suppress", "file_no_locals"),
    [
        pytest.param((__file__,), __file__, id="file"),
        pytest.param((json,), json.__file__, id="json"),
    ],
)
def test_json_tracebacks_suppress(
    suppress: tuple[str | ModuleType, ...],
    file_no_locals: str,
    capsys: pytest.CaptureFixture,
) -> None:
    """
    Console and JSON output look as expected

    This means also warnings are logged correctly.
    """
    try:
        try:
            json.loads(42)  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError("error shown to users") from e
    except ValueError as e:
        format_json = tracebacks.ExceptionDictTransformer(
            show_locals=True, suppress=suppress
        )
        result = format_json((type(e), e, e.__traceback__))
        for stack in result:
            for frame in stack["frames"]:
                no_locals = frame["filename"] == file_no_locals
                if no_locals:
                    assert "locals" not in frame
                else:
                    assert "locals" in frame


@pytest.mark.parametrize(
    ("hide_sunder", "hide_dunder", "expected"),
    [
        (False, False, {"_spam", "__eggs"}),
        (True, False, set()),  # Also hides "__eggs", b/c it starts with "_"!
        (False, True, {"_spam"}),
        (True, True, set()),
    ],
)
def test_json_tracebacks_skip_sunder_dunder(
    hide_sunder: bool, hide_dunder: bool, expected: set[str]
) -> None:
    """
    Local variables starting with "_" or "__" can be hidden from the locals
    dict.
    """

    def func() -> None:
        _spam = True
        __eggs = 3
        1 / 0

    try:
        func()
    except ZeroDivisionError as e:
        format_json = tracebacks.ExceptionDictTransformer(
            show_locals=True,
            locals_hide_sunder=hide_sunder,
            locals_hide_dunder=hide_dunder,
        )
        result = format_json((type(e), e, e.__traceback__))
        assert expected == set(result[0]["frames"][1]["locals"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"locals_max_length": -1},
        {"locals_max_string": -1},
        {"max_frames": -1},
        {"max_frames": 0},
        {"max_frames": 1},
        {"max_total_frames": -1},
        {"max_total_frames": 0},
        {"max_branches": -1},
        {"max_branches": 0},
        {"max_locals_bytes": -1},
    ],
)
def test_json_traceback_value_error(kwargs: dict[str, Any]) -> None:
    """
    Wrong arguments to ExceptionDictTransformer raise a ValueError that
    contains the name of the argument.
    """
    with pytest.raises(ValueError, match=next(iter(kwargs.keys()))):
        tracebacks.ExceptionDictTransformer(**kwargs)


def test_json_traceback_suppress_without_file() -> None:
    """
    Suppressing a module without a __file__ raises a ValueError that contains
    the name of the argument.
    """
    with (
        patch.object(json, "__file__", None),
        pytest.raises(ValueError, match="suppress"),
    ):
        tracebacks.ExceptionDictTransformer(suppress=(json,))


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_json_exception_groups() -> None:
    """
    When rendered as JSON, the "Trace.stacks" is stripped, so "exceptions" is a
    list of lists and not a list of objects (with a single "stacks" attribute.
    """

    lineno = get_next_lineno()

    async def t1() -> None:
        1 / 0

    async def t2() -> None:
        raise ValueError("Blam!")

    async def main():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(t1())
            tg.create_task(t2())

    try:
        asyncio.run(main())
    except Exception as e:
        format_json = tracebacks.ExceptionDictTransformer(show_locals=False)
        result = format_json((type(e), e, e.__traceback__))

    assert "ExceptionGroup" == result[0]["exc_type"]
    exceptions = result[0]["exceptions"]
    assert [
        [
            {
                "exc_type": "ZeroDivisionError",
                "exc_value": "division by zero",
                "exc_notes": [],
                "syntax_error": None,
                "is_cause": False,
                "frames": [
                    {
                        "filename": __file__,
                        "lineno": lineno + 2,
                        "name": "t1",
                    }
                ],
                "is_group": False,
                "exceptions": [],
            }
        ],
        [
            {
                "exc_type": "ValueError",
                "exc_value": "Blam!",
                "exc_notes": [],
                "syntax_error": None,
                "is_cause": False,
                "frames": [
                    {
                        "filename": __file__,
                        "lineno": lineno + 5,
                        "name": "t2",
                    }
                ],
                "is_group": False,
                "exceptions": [],
            }
        ],
    ] == exceptions


class TestLogException:
    """
    Higher level integration tests for `Logger.exception()`.
    """

    @pytest.fixture
    def cap_logs(self) -> structlog.testing.LogCapture:
        """
        Create a LogCapture to be used as processor and fixture for retrieving
        logs in tests.
        """
        return structlog.testing.LogCapture()

    @pytest.fixture
    def logger(
        self, cap_logs: structlog.testing.LogCapture
    ) -> structlog.Logger:
        """
        Create a logger with the dict_tracebacks and a LogCapture processor.
        """
        old_processors = structlog.get_config()["processors"]
        structlog.configure([structlog.processors.dict_tracebacks, cap_logs])
        logger = structlog.get_logger("dict_tracebacks")
        try:
            yield logger
        finally:
            structlog.configure(processors=old_processors)

    def test_log_explicit_exception(
        self, logger: structlog.Logger, cap_logs: structlog.testing.LogCapture
    ) -> None:
        """
        The log row contains a traceback when `Logger.exception()` is
        explicitly called with an exception instance.
        """
        try:
            1 / 0
        except ZeroDivisionError as e:
            logger.exception("onoes", exception=e)

        log_row = cap_logs.entries[0]

        assert log_row["exception"][0]["exc_type"] == "ZeroDivisionError"

    def test_log_implicit_exception(
        self, logger: structlog.Logger, cap_logs: structlog.testing.LogCapture
    ) -> None:
        """
        The log row contains a traceback when `Logger.exception()` is called
        in an `except` block but without explicitly passing the exception.
        """
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception("onoes")

        log_row = cap_logs.entries[0]

        assert log_row["exception"][0]["exc_type"] == "ZeroDivisionError"

    def test_no_exception(
        self, logger: structlog.Logger, cap_logs: structlog.testing.LogCapture
    ) -> None:
        """
        `Logger.exception()` should not be called outside an `except` block
        but this cases is gracefully handled and does not lead to an exception.

        See: https://github.com/hynek/structlog/issues/634
        """
        logger.error("onoes")

        assert [{"event": "onoes", "log_level": "error"}] == cap_logs.entries


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_reraise_error_from_exception_group():
    """
    There is no RecursionError when building the traceback for an exception
    that has been re-raised from an ExceptionGroup.
    """
    inner_lineno = None
    lineno = None

    try:
        try:
            inner_lineno = get_next_lineno()
            raise ExceptionGroup(  # noqa: F821
                "Some error occurred",
                [ValueError("value error")],
            )
        except ExceptionGroup as e:  # noqa: F821
            lineno = get_next_lineno()
            raise e.exceptions[0]  # noqa: B904
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert lineno is not None
    assert inner_lineno is not None
    assert 2 == len(trace.stacks)
    assert lineno == trace.stacks[0].frames[0].lineno
    assert (
        tracebacks.Stack(
            exc_type="ValueError",
            exc_value="value error",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_reraise_error_from_exception_group",
                    locals=None,
                )
            ],
            is_group=False,
            exceptions=[],
        )
        == trace.stacks[0]
    )
    assert (
        tracebacks.Stack(
            exc_type="ExceptionGroup",
            exc_value="Some error occurred (1 sub-exception)",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=inner_lineno,
                    name="test_reraise_error_from_exception_group",
                    locals=None,
                ),
            ],
            is_group=True,
            exceptions=[tracebacks.Trace(stacks=[])],
        )
        == trace.stacks[1]
    )


def test_exception_cycle():
    """
    There is no RecursionError when building the traceback for an exception
    that has itself in its cause chain.
    """
    inner_lineno = None
    lineno = None

    try:
        try:
            exc = ValueError("onoes")
            inner_lineno = get_next_lineno()
            raise exc
        except Exception as exc:
            lineno = get_next_lineno()
            raise exc from exc
    except Exception as e:
        trace = tracebacks.extract(type(e), e, e.__traceback__)

    assert lineno is not None
    assert inner_lineno is not None
    assert 1 == len(trace.stacks)
    assert lineno == trace.stacks[0].frames[0].lineno
    assert (
        tracebacks.Stack(
            exc_type="ValueError",
            exc_value="onoes",
            syntax_error=None,
            is_cause=False,
            frames=[
                tracebacks.Frame(
                    filename=__file__,
                    lineno=lineno,
                    name="test_exception_cycle",
                    locals=None,
                ),
                tracebacks.Frame(
                    filename=__file__,
                    lineno=inner_lineno,
                    name="test_exception_cycle",
                    locals=None,
                ),
            ],
            is_group=False,
            exceptions=[],
        )
        == trace.stacks[0]
    )


def _raise_chain(depth: int) -> Exception:
    """
    Raise and return an exception with a cause chain *depth* exceptions
    long.  Every exception has exactly one frame.
    """
    try:
        raise ValueError("leaf")
    except ValueError as e:
        exc = e
    for level in range(depth - 1):
        try:
            raise RuntimeError(f"level {level}") from exc
        except RuntimeError as e:
            exc = e
    return exc


def _raise_group(width: int) -> Exception:
    """
    Raise and return an ExceptionGroup with *width* members, each with a
    traceback of one frame.
    """
    exceptions = []
    for member in range(width):
        try:
            raise ValueError(f"member {member}")
        except ValueError as e:
            exceptions.append(e)
    return ExceptionGroup("wide", exceptions)  # noqa: F821


class _CountedRepr:
    """
    A class that counts how often its repr() is called.
    """

    renders = 0

    def __repr__(self) -> str:
        type(self).renders += 1
        return "x" * 10


def _locals_budget_inner() -> None:
    aa = _CountedRepr()
    bb = _CountedRepr()
    raise ValueError("boom")


def _locals_budget_outer() -> None:
    cc = _CountedRepr()
    _locals_budget_inner()


def test_budget_deep_cause_chain_branches() -> None:
    """
    A deep cause chain is truncated to max_branches stacks and the rest of
    the chain is reported in the last stack's "skipped_causes" count.  The
    hierarchy of the extracted part stays intact.
    """
    exc = _raise_chain(10)
    transformer = tracebacks.ExceptionDictTransformer(max_branches=3)

    result = transformer((type(exc), exc, exc.__traceback__))

    assert 3 == len(result)
    assert [False, True, True] == [stack["is_cause"] for stack in result]
    assert "RuntimeError" == result[0]["exc_type"]
    assert "level 8" == result[0]["exc_value"]
    # Every extracted exception keeps its overview and its frame.
    assert all(stack["frames"] for stack in result)
    assert "skipped_causes" not in result[0]
    assert "skipped_causes" not in result[1]
    assert 7 == result[2]["skipped_causes"]


def test_budget_total_frames_allocation_order() -> None:
    """
    The total frame budget is allocated to stacks in extraction order.
    Stacks that get no budget anymore only contain a marker frame with the
    number of skipped frames.
    """
    exc = _raise_chain(5)
    transformer = tracebacks.ExceptionDictTransformer(max_total_frames=2)

    result = transformer((type(exc), exc, exc.__traceback__))

    assert 5 == len(result)
    real_frames = [
        frame["name"]
        for stack in result
        for frame in stack["frames"]
        if frame["lineno"] != -1
    ]
    assert 2 == len(real_frames)
    assert result[0]["frames"][0]["lineno"] != -1
    assert result[1]["frames"][0]["lineno"] != -1
    marker = {"filename": "", "lineno": -1, "name": "Skipped frames: 1"}
    for stack in result[2:]:
        assert [marker] == stack["frames"]


def test_budget_total_frames_keeps_first_and_last() -> None:
    """
    If a stack's frames don't fit into the remaining total frame budget,
    the first and last frames are kept and the skipped middle is reported.
    """
    raise_lineno = None

    def dive(depth: int) -> None:
        nonlocal raise_lineno
        if depth == 0:
            raise_lineno = get_next_lineno()
            raise ValueError("deep")
        dive(depth - 1)

    try:
        dive(9)
    except ValueError as e:
        transformer = tracebacks.ExceptionDictTransformer(max_total_frames=4)
        result = transformer((type(e), e, e.__traceback__))

    frames = result[0]["frames"]
    assert 5 == len(frames)
    assert "test_budget_total_frames_keeps_first_and_last" == frames[0]["name"]
    assert "dive" == frames[1]["name"]
    assert {
        "filename": "",
        "lineno": -1,
        "name": "Skipped frames: 7",
    } == frames[2]
    assert "dive" == frames[3]["name"]
    assert "dive" == frames[4]["name"]
    assert raise_lineno == frames[4]["lineno"]


def test_budget_total_frames_keeps_innermost_if_only_one_fits() -> None:
    """
    If only one frame fits into the remaining total frame budget, the last
    (innermost) frame is kept.
    """
    raise_lineno = None

    def dive(depth: int) -> None:
        nonlocal raise_lineno
        if depth == 0:
            raise_lineno = get_next_lineno()
            raise ValueError("deep")
        dive(depth - 1)

    try:
        dive(9)
    except ValueError as e:
        transformer = tracebacks.ExceptionDictTransformer(max_total_frames=1)
        result = transformer((type(e), e, e.__traceback__))

    frames = result[0]["frames"]
    assert 2 == len(frames)
    assert {
        "filename": "",
        "lineno": -1,
        "name": "Skipped frames: 10",
    } == frames[0]
    assert "dive" == frames[1]["name"]
    assert raise_lineno == frames[1]["lineno"]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_budget_wide_exception_group_branches() -> None:
    """
    A wide exception group is truncated to max_branches members and the
    remaining members are reported in the group's "skipped_exceptions"
    count.  The group hierarchy stays visible.
    """
    exc = _raise_group(100)
    transformer = tracebacks.ExceptionDictTransformer(max_branches=10)

    result = transformer((type(exc), exc, exc.__traceback__))

    assert 1 == len(result)
    (group,) = result
    assert group["is_group"]
    assert "ExceptionGroup" == group["exc_type"]
    assert 9 == len(group["exceptions"])
    assert 91 == group["skipped_exceptions"]
    # Each extracted member keeps its own structure.
    for member_trace in group["exceptions"]:
        assert "ValueError" == member_trace[0]["exc_type"]
        assert 1 == len(member_trace[0]["frames"])


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_budget_nested_exception_groups() -> None:
    """
    Skipped exception group members are counted including their nested
    subtrees, and the counts are attached where the elision happened.
    """
    inner = ExceptionGroup("inner", _raise_group(3).exceptions)  # noqa: F821
    exc = ExceptionGroup(  # noqa: F821
        "outer", [inner, *_raise_group(2).exceptions]
    )

    # Only the root fits: all members (including the nested group and its
    # members) are counted on the root.
    transformer = tracebacks.ExceptionDictTransformer(max_branches=1)
    result = transformer((type(exc), exc, exc.__traceback__))

    assert 1 == len(result)
    assert [] == result[0]["exceptions"]
    assert 6 == result[0]["skipped_exceptions"]

    # The nested group is extracted, but its members are counted on it.
    transformer = tracebacks.ExceptionDictTransformer(max_branches=2)
    result = transformer((type(exc), exc, exc.__traceback__))

    (inner_trace,) = result[0]["exceptions"]
    assert inner_trace[0]["is_group"]
    assert 3 == inner_trace[0]["skipped_exceptions"]
    assert 2 == result[0]["skipped_exceptions"]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_budgets_are_opt_in() -> None:
    """
    Without budget arguments, all branches are extracted and no elision
    counts appear.
    """
    exc = _raise_group(50)
    transformer = tracebacks.ExceptionDictTransformer()

    result = transformer((type(exc), exc, exc.__traceback__))

    assert 50 == len(result[0]["exceptions"])
    assert "skipped_exceptions" not in result[0]
    assert "skipped_causes" not in result[0]
    for member_trace in result[0]["exceptions"]:
        assert "locals_omitted" not in member_trace[0]["frames"][0]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
def test_budget_bounds_serialized_size() -> None:
    """
    The combination of all budgets bounds the real serialized size of a
    wide exception group with locals.
    """

    def raise_with_locals(member: int) -> Exception:
        payload = "x" * 100
        try:
            raise ValueError(f"member {member}")
        except ValueError as e:
            return e

    exc = ExceptionGroup(  # noqa: F821
        "wide", [raise_with_locals(member) for member in range(200)]
    )
    exc_info = (type(exc), exc, exc.__traceback__)

    unbounded = tracebacks.ExceptionDictTransformer(show_locals=True)(exc_info)
    transformer = tracebacks.ExceptionDictTransformer(
        show_locals=True,
        max_total_frames=20,
        max_branches=10,
        max_locals_bytes=500,
    )
    result = transformer(exc_info)

    unbounded_size = len(json.dumps(unbounded))
    bounded_size = len(json.dumps(result))
    assert unbounded_size > 20_000
    assert bounded_size < 5_000
    assert 191 == result[0]["skipped_exceptions"]
    # Locals rendering stopped once the budget was exhausted.
    omitted = [
        frame["locals_omitted"]
        for trace in result[0]["exceptions"]
        for stack in trace
        for frame in stack["frames"]
        if "locals_omitted" in frame
    ]
    assert omitted and all(count > 0 for count in omitted)


def test_budget_locals_bytes_stops_rendering() -> None:
    """
    The locals budget stops rendering locals as soon as it is exhausted
    and reports the number of omitted locals per frame.  repr() is not
    called for omitted locals.
    """
    try:
        _locals_budget_outer()
    except ValueError as e:
        # Drop the test function's own frame for deterministic locals.
        exc_info = (type(e), e, e.__traceback__.tb_next)

    _CountedRepr.renders = 0
    # Each local costs 12 bytes: 2 for the key, 10 for the value.
    transformer = tracebacks.ExceptionDictTransformer(
        show_locals=True,
        locals_max_string=None,
        max_locals_bytes=24,
        use_rich=False,
    )
    result = transformer(exc_info)

    assert 2 == _CountedRepr.renders
    outer_frame, inner_frame = result[0]["frames"]
    assert {"cc": "xxxxxxxxxx"} == outer_frame["locals"]
    assert "locals_omitted" not in outer_frame
    assert {"aa": "xxxxxxxxxx"} == inner_frame["locals"]
    assert 1 == inner_frame["locals_omitted"]

    # A budget of zero renders nothing at all.
    transformer = tracebacks.ExceptionDictTransformer(
        show_locals=True, max_locals_bytes=0, use_rich=False
    )
    result = transformer(exc_info)

    assert 2 == _CountedRepr.renders
    outer_frame, inner_frame = result[0]["frames"]
    assert {} == outer_frame["locals"]
    assert 1 == outer_frame["locals_omitted"]
    assert {} == inner_frame["locals"]
    assert 2 == inner_frame["locals_omitted"]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
)
@pytest.mark.parametrize("use_rich", [True, False])
def test_budget_broken_str_and_repr(use_rich: bool) -> None:
    """
    An exception whose __str__() raises and a local whose __repr__()
    raises don't break the extraction of the other branches.
    """

    class BadStr(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no str")

    class BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    def raise_bad() -> Exception:
        bad = BadRepr()
        try:
            raise BadStr("bad")
        except BadStr as e:
            return e

    def raise_good() -> Exception:
        try:
            raise ValueError("good")
        except ValueError as e:
            return e

    exc = ExceptionGroup("mixed", [raise_bad(), raise_good()])  # noqa: F821
    transformer = tracebacks.ExceptionDictTransformer(
        show_locals=True,
        max_total_frames=100,
        max_branches=100,
        max_locals_bytes=1_000,
        use_rich=use_rich,
    )
    with patch.object(tracebacks, "rich", rich if use_rich else None):
        result = transformer((type(exc), exc, exc.__traceback__))

    bad_trace, good_trace = result[0]["exceptions"]
    assert "BadStr" == bad_trace[0]["exc_type"]
    assert "<str-error" in bad_trace[0]["exc_value"]
    assert "<repr-error" in bad_trace[0]["frames"][0]["locals"]["bad"]
    assert "ValueError" == good_trace[0]["exc_type"]
    assert "good" == good_trace[0]["exc_value"]


def test_budget_context_cycle() -> None:
    """
    Budgets and elision counts terminate and stay correct even if the
    context chain contains a cycle.
    """
    try:
        raise ValueError("a")
    except ValueError as a_exc:
        try:
            raise TypeError("b")
        except TypeError as b_exc:
            a_exc.__context__ = b_exc
            b_exc.__context__ = a_exc
            exc_info = (type(a_exc), a_exc, a_exc.__traceback__)

    transformer = tracebacks.ExceptionDictTransformer(max_branches=1)
    result = transformer(exc_info)

    assert 1 == len(result)
    assert "ValueError" == result[0]["exc_type"]
    assert 1 == result[0]["skipped_causes"]

    # Without a branch limit, the cycle is extracted once and stops.
    transformer = tracebacks.ExceptionDictTransformer(
        max_branches=10, max_total_frames=10
    )
    result = transformer(exc_info)

    assert 2 == len(result)
    assert "ValueError" == result[0]["exc_type"]
    assert "TypeError" == result[1]["exc_type"]
    assert "skipped_causes" not in result[1]


def test_budget_self_cause_cycle() -> None:
    """
    An exception that is its own cause doesn't lead to an infinite loop
    when budgets are enabled.
    """
    try:
        try:
            exc = ValueError("onoes")
            raise exc
        except Exception as exc:
            raise exc from exc
    except Exception as e:
        exc_info = (type(e), e, e.__traceback__)

    result = tracebacks.ExceptionDictTransformer(
        max_branches=10, max_total_frames=10
    )(exc_info)

    assert 1 == len(result)
    assert "ValueError" == result[0]["exc_type"]
