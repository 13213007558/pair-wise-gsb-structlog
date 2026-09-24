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
        {"max_branches": -1},
        {"locals_max_total_bytes": -1},
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


def _raise_deeply(depth: int) -> None:
    """
    Raise a ValueError with a traceback *depth* frames deep.
    """
    if depth <= 1:
        raise ValueError("boom")
    _raise_deeply(depth - 1)


def _raise_chained(depth: int) -> None:
    """
    Raise a RuntimeError with a __cause__ chain *depth* exceptions long.

    The top exception has a 2-frame traceback (its caller and itself) and
    every deeper cause has one frame more.
    """
    if depth == 1:
        raise ValueError("leaf")
    try:
        _raise_chained(depth - 1)
    except Exception as e:
        raise RuntimeError(f"layer {depth}") from e


def _exc_info(depth: int) -> tuple[type[BaseException], BaseException, Any]:
    """
    Call *_raise_chained* and return its exc_info.
    """
    try:
        _raise_chained(depth)
    except Exception as e:
        return (type(e), e, e.__traceback__)
    raise AssertionError("unreachable")  # pragma: no cover


class TestExceptionBudgets:
    """
    Whole-tree budgets of ExceptionDictTransformer.
    """

    def test_max_total_frames_deep_cause_chain(self) -> None:
        """
        The total number of frames across a deep cause chain is bounded by
        max_total_frames while the cause hierarchy stays visible and every
        stack that receives frames keeps its innermost one.
        """
        exc_info = _exc_info(10)

        result = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_total_frames=3
        )(exc_info)

        # The chain is fully visible: 10 stacks, all but the first are
        # causes.
        assert 10 == len(result)
        assert [False, *([True] * 9)] == [s["is_cause"] for s in result]

        # Exactly max_total_frames real frames are present.
        real_frames = [
            f for s in result for f in s["frames"] if f["lineno"] != -1
        ]
        assert 3 == len(real_frames)

        for stack in result:
            real = [f for f in stack["frames"] if f["lineno"] != -1]
            if not real:
                # All frames elided: a structured count is left behind.
                assert 1 == len(stack["frames"])
                assert stack["frames"][0]["name"].startswith(
                    "Skipped frames: "
                )
                continue
            # The innermost frame (where the exception happened) is kept.
            assert "_raise_chained" == real[-1]["name"]

        # The second stack receives a single frame from the budget: the
        # innermost one.  The elided frame is reported in place.
        assert "Skipped frames: 1" == result[1]["frames"][0]["name"]
        assert "_raise_chained" == result[1]["frames"][1]["name"]

    def test_max_total_frames_removes_frames_from_the_middle(self) -> None:
        """
        Within a stack, frames are removed from the middle out: the
        shallowest and the innermost frames are kept and the elided middle
        ones are reported in place.
        """
        try:
            _raise_deeply(10)
        except ValueError as e:
            exc_info = (type(e), e, e.__traceback__)

        result = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_total_frames=3
        )(exc_info)

        (stack,) = result
        assert 4 == len(stack["frames"])
        first, marker, *innermost = stack["frames"]
        # The shallowest frame (this test) and the two innermost ones are
        # kept.
        assert (
            "test_max_total_frames_removes_frames_from_the_middle"
            == (first["name"])
        )
        assert "_raise_deeply" == innermost[-1]["name"]
        # The 8 elided frames in the middle are reported in place.
        assert {
            "filename": "",
            "lineno": -1,
            "name": "Skipped frames: 8",
        } == marker

    def test_max_total_frames_is_smaller_than_unbudgeted(self) -> None:
        """
        The serialized size of a budgeted deep cause chain is much smaller
        than the unbudgeted one.
        """
        exc_info = _exc_info(50)

        full = tracebacks.ExceptionDictTransformer(show_locals=False)(exc_info)
        budgeted = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_total_frames=10
        )(exc_info)

        real_frames = [
            f for s in budgeted for f in s["frames"] if f["lineno"] != -1
        ]
        assert 10 == len(real_frames)

        full_size = len(json.dumps(full))
        budget_size = len(json.dumps(budgeted))
        assert budget_size < full_size

    def test_max_branches_truncates_cause_chain(self) -> None:
        """
        A cause chain beyond max_branches is replaced by a placeholder with
        structured counts of the omitted stacks and frames.
        """
        result = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_branches=2
        )(_exc_info(5))

        # 2 rendered stacks + 1 placeholder.
        assert 3 == len(result)
        assert "RuntimeError" == result[0]["exc_type"]
        assert result[1]["is_cause"]
        # The remaining 3 stacks have 2 frames each.
        assert {"truncated": {"stacks": 3, "frames": 6}} == result[2]

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
    )
    def test_max_branches_wide_exception_group(self) -> None:
        """
        In a wide exception group, children beyond max_branches are
        aggregated into one placeholder at the position where they were
        omitted.
        """
        exceptions = []
        for _ in range(20):
            try:
                _raise_deeply(2)
            except ValueError as e:
                exceptions.append(e)
        try:
            raise ExceptionGroup("wide", exceptions)  # noqa: F821
        except ExceptionGroup as e:  # noqa: F821
            exc_info = (type(e), e, e.__traceback__)

        result = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_branches=5
        )(exc_info)

        assert result[0]["is_group"]
        children = result[0]["exceptions"]
        # The group itself plus 4 children are rendered, the remaining 16
        # children (with 3 frames each) are aggregated into one placeholder.
        assert 5 == len(children)
        for child in children[:4]:
            assert "ValueError" == child[0]["exc_type"]
            assert 3 == len(child[0]["frames"])
        assert [{"truncated": {"stacks": 16, "frames": 48}}] == children[4]

        # Exactly max_branches branches were rendered.
        assert (
            5
            == sum(
                1
                for child in children[:4]
                for stack in child
                if "exc_type" in stack
            )
            + 1
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
    )
    def test_max_branches_nested_groups_keep_hierarchy(self) -> None:
        """
        Budgets are spent depth-first: a nested group's children are
        rendered before the outer group's later children, and placeholders
        appear where branches were omitted.
        """
        inner_exceptions = []
        for _ in range(3):
            try:
                _raise_deeply(1)
            except ValueError as e:
                inner_exceptions.append(e)
        inner = ExceptionGroup("inner", inner_exceptions)  # noqa: F821
        try:
            raise ExceptionGroup(  # noqa: F821
                "outer", [inner, ValueError("plain")]
            )
        except ExceptionGroup as e:  # noqa: F821
            exc_info = (type(e), e, e.__traceback__)

        result = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_branches=3
        )(exc_info)

        (outer,) = result
        assert outer["is_group"]
        # Budget: outer (1), inner group (2), its first child (3).
        (inner_stacks, plain_stacks) = outer["exceptions"]
        assert inner_stacks[0]["is_group"]
        # One child rendered, the other two aggregated (2 frames each).
        assert 2 == len(inner_stacks[0]["exceptions"])
        assert [{"truncated": {"stacks": 2, "frames": 4}}] == inner_stacks[0][
            "exceptions"
        ][1]
        # The outer group's second child is omitted entirely.  Its
        # traceback is empty because it was never raised.
        assert [{"truncated": {"stacks": 1, "frames": 0}}] == plain_stacks

    def test_budgets_with_cyclic_cause(self) -> None:
        """
        A cycle in the __cause__ chain neither hangs nor is counted twice
        when budgets are enabled.
        """
        try:
            raise ValueError("a")
        except ValueError as exc:
            a = exc
        try:
            raise ValueError("b")
        except ValueError as exc:
            b = exc
        a.__cause__ = b
        b.__cause__ = a

        result = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_branches=1
        )((type(a), a, a.__traceback__))

        # "a" is rendered; the rest of the cycle ("b"; "a" is already seen)
        # is reported as a placeholder.
        assert 2 == len(result)
        assert {"truncated": {"stacks": 1, "frames": 1}} == result[1]

    def test_budgets_with_cyclic_context(self) -> None:
        """
        A cycle in the __context__ chain terminates and is not counted
        twice when budgets are enabled.
        """
        try:
            raise ValueError("a")
        except ValueError as exc:
            a = exc
        try:
            raise ValueError("b")
        except ValueError as exc:
            b = exc
        a.__context__ = b
        b.__context__ = a

        result = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_branches=2
        )((type(a), a, a.__traceback__))

        # Both exceptions of the cycle are rendered exactly once.
        assert 2 == len(result)
        assert "a" == result[0]["exc_value"]
        assert "b" == result[1]["exc_value"]

    def test_locals_byte_budget_limits_repr_calls(self) -> None:
        """
        Once the locals byte budget is spent, no further repr()s are
        computed and the omitted locals are counted per frame.
        """

        def func() -> None:
            spam = "s" * 20  # noqa: F841
            eggs = "e" * 20  # noqa: F841
            bacon = "b" * 20  # noqa: F841
            1 / 0

        try:
            func()
        except ZeroDivisionError as e:
            exc_info = (type(e), e, e.__traceback__)

        real_to_repr = tracebacks.to_repr
        calls = []

        def counting_to_repr(obj: Any, **kwargs: Any) -> str:
            calls.append(obj)
            return real_to_repr(obj, **kwargs)

        with patch.object(tracebacks, "to_repr", counting_to_repr):
            result = tracebacks.ExceptionDictTransformer(
                show_locals=True, locals_max_total_bytes=1
            )(exc_info)

        # Exactly one local was rendered before the budget was spent.
        assert 1 == len(calls)

        rendered = omitted = 0
        for stack in result:
            for frame in stack["frames"]:
                rendered += len(frame.get("locals", {}))
                omitted += frame.get("locals_omitted", 0)
        assert 1 == rendered
        # Every other local was counted as omitted, not rendered.
        assert omitted > 0

    def test_locals_byte_budget_counts_bytes(self) -> None:
        """
        The total number of bytes spent on rendered locals stays within
        the budget plus at most one rendered local.
        """

        def func() -> None:
            spam = "s" * 50  # noqa: F841
            eggs = "e" * 50  # noqa: F841
            bacon = "b" * 50  # noqa: F841
            1 / 0

        try:
            func()
        except ZeroDivisionError as e:
            exc_info = (type(e), e, e.__traceback__)

        budget = 100
        result = tracebacks.ExceptionDictTransformer(
            show_locals=True, locals_max_total_bytes=budget
        )(exc_info)

        total = sum(
            len(value.encode("utf-8"))
            for stack in result
            for frame in stack["frames"]
            for value in frame.get("locals", {}).values()
        )
        # One rendered local can exceed the budget, but by at most its own
        # (truncated) size.
        assert total <= budget + tracebacks.LOCALS_MAX_STRING + 10
        # Some locals were omitted.
        assert any(
            frame.get("locals_omitted")
            for stack in result
            for frame in stack["frames"]
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
    )
    def test_budgets_bound_serialized_size(self) -> None:
        """
        A wide exception group with locals is bounded to a small serialized
        size when budgets are enabled.
        """
        exceptions = []
        for i in range(100):
            try:
                _raise_deeply(3)
            except ValueError as e:
                e.add_note(f"note {i}")
                exceptions.append(e)
        try:
            raise ExceptionGroup("huge", exceptions)  # noqa: F821
        except ExceptionGroup as e:  # noqa: F821
            exc_info = (type(e), e, e.__traceback__)

        full = tracebacks.ExceptionDictTransformer(show_locals=True)(exc_info)
        budgeted = tracebacks.ExceptionDictTransformer(
            show_locals=True,
            max_total_frames=10,
            max_branches=4,
            locals_max_total_bytes=200,
        )(exc_info)

        full_size = len(json.dumps(full))
        budget_size = len(json.dumps(budgeted))

        assert full_size > 10_000
        assert budget_size < 4_000
        assert budget_size < full_size // 10

    def test_generous_budgets_match_disabled_budgets(self) -> None:
        """
        Budgets that never bind don't change the output.
        """
        exc_info = _exc_info(3)

        plain = tracebacks.ExceptionDictTransformer(
            show_locals=True, max_frames=4
        )(exc_info)
        budgeted = tracebacks.ExceptionDictTransformer(
            show_locals=True,
            max_frames=4,
            max_total_frames=1_000_000,
            max_branches=1_000_000,
            locals_max_total_bytes=1_000_000,
        )(exc_info)

        assert plain == budgeted

    def test_locals_budget_still_applies_max_frames(self) -> None:
        """
        With only a locals budget enabled, max_frames is still applied to
        each stack exactly as without budgets.
        """
        try:
            _raise_deeply(10)
        except ValueError as e:
            exc_info = (type(e), e, e.__traceback__)

        plain = tracebacks.ExceptionDictTransformer(
            show_locals=False, max_frames=3
        )(exc_info)
        budgeted = tracebacks.ExceptionDictTransformer(
            show_locals=False,
            max_frames=3,
            locals_max_total_bytes=1_000_000,
        )(exc_info)

        assert plain == budgeted

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Requires Python 3.11 or higher"
    )
    def test_raising_str_and_repr_dont_break_other_branches(self) -> None:
        """
        An exception whose __str__ raises and a local whose __repr__ raises
        don't break the rendering of other branches when budgets are
        enabled.
        """

        class BadStr(Exception):
            def __str__(self) -> str:
                raise RuntimeError("no str")

        class BadRepr:
            def __repr__(self) -> str:
                raise RuntimeError("no repr")

        def func() -> None:
            bad = BadRepr()  # noqa: F841
            raise BadStr("onoes")

        exceptions = []
        for _ in range(3):
            try:
                func()
            except BadStr as e:
                exceptions.append(e)
        try:
            raise ExceptionGroup("bad ones", exceptions)  # noqa: F821
        except ExceptionGroup as e:  # noqa: F821
            exc_info = (type(e), e, e.__traceback__)

        result = tracebacks.ExceptionDictTransformer(
            show_locals=True,
            max_branches=2,
            locals_max_total_bytes=1_000_000,
        )(exc_info)

        child = result[0]["exceptions"][0][0]
        assert "BadStr" == child["exc_type"]
        assert "<str-error" in child["exc_value"]
        assert "<repr-error" in child["frames"][-1]["locals"]["bad"]
        # The remaining two children (2 frames each) become a placeholder.
        assert [{"truncated": {"stacks": 2, "frames": 4}}] == result[0][
            "exceptions"
        ][1]


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
