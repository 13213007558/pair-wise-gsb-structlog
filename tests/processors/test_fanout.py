# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

from __future__ import annotations

import copy
import json
import logging
import sys
import threading

from io import StringIO

import pytest

import structlog

from structlog.exceptions import DropEvent, FanOutError
from structlog.processors import Branch, FanOut, JSONRenderer, safe_deepcopy
from structlog.stdlib import ProcessorFormatter
from structlog.testing import CapturingLogger


def text_renderer(logger, method_name, event_dict):
    return "text:" + event_dict["event"]


def boom(logger, method_name, event_dict):
    raise ValueError("boom")


class TestFanOut:
    def test_common_processors_run_once_per_log_call(self):
        """
        Processors before FanOut run exactly once per log call, no matter
        how many branches follow. Each branch renders with its own chain
        into its own logger and the outer logger stays silent.
        """
        common_calls = []
        json_io, text_io, outer_io = StringIO(), StringIO(), StringIO()

        def common(logger, method_name, event_dict):
            common_calls.append(event_dict)
            event_dict["common"] = True
            return event_dict

        structlog.configure(
            processors=[
                common,
                FanOut(
                    Branch(structlog.PrintLogger(json_io), [JSONRenderer()]),
                    (structlog.PrintLogger(text_io), [text_renderer]),
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(outer_io),
        )

        structlog.get_logger().info("hello", answer=42)

        assert 1 == len(common_calls)
        assert {
            "event": "hello",
            "answer": 42,
            "common": True,
        } == json.loads(json_io.getvalue())
        assert "text:hello" == text_io.getvalue().strip()
        assert "" == outer_io.getvalue()

    def test_branches_are_isolated(self):
        """
        A branch mutating nested values affects neither the other branch
        nor the bound logger's long-term context.
        """
        seen_in_b = []

        def mutate_nested(logger, method_name, event_dict):
            event_dict["nested"]["x"] = "mutated"
            event_dict["branch"] = "a"
            return event_dict

        def capture(logger, method_name, event_dict):
            seen_in_b.append(copy.deepcopy(event_dict))
            raise DropEvent

        a_io = StringIO()
        structlog.configure(
            processors=[
                FanOut(
                    (
                        structlog.PrintLogger(a_io),
                        [mutate_nested, JSONRenderer()],
                    ),
                    (CapturingLogger(), [capture]),
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        log = structlog.get_logger(nested={"x": "original"})
        log.info("hello")

        assert "original" == seen_in_b[0]["nested"]["x"]
        assert "branch" not in seen_in_b[0]
        assert {"x": "original"} == structlog.get_context(log)["nested"]
        assert "mutated" in a_io.getvalue()

    def test_drop_event_only_drops_own_branch(self):
        """
        DropEvent raised in one branch does not prevent the other branches
        from running.
        """
        b_io = StringIO()

        def drop(logger, method_name, event_dict):
            raise DropEvent

        structlog.configure(
            processors=[
                FanOut(
                    (structlog.PrintLogger(StringIO()), [drop]),
                    (structlog.PrintLogger(b_io), [JSONRenderer()]),
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        structlog.get_logger().info("hello")

        assert "hello" in b_io.getvalue()

    def test_error_raise_stops_remaining_branches(self):
        """
        With on_error="raise" (the default), the first exception
        propagates immediately and remaining branches are skipped.
        """
        ran_b = False

        def mark_b(logger, method_name, event_dict):
            nonlocal ran_b
            ran_b = True
            return event_dict

        structlog.configure(
            processors=[
                FanOut(
                    (structlog.PrintLogger(StringIO()), [boom]),
                    (structlog.PrintLogger(StringIO()), [mark_b]),
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        with pytest.raises(ValueError, match="boom"):
            structlog.get_logger().info("hello")

        assert not ran_b

    def test_error_collect_aggregates_after_all_branches(self):
        """
        With on_error="collect", all branches run and the errors are
        aggregated into a FanOutError afterwards.
        """
        ran_b = False
        ok_io = StringIO()

        def mark_b(logger, method_name, event_dict):
            nonlocal ran_b
            ran_b = True
            raise KeyError("b failed")

        structlog.configure(
            processors=[
                FanOut(
                    (structlog.PrintLogger(StringIO()), [boom]),
                    (structlog.PrintLogger(StringIO()), [mark_b]),
                    (structlog.PrintLogger(ok_io), [JSONRenderer()]),
                    on_error="collect",
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        with pytest.raises(FanOutError) as ei:
            structlog.get_logger().info("hello")

        assert 2 == len(ei.value.errors)
        assert isinstance(ei.value.errors[0], ValueError)
        assert isinstance(ei.value.errors[1], KeyError)
        assert ran_b
        assert "hello" in ok_io.getvalue()

    def test_invalid_on_error(self):
        """
        An unknown on_error value is rejected.
        """
        with pytest.raises(ValueError, match="on_error"):
            FanOut((CapturingLogger(), []), on_error="yolo")

    def test_custom_copier_is_used_per_branch(self):
        """
        The copier is called once per branch with the common event dict.
        """
        copies = []

        def copier(event_dict):
            copies.append(event_dict)
            return dict(event_dict)

        structlog.configure(
            processors=[
                FanOut(
                    (structlog.PrintLogger(StringIO()), [JSONRenderer()]),
                    (structlog.PrintLogger(StringIO()), [JSONRenderer()]),
                    copier=copier,
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        structlog.get_logger().info("hello")

        assert 2 == len(copies)

    def test_dispatch_return_forms(self):
        """
        Branch-final processors may return a str, a tuple of
        (args, kwargs), or a dict of kwargs -- just like regular final
        processors.
        """
        str_logger = CapturingLogger()
        tuple_logger = CapturingLogger()
        dict_logger = CapturingLogger()

        structlog.configure(
            processors=[
                FanOut(
                    (str_logger, [text_renderer]),
                    (tuple_logger, [lambda *_: (("args!",), {"kw": 1})]),
                    (dict_logger, [lambda *_: {"event": "as-kwarg"}]),
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        structlog.get_logger().info("hello")

        assert [("info", ("text:hello",), {})] == [
            (c.method_name, c.args, c.kwargs) for c in str_logger.calls
        ]
        assert [("info", ("args!",), {"kw": 1})] == [
            (c.method_name, c.args, c.kwargs) for c in tuple_logger.calls
        ]
        assert [("info", (), {"event": "as-kwarg"})] == [
            (c.method_name, c.args, c.kwargs) for c in dict_logger.calls
        ]

    def test_dispatch_invalid_return_value(self):
        """
        A branch-final processor returning something not dispatchable
        raises a ValueError.
        """
        structlog.configure(
            processors=[
                FanOut((CapturingLogger(), [lambda *_: 42])),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        with pytest.raises(ValueError, match="didn't return an appropriate"):
            structlog.get_logger().info("hello")

    def test_uncopyable_values_are_shared(self):
        """
        Values that can't be deep-copied (like exc_info tuples containing
        tracebacks) don't break fan-out; they are shared by reference.
        """
        b_io = StringIO()

        def mutate(logger, method_name, event_dict):
            event_dict["nested"]["x"] = "mutated"
            return event_dict

        structlog.configure(
            processors=[
                FanOut(
                    (
                        structlog.PrintLogger(StringIO()),
                        [mutate, JSONRenderer()],
                    ),
                    (structlog.PrintLogger(b_io), [JSONRenderer()]),
                ),
            ],
            logger_factory=lambda: structlog.PrintLogger(StringIO()),
        )

        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()

        structlog.get_logger().info(
            "hello", exc_info=exc_info, nested={"x": "original"}
        )

        out = json.loads(b_io.getvalue())
        assert "original" == out["nested"]["x"]

    def test_processor_formatter_branches(self):
        """
        Branches can end in ProcessorFormatter.wrap_for_formatter and
        dispatch to stdlib loggers with ProcessorFormatter handlers.
        """
        json_io, text_io = StringIO(), StringIO()

        json_handler = logging.StreamHandler(json_io)
        json_handler.setFormatter(
            ProcessorFormatter(
                processors=[
                    ProcessorFormatter.remove_processors_meta,
                    JSONRenderer(),
                ]
            )
        )
        text_handler = logging.StreamHandler(text_io)
        text_handler.setFormatter(
            ProcessorFormatter(
                processors=[
                    ProcessorFormatter.remove_processors_meta,
                    text_renderer,
                ]
            )
        )

        json_logger = logging.getLogger("test.fanout.json")
        json_logger.addHandler(json_handler)
        json_logger.setLevel(logging.DEBUG)
        json_logger.propagate = False

        text_logger = logging.getLogger("test.fanout.text")
        text_logger.addHandler(text_handler)
        text_logger.setLevel(logging.DEBUG)
        text_logger.propagate = False

        common_calls = []

        def common(logger, method_name, event_dict):
            common_calls.append(event_dict)
            return event_dict

        structlog.configure(
            processors=[
                common,
                FanOut(
                    (json_logger, [ProcessorFormatter.wrap_for_formatter]),
                    (text_logger, [ProcessorFormatter.wrap_for_formatter]),
                ),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )

        structlog.get_logger("test.fanout.main").info("hello", answer=42)

        assert 1 == len(common_calls)
        assert 42 == json.loads(json_io.getvalue())["answer"]
        assert "text:hello" == text_io.getvalue().strip()

    def test_record_not_mutated_by_repeated_formatting(self):
        """
        A LogRecord formatted by multiple ProcessorFormatter handlers
        keeps its msg/args intact, so every handler sees the same event
        dict.
        """
        io1, io2 = StringIO(), StringIO()
        logger = logging.getLogger("test.fanout.dup")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        for io in (io1, io2):
            handler = logging.StreamHandler(io)
            handler.setFormatter(
                ProcessorFormatter(
                    processors=[
                        ProcessorFormatter.remove_processors_meta,
                        JSONRenderer(),
                    ]
                )
            )
            logger.addHandler(handler)

        event_dict = {"event": "hello", "answer": 42}
        args, kwargs = ProcessorFormatter.wrap_for_formatter(
            logger, "info", event_dict
        )
        logger.info(*args, **kwargs)

        assert {"event": "hello", "answer": 42} == event_dict
        assert io1.getvalue() == io2.getvalue()
        assert 42 == json.loads(io2.getvalue())["answer"]


class TestSafeDeepcopy:
    def test_copies_nested_values(self):
        """
        Nested containers are deep-copied, not shared.
        """
        ed = {"nested": {"x": [1, 2]}, "plain": 1}
        copied = safe_deepcopy(ed)

        assert ed == copied
        assert copied["nested"] is not ed["nested"]
        assert copied["nested"]["x"] is not ed["nested"]["x"]

    def test_uncopyable_values_are_shared(self):
        """
        Values that can't be deep-copied (e.g. exc_info tuples containing
        tracebacks) are shared by reference instead of failing.
        """
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()

        ed = {"exc_info": exc_info, "nested": {"x": 1}}
        copied = safe_deepcopy(ed)

        assert copied["exc_info"] is exc_info
        assert copied["nested"] is not ed["nested"]

    def test_whole_dict_fallback(self, monkeypatch):
        """
        If deep-copying the whole dict fails, copying falls back to
        key-by-key so copyable keys are still isolated.
        """
        ed = {"good": {"x": 1}, "bad": threading.Lock()}

        real_deepcopy = copy.deepcopy

        def flaky_deepcopy(value):
            if value is ed:
                raise TypeError("nope")
            return real_deepcopy(value)

        monkeypatch.setattr(copy, "deepcopy", flaky_deepcopy)

        copied = safe_deepcopy(ed)

        assert copied["good"] is not ed["good"]
        assert copied["bad"] is ed["bad"]
