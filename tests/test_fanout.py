# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

import pytest

from structlog import BoundLogger
from structlog.exceptions import DropEvent
from structlog.fanout import (
    FanOut,
    FanOutBranch,
    FanOutError,
    FanOutReentrancyError,
)
from structlog.testing import ReturnLogger


class RecordingSink:
    """
    A sink that records exactly what it is called with.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, method_name, *args, **kwargs):
        self.calls.append((method_name, args, kwargs))


def build_fanout_logger(branches, context=None, pre_processors=(), **kw):
    """
    Build a BoundLoggerBase whose chain ends in a FanOut processor.
    """
    return BoundLogger(
        ReturnLogger(),
        [*pre_processors, FanOut(branches, **kw)],
        context if context is not None else {},
    )


class TestFanOutDemo:
    def test_audit_console_and_dropped_branch(self):
        """
        The audit/console use case: common pre-processing runs once, the
        audit branch mutates nested data, a second branch drops the event,
        and the console branch still receives the original, complete event.
        """
        audit_sink = RecordingSink()
        dropped_sink = RecordingSink()
        console_sink = RecordingSink()
        pre_calls = []

        def add_timestamp(logger, method_name, event_dict):
            pre_calls.append(1)
            event_dict["timestamp"] = "2026-09-23T00:00:00Z"
            return event_dict

        def mutate_nested(logger, method_name, event_dict):
            event_dict["request"]["id"] = 99
            event_dict["request"]["tags"].append("tampered")
            return event_dict

        def drop_it(logger, method_name, event_dict):
            raise DropEvent

        def redact(logger, method_name, event_dict):
            event_dict["password"] = "***"
            return event_dict

        log = build_fanout_logger(
            [
                FanOutBranch([mutate_nested], audit_sink, name="audit"),
                FanOutBranch([drop_it], dropped_sink, name="bit-bucket"),
                FanOutBranch([redact], console_sink, name="console"),
            ],
            context={"request": {"id": 1, "tags": ["login"]}},
            pre_processors=[add_timestamp],
        )

        assert log.info("user_login", password="hunter2") is None

        # Common pre-processing ran exactly once.
        assert 1 == len(pre_calls)

        # The audit branch saw its own mutations.
        assert [
            (
                "info",
                (),
                {
                    "event": "user_login",
                    "password": "hunter2",
                    "request": {"id": 99, "tags": ["login", "tampered"]},
                    "timestamp": "2026-09-23T00:00:00Z",
                },
            )
        ] == audit_sink.calls

        # The dropping branch swallowed its copy only.
        assert [] == dropped_sink.calls

        # The console branch still received the original, complete event
        # plus its own redaction.
        assert [
            (
                "info",
                (),
                {
                    "event": "user_login",
                    "password": "***",
                    "request": {"id": 1, "tags": ["login"]},
                    "timestamp": "2026-09-23T00:00:00Z",
                },
            )
        ] == console_sink.calls

        # The bound context was not touched by any branch.
        assert {"request": {"id": 1, "tags": ["login"]}} == log._context


class TestSnapshots:
    def test_nested_containers_are_independent_per_branch(self):
        """
        Mutations of nested dicts and lists in one branch are invisible to
        later branches and to the bound context.
        """
        sink1 = RecordingSink()
        sink2 = RecordingSink()

        def mutate(logger, method_name, event_dict):
            event_dict["deep"]["list"].append("x")
            event_dict["deep"]["dict"]["k"] = "changed"
            event_dict["top"] = "changed"
            return event_dict

        log = build_fanout_logger(
            [FanOutBranch([mutate], sink1), FanOutBranch([], sink2)],
            context={"deep": {"list": [1], "dict": {"k": "v"}}, "top": "orig"},
        )

        log.info("hi")

        assert {
            "event": "hi",
            "deep": {"list": [1, "x"], "dict": {"k": "changed"}},
            "top": "changed",
        } == sink1.calls[0][2]
        assert {
            "event": "hi",
            "deep": {"list": [1], "dict": {"k": "v"}},
            "top": "orig",
        } == sink2.calls[0][2]
        assert {
            "deep": {"list": [1], "dict": {"k": "v"}},
            "top": "orig",
        } == log._context

    def test_user_objects_are_shared_read_only(self):
        """
        Arbitrary user objects are passed to every branch as-is; their
        __deepcopy__ hook is never called.
        """

        class Sentinel:
            deepcopy_called = False

            def __deepcopy__(self, memo):
                self.deepcopy_called = True
                raise AssertionError("__deepcopy__ must not be called")

        sentinel = Sentinel()
        seen = []

        def record(logger, method_name, event_dict):
            seen.append(event_dict["obj"])
            return event_dict

        log = build_fanout_logger(
            [FanOutBranch([record], RecordingSink()) for _ in range(2)],
            context={"obj": sentinel},
        )

        log.info("hi")

        assert [sentinel, sentinel] == seen
        assert seen[0] is sentinel
        assert not sentinel.deepcopy_called

    def test_snapshot_adapter_handles_special_values(self):
        """
        A snapshot adapter is called for non-container values and its return
        value is put into the branch's snapshot.
        """

        class Token:
            def __init__(self, value):
                self.value = value

        def adapter(value):
            if isinstance(value, Token):
                return Token(value.value)
            return value

        sink = RecordingSink()
        token = Token("secret")
        log = build_fanout_logger(
            [FanOutBranch([], sink)],
            context={"token": token, "nested": {"inner": token}},
            snapshot_adapter=adapter,
        )

        log.info("hi")

        kwargs = sink.calls[0][2]
        assert kwargs["token"] is not token
        assert "secret" == kwargs["token"].value
        assert kwargs["nested"]["inner"] is not token
        assert "secret" == kwargs["nested"]["inner"].value

    def test_snapshot_adapter_not_called_for_containers(self):
        """
        The adapter is not called for built-in containers or dict keys.
        """
        adapted = []

        def adapter(value):
            adapted.append(value)
            return value

        sink = RecordingSink()
        log = build_fanout_logger(
            [FanOutBranch([], sink)],
            context={"a_dict": {"k": "v"}, "a_list": [1]},
            snapshot_adapter=adapter,
        )

        log.info("hi")

        assert not any(isinstance(v, (dict, list)) for v in adapted)
        assert "k" not in adapted


class TestDelivery:
    @pytest.mark.parametrize(
        ("rendered", "expected_args", "expected_kwargs"),
        [
            ("text", ("text",), {}),
            (b"bytes", (b"bytes",), {}),
            (bytearray(b"ba"), (bytearray(b"ba"),), {}),
            ((("a", 1), {"k": "v"}), ("a", 1), {"k": "v"}),
            ({"event": "x", "n": 1}, (), {"event": "x", "n": 1}),
        ],
    )
    def test_return_forms_reach_sink(
        self, rendered, expected_args, expected_kwargs
    ):
        """
        str, bytes, bytearray, (args, kwargs) tuples, and dicts from a
        branch's last processor are normalized like a bound logger does and
        reach the sink.
        """
        sink = RecordingSink()
        log = build_fanout_logger([FanOutBranch([lambda *_: rendered], sink)])

        log.info("hi")

        assert [("info", expected_args, expected_kwargs)] == sink.calls

    def test_invalid_branch_return_value(self):
        """
        A branch whose last processor returns something a bound logger
        could not pass on raises a ValueError.
        """
        log = build_fanout_logger(
            [FanOutBranch([lambda *_: 42], RecordingSink())]
        )

        with pytest.raises(ValueError, match="didn't return"):
            log.info("hi")

    def test_method_name_is_passed_through(self):
        """
        The sink is called with the method name the user called.
        """
        sink = RecordingSink()
        log = build_fanout_logger([FanOutBranch([], sink)])

        log.warning("hi")

        assert "warning" == sink.calls[0][0]

    def test_non_callable_sink_rejected(self):
        """
        A sink that is not callable raises a TypeError.
        """
        with pytest.raises(TypeError, match="sink must be callable"):
            FanOutBranch([], "not-a-sink")


class TestDropEvent:
    def test_drop_event_only_drops_current_branch(self):
        """
        DropEvent raised in one branch does not affect the other branches.
        """
        sink = RecordingSink()

        def drop(logger, method_name, event_dict):
            raise DropEvent

        log = build_fanout_logger(
            [
                FanOutBranch([drop], RecordingSink()),
                FanOutBranch([], sink),
            ]
        )

        assert log.info("hi") is None
        assert 1 == len(sink.calls)


class TestErrorStrategies:
    def test_invalid_on_error(self):
        """
        An unknown on_error strategy raises a ValueError.
        """
        with pytest.raises(ValueError, match="on_error"):
            FanOut([], on_error="ignore")

    def test_fail_fast_stops_remaining_branches(self):
        """
        With fail_fast, the first exception propagates immediately and
        remaining branches are not attempted.
        """
        sink = RecordingSink()

        def boom(logger, method_name, event_dict):
            raise ValueError("nope")

        log = build_fanout_logger(
            [
                FanOutBranch([boom], RecordingSink(), name="bad"),
                FanOutBranch([], sink, name="good"),
            ],
            on_error="fail_fast",
        )

        with pytest.raises(ValueError, match="nope"):
            log.info("hi")

        assert [] == sink.calls

    def test_collect_reports_per_branch_results(self):
        """
        With collect, all branches are attempted exactly once and a
        FanOutError with per-branch results is raised afterwards.
        """
        ok_sink = RecordingSink()
        attempts = []

        def boom(logger, method_name, event_dict):
            attempts.append(1)
            raise ValueError("nope")

        def drop(logger, method_name, event_dict):
            raise DropEvent

        log = build_fanout_logger(
            [
                FanOutBranch([boom], RecordingSink(), name="bad"),
                FanOutBranch([], ok_sink, name="good"),
                FanOutBranch([drop], RecordingSink(), name="dropper"),
            ],
            on_error="collect",
        )

        with pytest.raises(FanOutError) as exc_info:
            log.info("hi")

        results = exc_info.value.results
        assert ["bad", "good", "dropper"] == [r.branch for r in results]
        assert ["failed", "ok", "dropped"] == [r.status for r in results]
        assert isinstance(results[0].exception, ValueError)
        assert "bad" in str(exc_info.value)

        # The failed branch was attempted exactly once: no replay.
        assert 1 == len(attempts)
        # The healthy branch still received the event.
        assert 1 == len(ok_sink.calls)

    def test_collect_without_failures_does_not_raise(self):
        """
        With collect and no failing branches, no FanOutError is raised.
        """
        sink = RecordingSink()
        log = build_fanout_logger([FanOutBranch([], sink)], on_error="collect")

        assert log.info("hi") is None
        assert 1 == len(sink.calls)

    def test_unnamed_branches_are_labeled_by_index(self):
        """
        Branches without a name are labeled by their index in results.
        """

        def boom(logger, method_name, event_dict):
            raise ValueError("nope")

        log = build_fanout_logger(
            [FanOutBranch([boom], RecordingSink())],
            on_error="collect",
        )

        with pytest.raises(FanOutError) as exc_info:
            log.info("hi")

        assert "#0" == exc_info.value.results[0].branch


class TestReentrancy:
    def test_reentrant_sink_is_reported(self):
        """
        A sink that logs through the same bound logger raises a
        FanOutReentrancyError instead of recursing infinitely.
        """
        sink_calls = []

        def sink(method_name, *args, **kwargs):
            sink_calls.append(method_name)
            log.info("recursive")

        log = build_fanout_logger([FanOutBranch([], sink)])

        with pytest.raises(FanOutReentrancyError, match="re-entered"):
            log.info("boom")

        assert ["info"] == sink_calls

    def test_reentrancy_error_propagates_even_with_collect(self):
        """
        FanOutReentrancyError is a programming error and is never collected.
        """

        def sink(method_name, *args, **kwargs):
            log.info("recursive")

        log = build_fanout_logger([FanOutBranch([], sink)], on_error="collect")

        with pytest.raises(FanOutReentrancyError, match="re-entered"):
            log.info("boom")

    def test_guard_is_reset_after_delivery(self):
        """
        After a delivery finishes, the same logger can be used again.
        """
        sink = RecordingSink()
        log = build_fanout_logger([FanOutBranch([], sink)])

        log.info("one")
        log.info("two")

        assert 2 == len(sink.calls)


class TestOuterChain:
    def test_default_stops_outer_chain(self):
        """
        By default the event is fully handled by the branches and the outer
        chain stops.
        """
        sink = RecordingSink()

        def after(logger, method_name, event_dict):
            raise AssertionError("must not run")

        bl = BoundLogger(
            ReturnLogger(),
            [FanOut([FanOutBranch([], sink)]), after],
            {},
        )

        assert bl.info("hi") is None
        assert 1 == len(sink.calls)

    def test_pass_through_continues_outer_chain(self):
        """
        With pass_through=True, the original event dictionary continues
        down the outer processor chain after all branches ran.
        """
        sink = RecordingSink()
        seen = []

        def after(logger, method_name, event_dict):
            seen.append(dict(event_dict))
            return "final"

        bl = BoundLogger(
            ReturnLogger(),
            [FanOut([FanOutBranch([], sink)], pass_through=True), after],
            {},
        )

        assert "final" == bl.info("hi", x=1)
        assert [{"event": "hi", "x": 1}] == seen
        assert [("info", (), {"event": "hi", "x": 1})] == sink.calls
