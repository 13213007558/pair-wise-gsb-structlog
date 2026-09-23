# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

import pytest

import structlog

from structlog import DropEvent
from structlog.fanout import (
    Branch,
    FanOut,
    FanOutError,
    FanOutReentrancyError,
)


class RecordingSink:
    """
    Records every log method call as (method_name, args, kwargs).
    """

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def make_logger(fanout, *pre_processors):
    return structlog.wrap_logger(None, processors=[*pre_processors, fanout])


class TestSnapshot:
    def test_nested_containers_are_isolated_between_branches(self):
        """
        Mutating nested dicts/lists in one branch affects neither other
        branches nor the bound context.
        """
        first, second = RecordingSink(), RecordingSink()

        def mutate(logger, method_name, event_dict):
            event_dict["payload"]["nested"]["n"] = 99
            event_dict["payload"]["items"].append(3)
            return event_dict

        log = make_logger(
            FanOut(
                [
                    Branch([mutate], first, name="first"),
                    Branch([], second, name="second"),
                ]
            )
        ).bind(payload={"nested": {"n": 1}, "items": [1, 2]})

        log.info("hello")

        assert [
            (
                "info",
                (),
                {
                    "event": "hello",
                    "payload": {"nested": {"n": 99}, "items": [1, 2, 3]},
                },
            )
        ] == first.calls
        assert [
            (
                "info",
                (),
                {
                    "event": "hello",
                    "payload": {"nested": {"n": 1}, "items": [1, 2]},
                },
            )
        ] == second.calls
        assert {"payload": {"nested": {"n": 1}, "items": [1, 2]}} == (
            structlog.get_context(log)
        )

    def test_user_objects_are_shared_and_never_deep_copied(self):
        """
        Non-container values are passed by reference to every branch and
        their __deepcopy__ hooks are not called.
        """

        class ReadOnly:
            def __deepcopy__(self, memo):
                pytest.fail("__deepcopy__ must not be called")

        obj = ReadOnly()
        seen = []

        def remember(logger, method_name, event_dict):
            seen.append(event_dict["obj"])
            return event_dict

        log = make_logger(
            FanOut(
                [
                    Branch([remember], RecordingSink()),
                    Branch([remember], RecordingSink()),
                ]
            )
        )

        log.info("hello", obj=obj)

        assert [obj, obj] == seen
        assert all(s is obj for s in seen)

    def test_snapshot_adapter_handles_special_values(self):
        """
        A custom snapshot adapter is called for non-container values and can
        copy them per branch.
        """

        class Box:
            def __init__(self, value):
                self.value = value

        def adapter(value):
            if isinstance(value, Box):
                return Box(value.value)

            return value

        boxes = []

        def mutate(logger, method_name, event_dict):
            event_dict["box"].value = 2
            return event_dict

        def remember(logger, method_name, event_dict):
            boxes.append(event_dict["box"])
            return event_dict

        log = make_logger(
            FanOut(
                [
                    Branch([mutate], RecordingSink()),
                    Branch([remember], RecordingSink()),
                ],
                snapshot_adapter=adapter,
            )
        )

        log.info("hello", box=Box(1))

        assert [1] == [b.value for b in boxes]


class TestRendererReturnForms:
    @pytest.mark.parametrize(
        ("final", "expected"),
        [
            ("text", (("text",), {})),
            (b"bytes", ((b"bytes",), {})),
            (bytearray(b"ba"), ((bytearray(b"ba"),), {})),
            ((("a", 1), {"k": 2}), (("a", 1), {"k": 2})),
            ({"event": "x", "n": 1}, ((), {"event": "x", "n": 1})),
        ],
        ids=["str", "bytes", "bytearray", "tuple", "dict"],
    )
    def test_return_forms_reach_sink(self, final, expected):
        """
        str, bytes, bytearray, (args, kwargs) tuples, and dicts from the
        last branch processor are all delivered to the sink.
        """
        sink = RecordingSink()

        def renderer(logger, method_name, event_dict):
            return final

        log = make_logger(FanOut([Branch([renderer], sink)]))

        log.info("hello")

        assert [("info", *expected)] == sink.calls

    def test_callable_sink(self):
        """
        A plain callable without log methods is called directly.
        """
        calls = []
        log = make_logger(
            FanOut([Branch([], lambda *args, **kw: calls.append((args, kw)))])
        )

        log.info("hello", n=1)

        assert [((), {"event": "hello", "n": 1})] == calls

    def test_invalid_return_value(self):
        """
        An invalid final return value raises ValueError in fail-fast mode.
        """
        log = make_logger(FanOut([Branch([lambda *_: 42], RecordingSink())]))

        with pytest.raises(ValueError, match="didn't return an appropriate"):
            log.info("hello")


class TestDropEvent:
    def test_drop_event_drops_only_current_branch(self):
        """
        DropEvent in one branch drops just that branch's event.
        """
        before, dropped, after = (
            RecordingSink(),
            RecordingSink(),
            RecordingSink(),
        )

        def drop(logger, method_name, event_dict):
            raise DropEvent

        log = make_logger(
            FanOut(
                [
                    Branch([], before),
                    Branch([drop], dropped),
                    Branch([], after),
                ]
            )
        )

        log.info("hello")

        assert 1 == len(before.calls)
        assert [] == dropped.calls
        assert 1 == len(after.calls)

    def test_outer_wrapped_logger_is_not_called(self):
        """
        FanOut is terminal: the outer wrapped logger never sees the event.
        """
        outer = RecordingSink()
        log = structlog.wrap_logger(
            outer, processors=[FanOut([Branch([], RecordingSink())])]
        )

        log.info("hello")

        assert [] == outer.calls


class TestErrorStrategies:
    def test_fail_fast_propagates_and_skips_remaining(self):
        """
        With fail-fast, the first exception propagates and later branches
        don't run.
        """
        before, after = RecordingSink(), RecordingSink()

        def boom(logger, method_name, event_dict):
            raise ValueError("nope")

        log = make_logger(
            FanOut(
                [
                    Branch([], before),
                    Branch([boom], RecordingSink()),
                    Branch([], after),
                ]
            )
        )

        with pytest.raises(ValueError, match="nope"):
            log.info("hello")

        assert 1 == len(before.calls)
        assert [] == after.calls

    def test_collect_reports_per_branch_results_without_replay(self):
        """
        With collect, all branches run exactly once and a FanOutError with
        per-branch results is raised afterwards.
        """
        ok_first, ok_last = RecordingSink(), RecordingSink()
        failure = ValueError("nope")
        calls = {"failing": 0}

        def boom(logger, method_name, event_dict):
            calls["failing"] += 1
            raise failure

        log = make_logger(
            FanOut(
                [
                    Branch([], ok_first, name="first"),
                    Branch([boom], RecordingSink(), name="bad"),
                    Branch([], ok_last, name="last"),
                ],
                on_error="collect",
            )
        )

        with pytest.raises(FanOutError, match="1 of 3") as exc_info:
            log.info("hello")

        results = exc_info.value.results
        assert ["ok", "failed", "ok"] == [r.status for r in results]
        assert ["first", "bad", "last"] == [r.name for r in results]
        assert results[1].exception is failure
        assert 1 == calls["failing"]
        assert 1 == len(ok_first.calls)
        assert 1 == len(ok_last.calls)

    def test_collect_records_dropped_branches(self):
        """
        Dropped branches show up as "dropped" in the per-branch results and
        don't count as failures.
        """

        def drop(logger, method_name, event_dict):
            raise DropEvent

        def boom(logger, method_name, event_dict):
            raise ValueError("nope")

        log = make_logger(
            FanOut(
                [
                    Branch([drop], RecordingSink(), name="dropper"),
                    Branch([boom], RecordingSink(), name="bad"),
                ],
                on_error="collect",
            )
        )

        with pytest.raises(FanOutError, match="1 of 2") as exc_info:
            log.info("hello")

        assert ["dropped", "failed"] == [
            r.status for r in exc_info.value.results
        ]

    def test_invalid_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            FanOut([], on_error="bogus")


class TestReentrancy:
    def test_sink_logging_through_same_logger_is_detected(self):
        """
        A sink that logs through the same fan-out logger raises a clear
        FanOutReentrancyError instead of recursing forever.
        """
        log = None

        def sink(*args, **kwargs):
            log.info("again")

        log = make_logger(FanOut([Branch([], sink)]))

        with pytest.raises(FanOutReentrancyError, match="Re-entrant"):
            log.info("boom")

    def test_reentrancy_is_detected_in_collect_mode_too(self):
        log = None

        def sink(*args, **kwargs):
            log.info("again")

        log = make_logger(FanOut([Branch([], sink)], on_error="collect"))

        with pytest.raises(FanOutReentrancyError, match="Re-entrant"):
            log.info("boom")

    def test_logger_is_usable_again_after_dispatch(self):
        """
        The re-entrancy guard is reset after each dispatch.
        """
        sink = RecordingSink()
        log = make_logger(FanOut([Branch([], sink)]))

        log.info("one")
        log.info("two")

        assert 2 == len(sink.calls)


def test_audit_console_archive_demo():
    """
    The motivating use case: shared pre-processing runs once, then the event
    fans out to an audit sink (mutates nested fields), a console sink (drops
    the event), and an archive sink (still receives the original data).
    """
    audit, console, archive = (
        RecordingSink(),
        RecordingSink(),
        RecordingSink(),
    )
    common_runs = 0

    def add_common(logger, method_name, event_dict):
        nonlocal common_runs
        common_runs += 1
        event_dict["trace_id"] = "t-1"
        return event_dict

    def mutate_nested(logger, method_name, event_dict):
        event_dict["payload"]["nested"]["n"] = 99
        return event_dict

    def drop(logger, method_name, event_dict):
        raise DropEvent

    log = make_logger(
        FanOut(
            [
                Branch([mutate_nested], audit, name="audit"),
                Branch([drop], console, name="console"),
                Branch([], archive, name="archive"),
            ]
        ),
        add_common,
    ).bind(payload={"nested": {"n": 1}})

    log.info("hello")

    assert 1 == common_runs
    assert [
        (
            "info",
            (),
            {
                "event": "hello",
                "trace_id": "t-1",
                "payload": {"nested": {"n": 99}},
            },
        )
    ] == audit.calls
    assert [] == console.calls
    assert [
        (
            "info",
            (),
            {
                "event": "hello",
                "trace_id": "t-1",
                "payload": {"nested": {"n": 1}},
            },
        )
    ] == archive.calls
    assert {"payload": {"nested": {"n": 1}}} == structlog.get_context(log)
