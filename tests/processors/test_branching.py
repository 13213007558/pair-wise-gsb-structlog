# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

from __future__ import annotations

import io
import json
import logging

import pytest

from structlog import DropEvent, configure, get_logger
from structlog._base import BoundLoggerBase
from structlog._config import _CONFIG
from structlog.processors import (
    Branch,
    BranchProcessingError,
    BranchingProcessor,
    JSONRenderer,
    copy_event_dict,
)
from structlog.stdlib import (
    BoundLogger,
    LoggerFactory,
    ProcessorFormatter,
)
from structlog.testing import ReturnLogger


def build_bound_logger(processors, context=None):
    return BoundLoggerBase(
        ReturnLogger(),
        processors,
        _CONFIG.default_context_class() if context is None else context,
    )


class TestBranchingProcessor:
    def test_common_chain_runs_once_and_branches_are_isolated(self):
        common_calls = []
        outputs = []

        def common(_, __, event_dict):
            common_calls.append(event_dict)
            return event_dict

        def mutate(_, __, event_dict):
            event_dict["payload"]["value"] = "json"
            event_dict["only"] = "json"
            return event_dict

        def drop(_, __, event_dict):
            raise DropEvent

        class CapturingLogger:
            def info(self, *args, **kwargs):
                outputs.append((args, kwargs))

        branching = BranchingProcessor(
            [
                Branch(
                    CapturingLogger(),
                    (
                        mutate,
                        lambda _, __, event_dict: json.dumps(event_dict),
                    ),
                ),
                Branch(CapturingLogger(), (drop,)),
                Branch(
                    CapturingLogger(),
                    (lambda _, __, event_dict: event_dict["event"],),
                ),
            ]
        )
        original = build_bound_logger(
            [common, branching],
            {"payload": {"value": "original"}},
        )

        with pytest.raises(DropEvent):
            original._process_event("info", "hello", {})

        assert 1 == len(common_calls)
        assert 2 == len(outputs)
        json_event = json.loads(outputs[0][0][0])
        assert "json" == json_event["payload"]["value"]
        assert "json" == json_event["only"]
        assert ("hello",) == outputs[1][0]
        assert "original" == original._context["payload"]["value"]
        assert "only" not in original._context

    def test_drop_event_only_stops_current_branch(self):
        calls = []

        def record(name):
            def processor(_, __, event_dict):
                calls.append(name)
                return event_dict

            return processor

        branching = BranchingProcessor(
            [
                Branch(ReturnLogger(), (record("first"),)),
                Branch(
                    ReturnLogger(),
                    (record("dropped"), lambda *args: (_ for _ in ()).throw(DropEvent)),
                ),
                Branch(ReturnLogger(), (record("third"),)),
            ]
        )

        with pytest.raises(DropEvent):
            branching(ReturnLogger(), "info", {"event": "hello"})

        assert ["first", "dropped", "third"] == calls

    def test_raise_early_stops_remaining_branches(self):
        calls = []

        def fail(name):
            def processor(_, __, event_dict):
                calls.append(name)
                raise RuntimeError(name)

            return processor

        branching = BranchingProcessor(
            [
                Branch(ReturnLogger(), (fail("first"),)),
                Branch(ReturnLogger(), (fail("second"),)),
            ]
        )

        with pytest.raises(RuntimeError, match="first"):
            branching(ReturnLogger(), "info", {"event": "hello"})

        assert ["first"] == calls

    def test_collects_errors_from_remaining_branches(self):
        calls = []

        def fail(name):
            def processor(_, __, event_dict):
                calls.append(name)
                raise RuntimeError(name)

            return processor

        branching = BranchingProcessor(
            [
                Branch(ReturnLogger(), (fail("first"),)),
                Branch(ReturnLogger(), ()),
                Branch(ReturnLogger(), (fail("second"),)),
            ],
            raise_early=False,
        )

        with pytest.raises(BranchProcessingError) as exc_info:
            branching(ReturnLogger(), "info", {"event": "hello"})

        assert ["first", "second"] == [str(error) for error in calls]
        assert 2 == len(exc_info.value.exceptions)

    def test_default_copier_shares_opaque_exception_values(self):
        error = RuntimeError("boom")
        event_dict = {"error": error, "payload": {"value": 1}}

        copied = copy_event_dict(event_dict)

        assert error is copied["error"]
        assert event_dict["payload"] is not copied["payload"]

    def test_custom_copier_is_used_for_every_branch(self):
        copies = []

        def copier(event_dict):
            copied = event_dict.copy()
            copies.append(copied)
            return copied

        branching = BranchingProcessor(
            [
                Branch(ReturnLogger(), ()),
                Branch(ReturnLogger(), ()),
            ],
            copier=copier,
        )

        with pytest.raises(DropEvent):
            branching(ReturnLogger(), "info", {"event": "hello"})

        assert 2 == len(copies)
        assert copies[0] is not copies[1]


class TestProcessorFormatterBranching:
    def test_each_branch_uses_its_own_log_record_and_formatter(self):
        common_calls = []

        def common(_, __, event_dict):
            common_calls.append(event_dict)
            return event_dict

        def mutate(_, __, event_dict):
            event_dict["payload"]["destination"] = "collector"
            return event_dict

        def drop(_, __, event_dict):
            raise DropEvent

        json_stream = io.StringIO()
        text_stream = io.StringIO()
        dropped_stream = io.StringIO()
        json_logger = logging.getLogger(
            "test_branching_processor_formatter_json"
        )
        text_logger = logging.getLogger(
            "test_branching_processor_formatter_text"
        )
        dropped_logger = logging.getLogger(
            "test_branching_processor_formatter_dropped"
        )

        for logger, stream in (
            (json_logger, json_stream),
            (text_logger, text_stream),
            (dropped_logger, dropped_stream),
        ):
            logger.handlers.clear()
            logger.propagate = False
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                ProcessorFormatter(
                    processors=[
                        ProcessorFormatter.remove_processors_meta,
                        JSONRenderer(),
                    ]
                )
                if logger is json_logger
                else ProcessorFormatter(
                    processors=[
                        ProcessorFormatter.remove_processors_meta,
                        lambda _, __, event_dict: event_dict["event"],
                    ]
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        configure(
            processors=[
                common,
                BranchingProcessor(
                    [
                        Branch(
                            json_logger,
                            (
                                mutate,
                                ProcessorFormatter.wrap_for_formatter,
                            ),
                        ),
                        Branch(
                            text_logger,
                            (ProcessorFormatter.wrap_for_formatter,),
                        ),
                        Branch(
                            dropped_logger,
                            (drop, ProcessorFormatter.wrap_for_formatter),
                        ),
                    ]
                ),
            ],
            logger_factory=LoggerFactory(),
            wrapper_class=BoundLogger,
        )

        get_logger().info("hello", payload={"destination": "terminal"})

        assert 1 == len(common_calls)
        json_event = json.loads(json_stream.getvalue())
        assert "collector" == json_event["payload"]["destination"]
        assert "hello\n" == text_stream.getvalue()
        assert "" == dropped_stream.getvalue()
        assert "terminal" == common_calls[0]["payload"]["destination"]
