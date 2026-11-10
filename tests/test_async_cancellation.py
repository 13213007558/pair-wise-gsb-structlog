# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Regression tests for the cancellation and exception semantics of the
``a``-prefixed async log methods.

The tests deliberately avoid ``sleep()`` based scheduling guesswork. Instead
the wrapped sink cooperates through three :class:`threading.Event` checkpoints
that make the state transitions of an in-flight log call deterministic:

1. ``entered`` is set when the executor thread starts running the sink,
2. ``release`` gates the middle of the sink, so a test can cancel the asyncio
   future while the sink is demonstrably executing, and
3. ``written`` is set only after the sink's write step has completed.

Cancelling a coroutine before it reaches ``loop.run_in_executor()`` gives a
fourth, equally deterministic node: the work item is never submitted.

The checkpoints are threading events because the sink runs on an executor
thread. Waiting for them happens through :func:`asyncio.to_thread`, so the
event loop stays free to process executor callbacks while a test parks.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from structlog import make_filtering_bound_logger
from structlog.stdlib import BoundLogger

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id"
)

WAIT_TIMEOUT = 5


async def wait_for_event(event: threading.Event) -> None:
    """
    Park until *event* is set without blocking the event loop.
    """
    assert await asyncio.to_thread(
        event.wait, WAIT_TIMEOUT
    ), "timed out waiting for a sink checkpoint"


class EventGatedSink:
    """
    A stand-in for a real logging sink whose progress is fully controllable.

    Every log method records that the write step happened. The write itself is
    bracketed by threading events, so tests can assert the exact relationship
    between ``Future.cancel()`` and an actual write without sleeping.
    """

    def __init__(
        self,
        error: BaseException | None = None,
        *,
        auto_release: bool = False,
    ) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.written = threading.Event()
        self.observed_context: dict[str, str] = {}
        self.error = error
        # Tests that do not need to inspect the mid-flight state let the sink
        # pass its gate immediately instead of releasing the event by hand.
        self.auto_release = auto_release
        self.writes: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, method_name: str, *args: Any, **kw: Any) -> None:
        self.entered.set()
        self.observed_context = {
            "request_id": request_id_var.get("unset"),
            "thread": threading.current_thread().name,
        }
        if self.auto_release:
            self.release.set()
        # Block here while the test drives cancellation from the loop thread.
        # The event, not a timeout, decides when we continue.
        if not self.release.wait(WAIT_TIMEOUT):
            return
        if self.error is not None:
            raise self.error
        self.writes.append((method_name, args, kw))
        self.written.set()

    def debug(self, *args: Any, **kw: Any) -> None:
        self._record("debug", *args, **kw)

    def info(self, *args: Any, **kw: Any) -> None:
        self._record("info", *args, **kw)

    def warning(self, *args: Any, **kw: Any) -> None:
        self._record("warning", *args, **kw)

    def error(self, *args: Any, **kw: Any) -> None:
        self._record("error", *args, **kw)

    def critical(self, *args: Any, **kw: Any) -> None:
        self._record("critical", *args, **kw)


def _record_context(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    event_dict["request_id"] = request_id_var.get("unset")
    return event_dict


def _build_stdlib_logger(sink: EventGatedSink) -> BoundLogger:
    return BoundLogger(sink, [_record_context], {})


def _build_native_logger(sink: EventGatedSink) -> Any:
    cls = make_filtering_bound_logger(logging.DEBUG)
    return cls(sink, [_record_context], {})


@pytest.fixture(params=["stdlib", "native"])
def logger_kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _make_logger(kind: str, sink: EventGatedSink) -> Any:
    if kind == "stdlib":
        return _build_stdlib_logger(sink)
    return _build_native_logger(sink)


@pytest.fixture
def sink() -> Iterator[EventGatedSink]:
    gate = EventGatedSink()
    yield gate
    # Unblock any thread that is still waiting so executor teardown cannot
    # deadlock, even when a test failed before releasing the gate itself.
    gate.release.set()


class TestCancellationNodes:
    async def test_cancel_before_submission_never_writes(
        self,
        logger_kind: str,
        sink: EventGatedSink,
    ) -> None:
        """
        If the coroutine is cancelled before ``run_in_executor()`` is reached,
        the work item is never submitted and no write can happen.
        """
        logger = _make_logger(logger_kind, sink)
        reached_gate = threading.Event()

        async def suspend_before_logging() -> None:
            reached_gate.set()
            await asyncio.sleep(10)

        async def request() -> None:
            await suspend_before_logging()
            await logger.ainfo("never submitted")

        task = asyncio.create_task(request())
        await wait_for_event(reached_gate)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled()
        assert not sink.entered.is_set()
        assert sink.writes == []

    async def test_cancel_while_sink_runs_does_not_undo_write(
        self,
        logger_kind: str,
        sink: EventGatedSink,
    ) -> None:
        """
        Cancellation while the sink is executing gives the caller
        ``CancelledError``, but the executor thread keeps running and the
        write still completes. Cancellation cannot revoke output that is in
        flight.
        """
        logger = _make_logger(logger_kind, sink)
        task = asyncio.create_task(logger.ainfo("in flight"))

        # Deterministically wait until the sink is executing.
        await wait_for_event(sink.entered)
        assert not sink.written.is_set()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

        # The caller has given up, but the thread is still blocked inside the
        # sink. Releasing it proves the write happens after cancellation.
        sink.release.set()
        await wait_for_event(sink.written)
        assert sink.writes == [
            ("info", (), {"event": "in flight", "request_id": "unset"})
        ]

    async def test_cancel_after_write_completed_returns_false(
        self,
        logger_kind: str,
        sink: EventGatedSink,
    ) -> None:
        """
        Once the executor work has finished, the future can no longer be
        cancelled: the caller observes the normal result, not
        ``CancelledError``.
        """
        logger = _make_logger(logger_kind, sink)
        task = asyncio.create_task(logger.ainfo("done"))

        await wait_for_event(sink.entered)
        sink.release.set()
        await wait_for_event(sink.written)
        # Let the loop process the executor's done callback before cancelling.
        await asyncio.sleep(0)

        assert task.cancel() is False
        assert not task.cancelled()
        assert task.done()
        await task  # completes normally, no exception
        assert len(sink.writes) == 1


class TestSinkErrors:
    async def test_sink_error_propagates_when_awaited(
        self,
        logger_kind: str,
    ) -> None:
        """
        An exception raised by the sink in the executor is re-raised by the
        ``await`` when the task is not cancelled.
        """
        failing = EventGatedSink(
            error=ValueError("sink failed"), auto_release=True
        )
        logger = _make_logger(logger_kind, failing)

        with pytest.raises(ValueError, match="sink failed"):
            await logger.ainfo("boom")

    async def test_held_task_exposes_sink_error(
        self,
        logger_kind: str,
    ) -> None:
        """
        An application that keeps the task handle can observe the sink's
        exception through ``Task.exception()`` instead of awaiting it.
        """
        failing = EventGatedSink(error=RuntimeError("held"), auto_release=True)
        logger = _make_logger(logger_kind, failing)
        task = asyncio.create_task(logger.ainfo("boom"))

        await wait_for_event(failing.entered)
        await asyncio.sleep(0)

        assert task.done()
        assert isinstance(task.exception(), RuntimeError)

    async def test_cancellation_masks_in_flight_sink_error(
        self,
        logger_kind: str,
    ) -> None:
        """
        If the sink is still executing when the task is cancelled, the holder
        only sees ``CancelledError``. Whether the sink later raises is
        unobservable through the cancelled task and must not be inferred from
        the cancellation.
        """
        failing = EventGatedSink(error=RuntimeError("late failure"))
        logger = _make_logger(logger_kind, failing)
        task = asyncio.create_task(logger.ainfo("boom"))

        await wait_for_event(failing.entered)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

        # Let the sink finish and raise inside the abandoned work item.
        failing.release.set()
        await asyncio.sleep(0)


class TestApplicationObservation:
    async def test_fire_and_forget_write_survives_lost_coroutine(
        self,
        logger_kind: str,
        sink: EventGatedSink,
    ) -> None:
        """
        Dropping the reference to a logging coroutine (not awaiting it) does
        not stop the submitted work item. The only reliable way to learn that
        the write happened is to observe the sink itself.
        """
        logger = _make_logger(logger_kind, sink)

        async def fire_and_forget() -> None:
            await logger.ainfo("unawaited")

        task = asyncio.create_task(fire_and_forget())
        await wait_for_event(sink.entered)
        del task

        sink.release.set()
        await wait_for_event(sink.written)
        assert sink.writes == [
            ("info", (), {"event": "unawaited", "request_id": "unset"})
        ]

    async def test_held_task_keeps_result_observable(
        self,
        logger_kind: str,
        sink: EventGatedSink,
    ) -> None:
        """
        An application that keeps the task handle decides when to observe the
        outcome: awaiting it after the write completes returns normally and
        the sink state confirms the output, with no cancellation involved.
        """
        logger = _make_logger(logger_kind, sink)
        task = asyncio.create_task(logger.ainfo("tracked"))

        await wait_for_event(sink.entered)
        assert not task.done()
        sink.release.set()
        await wait_for_event(sink.written)

        # Awaiting the retained task is how an application waits until the
        # processor chain and sink call have completed.
        await task
        assert task.done()
        assert task.exception() is None
        assert len(sink.writes) == 1


class TestContextRestoration:
    async def test_worker_runs_with_copied_context(
        self,
        logger_kind: str,
        sink: EventGatedSink,
    ) -> None:
        """
        The executor thread runs inside ``copy_context()``, so contextvars
        bound at call time are visible to the processor chain and sink even
        though the work runs on another thread.
        """
        logger = _make_logger(logger_kind, sink)
        token = request_id_var.set("req-42")
        try:
            task = asyncio.create_task(logger.ainfo("context"))
            await wait_for_event(sink.entered)
            sink.release.set()
        finally:
            request_id_var.reset(token)

        await wait_for_event(sink.written)
        await task

        assert sink.observed_context["request_id"] == "req-42"
        assert sink.observed_context["thread"] != threading.main_thread().name
        assert sink.writes[0][2]["request_id"] == "req-42"

    async def test_contextvars_are_reset_after_cancellation(
        self,
        logger_kind: str,
        sink: EventGatedSink,
    ) -> None:
        """
        The ``finally`` block that resets the internal callsite contextvars
        still runs when the awaiting coroutine is cancelled mid-flight.
        Cancellation does not leave the loop thread's context polluted.
        """
        from structlog.contextvars import (
            _ASYNC_CALLING_STACK,
            _ASYNC_CALLING_THREAD,
        )

        logger = _make_logger(logger_kind, sink)
        task = asyncio.create_task(logger.ainfo("reset me"))
        await wait_for_event(sink.entered)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        sink.release.set()
        await wait_for_event(sink.written)

        with pytest.raises(LookupError):
            _ASYNC_CALLING_STACK.get()
        with pytest.raises(LookupError):
            _ASYNC_CALLING_THREAD.get()

    async def test_worker_context_mutations_do_not_escape(
        self,
        logger_kind: str,
    ) -> None:
        """
        Mutating a contextvar inside the copied worker context cannot affect
        the loop thread: the copy created at submission time is isolated.
        """

        def mutating_processor(
            logger_: Any, method_name: str, event_dict: dict[str, Any]
        ) -> dict[str, Any]:
            request_id_var.set("mutated-in-worker")
            event_dict["worker_value"] = request_id_var.get()
            return event_dict

        isolated_sink = EventGatedSink(auto_release=True)
        if logger_kind == "stdlib":
            logger = BoundLogger(isolated_sink, [mutating_processor], {})
        else:
            cls = make_filtering_bound_logger(logging.DEBUG)
            logger = cls(isolated_sink, [mutating_processor], {})

        token = request_id_var.set("outer")
        try:
            task = asyncio.create_task(logger.ainfo("isolated"))
            await wait_for_event(isolated_sink.written)
            await task
            # The worker mutated only its copied context; the loop thread
            # still sees its own value while the token is active.
            assert request_id_var.get() == "outer"
        finally:
            request_id_var.reset(token)

        assert isolated_sink.writes[0][2]["worker_value"] == (
            "mutated-in-worker"
        )
        with pytest.raises(LookupError):
            request_id_var.get()
