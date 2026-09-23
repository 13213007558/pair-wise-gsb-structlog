# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Fan out log events to multiple independent processor chains and sinks.

Usually, all processors form a single chain and every processor sees the
mutations of its predecessors. Sometimes that's not what you want -- for
example when an event must go to an audit log *and* to the console, but the
console output needs redaction that the audit log must not see.

`FanOut` is an explicit fan-out node for the processor chain: the
processors configured *before* it run once, and each `FanOutBranch`
then receives its own snapshot of the event dictionary, runs its own
processor chain, and delivers the result to its own sink.

.. versionadded:: 26.2.0
"""

from __future__ import annotations

import threading

from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple

from .exceptions import DropEvent
from .typing import EventDict, Processor, WrappedLogger


__all__ = [
    "BranchResult",
    "FanOut",
    "FanOutBranch",
    "FanOutError",
    "FanOutReentrancyError",
]


Sink = Callable[..., Any]
"""
A callable that receives the final output of a `FanOutBranch`.

It is called as ``sink(method_name, *args, **kwargs)`` where
*args* and *kwargs* are the normalized return value of the branch's last
processor -- exactly like a wrapped logger would be called by a bound
logger.

.. versionadded:: 26.2.0
"""

SnapshotAdapter = Callable[[Any], Any]
"""
A callable that returns a snapshot of a single non-container value.

See `FanOut` for when it is called.

.. versionadded:: 26.2.0
"""


def _snapshot(
    value: Any, adapter: SnapshotAdapter | None, memo: dict[int, Any]
) -> Any:
    """
    Return an independent snapshot of *value*.

    Built-in containers (dicts, lists, tuples, sets, and frozensets) are
    copied recursively. Anything else is either passed to *adapter* (if one
    is configured) or returned as-is, on the understanding that it is
    treated as read-only. In particular, ``__deepcopy__`` hooks
    of user objects are never called.
    """
    if isinstance(value, dict):
        copied = memo.get(id(value))
        if copied is None:
            copied = {}
            memo[id(value)] = copied
            for key, item in value.items():
                copied[key] = _snapshot(item, adapter, memo)
        return copied

    if isinstance(value, list):
        copied = memo.get(id(value))
        if copied is None:
            copied = []
            memo[id(value)] = copied
            copied.extend(_snapshot(item, adapter, memo) for item in value)
        return copied

    if isinstance(value, tuple):
        return tuple(_snapshot(item, adapter, memo) for item in value)

    if isinstance(value, (set, frozenset)):
        return type(value)(_snapshot(item, adapter, memo) for item in value)

    if adapter is not None:
        return adapter(value)

    return value


def _finalize(result: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Normalize a branch's final processor return value into (args, kwargs).

    Mirrors the rules of `structlog.BoundLoggerBase._process_event`.
    """
    if isinstance(result, (str, bytes, bytearray)):
        return (result,), {}

    if isinstance(result, tuple):
        # Assume the last processor returned a tuple of (args, kwargs)
        # and pass it right through.
        return result

    if isinstance(result, dict):
        return (), result

    msg = (
        "Last processor of a fan-out branch didn't return an appropriate "
        "value. Valid return values are a dict, a tuple of (args, kwargs), "
        "bytes, or a str."
    )
    raise ValueError(msg)


class FanOutReentrancyError(Exception):
    """
    A sink or branch processor logged through the same bound logger that
    contains the `FanOut` processor that is currently delivering an
    event.

    This would lead to infinite recursion, so it is detected and raised
    instead.

    .. versionadded:: 26.2.0
    """


class BranchResult(NamedTuple):
    """
    The outcome of delivering an event to one `FanOutBranch`.

    .. versionadded:: 26.2.0
    """

    branch: str
    """The branch's name, or "#<index>" if it has none."""

    status: Literal["ok", "dropped", "failed"]
    """
    Whether the branch delivered the event, dropped it via
    `structlog.DropEvent`, or failed with an exception.
    """

    value: Any = None
    """The sink's return value if *status* is "ok"."""

    exception: Exception | None = None
    """The exception the branch raised if *status* is "failed"."""


class FanOutError(Exception):
    """
    One or more fan-out branches failed while using the "collect" error
    strategy.

    Raised only after all branches have been attempted. Failed branches are
    not replayed.

    Attributes:
        results: A `BranchResult` for every branch, in configuration
            order.

    .. versionadded:: 26.2.0
    """

    def __init__(self, results: Sequence[BranchResult]):
        self.results = list(results)
        failed = [r for r in self.results if r.status == "failed"]
        msg = "Fan-out branches failed: " + ", ".join(
            f"{r.branch} raised {r.exception!r}" for r in failed
        )
        super().__init__(msg)


class FanOutBranch:
    """
    One delivery branch of a `FanOut` processor.

    Args:
        processors:
            The branch's own processor chain. Its last processor must return
            a value that a bound logger could pass to a wrapped logger: a
            `dict`, a tuple of (args, kwargs), `bytes`, or a
            `str`.

        sink:
            A callable that receives the branch's final output. It is called
            as ``sink(method_name, *args, **kwargs)``.

        name:
            An optional human-readable name used in `BranchResult` and
            error messages.

    .. versionadded:: 26.2.0
    """

    def __init__(
        self,
        processors: Sequence[Processor],
        sink: Sink,
        *,
        name: str | None = None,
    ):
        if not callable(sink):
            msg = f"sink must be callable, got {sink!r}"
            raise TypeError(msg)
        self.processors = list(processors)
        self.sink = sink
        self.name = name

    def __repr__(self) -> str:
        return f"<FanOutBranch(name={self.name!r}, sink={self.sink!r})>"

    def _deliver(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> Any:
        for proc in self.processors:
            event_dict = proc(logger, method_name, event_dict)

        args, kwargs = _finalize(event_dict)
        return self.sink(method_name, *args, **kwargs)


class FanOut:
    """
    A processor that fans an event out to multiple independent branches.

    The processors configured *before* this one run once. Then each branch
    receives its own snapshot of the event dictionary, runs its own
    processor chain, and delivers the result to its own sink.

    Snapshots copy built-in containers (dicts, lists, tuples, sets, and
    frozensets) recursively, so a branch may freely mutate nested containers
    without affecting the other branches or the bound logger's context. Any
    other object is treated as read-only and shared between branches as-is;
    its ``__deepcopy__`` hook is never called. Use
    *snapshot_adapter* to customize the snapshotting of such values.

    `structlog.DropEvent` raised within a branch drops the event for
    that branch only.

    Args:
        branches: The `FanOutBranch` instances to deliver the event
            to.

        on_error:
            How to handle exceptions (other than
            `structlog.DropEvent`) raised by a branch:

            - "fail_fast" (default): re-raise the first exception
              immediately; remaining branches are not attempted.
            - "collect": attempt all branches, then raise a
              `FanOutError` carrying a `BranchResult` per branch
              if any branch failed. Failed branches are not replayed.

            `FanOutReentrancyError` always propagates immediately,
            regardless of this setting.

        snapshot_adapter:
            An optional callable that is called with every non-container
            value of the event dictionary and must return the value to put
            into the snapshot. It is not called for built-in containers or
            dictionary keys.

        pass_through:
            If `True`, return the event dictionary to the outer
            processor chain after all branches have been delivered, so
            processing continues with the remaining processors and the
            wrapped logger. If `False` (default), the event is
            considered fully handled and the outer chain is stopped by
            raising `structlog.DropEvent`.

    .. versionadded:: 26.2.0
    """

    def __init__(
        self,
        branches: Sequence[FanOutBranch],
        *,
        on_error: Literal["fail_fast", "collect"] = "fail_fast",
        snapshot_adapter: SnapshotAdapter | None = None,
        pass_through: bool = False,
    ):
        if on_error not in ("fail_fast", "collect"):
            msg = (
                f"on_error must be 'fail_fast' or 'collect', got {on_error!r}"
            )
            raise ValueError(msg)

        self._branches = list(branches)
        self._on_error = on_error
        self._snapshot_adapter = snapshot_adapter
        self._pass_through = pass_through
        self._local = threading.local()

    def __call__(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        if getattr(self._local, "active", False):
            msg = (
                "FanOut was re-entered while delivering an event. A sink or "
                "branch processor must not log through a bound logger whose "
                "processor chain contains this FanOut processor."
            )
            raise FanOutReentrancyError(msg)

        self._local.active = True
        try:
            results = []
            for index, branch in enumerate(self._branches):
                label = branch.name if branch.name is not None else f"#{index}"
                snapshot = _snapshot(event_dict, self._snapshot_adapter, {})
                try:
                    value = branch._deliver(logger, method_name, snapshot)
                except DropEvent:
                    results.append(BranchResult(label, "dropped"))
                except FanOutReentrancyError:
                    raise
                except Exception as exc:
                    if self._on_error == "fail_fast":
                        raise
                    results.append(
                        BranchResult(label, "failed", exception=exc)
                    )
                else:
                    results.append(BranchResult(label, "ok", value=value))

            if self._on_error == "collect" and any(
                r.status == "failed" for r in results
            ):
                raise FanOutError(results)

            if self._pass_through:
                return event_dict

            raise DropEvent
        finally:
            self._local.active = False
