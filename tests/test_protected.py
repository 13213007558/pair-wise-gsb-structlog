# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

import asyncio

import pytest

import structlog

from structlog.contextvars import (
    bind_contextvars,
    bound_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
)
from structlog.exceptions import ProtectedKeyError
from structlog.protected import (
    configure_protection,
    get_protected_keys,
    protect_keys,
    unprotect_keys,
)
from structlog.stdlib import AsyncBoundLogger
from structlog.testing import CapturingLogger


@pytest.fixture(autouse=True)
def _clear_contextvars():
    """
    Make sure all tests start with a clean context-local slate.

    Protection itself is reset by the global reset_defaults fixture.
    """
    clear_contextvars()


@pytest.fixture(name="cl")
def _cl():
    return CapturingLogger()


@pytest.fixture(name="bl")
def _bl(cl):
    return structlog.BoundLogger(cl, [], {})


def last_kwargs(cl):
    return cl.calls[-1].kwargs


class TestProtectKeys:
    def test_event_kwarg_cannot_override_bound(self, bl, cl):
        """
        An event kwarg conflicting with a bound protected key is moved to
        the conflicts key; the bound value wins.
        """
        protect_keys("actor_id")

        bl.bind(actor_id="real-user").info(
            "login", actor_id="spoofed", action="login"
        )

        assert {
            "event": "login",
            "actor_id": "real-user",
            "action": "login",
            "protected_conflicts": {"actor_id": "spoofed"},
        } == last_kwargs(cl)

    def test_no_protection_by_default(self, bl, cl):
        """
        Without protect_keys, event kwargs override bound context as
        before.
        """
        bl.bind(actor_id="bound").info("login", actor_id="kwarg")

        assert {"event": "login", "actor_id": "kwarg"} == last_kwargs(cl)

    def test_unbound_protected_key_passes_through(self, bl, cl):
        """
        A declared but unbound protected key can be set by event kwargs
        without being treated as a conflict.
        """
        protect_keys("actor_id")

        bl.info("login", actor_id="from-payload")

        assert {"event": "login", "actor_id": "from-payload"} == last_kwargs(
            cl
        )

    def test_nested_values(self, bl, cl):
        """
        Nested bound values are protected as a whole; a conflicting nested
        payload is displaced in full.
        """
        protect_keys("user")
        bound = {"id": 1, "roles": ["admin"]}

        bl.bind(user=bound).info("x", user={"id": 2, "roles": ["root"]})

        assert {
            "event": "x",
            "user": {"id": 1, "roles": ["admin"]},
            "protected_conflicts": {"user": {"id": 2, "roles": ["root"]}},
        } == last_kwargs(cl)

    def test_equal_value_is_not_a_conflict(self, bl, cl):
        """
        An event kwarg equal to the bound value is not treated as a
        conflict.
        """
        protect_keys("actor_id")

        bl.bind(actor_id="same").info("x", actor_id="same")

        assert {"event": "x", "actor_id": "same"} == last_kwargs(cl)

    def test_rebind_updates_trusted_value(self, bl, cl):
        """
        Trusted code can re-bind a protected key; the latest bound value is
        enforced.
        """
        protect_keys("actor_id")

        bl.bind(actor_id="first").bind(actor_id="second").info(
            "x", actor_id="spoofed"
        )

        assert {
            "event": "x",
            "actor_id": "second",
            "protected_conflicts": {"actor_id": "spoofed"},
        } == last_kwargs(cl)

    def test_unbind_releases_protection(self, bl, cl):
        """
        After explicitly unbinding a protected key, event kwargs may set it
        again.
        """
        protect_keys("actor_id")

        bl.bind(actor_id="bound").unbind("actor_id").info(
            "x", actor_id="from-payload"
        )

        assert {"event": "x", "actor_id": "from-payload"} == last_kwargs(cl)

    def test_try_unbind_releases_protection(self, bl, cl):
        """
        try_unbind on a protected key releases it just like unbind.
        """
        protect_keys("actor_id")

        bl.bind(actor_id="bound").try_unbind("actor_id").info(
            "x", actor_id="from-payload"
        )

        assert {"event": "x", "actor_id": "from-payload"} == last_kwargs(cl)

    def test_new_clears_protection(self, bl, cl):
        """
        new() clears the bound context, so previously bound protected
        values no longer shield the key.
        """
        protect_keys("actor_id")

        bl.bind(actor_id="bound").new().info("x", actor_id="from-payload")

        assert {"event": "x", "actor_id": "from-payload"} == last_kwargs(cl)

    def test_new_can_bind_protected(self, bl, cl):
        """
        Values bound via new() are protected like values bound via bind().
        """
        protect_keys("actor_id")

        bl.new(actor_id="bound").info("x", actor_id="spoofed")

        assert {
            "event": "x",
            "actor_id": "bound",
            "protected_conflicts": {"actor_id": "spoofed"},
        } == last_kwargs(cl)

    def test_unprotect_keys(self, bl, cl):
        """
        unprotect_keys removes protection again.
        """
        protect_keys("actor_id")
        unprotect_keys("actor_id")

        assert frozenset() == get_protected_keys()

        bl.bind(actor_id="bound").info("x", actor_id="kwarg")

        assert {"event": "x", "actor_id": "kwarg"} == last_kwargs(cl)

    def test_get_protected_keys(self):
        """
        get_protected_keys returns the declared keys.
        """
        protect_keys("actor_id", "tenant")

        assert frozenset({"actor_id", "tenant"}) == get_protected_keys()


class TestConflictHandling:
    def test_custom_conflict_key(self, bl, cl):
        """
        Displaced values are collected under a custom conflict key.
        """
        protect_keys("actor_id")
        configure_protection(conflict_key="dropped_payload_fields")

        bl.bind(actor_id="real").info("x", actor_id="spoofed")

        assert {
            "event": "x",
            "actor_id": "real",
            "dropped_payload_fields": {"actor_id": "spoofed"},
        } == last_kwargs(cl)

    def test_raise_mode(self, bl, cl):
        """
        With on_conflict="raise", a conflicting event kwarg raises
        ProtectedKeyError and nothing is logged.
        """
        protect_keys("actor_id")
        configure_protection(on_conflict="raise")

        with pytest.raises(ProtectedKeyError) as ei:
            bl.bind(actor_id="real").info("x", actor_id="spoofed")

        assert {"actor_id": "spoofed"} == ei.value.conflicts
        assert "actor_id" in str(ei.value)
        assert [] == cl.calls

    def test_raise_mode_no_conflict_no_raise(self, bl, cl):
        """
        on_conflict="raise" does not affect non-conflicting log calls.
        """
        protect_keys("actor_id")
        configure_protection(on_conflict="raise")

        bl.bind(actor_id="real").info("x", action="login")

        assert {"event": "x", "actor_id": "real", "action": "login"} == (
            last_kwargs(cl)
        )

    def test_invalid_on_conflict(self):
        """
        configure_protection validates on_conflict.
        """
        with pytest.raises(ValueError, match="on_conflict"):
            configure_protection(on_conflict="ignore")

    def test_invalid_conflict_key(self):
        """
        configure_protection validates conflict_key.
        """
        with pytest.raises(ValueError, match="conflict_key"):
            configure_protection(conflict_key="")

    def test_partial_configure_keeps_values(self, bl, cl):
        """
        configure_protection with a single argument keeps the other
        setting.
        """
        protect_keys("actor_id")
        configure_protection(conflict_key="conflicts")
        configure_protection(on_conflict="raise")

        with pytest.raises(ProtectedKeyError):
            bl.bind(actor_id="real").info("x", actor_id="spoofed")

    def test_reset_defaults_resets_protection(self, bl, cl):
        """
        structlog.reset_defaults removes protected keys and restores
        default conflict handling.
        """
        protect_keys("actor_id")
        configure_protection(on_conflict="raise", conflict_key="c")

        structlog.reset_defaults()

        assert frozenset() == get_protected_keys()
        bl.bind(actor_id="bound").info("x", actor_id="kwarg")
        assert {"event": "x", "actor_id": "kwarg"} == last_kwargs(cl)


class TestContextvars:
    def test_merge_contextvars_protected(self, cl):
        """
        A protected context-local value wins over event kwargs; the
        displaced kwarg is collected under the conflict key.
        """
        protect_keys("actor_id")
        bl = structlog.BoundLogger(cl, [merge_contextvars], {})
        bind_contextvars(actor_id="middleware-user")

        bl.info("login", actor_id="spoofed")

        assert {
            "event": "login",
            "actor_id": "middleware-user",
            "protected_conflicts": {"actor_id": "spoofed"},
        } == last_kwargs(cl)

    def test_merge_contextvars_unprotected_unchanged(self, cl):
        """
        Without protection, existing event dict values keep winning over
        context-local values (setdefault semantics).
        """
        bl = structlog.BoundLogger(cl, [merge_contextvars], {})
        bind_contextvars(actor_id="ctx")

        bl.info("x", actor_id="kwarg")

        assert {"event": "x", "actor_id": "kwarg"} == last_kwargs(cl)

    def test_unbind_contextvars_releases_protection(self, cl):
        """
        After unbind_contextvars, event kwargs may set the key again.
        """
        protect_keys("actor_id")
        bl = structlog.BoundLogger(cl, [merge_contextvars], {})
        bind_contextvars(actor_id="ctx")
        unbind_contextvars("actor_id")

        bl.info("x", actor_id="from-payload")

        assert {"event": "x", "actor_id": "from-payload"} == last_kwargs(cl)

    def test_merge_contextvars_raise_mode(self, cl):
        """
        on_conflict="raise" also applies to context-local merges.
        """
        protect_keys("actor_id")
        configure_protection(on_conflict="raise")
        bl = structlog.BoundLogger(cl, [merge_contextvars], {})
        bind_contextvars(actor_id="ctx")

        with pytest.raises(ProtectedKeyError):
            bl.info("x", actor_id="spoofed")

    async def test_concurrent_requests_keep_their_actor(self, cl):
        """
        Two concurrent requests binding different actor_ids via
        contextvars each keep their own actor_id, even if payloads try to
        override it.
        """
        protect_keys("actor_id")
        bl = structlog.BoundLogger(cl, [merge_contextvars], {})
        barrier = asyncio.Barrier(2)

        async def request(actor, spoof):
            with bound_contextvars(actor_id=actor):
                # Interleave both requests between binding and logging.
                await barrier.wait()
                bl.info("request", actor_id=spoof, path=f"/{actor}")

        await asyncio.gather(
            request("user-a", "evil-a"), request("user-b", "evil-b")
        )

        events = sorted(
            (call.kwargs for call in cl.calls), key=lambda kw: kw["path"]
        )
        assert [
            {
                "event": "request",
                "actor_id": "user-a",
                "path": "/user-a",
                "protected_conflicts": {"actor_id": "evil-a"},
            },
            {
                "event": "request",
                "actor_id": "user-b",
                "path": "/user-b",
                "protected_conflicts": {"actor_id": "evil-b"},
            },
        ] == events


class TestStdlib:
    def test_stdlib_bound_logger(self, cl):
        """
        structlog.stdlib.BoundLogger enforces protected keys like the
        generic one.
        """
        protect_keys("actor_id")
        bl = structlog.stdlib.BoundLogger(cl, [], {})

        bl.bind(actor_id="real").info("x", actor_id="spoofed")

        assert {
            "event": "x",
            "actor_id": "real",
            "protected_conflicts": {"actor_id": "spoofed"},
        } == last_kwargs(cl)

    async def test_async_log_call_keeps_actor(self, cl):
        """
        An async log call on a stdlib AsyncBoundLogger carries the
        protected context-local actor_id into the worker thread and does
        not let the payload override it.
        """
        protect_keys("actor_id")
        abl = AsyncBoundLogger(cl, [merge_contextvars], {})

        with bound_contextvars(actor_id="middleware-user"):
            await abl.info("login", actor_id="spoofed")

        assert {
            "event": "login",
            "actor_id": "middleware-user",
            "protected_conflicts": {"actor_id": "spoofed"},
        } == last_kwargs(cl)


class TestTrustedProcessors:
    def test_processor_can_still_modify_protected_key(self, cl):
        """
        Protection only guards the merge of event kwargs; trusted
        processors still see and can change the final event, including
        protected keys and the conflicts key.
        """

        def audit_processor(logger, method_name, event_dict):
            event_dict["actor_id"] = "rewritten-by-processor"
            return event_dict

        protect_keys("actor_id")
        bl = structlog.BoundLogger(cl, [audit_processor], {})

        bl.bind(actor_id="real").info("x", actor_id="spoofed")

        assert {
            "event": "x",
            "actor_id": "rewritten-by-processor",
            "protected_conflicts": {"actor_id": "spoofed"},
        } == last_kwargs(cl)
