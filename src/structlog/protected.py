# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Protected context fields: keep integration-bound fields (for example an
``actor_id`` bound by authentication middleware) from being silently
overridden by event keyword arguments like ``log.info("msg",
**user_payload)``.

Protection is enforced where untrusted event keyword arguments meet trusted
bound context:

- in `structlog.BoundLoggerBase._process_event` (used by *all* bound
  loggers, including `structlog.stdlib.BoundLogger` and its async variant),
- and in `structlog.contextvars.merge_contextvars` for context-local
  values.

.. important::

    This is **not** tamper-proof log storage. It protects the *provenance*
    of declared fields at merge time. Trusted processors still run
    afterwards and can modify any key of the final event, including
    protected ones.

.. versionadded:: 26.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Literal

from .exceptions import ProtectedKeyError


__all__ = [
    "configure_protection",
    "get_protected_keys",
    "protect_keys",
    "reset_protection",
    "unprotect_keys",
]

DEFAULT_CONFLICT_KEY = "protected_conflicts"
"""
Default name of the event dict key under which displaced event keyword
arguments are collected when *on_conflict* is ``"move"``.
"""

_PROTECTED_KEYS: set[str] = set()
_on_conflict: Literal["move", "raise"] = "move"
_conflict_key: str = DEFAULT_CONFLICT_KEY


def protect_keys(*keys: str) -> None:
    """
    Declare *keys* as protected context fields.

    Once declared, a bound value for any of these keys -- no matter whether
    bound via `structlog.BoundLoggerBase.bind`, `new`, or
    `structlog.contextvars.bind_contextvars` -- can not be overridden by
    event keyword arguments. Conflicting event keyword arguments are
    handled according to `configure_protection`.

    Trusted code can still replace the value by binding it again, or remove
    the protection for a logger by unbinding the key.

    If a declared key is not bound at log time, event keyword arguments for
    it pass through normally.
    """
    _PROTECTED_KEYS.update(keys)


def unprotect_keys(*keys: str) -> None:
    """
    Remove *keys* from the set of protected context fields.

    Unknown keys are ignored.
    """
    _PROTECTED_KEYS.difference_update(keys)


def get_protected_keys() -> frozenset[str]:
    """
    Return the currently protected context field names.
    """
    return frozenset(_PROTECTED_KEYS)


def configure_protection(
    *,
    on_conflict: Literal["move", "raise"] | None = None,
    conflict_key: str | None = None,
) -> None:
    """
    Configure how conflicts between protected bound values and event
    keyword arguments are handled.

    Args:
        on_conflict:
            ``"move"`` (the default) keeps the bound value and moves the
            conflicting event keyword argument into a dict under
            *conflict_key*. ``"raise"`` raises
            `structlog.exceptions.ProtectedKeyError` instead. Conflicting
            values are never silently dropped.

        conflict_key:
            Name of the event dict key that collects displaced event
            keyword arguments when *on_conflict* is ``"move"``. Default:
            ``"protected_conflicts"``.

    Arguments that are `None` keep their current value.
    """
    global _on_conflict, _conflict_key

    if on_conflict is not None:
        if on_conflict not in ("move", "raise"):
            msg = (
                "on_conflict must be 'move' or 'raise', "
                f"not {on_conflict!r}."
            )
            raise ValueError(msg)
        _on_conflict = on_conflict

    if conflict_key is not None:
        if not isinstance(conflict_key, str) or not conflict_key:
            msg = (
                "conflict_key must be a non-empty string, "
                f"not {conflict_key!r}."
            )
            raise ValueError(msg)
        _conflict_key = conflict_key


def reset_protection() -> None:
    """
    Remove all protected keys and reset conflict handling to defaults.

    Called by `structlog.reset_defaults`.
    """
    global _on_conflict, _conflict_key

    _PROTECTED_KEYS.clear()
    _on_conflict = "move"
    _conflict_key = DEFAULT_CONFLICT_KEY


def _values_equal(bound: Any, incoming: Any) -> bool:
    try:
        return bool(bound == incoming)
    except Exception:
        return False


def _apply_protected(
    event_dict: MutableMapping[str, Any], trusted: Mapping[str, Any]
) -> None:
    """
    Re-assert *trusted* values for protected keys in *event_dict*.

    Values in *event_dict* that are displaced by a trusted value are moved
    under the configured conflict key, or a
    `structlog.exceptions.ProtectedKeyError` is raised, depending on the
    configuration. Does nothing if no keys are protected.
    """
    if not _PROTECTED_KEYS:
        return

    conflicts = {
        key: event_dict[key]
        for key in _PROTECTED_KEYS
        if key in trusted
        and key in event_dict
        and not _values_equal(trusted[key], event_dict[key])
    }

    if conflicts and _on_conflict == "raise":
        raise ProtectedKeyError(conflicts)

    for key in _PROTECTED_KEYS:
        if key in trusted:
            event_dict[key] = trusted[key]

    if conflicts:
        existing = event_dict.get(_conflict_key)
        if isinstance(existing, dict):
            event_dict[_conflict_key] = {**existing, **conflicts}
        else:
            event_dict[_conflict_key] = conflicts
