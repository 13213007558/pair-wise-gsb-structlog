# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Exceptions factored out to avoid import loops.
"""

from __future__ import annotations


class DropEvent(BaseException):
    """
    If raised by an processor, the event gets silently dropped.

    Derives from BaseException because it's technically not an error.
    """


class ProtectedKeyError(Exception):
    """
    An event keyword argument tried to override a protected context field
    while `structlog.protected` is configured with
    ``on_conflict="raise"``.

    The conflicting event keyword arguments are available as the
    `conflicts` attribute.

    .. versionadded:: 26.2.0
    """

    def __init__(self, conflicts: dict[str, object]) -> None:
        self.conflicts = conflicts
        keys = ", ".join(sorted(conflicts))
        super().__init__(
            "Event keyword argument(s) tried to override protected "
            f"context field(s): {keys}"
        )


class NoConsoleRendererConfiguredError(Exception):
    """
    A user asked for the current `structlog.dev.ConsoleRenderer` but none is
    configured.

    .. versionadded:: 25.5.0
    """


class MultipleConsoleRenderersConfiguredError(Exception):
    """
    A user asked for the current `structlog.dev.ConsoleRenderer` and more than one is configured.

    .. versionadded:: 25.5.0
    """
