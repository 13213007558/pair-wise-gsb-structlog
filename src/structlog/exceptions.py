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


class FanOutError(Exception):
    """
    One or more branches of `structlog.processors.FanOut` raised an
    exception while it was configured with ``on_error="collect"``.

    The original exceptions are available in the `errors` attribute.

    .. versionadded:: 26.2.0
    """

    def __init__(self, errors: list[Exception]):
        self.errors = tuple(errors)
        super().__init__(
            f"{len(self.errors)} fan-out branch(es) failed: "
            + ", ".join(repr(e) for e in self.errors)
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
