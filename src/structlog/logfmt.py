# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Incremental parsing of logfmt_ records as produced by
:class:`structlog.processors.LogfmtRenderer`.

.. _logfmt: https://brandur.org/logfmt
"""

from __future__ import annotations

import codecs


__all__ = ["LogfmtParseError", "LogfmtParser"]


class LogfmtParseError(ValueError):
    """
    A logfmt record could not be parsed.

    Attributes:
        message: What went wrong, without location information.

        offset:
            Absolute character offset within the parsed stream at which the
            problem was detected. For unterminated quoted values this is the
            position of the opening quote.

    .. versionadded:: 26.2.0
    """

    message: str
    offset: int

    def __init__(self, message: str, offset: int) -> None:
        super().__init__(f"{message} (at offset {offset})")
        self.message = message
        self.offset = offset


_ESCAPES = {"n": "\n", '"': '"', "\\": "\\"}


def _unescape(char: str) -> str:
    r"""
    Resolve the character following a backslash.

    Known escapes (``\n``, ``\"``, ``\\``) are
    resolved. Anything else keeps the backslash, so unknown escapes
    round-trip instead of silently losing information.
    """
    return _ESCAPES.get(char, "\\" + char)


class LogfmtParser:
    r"""
    Incrementally parse logfmt_ records into lists of fields.

    Feed text or UTF-8 bytes in arbitrary chunks using :meth:`feed`; a
    record may span any number of chunks and byte chunks may split multi-byte
    UTF-8 characters. Each completed record (records are delimited by
    newlines) is returned as a list of ``(key, value)`` tuples in
    their original order -- duplicate keys are preserved and it is up to the
    caller how to map them (e.g. ``dict(fields)`` keeps the last
    value). Bare keys (``flag``) have a value of None, empty values
    (``key=``) parse to ``""``. Empty lines are skipped.

    Escapes mirror :class:`structlog.processors.LogfmtRenderer`:
    ``\n`` is a newline, ``\"`` a quote, and ``\\`` a
    backslash. Unknown escapes keep their backslash instead of dropping it.
    Anything produced by :class:`structlog.processors.LogfmtRenderer`
    round-trips through this parser.

    Args:
        strict:
            When True (the default), a malformed record raises
            :class:`LogfmtParseError`. No data is lost: records parsed
            before the error are returned by the next call, and parsing
            resumes after the malformed record, so the following records are
            not swallowed. When False, malformed records are silently
            skipped.

        max_record_size:
            Maximum number of characters a single not-yet-terminated record
            may buffer before it is rejected. Guards against unbounded memory
            growth from never-ending records.

    .. versionadded:: 26.2.0
    """

    def __init__(
        self, *, strict: bool = True, max_record_size: int = 1_048_576
    ) -> None:
        if max_record_size < 1:
            msg = "max_record_size must be at least 1"
            raise ValueError(msg)

        self._strict = strict
        self._max_record_size = max_record_size
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buf = ""
        self._record_start = 0
        self._discarding = False
        self._pending: list[list[tuple[str, str | None]]] = []

    def feed(
        self, data: str | bytes | bytearray | memoryview
    ) -> list[list[tuple[str, str | None]]]:
        """
        Feed a chunk of text or UTF-8 bytes and return completed records.
        """
        if isinstance(data, str):
            text = data
        elif isinstance(data, bytes | bytearray | memoryview):
            text = self._decoder.decode(bytes(data))
        else:
            msg = f"data must be str or bytes-like, not {type(data).__name__}"
            raise TypeError(msg)

        self._buf += text

        return self._drain(final=False)

    def flush(self) -> list[list[tuple[str, str | None]]]:
        """
        Signal end of input and return any remaining records.

        A trailing record without a final newline is parsed, and incomplete
        UTF-8 sequences raise :class:`UnicodeDecodeError`.
        """
        self._buf += self._decoder.decode(b"", final=True)

        return self._drain(final=True)

    def _drain(self, final: bool) -> list[list[tuple[str, str | None]]]:
        records: list[list[tuple[str, str | None]]] = []

        while self._buf:
            nl = self._buf.find("\n")

            if self._discarding:
                # Discard the rest of a rejected record.
                if nl < 0:
                    self._record_start += len(self._buf)
                    self._buf = ""
                    break

                self._record_start += nl + 1
                self._buf = self._buf[nl + 1 :]
                self._discarding = False
                continue

            if nl < 0:
                self._enforce_size_limit(records)
                break

            line = self._buf[:nl]
            base = self._record_start
            self._record_start += nl + 1
            self._buf = self._buf[nl + 1 :]

            if line.strip(" "):
                self._parse_into(records, line, base)

        if final:
            self._finish(records)

        return self._take_pending() + records

    def _enforce_size_limit(
        self, records: list[list[tuple[str, str | None]]]
    ) -> None:
        if len(self._buf) <= self._max_record_size:
            return

        offset = self._record_start
        self._record_start += len(self._buf)
        self._buf = ""
        self._discarding = True
        if self._strict:
            self._pending = records
            raise LogfmtParseError(
                "record exceeds maximum size of "
                f"{self._max_record_size} characters",
                offset,
            )

    def _parse_into(
        self,
        records: list[list[tuple[str, str | None]]],
        line: str,
        base: int,
    ) -> None:
        try:
            records.append(self._parse_record(line, base))
        except LogfmtParseError:
            if self._strict:
                self._pending = records
                raise

    def _finish(self, records: list[list[tuple[str, str | None]]]) -> None:
        if self._discarding:
            self._discarding = False
            self._buf = ""

            return

        if not self._buf.strip(" "):
            self._buf = ""

            return

        line, base = self._buf, self._record_start
        self._buf = ""
        self._record_start += len(line)
        self._parse_into(records, line, base)

    def _take_pending(self) -> list[list[tuple[str, str | None]]]:
        pending, self._pending = self._pending, []

        return pending

    def _parse_record(
        self, line: str, base: int
    ) -> list[tuple[str, str | None]]:
        fields: list[tuple[str, str | None]] = []
        pos = 0
        end = len(line)

        while pos < end:
            while pos < end and line[pos] == " ":
                pos += 1
            if pos >= end:
                break

            key_start = pos
            while pos < end and line[pos] not in " =":
                pos += 1
            key = line[key_start:pos]

            if not key:
                raise LogfmtParseError("missing key", base + pos)

            if pos >= end or line[pos] == " ":
                fields.append((key, None))
                continue

            # line[pos] == "="
            pos += 1
            if pos < end and line[pos] == '"':
                value, pos = self._parse_quoted(line, pos, base)
            else:
                chars: list[str] = []
                while pos < end and line[pos] != " ":
                    if line[pos] == "\\" and pos + 1 < end:
                        chars.append(_unescape(line[pos + 1]))
                        pos += 2
                    else:
                        chars.append(line[pos])
                        pos += 1
                value = "".join(chars)

            fields.append((key, value))

        return fields

    @staticmethod
    def _parse_quoted(line: str, pos: int, base: int) -> tuple[str, int]:
        quote_offset = base + pos
        pos += 1  # Opening quote.
        chars: list[str] = []
        end = len(line)

        while pos < end:
            char = line[pos]
            if char == '"':
                pos += 1
                if pos < end and line[pos] != " ":
                    raise LogfmtParseError(
                        "unexpected character after closing quote", base + pos
                    )

                return "".join(chars), pos

            if char == "\\":
                if pos + 1 >= end:
                    break
                chars.append(_unescape(line[pos + 1]))
                pos += 2
            else:
                chars.append(char)
                pos += 1

        raise LogfmtParseError("unterminated quoted value", quote_offset)
