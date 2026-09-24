# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Incremental logfmt parsing.

This module is the read-side counterpart of
`structlog.processors.LogfmtRenderer`: anything the renderer emits can be
parsed back into the same fields (modulo the type coercions documented on
`LogfmtParser`), so logfmt escaping rules live in exactly one library.

.. versionadded:: 26.2.0
"""

from __future__ import annotations

import codecs

from typing import Any


__all__ = ["LogfmtError", "LogfmtParser", "Record"]


#: One parsed logfmt record: an ordered list of ``(key, value)`` pairs.
#:
#: Values are strings, or `True` for bare keys (flags).  Duplicate keys are
#: kept in order of appearance; convert to a dict yourself if you want a
#: different mapping strategy (e.g. ``dict(record)`` for last-wins).
Record = list[tuple[str, Any]]

_DEFAULT_MAX_RECORD_SIZE = 1_048_576  # 1 MiB of decoded text


class LogfmtError(ValueError):
    """
    A logfmt record could not be parsed.

    Attributes:
        offset:
            Absolute character offset within the decoded stream at which the
            problem was detected.

        record_offset:
            Absolute character offset within the decoded stream at which the
            offending record started.

        record:
            The text of the offending record.  Empty if the record was
            discarded before it could be buffered (e.g. because it exceeded
            the size limit).

    .. versionadded:: 26.2.0
    """

    def __init__(
        self,
        message: str,
        *,
        offset: int,
        record_offset: int,
        record: str = "",
    ):
        super().__init__(message)
        self.offset = offset
        self.record_offset = record_offset
        self.record = record


def _unescape(escaped: str) -> str:
    r"""
    Resolve backslash escapes within a quoted value.

    Known escapes are ``\\``, ``\"``, and ``\n`` -- exactly the
    ones `structlog.processors.LogfmtRenderer` produces.  Unknown escapes
    are kept verbatim: the backslash is never silently dropped.
    """
    if "\\" not in escaped:
        return escaped

    out: list[str] = []
    i = 0
    n = len(escaped)
    while i < n:
        c = escaped[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue

        nxt = escaped[i + 1]
        if nxt == "n":
            out.append("\n")
        elif nxt in ('"', "\\"):
            out.append(nxt)
        else:
            # Unknown escape: keep both characters.
            out.append(c)
            out.append(nxt)
        i += 2

    return "".join(out)


def _parse_record(record: str, record_offset: int) -> Record:
    """
    Parse one logfmt record (a single line, without the newline).

    Pairs are separated by spaces.  Raises `LogfmtError` on malformed
    input.
    """
    fields: Record = []
    i = 0
    n = len(record)

    while True:
        while i < n and record[i] == " ":
            i += 1
        if i >= n:
            return fields

        key_start = i
        while i < n and record[i] not in "= ":
            i += 1
        key = record[key_start:i]
        if not key:
            raise LogfmtError(
                "Empty key.",
                offset=record_offset + i,
                record_offset=record_offset,
                record=record,
            )

        if i >= n or record[i] == " ":
            # Bare key: a flag, as rendered from {"key": True}.
            fields.append((key, True))
            continue

        # record[i] == "="
        i += 1
        if i < n and record[i] == '"':
            value, i = _parse_quoted(record, i, key, record_offset)
            fields.append((key, value))
        else:
            value_start = i
            while i < n and record[i] != " ":
                i += 1
            # Unquoted values are taken verbatim: backslashes are literal.
            fields.append((key, record[value_start:i]))


def _parse_quoted(
    record: str, i: int, key: str, record_offset: int
) -> tuple[str, int]:
    """
    Parse a quoted value starting at the opening quote at position *i*.

    Returns the unescaped value and the position after the closing quote.
    """
    quote_offset = record_offset + i
    n = len(record)
    i += 1
    value_start = i
    while True:
        if i >= n:
            raise LogfmtError(
                f"Unterminated quoted value for key {key!r}.",
                offset=quote_offset,
                record_offset=record_offset,
                record=record,
            )
        if record[i] == "\\":
            i += 2
            continue
        if record[i] == '"':
            break
        i += 1

    value = _unescape(record[value_start:i])
    i += 1
    if i < n and record[i] != " ":
        raise LogfmtError(
            f"Unexpected character {record[i]!r} after closing quote.",
            offset=record_offset + i,
            record_offset=record_offset,
            record=record,
        )

    return value, i


class LogfmtParser:
    r"""
    Incrementally parse logfmt_ into records.

    .. _logfmt: https://brandur.org/logfmt

    Feed text or UTF-8 encoded bytes using `feed()` and get back the
    records that were completed by the chunk.  A record may span any number
    of chunks, and byte chunks may split multi-byte UTF-8 characters.  Call
    `flush()` at end of stream to parse a trailing record that is not
    terminated by a newline.

    Records are newline-delimited and parsed into ordered lists of
    ``(key, value)`` pairs (see `Record`).  The parser mirrors
    `structlog.processors.LogfmtRenderer`:

    - Quoted values understand the escapes ``\\``, ``\"``, and
      ``\n``.  Unknown escapes are kept verbatim -- the backslash is
      never silently dropped.
    - Unquoted values are taken verbatim (backslashes are literal).
    - ``key=`` parses to an empty string, a bare ``key`` to
      `True` (mirroring *bool_as_flag*).
    - Duplicate keys are preserved in order; choose your own mapping
      strategy, e.g. ``dict(record)`` for last-wins.

    Args:
        strict:
            If `True` (default), malformed records raise `LogfmtError`.
            If `False`, bad records are skipped, their errors are
            collected in `errors`, and parsing continues with the next
            record.

        max_record_size:
            Maximum size of one record in decoded characters before its
            terminating newline is seen.  Records that grow beyond this
            limit are rejected with a `LogfmtError` so an unterminated
            record cannot exhaust memory.  `None` disables the limit.

    Attributes:
        errors:
            Errors of skipped records, in stream order.  Only populated in
            non-strict mode.

    .. versionadded:: 26.2.0
    """

    def __init__(
        self,
        *,
        strict: bool = True,
        max_record_size: int | None = _DEFAULT_MAX_RECORD_SIZE,
    ):
        self.strict = strict
        self.max_record_size = max_record_size
        self.errors: list[LogfmtError] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")(
            "strict" if strict else "replace"
        )
        self._buf: list[str] = []
        self._buf_len = 0
        # Absolute offset of the first buffered character.
        self._buf_offset = 0
        # Discarding an oversized record until its newline arrives.
        self._discarding = False

    def feed(self, data: str | bytes) -> list[Record]:
        """
        Feed a chunk of text or UTF-8 bytes.

        Returns the records completed by this chunk.  Raises `LogfmtError`
        on malformed records in strict mode.
        """
        if isinstance(data, (bytes, bytearray, memoryview)):
            text = self._decoder.decode(bytes(data))
        else:
            text = data

        records: list[Record] = []
        start = 0
        while True:
            nl = text.find("\n", start)
            if nl < 0:
                break

            self._feed_segment(text[start:nl])
            try:
                record = self._finish_record()
            finally:
                self._buf_offset += 1  # the newline itself

            if record is not None:
                records.append(record)
            start = nl + 1

        self._feed_segment(text[start:])
        return records

    def flush(self) -> list[Record]:
        """
        Signal end of stream and parse any trailing unterminated record.

        Returns the remaining records (at most one).  Raises `LogfmtError`
        on malformed records or truncated UTF-8 in strict mode.
        """
        try:
            tail = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as e:
            raise LogfmtError(
                f"Truncated UTF-8 at end of stream: {e}.",
                offset=self._buf_offset + self._buf_len,
                record_offset=self._buf_offset,
            ) from None

        self._feed_segment(tail)
        record = self._finish_record()
        return [record] if record is not None else []

    def _feed_segment(self, segment: str) -> None:
        """
        Buffer a newline-free segment, enforcing the record size limit.
        """
        if not segment or self._discarding:
            return

        if (
            self.max_record_size is not None
            and self._buf_len + len(segment) > self.max_record_size
        ):
            record_offset = self._buf_offset
            self._buf.clear()
            self._buf_len = 0
            self._discarding = True
            self._error(
                LogfmtError(
                    "Record exceeds max_record_size of "
                    f"{self.max_record_size} characters.",
                    offset=record_offset,
                    record_offset=record_offset,
                )
            )
            return

        self._buf.append(segment)
        self._buf_len += len(segment)

    def _finish_record(self) -> Record | None:
        """
        Parse and reset the buffered record, if any.
        """
        record_text = "".join(self._buf)
        record_offset = self._buf_offset
        self._buf.clear()
        self._buf_offset += self._buf_len
        self._buf_len = 0
        was_discarding = self._discarding
        self._discarding = False

        if was_discarding or not record_text.strip():
            return None

        return self._parse(record_text, record_offset)

    def _parse(self, record_text: str, record_offset: int) -> Record | None:
        """
        Parse one record, honoring strict/non-strict error handling.
        """
        try:
            return _parse_record(record_text, record_offset)
        except LogfmtError as e:
            self._error(e)
            return None

    def _error(self, error: LogfmtError) -> None:
        if self.strict:
            raise error

        self.errors.append(error)
