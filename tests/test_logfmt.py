# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Tests for structlog.logfmt.
"""

import pytest

from structlog.logfmt import LogfmtParseError, LogfmtParser
from structlog.processors import LogfmtRenderer


def parse(text, **kwargs):
    """
    Parse *text* in one shot and return all records.
    """
    parser = LogfmtParser(**kwargs)
    records = parser.feed(text)
    records += parser.flush()

    return records


SAMPLE = (
    'a=1 b="two words" flag empty= dup=1 dup=2 q="esc \\" \\\\ \\n"\n'
    'uni="héllo→世界" bare=C:\\path unknown="\\t"\n'
    'last="no trailing newline"'
)

SAMPLE_BYTES = SAMPLE.encode()

SAMPLE_RECORDS = [
    [
        ("a", "1"),
        ("b", "two words"),
        ("flag", None),
        ("empty", ""),
        ("dup", "1"),
        ("dup", "2"),
        ("q", 'esc " \\ \n'),
    ],
    [
        ("uni", "héllo→世界"),
        ("bare", "C:\\path"),
        ("unknown", "\\t"),
    ],
    [("last", "no trailing newline")],
]


class TestParse:
    def test_sample_one_shot(self):
        """
        The sample input parses into the expected records.
        """
        assert parse(SAMPLE) == SAMPLE_RECORDS

    def test_simple_record(self):
        """
        A simple record parses into a list of key/value tuples.
        """
        assert parse("a=1 b=two\n") == [[("a", "1"), ("b", "two")]]

    def test_quoted_value(self):
        """
        Quoted values may contain spaces and equal signs.
        """
        assert parse('a="hello world" b="x=y"\n') == [
            [("a", "hello world"), ("b", "x=y")]
        ]

    def test_quoted_escapes(self):
        r"""
        \n, \" and \\ are resolved within quoted values.
        """
        assert parse('a="x\\ny\\"z\\\\"\n') == [[("a", 'x\ny"z\\')]]

    def test_unknown_escape_keeps_backslash(self):
        r"""
        Unknown escapes like \t keep their backslash instead of dropping it.
        """
        assert parse('a="\\t"\n') == [[("a", "\\t")]]
        assert parse("a=x\\ty\n") == [[("a", "x\\ty")]]

    def test_empty_value(self):
        """
        A key followed by a bare equal sign has an empty string value.
        """
        assert parse("a= b=1\n") == [[("a", ""), ("b", "1")]]

    def test_bare_key(self):
        """
        A key without a value parses to None.
        """
        assert parse("flag a=1\n") == [[("flag", None), ("a", "1")]]

    def test_duplicate_keys_preserved_in_order(self):
        """
        Duplicate keys are preserved in order; the caller chooses the
        mapping strategy (e.g. last wins via dict()).
        """
        (fields,) = parse("a=1 a=2 a=3\n")

        assert fields == [("a", "1"), ("a", "2"), ("a", "3")]
        assert dict(fields) == {"a": "3"}

    def test_multiple_records(self):
        """
        Records are delimited by newlines.
        """
        assert parse("a=1\nb=2\n") == [[("a", "1")], [("b", "2")]]

    def test_empty_lines_are_skipped(self):
        """
        Empty and whitespace-only lines produce no records.
        """
        assert parse("a=1\n\n  \nb=2\n") == [[("a", "1")], [("b", "2")]]

    def test_trailing_record_without_newline(self):
        """
        A final record without a trailing newline is parsed on flush().
        """
        assert parse("a=1") == [[("a", "1")]]

    def test_equal_sign_in_bare_value(self):
        """
        Bare values may contain equal signs.
        """
        assert parse("a=b=c\n") == [[("a", "b=c")]]

    def test_multiple_spaces_between_fields(self):
        """
        Runs of spaces separate fields.
        """
        assert parse("a=1   b=2\n") == [[("a", "1"), ("b", "2")]]

    def test_trailing_backslash_in_bare_value(self):
        """
        A trailing backslash in a bare value is kept literally.
        """
        assert parse("a=x\\\n") == [[("a", "x\\")]]


class TestChunking:
    def test_record_spans_chunks(self):
        """
        A record is only returned once it is complete.
        """
        parser = LogfmtParser()

        assert parser.feed('a="hel') == []
        assert parser.feed('lo" b=2') == []
        assert parser.feed("\n") == [[("a", "hello"), ("b", "2")]]

    @pytest.mark.parametrize("split", range(len(SAMPLE) + 1))
    def test_every_text_split_position(self, split):
        """
        Splitting the input into two text chunks at any position yields the
        same records as parsing it in one piece.
        """
        parser = LogfmtParser()
        records = parser.feed(SAMPLE[:split])
        records += parser.feed(SAMPLE[split:])
        records += parser.flush()

        assert records == SAMPLE_RECORDS

    @pytest.mark.parametrize("split", range(len(SAMPLE_BYTES) + 1))
    def test_every_byte_split_position(self, split):
        """
        Byte chunks may be split at any position, even in the middle of a
        multi-byte UTF-8 character.
        """
        parser = LogfmtParser()
        records = parser.feed(SAMPLE_BYTES[:split])
        records += parser.feed(SAMPLE_BYTES[split:])
        records += parser.flush()

        assert records == SAMPLE_RECORDS

    def test_byte_by_byte(self):
        """
        Feeding one byte at a time works.
        """
        parser = LogfmtParser()
        records = []
        for byte in SAMPLE_BYTES:
            records += parser.feed(bytes([byte]))
        records += parser.flush()

        assert records == SAMPLE_RECORDS

    def test_char_by_char(self):
        """
        Feeding one character at a time works.
        """
        parser = LogfmtParser()
        records = []
        for char in SAMPLE:
            records += parser.feed(char)
        records += parser.flush()

        assert records == SAMPLE_RECORDS

    def test_mixed_text_and_bytes(self):
        """
        Text and byte chunks can be mixed.
        """
        parser = LogfmtParser()
        records = parser.feed(SAMPLE[:10])
        records += parser.feed(SAMPLE[10:].encode())
        records += parser.flush()

        assert records == SAMPLE_RECORDS

    def test_bytearray_and_memoryview(self):
        """
        Bytes-like objects other than bytes are accepted.
        """
        parser = LogfmtParser()

        assert parser.feed(bytearray(b"a=1\n")) == [[("a", "1")]]
        assert parser.feed(memoryview(b"b=2\n")) == [[("b", "2")]]


class TestErrors:
    def test_unterminated_quote_strict(self):
        """
        An unterminated quote at the end of a record raises an error whose
        offset points at the opening quote. The following record is not
        swallowed, and records parsed before the error are not lost.
        """
        parser = LogfmtParser()

        with pytest.raises(LogfmtParseError) as ei:
            parser.feed('x=1\ny="abc\nz=2\n')

        assert ei.value.message == "unterminated quoted value"
        assert ei.value.offset == 6

        assert parser.flush() == [[("x", "1")], [("z", "2")]]

    def test_unterminated_quote_at_eof(self):
        """
        An unterminated quote at end of input raises on flush().
        """
        parser = LogfmtParser()
        parser.feed('a="unterminated')

        with pytest.raises(LogfmtParseError) as ei:
            parser.flush()

        assert ei.value.offset == 2

    def test_unterminated_quote_skipped_in_non_strict(self):
        """
        In non-strict mode, a record with an unterminated quote is skipped
        and parsing continues with the next record.
        """
        assert parse('a="bad\nb=2\n', strict=False) == [[("b", "2")]]

    def test_unexpected_character_after_closing_quote(self):
        """
        Garbage after a closing quote is an error with the offset of the
        offending character.
        """
        with pytest.raises(LogfmtParseError) as ei:
            parse('a="x"y\n')

        assert ei.value.message == "unexpected character after closing quote"
        assert ei.value.offset == 5

    def test_missing_key(self):
        """
        A value without a key is an error.
        """
        with pytest.raises(LogfmtParseError) as ei:
            parse("=1\n")

        assert ei.value.message == "missing key"
        assert ei.value.offset == 0

    def test_trailing_backslash_in_quoted_value(self):
        """
        A quoted value ending in a backslash is unterminated.
        """
        with pytest.raises(LogfmtParseError) as ei:
            parse('a="x\\\n')

        assert ei.value.message == "unterminated quoted value"

    def test_error_carries_message_and_offset(self):
        """
        LogfmtParseError exposes message and offset, is a ValueError, and
        renders the offset in its string form.
        """
        error = LogfmtParseError("some problem", 42)

        assert error.message == "some problem"
        assert error.offset == 42
        assert "42" in str(error)
        assert isinstance(error, ValueError)

    def test_invalid_data_type(self):
        """
        Feeding anything but str or bytes-like objects raises TypeError.
        """
        parser = LogfmtParser()

        with pytest.raises(TypeError, match="str or bytes-like"):
            parser.feed(42)

    def test_invalid_utf8(self):
        """
        Invalid UTF-8 raises UnicodeDecodeError.
        """
        parser = LogfmtParser()

        with pytest.raises(UnicodeDecodeError):
            parser.feed(b"\xff")

    def test_truncated_utf8_at_flush(self):
        """
        An incomplete UTF-8 sequence at end of input raises on flush().
        """
        parser = LogfmtParser()
        parser.feed("é".encode()[:1])

        with pytest.raises(UnicodeDecodeError):
            parser.flush()


class TestMaxRecordSize:
    def test_overlong_record_strict(self):
        """
        A record exceeding max_record_size without terminating raises in
        strict mode; parsing continues after the record's newline.
        """
        parser = LogfmtParser(max_record_size=10)

        assert parser.feed("ok=1\n") == [[("ok", "1")]]

        with pytest.raises(LogfmtParseError) as ei:
            parser.feed("bad=" + "x" * 20)

        assert ei.value.offset == 5

        assert parser.feed("more\nnext=2\n") == [[("next", "2")]]

    def test_overlong_record_skipped_in_non_strict(self):
        """
        In non-strict mode, an overlong record is discarded silently until
        its newline.
        """
        parser = LogfmtParser(strict=False, max_record_size=10)

        assert parser.feed("bad=" + "x" * 20) == []
        assert parser.feed("more\nnext=2\n") == [[("next", "2")]]

    def test_record_at_limit_is_ok(self):
        """
        A record of exactly max_record_size characters is accepted.
        """
        record = "a=" + "x" * 8

        assert parse(record + "\n", max_record_size=10) == [[("a", "x" * 8)]]

    def test_limit_counts_across_chunks(self):
        """
        The limit applies to records assembled from multiple chunks.
        """
        parser = LogfmtParser(max_record_size=10)

        assert parser.feed("a=xx") == []

        with pytest.raises(LogfmtParseError):
            parser.feed("x" * 8)

    def test_limit_must_be_positive(self):
        """
        max_record_size must be at least 1.
        """
        with pytest.raises(ValueError, match="max_record_size"):
            LogfmtParser(max_record_size=0)


ROUND_TRIP_VALUES = [
    "simple",
    "with space",
    'with"quote',
    "with=equals",
    "back\\slash",
    "literal \\n sequence",
    "real\nnewline",
    "double \\\\ backslash",
    "trailing\\",
    "C:\\new\\folder",
    "tab\there",
    "unicode → héllo 世界",
    "",
    " ",
    "\\",
    "\\n",
    "\\\\",
    '"',
    "=",
    'mix \\n and \n and " and \\',
]


class TestRoundTrip:
    @pytest.mark.parametrize("value", ROUND_TRIP_VALUES)
    def test_renderer_output_round_trips(self, value):
        """
        Every string value rendered by LogfmtRenderer parses back to the
        original value.
        """
        rendered = LogfmtRenderer()(None, None, {"key": value})

        assert parse(rendered + "\n") == [[("key", value)]]

    def test_event_dict_round_trip(self):
        """
        A whole event dict of string values round-trips.
        """
        event_dict = {
            "a": "1",
            "b": "hello world",
            "c": 'quote "and\\backslash',
            "d": "line\nbreak",
            "e": "",
        }

        rendered = LogfmtRenderer(sort_keys=True)(None, None, event_dict)
        (fields,) = parse(rendered + "\n")

        assert dict(fields) == event_dict

    def test_bool_flag_parses_to_none(self):
        """
        A flag rendered from True parses back as a bare key whose value is
        None -- logfmt does not preserve types.
        """
        rendered = LogfmtRenderer()(None, None, {"flag": True})

        assert parse(rendered + "\n") == [[("flag", None)]]
