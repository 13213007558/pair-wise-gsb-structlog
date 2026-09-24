# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

from __future__ import annotations

import pytest

from structlog.logfmt import LogfmtError, LogfmtParser
from structlog.processors import LogfmtRenderer


def parse_all(text, **kwargs):
    """
    Feed *text* at once, flush, and return all records.
    """
    parser = LogfmtParser(**kwargs)
    records = parser.feed(text)
    records += parser.flush()
    return records


class TestParseRecords:
    def test_simple(self):
        assert [[("a", "1"), ("b", "2")]] == parse_all("a=1 b=2\n")

    def test_multiple_records(self):
        assert [[("a", "1")], [("b", "2")]] == parse_all("a=1\nb=2\n")

    def test_empty_value(self):
        assert [[("a", "")]] == parse_all("a=\n")

    def test_bare_key_is_flag(self):
        assert [[("flag", True)]] == parse_all("flag\n")

    def test_quoted_value(self):
        assert [[("a", "x y"), ("b", "x=y")]] == parse_all(
            'a="x y" b="x=y"\n'
        )

    def test_empty_quoted_value(self):
        assert [[("a", "")]] == parse_all('a=""\n')

    def test_unquoted_backslash_is_literal(self):
        assert [[("a", r"x\y")]] == parse_all("a=x\\y\n")

    def test_quoted_escapes(self):
        assert [[("a", 'x"y'), ("b", "x\\y"), ("c", "x\ny")]] == parse_all(
            'a="x\\"y" b="x\\\\y" c="x\\ny"\n'
        )

    def test_unknown_escape_keeps_backslash(self):
        """
        \t is not a known escape: both characters are kept verbatim.
        """
        assert [[("a", r"x\ty")]] == parse_all('a="x\\ty"\n')

    def test_unquoted_equal_sign(self):
        assert [[("a", "b=c")]] == parse_all("a=b=c\n")

    def test_duplicate_keys_keep_order(self):
        """
        Duplicates are preserved in order; the caller picks a mapping
        strategy.
        """
        (record,) = parse_all("a=1 a=2 a=3\n")

        assert [("a", "1"), ("a", "2"), ("a", "3")] == record
        assert {"a": "3"} == dict(record)

    def test_blank_lines_are_skipped(self):
        assert [[("a", "1")]] == parse_all("\n  \na=1\n\n")

    def test_trailing_record_without_newline(self):
        assert [[("a", "1")]] == parse_all("a=1")

    def test_empty_input(self):
        assert [] == parse_all("")

    def test_error_is_value_error(self):
        assert issubclass(LogfmtError, ValueError)


STREAM = "\n".join(
    [
        'event="hello world" level=info flag empty= n=1',
        'msg="esc \\" \\\\ \\n" key="哈 喽" emoji="🐍"',
        "a=1 a=2 a=3",
        "",  # trailing newline
    ]
)

EXPECTED = [
    [
        ("event", "hello world"),
        ("level", "info"),
        ("flag", True),
        ("empty", ""),
        ("n", "1"),
    ],
    [("msg", 'esc " \\ \n'), ("key", "哈 喽"), ("emoji", "🐍")],
    [("a", "1"), ("a", "2"), ("a", "3")],
]


class TestIncrementalText:
    @pytest.mark.parametrize("split", range(len(STREAM) + 1))
    def test_every_split_position(self, split):
        """
        Splitting the stream at any character boundary yields the same
        records.
        """
        parser = LogfmtParser()
        records = parser.feed(STREAM[:split])
        records += parser.feed(STREAM[split:])
        records += parser.flush()

        assert EXPECTED == records

    def test_char_by_char(self):
        parser = LogfmtParser()
        records = []
        for char in STREAM:
            records += parser.feed(char)
        records += parser.flush()

        assert EXPECTED == records

    def test_feed_returns_only_newly_completed_records(self):
        parser = LogfmtParser()

        assert [[("a", "1")]] == parser.feed("a=1\nb=2")
        assert [] == parser.feed("")
        assert [[("b", "2")]] == parser.feed("\n")
        assert [] == parser.flush()


STREAM_BYTES = STREAM.encode("utf-8")


class TestIncrementalBytes:
    @pytest.mark.parametrize("split", range(len(STREAM_BYTES) + 1))
    def test_every_byte_split_position(self, split):
        """
        Splitting the encoded stream at any byte -- even inside a
        multi-byte UTF-8 character -- yields the same records.
        """
        parser = LogfmtParser()
        records = parser.feed(STREAM_BYTES[:split])
        records += parser.feed(STREAM_BYTES[split:])
        records += parser.flush()

        assert EXPECTED == records

    def test_byte_by_byte(self):
        parser = LogfmtParser()
        records = []
        for i in range(len(STREAM_BYTES)):
            records += parser.feed(STREAM_BYTES[i : i + 1])
        records += parser.flush()

        assert EXPECTED == records

    def test_split_inside_multibyte_character(self):
        """
        A single 3-byte character fed byte by byte is decoded once
        complete.
        """
        data = "k=哈\n".encode("utf-8")
        parser = LogfmtParser()
        records = []
        for i in range(len(data)):
            records += parser.feed(data[i : i + 1])
        records += parser.flush()

        assert [[("k", "哈")]] == records


class TestErrors:
    def test_unterminated_quote_strict(self):
        parser = LogfmtParser()

        with pytest.raises(LogfmtError) as ei:
            parser.feed('a=1 b="oops\n')

        error = ei.value
        assert "Unterminated quoted value" in str(error)
        assert 0 == error.record_offset
        assert 6 == error.offset  # position of the opening quote
        assert 'a=1 b="oops' == error.record

    def test_unterminated_quote_does_not_swallow_next_record(self):
        """
        A newline always ends a record: an unclosed quote does not leak
        into the following record.
        """
        parser = LogfmtParser(strict=False)

        records = parser.feed('b="oops\nc=3\n')

        assert [[("c", "3")]] == records
        assert 1 == len(parser.errors)
        assert isinstance(parser.errors[0], LogfmtError)

    def test_error_offsets_account_for_previous_records(self):
        parser = LogfmtParser(strict=False)
        parser.feed("ok=1\n")
        parser.feed('bad="x\n')

        (error,) = parser.errors

        assert 5 == error.record_offset
        assert 9 == error.offset

    def test_error_offsets_count_decoded_characters(self):
        parser = LogfmtParser(strict=False)
        parser.feed("哈=1\n".encode("utf-8"))
        parser.feed('b="x\n')

        (error,) = parser.errors

        assert 4 == error.record_offset
        assert 6 == error.offset

    def test_empty_key(self):
        with pytest.raises(LogfmtError, match="Empty key"):
            parse_all("=x\n")

    def test_unexpected_character_after_quote(self):
        with pytest.raises(LogfmtError, match="Unexpected character"):
            parse_all('a="x"y\n')

    def test_non_strict_skips_bad_records_and_continues(self):
        parser = LogfmtParser(strict=False)

        records = parser.feed('good=1\nbad="x\nalso_good=2\n')

        assert [[("good", "1")], [("also_good", "2")]] == records
        assert 1 == len(parser.errors)

    def test_truncated_utf8_at_flush(self):
        parser = LogfmtParser()
        parser.feed(b"k=\xe5")

        with pytest.raises(LogfmtError, match="Truncated UTF-8"):
            parser.flush()

    def test_non_strict_replaces_invalid_utf8(self):
        parser = LogfmtParser(strict=False)
        records = parser.feed(b"k=\xe5\n")
        records += parser.flush()

        assert [[("k", "\ufffd")]] == records


class TestRecordSizeLimit:
    def test_strict_raises_when_record_too_big(self):
        parser = LogfmtParser(max_record_size=10)

        with pytest.raises(LogfmtError, match="max_record_size"):
            parser.feed("x" * 11)

    def test_limit_is_enforced_across_chunks(self):
        """
        An unterminated record cannot grow without bounds, no matter how
        many chunks it arrives in.
        """
        parser = LogfmtParser(max_record_size=10)
        parser.feed("x" * 6)

        with pytest.raises(LogfmtError, match="max_record_size"):
            parser.feed("y" * 5)

    def test_oversized_record_is_skipped_and_parsing_continues(self):
        parser = LogfmtParser(strict=False, max_record_size=10)

        records = parser.feed("x" * 50 + "\nok=1\n")

        assert [[("ok", "1")]] == records
        assert 1 == len(parser.errors)

    def test_oversized_record_spanning_chunks_is_discarded(self):
        parser = LogfmtParser(strict=False, max_record_size=10)
        parser.feed("x" * 20)
        records = parser.feed("y" * 20 + "\nok=1\n")

        assert [[("ok", "1")]] == records
        assert 1 == len(parser.errors)

    def test_record_at_limit_is_accepted(self):
        parser = LogfmtParser(max_record_size=10)

        assert [[("a", "12345678")]] == parser.feed("a=12345678\n")

    def test_limit_can_be_disabled(self):
        parser = LogfmtParser(max_record_size=None)

        assert [[("a", "x" * 100_000)]] == parser.feed(
            "a=" + "x" * 100_000 + "\n"
        )

    def test_default_limit_exists(self):
        parser = LogfmtParser()

        with pytest.raises(LogfmtError, match="max_record_size"):
            parser.feed("x" * (1_048_576 + 1))


ROUND_TRIP_VALUES = [
    "",
    "simple",
    "with space",
    'with"quote',
    "with\\backslash",
    "with\nnewline",
    "with\ttab",
    "with=equal",
    "哈 喽",
    "🐍",
    "true",
    "false",
    "null",
    "\\n",
    "\\",
    "\\\\",
    "\\n\\n",
    'mixed " \\ \n = 哈',
    "trailing\\",
    " leading",
    "trailing ",
]


class TestRoundTrip:
    @pytest.mark.parametrize("value", ROUND_TRIP_VALUES)
    def test_value_round_trip(self, value):
        """
        Every value LogfmtRenderer renders parses back to the same value.
        """
        line = LogfmtRenderer()(None, "", {"key": value})

        (record,) = parse_all(line + "\n")

        assert [("key", value)] == record

    def test_full_event_dict_round_trip(self):
        event_dict = {
            "event": "hello world",
            "level": "info",
            "flag": True,
            "not_flag": False,
            "nothing": None,
            "count": 42,
            "path": r"C:\Program Files\App",
            "quote": 'say "hi"',
            "newline": "line1\nline2",
            "unicode": "哈 喽 🐍",
            "empty": "",
        }

        line = LogfmtRenderer()(None, "", dict(event_dict))
        (record,) = parse_all(line + "\n")

        expected = {
            key: (
                True if value is True else "" if value is None else str(value)
            )
            for key, value in event_dict.items()
        }
        # False renders as the text "false", which parses back as a string.
        expected["not_flag"] = "false"
        assert expected == dict(record)

        # Re-rendering the parsed record reproduces the exact same line.
        assert line == LogfmtRenderer()(None, "", dict(record))

    def test_rendered_stream_round_trip_every_split(self):
        """
        A rendered multi-record stream parses to the same records no
        matter where the input is chunked.
        """
        stream = "".join(
            LogfmtRenderer()(
                None, "", {"i": i, "v": f"val {i}", "q": 'x"y\\z\n'}
            )
            + "\n"
            for i in range(5)
        )
        expected = parse_all(stream)

        for split in range(len(stream) + 1):
            parser = LogfmtParser()
            records = parser.feed(stream[:split])
            records += parser.feed(stream[split:])
            records += parser.flush()

            assert expected == records
