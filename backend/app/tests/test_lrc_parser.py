"""`.lrc` sidecar parsing."""
import pytest

from app.services.lyrics.lrc import (
    decode_lrc_bytes,
    parse_lrc,
    parse_lrc_bytes,
)


class TestBasicLrc:
    def test_metadata_tags(self):
        data = parse_lrc(
            "[ti:One More Time]\n[ar:Daft Punk]\n[al:Discovery]\n[by:me]\n"
            "[00:12.00]hello\n"
        )
        assert data.title == "One More Time"
        assert data.artist == "Daft Punk"
        assert data.album == "Discovery"
        assert data.author == "me"
        assert data.offset_ms == 0

    def test_line_timestamps(self):
        data = parse_lrc("[00:12.00]one\n[00:17.20]two\n")
        assert data.timed
        assert [(ln.start, ln.text) for ln in data.timed_lines()] == [
            (12.0, "one"),
            (17.2, "two"),
        ]

    def test_hundredths_and_milliseconds(self):
        data = parse_lrc(
            "[00:12.5]a\n[00:12.50]b\n[00:12.500]c\n"
        )
        starts = [ln.start for ln in data.timed_lines()]
        assert starts == [12.5, 12.5, 12.5]

    def test_minutes_beyond_59(self):
        data = parse_lrc("[100:00.00]long\n")
        assert data.timed_lines()[0].start == 6000.0

    def test_extended_hour_stamps(self):
        data = parse_lrc("[01:02:03.500]deep cut\n")
        assert data.timed_lines()[0].start == pytest.approx(3723.5)

    def test_multiple_stamps_per_line_emit_repeats(self):
        data = parse_lrc("[00:10.00][01:40.00]chorus line\n")
        assert [ln.start for ln in data.timed_lines()] == [10.0, 100.0]
        assert all(ln.text == "chorus line" for ln in data.timed_lines())

    def test_section_headers_stripped(self):
        data = parse_lrc("[00:10.00][Chorus]\n[00:12.00]real line\n")
        assert data.timed_lines()[0].text == ""

    def test_text_joins_non_blank_lines(self):
        data = parse_lrc("[00:10.00]one\n\n[00:12.00]two\n")
        assert data.text == "one\ntwo"

    def test_blank_timed_line_is_kept(self):
        # A timestamped empty line is an authored instrumental gap.
        data = parse_lrc("[00:10.00]one\n[00:20.00]\n[00:30.00]three\n")
        assert len(data.timed_lines()) == 3
        assert data.timed_lines()[1].text == ""


class TestEndTimes:
    def test_end_is_next_line_start(self):
        data = parse_lrc("[00:10.00]one\n[00:14.00]two\n")
        js = data.to_json()
        assert js[0]["end"] == 14.0
        assert js[1]["end"] is None


class TestOffset:
    def test_positive_offset_advances_lyrics(self):
        data = parse_lrc("[offset:+500]\n[00:12.00]one\n")
        assert data.offset_ms == 500
        assert data.timed_lines()[0].start == pytest.approx(11.5)

    def test_negative_offset_delays_lyrics(self):
        data = parse_lrc("[offset:-300]\n[00:12.00]one\n")
        assert data.timed_lines()[0].start == pytest.approx(12.3)

    def test_offset_clamps_at_zero(self):
        data = parse_lrc("[offset:+2000]\n[00:01.00]one\n")
        assert data.timed_lines()[0].start == 0.0

    def test_offset_applies_to_enhanced_words(self):
        data = parse_lrc(
            "[offset:+1000]\n[00:10.00]<00:10.00>a <00:11.00>b\n"
        )
        assert [w.start for w in data.timed_lines()[0].words] == [9.0, 10.0]


class TestEnhancedWordLevel:
    def test_word_timestamps(self):
        data = parse_lrc("[00:12.00]<00:12.00>one <00:12.40>two\n")
        words = data.timed_lines()[0].words
        assert [(w.start, w.text) for w in words] == [
            (12.0, "one"),
            (12.4, "two"),
        ]

    def test_multi_word_span_is_split_evenly(self):
        data = parse_lrc("[00:12.00]<00:12.00>a b c <00:13.20>d\n")
        words = data.timed_lines()[0].words
        assert [w.text for w in words] == ["a", "b", "c", "d"]
        assert words[0].start == pytest.approx(12.0)
        assert words[1].start == pytest.approx(12.4)
        assert words[2].start == pytest.approx(12.8)
        assert words[3].start == pytest.approx(13.2)

    def test_untimed_lead_in(self):
        data = parse_lrc("[00:12.00]oh <00:12.50>yeah\n")
        words = data.timed_lines()[0].words
        assert words[0].start is None
        assert words[0].text == "oh"
        assert words[1].start == pytest.approx(12.5)

    def test_word_level_flag(self):
        assert parse_lrc("[00:01.00]<00:01.00>a\n").word_level
        assert not parse_lrc("[00:01.00]a\n").word_level


class TestUnsynced:
    def test_no_timestamps_still_yields_text(self):
        data = parse_lrc("just some\nplain lyrics\n")
        assert not data.timed
        assert data.text == "just some\nplain lyrics"
        assert data.to_json() == []


class TestColonAmbiguity:
    def test_three_part_read_as_mm_ss_ff_by_default(self):
        # No fractional stamps anywhere -> old Winamp [mm:ss:ff] family.
        data = parse_lrc("[03:45:00]mid\n")
        assert data.timed_lines()[0].start == pytest.approx(225.0)

    def test_three_part_read_as_h_m_s_when_fractions_present(self):
        data = parse_lrc("[00:10.00]early\n[01:02:03.500]late\n")
        assert data.timed_lines()[1].start == pytest.approx(3723.5)

    def test_override_forces_mm_ss_ff(self):
        data = parse_lrc("[03:45:00]mid\n", colon_hundredths=True)
        assert data.timed_lines()[0].start == pytest.approx(225.0)

    def test_override_forces_h_m_s(self):
        data = parse_lrc("[03:45:00]mid\n", colon_hundredths=False)
        assert data.timed_lines()[0].start == pytest.approx(13_500.0)


class TestEncoding:
    def test_utf8_bom(self):
        data = parse_lrc_bytes("\ufeff[00:01.00]hi\n".encode("utf-8"))
        assert data.text == "hi"

    def test_gb18030_chinese(self):
        source = "[00:12.00]我爱北京天安门\n"
        data = parse_lrc_bytes(source.encode("gb18030"))
        assert data.text == "我爱北京天安门"

    def test_crlf(self):
        data = parse_lrc("[00:01.00]a\r\n[00:02.00]b\r\n")
        assert [ln.text for ln in data.timed_lines()] == ["a", "b"]

    def test_decode_never_raises(self):
        assert decode_lrc_bytes(b"\xff\xfe\x00garbage") is not None
