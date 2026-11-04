"""`.lrc`-aware (line-timed) lyric alignment: drift correction, chorus
disambiguation, and the Whisper -> `.lrc` timing fallback chain."""
from __future__ import annotations

import pytest

from app.models.lyrics import LyricsAlignmentStatus
from app.services.lyrics_alignment.aligner import align_lyrics


def _seg(words: list[tuple[str, float, float, float]]) -> list[dict]:
    return [{
        "start": words[0][1] if words else 0.0,
        "end": words[-1][2] if words else 0.0,
        "text": " ".join(w[0] for w in words),
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.5,
        "temperature": 0.0,
        "words": [
            {"word": w, "start": s, "end": e, "probability": p}
            for (w, s, e, p) in words
        ],
    }]


def _lines(*specs) -> list[dict]:
    """(start, end, text) triples -> the persisted line_timestamps shape."""
    return [
        {"start": s, "end": e, "text": t, "words": None}
        for (s, e, t) in specs
    ]


class TestDriftCorrection:
    def test_constant_offset_is_estimated_and_applied(self):
        # The whole .lrc sits 1.5 s late — the classic hand-made offset.
        lines = _lines(
            (2.0, 4.0, "one two three"),
            (4.0, 6.0, "four five six"),
            (6.0, 8.0, "seven eight nine"),
        )
        segs = _seg([
            ("one", 0.5, 0.7, 0.9), ("two", 0.85, 1.0, 0.9), ("three", 1.2, 1.4, 0.9),
            ("four", 2.5, 2.7, 0.9), ("five", 2.85, 3.0, 0.9), ("six", 3.2, 3.4, 0.9),
            ("seven", 4.5, 4.7, 0.9), ("eight", 4.85, 5.0, 0.9), ("nine", 5.2, 5.4, 0.9),
        ])
        out = align_lyrics(segs, "", line_times=lines)
        assert out["offset"] == pytest.approx(-1.5)
        assert out["slope"] == pytest.approx(1.0)
        starts = [a["start"] for a in out["aligned_words"]]
        assert starts == pytest.approx([0.5, 0.85, 1.2, 2.5, 2.85, 3.2, 4.5, 4.85, 5.2])

    def test_stretched_lrc_gets_a_slope(self):
        # The .lrc came off a different master: 10% long, 2 s early.
        def t(x: float) -> float:
            return 1.1 * x - 2.0

        lines = []
        segs = []
        for base in (10.0, 20.0, 30.0, 40.0):
            lines.append({
                "start": base,
                "end": base + 1.0,
                "text": "a b",
                "words": [
                    {"start": base, "text": "a"},
                    {"start": base + 0.5, "text": "b"},
                ],
            })
            segs.extend(_seg([("a", t(base), t(base) + 0.2, 0.9)]))
            segs.extend(_seg([("b", t(base + 0.5), t(base + 0.5) + 0.2, 0.9)]))

        out = align_lyrics(segs, "", line_times=lines)
        assert out["slope"] == pytest.approx(1.1, abs=1e-6)
        assert out["offset"] == pytest.approx(-2.0, abs=1e-6)

    def test_persisted_warp_is_honoured_over_the_estimate(self):
        lines = _lines((2.0, 4.0, "one two three"), (4.0, 6.0, "four five six"))
        segs = _seg([
            ("one", 0.5, 0.7, 0.9), ("two", 0.85, 1.0, 0.9), ("three", 1.2, 1.4, 0.9),
            ("four", 2.5, 2.7, 0.9), ("five", 2.85, 3.0, 0.9), ("six", 3.2, 3.4, 0.9),
        ])
        out = align_lyrics(
            segs, "", line_times=lines, alignment_offset=0.0, alignment_slope=1.0
        )
        assert out["offset"] == 0.0
        assert out["slope"] == 1.0

    def test_no_anchors_yields_identity_warp(self):
        out = align_lyrics([], "", line_times=_lines((1.0, 2.0, "hello")))
        assert out["offset"] == 0.0
        assert out["slope"] == 1.0


class TestRepeatedChoruses:
    def test_each_occurrence_keeps_its_own_timings(self):
        """The motivating case: a chorus sung three times where Whisper
        drops two words from the middle occurrence. A free global match
        would borrow the third occurrence's timings for the second; the
        per-line partition cannot."""
        lines = _lines(
            (0.0, 4.0, "hey now hey now"),
            (10.0, 14.0, "hey now hey now"),
            (20.0, 24.0, "hey now hey now"),
        )
        segs = (
            _seg([
                ("hey", 0.0, 0.4, 0.9), ("now", 0.5, 0.9, 0.9),
                ("hey", 1.0, 1.4, 0.9), ("now", 1.5, 1.9, 0.9),
            ])
            # Middle chorus: Whisper only caught the first half.
            + _seg([("hey", 10.0, 10.4, 0.9), ("now", 10.5, 10.9, 0.9)])
            + _seg([
                ("hey", 20.0, 20.4, 0.9), ("now", 20.5, 20.9, 0.9),
                ("hey", 21.0, 21.4, 0.9), ("now", 21.5, 21.9, 0.9),
            ])
        )
        out = align_lyrics(segs, "", line_times=lines)
        words = out["aligned_words"]
        assert [a["word"] for a in words] == ["hey", "now", "hey", "now"] * 3

        middle = words[4:8]
        # The two Whisper-backed words land on the middle chorus…
        assert middle[0]["start"] == pytest.approx(10.0)
        assert middle[1]["start"] == pytest.approx(10.5)
        # …and the two Whisper missed are recovered from the .lrc stamp,
        # NOT borrowed from the third chorus at 20 s.
        assert middle[2]["source"] == "lrc_line"
        assert middle[3]["source"] == "lrc_line"
        for a in middle:
            assert 10.0 <= a["start"] < 13.0

    def test_no_cross_contamination_between_adjacent_identical_lines(self):
        lines = _lines((0.0, 2.0, "la la"), (10.0, 12.0, "la la"))
        segs = _seg([("la", 0.0, 0.3, 0.9), ("la", 0.4, 0.7, 0.9)]) + _seg([
            ("la", 10.0, 10.3, 0.9), ("la", 10.4, 10.7, 0.9),
        ])
        out = align_lyrics(segs, "", line_times=lines)
        starts = [a["start"] for a in out["aligned_words"]]
        assert starts[0] < 2.0 and starts[1] < 2.0
        assert starts[2] >= 10.0 and starts[3] >= 10.0


class TestFallbackChain:
    def test_missing_word_falls_back_to_lrc_line_timing(self):
        lines = _lines((0.0, 3.0, "hold the line"))
        segs = _seg([("hold", 0.0, 0.3, 0.95), ("line", 1.0, 1.3, 0.95)])
        out = align_lyrics(segs, "", line_times=lines)
        by_word = {a["word"]: a for a in out["aligned_words"]}
        assert by_word["hold"]["source"] == "whisper_match"
        assert by_word["line"]["source"] == "whisper_match"
        assert by_word["the"]["source"] == "lrc_line"
        # Recovered timing sits between its neighbours, not at 0/None.
        assert 0.3 < by_word["the"]["start"] < 1.0

    def test_enhanced_word_tags_are_authoritative(self):
        lines = [{
            "start": 5.0, "end": 7.0, "text": "hello world",
            "words": [{"start": 5.0, "text": "hello"}, {"start": 5.6, "text": "world"}],
        }]
        out = align_lyrics([], "", line_times=lines)
        words = out["aligned_words"]
        assert [a["source"] for a in words] == ["lrc_word", "lrc_word"]
        assert [a["start"] for a in words] == pytest.approx([5.0, 5.6])
        assert all(a["confidence"] == 1.0 for a in words)

    def test_enhanced_word_timing_beats_no_whisper_but_loses_to_whisper(self):
        lines = [{
            "start": 5.0, "end": 7.0, "text": "hello world",
            "words": [{"start": 5.0, "text": "hello"}, {"start": 5.6, "text": "world"}],
        }]
        segs = _seg([("hello", 5.05, 5.4, 0.9), ("world", 5.65, 6.0, 0.9)])
        out = align_lyrics(segs, "", line_times=lines)
        assert [a["source"] for a in out["aligned_words"]] == [
            "whisper_match", "whisper_match",
        ]
        assert [a["start"] for a in out["aligned_words"]] == pytest.approx([5.05, 5.65])

    def test_no_whisper_at_all_recoverable_from_lrc(self):
        lines = _lines((0.0, 3.0, "one two three"), (3.0, 6.0, "four five six"))
        out = align_lyrics([], "", line_times=lines)
        words = out["aligned_words"]
        assert len(words) == 6
        assert all(a["start"] is not None for a in words)
        assert all(a["source"] == "lrc_line" for a in words)
        # Line 2's words stay in line 2's neighbourhood.
        assert all(3.0 <= a["start"] < 6.0 for a in words[3:])

    def test_no_reference_at_all_is_an_error(self):
        # Nothing to align against — matches the empty-text contract.
        out = align_lyrics([], "", line_times=[])
        assert out["aligned_words"] == []
        assert out["alignment_status"] == LyricsAlignmentStatus.error


class TestUnspacedScripts:
    def test_japanese_whisper_word_splits_to_characters(self):
        # Whisper emits CJK as one unspaced "word"; it must still align
        # token-for-token against the reference.
        lines = _lines((0.0, 2.0, "東京の夜"))
        segs = _seg([("東京の夜", 0.0, 1.6, 0.9)])
        out = align_lyrics(segs, "", line_times=lines)
        assert [a["word"] for a in out["aligned_words"]] == ["東", "京", "の", "夜"]
        assert all(a["source"] == "whisper_match" for a in out["aligned_words"])
        assert out["aligned_words"][0]["start"] == pytest.approx(0.0)
        assert out["aligned_words"][-1]["end"] == pytest.approx(1.6)

    def test_chinese_matches_without_lrc(self):
        segs = _seg([("我爱北京", 0.0, 1.6, 0.9)])
        out = align_lyrics(segs, "我爱北京天安门")
        assert [a["word"] for a in out["aligned_words"]] == list("我爱北京天安门")
        assert out["aligned_words"][0]["source"] == "whisper_match"

    def test_accented_latin_still_matches(self):
        segs = _seg([("cafe", 0.0, 0.4, 0.9)])
        out = align_lyrics(segs, "café")
        assert out["aligned_words"][0]["source"] == "whisper_match"
        assert out["aligned_words"][0]["word"] == "café"

    def test_soundex_not_applied_across_scripts(self):
        # A CJK token must never be scored as a phonetic substitution.
        segs = _seg([("東京", 0.0, 0.8, 0.9)])
        out = align_lyrics(segs, "東X")
        assert all(
            a["source"] != "whisper_substitution" or a["confidence"] < 0.5
            for a in out["aligned_words"]
        )
