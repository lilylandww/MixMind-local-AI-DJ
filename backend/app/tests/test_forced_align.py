"""Pure mapping logic of the MMS_FA forced-alignment adapter.

The torch layer is deliberately thin and version-fragile; the part that
would silently corrupt DJ cut points is the frame-span -> word-span
grouping, so that is what is pinned here. No torch required.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from app.services.lyrics_alignment.forced_align import (
    ForcedAlignUnavailable,
    align_words,
    assign_word_times,
    merge_token_spans,
    _tokenise_words,
)


class TestMergeTokenSpans:
    def test_tuple_items_run_to_the_next_start(self):
        items = [(0, 0, 1.0), (1, 10, 1.0), (2, 25, 1.0)]
        spans = merge_token_spans(items, frame_rate=10.0, frame_count=30)
        assert spans == [(0.0, 1.0), (1.0, 2.5), (2.5, 3.0)]

    def test_attribute_items_are_supported(self):
        Item = namedtuple("Item", ["token", "start", "score"])
        items = [Item("a", 0, 1.0), Item("b", 100, 1.0)]
        spans = merge_token_spans(items, frame_rate=100.0, frame_count=250)
        assert spans == [(0.0, 1.0), (1.0, 2.5)]

    def test_last_token_runs_to_the_end_of_the_emission(self):
        spans = merge_token_spans([(0, 5, 1.0)], frame_rate=10.0, frame_count=40)
        assert spans == [(0.5, 4.0)]

    def test_unknown_item_shape_is_refused_not_guessed(self):
        with pytest.raises(ForcedAlignUnavailable):
            merge_token_spans([object()], frame_rate=1.0, frame_count=10)


class TestAssignWordTimes:
    def test_word_span_runs_first_char_to_last_char(self):
        words = ["a", "bb", "c"]
        spans = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
        out = assign_word_times(words, spans, [1, 2, 1])
        assert [(w["start"], w["end"]) for w in out] == [
            (0.0, 1.0), (1.0, 3.0), (3.0, 4.0),
        ]

    def test_words_keep_the_callers_orthography(self):
        # The flat text was romanised; the caller's word is what we show.
        out = assign_word_times(["日本"], [(0.0, 1.0), (1.0, 2.0)], [2])
        assert out[0]["word"] == "日本"
        assert (out[0]["start"], out[0]["end"]) == (0.0, 2.0)

    def test_zero_length_word_gets_no_timing(self):
        out = assign_word_times(["hello", "...", "world"],
                                [(0.0, 1.0), (1.0, 2.0)], [1, 0, 1])
        assert out[1]["start"] is None and out[1]["end"] is None
        assert out[0]["start"] == 0.0 and out[2]["start"] == 1.0

    def test_truncated_span_stream_never_borrows_neighbours(self):
        out = assign_word_times(["a", "b"], [(0.0, 1.0)], [1, 1])
        assert out[0]["start"] == 0.0
        assert out[1]["start"] is None


class _StubBundle:
    def __init__(self, table, romaniser=None):
        self._table = table
        self._romaniser = romaniser

    def get_dict(self, star=None):
        return self._table

    def get_to_phone(self):
        return self._romaniser


_LATIN = {chr(c): c - 97 for c in range(97, 123)}


class TestTokeniseWords:
    def test_ids_and_counts_partition_exactly(self):
        ids, counts = _tokenise_words(["Hello", "world"], _StubBundle(_LATIN))
        assert counts == [5, 5]
        assert len(ids) == sum(counts)
        assert ids[:5] == [_LATIN[c] for c in "hello"]

    def test_punctuation_only_word_contributes_nothing(self):
        ids, counts = _tokenise_words(["hi", "..."], _StubBundle(_LATIN))
        assert counts == [2, 0]
        assert len(ids) == 2

    def test_uncovered_script_is_refused_not_guessed(self):
        with pytest.raises(ForcedAlignUnavailable):
            _tokenise_words(["日本"], _StubBundle(_LATIN))

    def test_romaniser_keeps_counts_aligned_with_the_callers_words(self):
        romaniser = lambda form: "nihon"  # noqa: E731
        ids, counts = _tokenise_words(
            ["日本"], _StubBundle(_LATIN, romaniser=romaniser)
        )
        assert counts == [5]
        assert ids == [_LATIN[c] for c in "nihon"]


class TestAlignWordsGuards:
    def test_empty_input_never_touches_torch(self):
        assert align_words([], Path("/nope.wav")) == []
        assert align_words(["..."], Path("/nope.wav")) == []
