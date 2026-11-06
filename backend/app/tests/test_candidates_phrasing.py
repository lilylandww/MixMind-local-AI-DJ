"""Phrase-grid tests: seam candidates land on 8/16-bar phrase boundaries.

120 BPM, 4/4 → 2 s per bar. Downbeats every 2 s. Sections re-anchor the
phrase counter, so grid points are section starts plus every Nth downbeat
within the section.
"""

from __future__ import annotations

from app.services.mixer.candidates import (
    build_in_candidates,
    build_out_candidates,
    cap_in_candidates_for_out,
    max_seam_time,
    phrase_grid,
    _snap_to_phrase,
    _snap_to_phrase_at_or_before,
)
from app.services.mixer.types import AnalysisBundle

SPB = 2.0  # seconds per bar at 120 BPM 4/4
DUR = 240.0
DOWNBEATS = [i * SPB for i in range(int(DUR / SPB))]
SECTIONS = [
    {"start": 0.0, "end": 30.0, "label": "intro"},
    {"start": 30.0, "end": 150.0, "label": "body"},
    {"start": 150.0, "end": 240.0, "label": "outro"},
]


def _bundle(duration: float = DUR, sections=None) -> AnalysisBundle:
    sections = SECTIONS if sections is None else sections
    downbeats = [i * SPB for i in range(int(duration / SPB))]
    return AnalysisBundle(
        bpm=120.0, key="C", camelot_key="8B", time_signature=4,
        beat_grid=[i * 0.5 for i in range(int(duration / 0.5))],
        downbeats=downbeats, sections=sections, duration=duration,
    )


def test_phrase_grid_anchors_at_sections_16_bars():
    grid = phrase_grid(DOWNBEATS, SECTIONS)
    # Section anchors: 0, 30, 150. 16 bars = 32 s between boundaries.
    assert 0.0 in grid and 30.0 in grid and 150.0 in grid
    assert 62.0 in grid and 94.0 in grid and 126.0 in grid  # body phrases
    assert 182.0 in grid and 214.0 in grid                   # outro phrases
    assert 32.0 not in grid  # intro's next 16-bar step is past the section


def test_phrase_grid_falls_back_to_8_bars_on_short_songs():
    short_downbeats = [i * SPB for i in range(30)]  # 60 s song
    grid = phrase_grid(short_downbeats, [{"start": 0.0, "end": 60.0}])
    # A 16-bar grid would only have 0 and 32 (2 boundaries) — too sparse.
    assert grid == [0.0, 16.0, 32.0, 48.0]


def test_phrase_grid_without_data():
    assert phrase_grid([], SECTIONS) == []
    no_sections = phrase_grid(DOWNBEATS[:40], [])
    assert no_sections[0] == 0.0
    assert all(t % (16 * SPB) == 0 for t in no_sections)


def test_snap_helpers():
    grid = phrase_grid(DOWNBEATS, SECTIONS)
    assert _snap_to_phrase(30.0, grid, DOWNBEATS) == 30.0     # exact
    assert _snap_to_phrase(31.0, grid, DOWNBEATS) == 62.0     # next boundary
    # Ceiling blocks the boundary -> plain downbeat snap.
    assert _snap_to_phrase(31.0, grid, DOWNBEATS, ceiling=40.0) == 32.0
    assert _snap_to_phrase_at_or_before(63.0, grid, DOWNBEATS) == 62.0
    # Empty grid -> downbeat behavior.
    assert _snap_to_phrase(31.0, [], DOWNBEATS) == 32.0


def test_out_candidates_land_on_phrase_grid():
    b = _bundle()
    grid = phrase_grid(b.downbeats, b.sections)
    ceiling = max_seam_time(b.duration, b.bpm, b.time_signature)
    cands = build_out_candidates(b, [0.5] * int(DUR), [])
    assert cands
    for c in cands:
        assert c.time <= ceiling
        assert c.time in grid or c.time in b.downbeats
    # At least one candidate advertises its phrase alignment to the LLM.
    assert any("phrase-aligned" in c.description for c in cands)


def test_in_candidates_land_on_phrase_grid():
    b = _bundle()
    grid = phrase_grid(b.downbeats, b.sections)
    energy = [0.2] * 30 + [1.0] * 120 + [0.3] * 90  # body = the drop
    cands = build_in_candidates(b, energy, [])
    assert cands
    for c in cands:
        assert c.time in grid or c.time in b.downbeats
    # The intro-end candidate snapped onto the section-30 phrase anchor.
    assert any(c.time == 30.0 for c in cands)


def test_out_candidates_full_mode_is_unchanged_by_the_fraction_knob():
    b = _bundle()
    energy = [0.5] * int(DUR)
    default = build_out_candidates(b, energy, [])
    explicit_full = build_out_candidates(b, energy, [], max_out_fraction=1.0)
    assert [c.time for c in default] == [c.time for c in explicit_full]


def test_out_candidates_capped_to_half_the_track():
    b = _bundle()
    cands = build_out_candidates(b, [0.5] * int(DUR), [], max_out_fraction=0.5)
    assert cands
    assert all(c.time <= DUR * 0.5 for c in cands)
    # The "latest allowed" exit survives and is close to the cap. This is
    # the regression that the earlier threshold (> 80 s) missed: snapping
    # back to the previous phrase produced 94 s and passed.
    assert max(c.time for c in cands) >= DUR * 0.5 - 2


def test_capped_out_candidates_survive_non_bar_aligned_durations():
    # 4:01 with a coarse intro/body/outro: the cap (120.5 s) isn't a bar
    # boundary. Snapping forward used to overshoot and drop the only
    # candidate, producing an empty menu and a silent full-length fallback.
    b = _bundle(duration=241.0)
    cands = build_out_candidates(b, [0.5] * 241, [], max_out_fraction=0.5)
    assert cands
    assert max(c.time for c in cands) >= 241.0 * 0.5 - 2
    assert all(c.time <= 241.0 * 0.5 for c in cands)


def test_capped_out_candidates_respect_a_musical_floor():
    # Without a floor, a coarse section list under a 50% cap offers
    # "mix out at 0:00" — no DJ leaves a track in its first bars.
    b = _bundle()
    cands = build_out_candidates(b, [0.5] * int(DUR), [], max_out_fraction=0.5)
    assert cands
    assert all(c.time >= DUR * 0.2 for c in cands)


def test_capped_out_candidates_prefer_late_section_boundaries():
    sections = [
        {"start": 0.0, "end": 30.0, "label": "intro"},
        {"start": 30.0, "end": 90.0, "label": "verse"},
        {"start": 90.0, "end": 150.0, "label": "chorus"},
        {"start": 150.0, "end": 240.0, "label": "outro"},
    ]
    b = _bundle(sections=sections)
    cands = build_out_candidates(b, [0.5] * int(DUR), [], max_out_fraction=0.5)
    # 90 s is the last section start under the 120 s cap, so it stays in
    # the menu. The capped "latest possible" point at 120 s also survives:
    # that is the exit that makes the mode actually ~half.
    assert any(c.time == 90.0 for c in cands)
    assert any(c.time == 120.0 for c in cands)
    assert max(c.time for c in cands) == 120.0


def test_cap_in_candidates_clears_the_next_out():
    b = _bundle(sections=[
        {"start": 0.0, "end": 30.0, "label": "intro"},
        {"start": 30.0, "end": 90.0, "label": "verse"},
        {"start": 90.0, "end": 150.0, "label": "chorus"},
        {"start": 150.0, "end": 240.0, "label": "outro"},
    ])
    energy = [0.3] * 30 + [0.5] * 60 + [0.95] * 60 + [0.3] * 90
    in_cands = build_in_candidates(b, energy, [])
    assert any(c.time == 90.0 for c in in_cands)

    # The next pair can exit at 90 s, so a 90 s entry plus any blend would
    # overlap; only the post-intro entry survives.
    kept = cap_in_candidates_for_out(
        in_cands, 90.0, 2.0, needs_ramp=True
    )
    assert [c.time for c in kept] == [30.0]

    # If the next pair's earliest exit is the 120 s cap, the 90 s entry
    # is still legal and should remain available.
    kept = cap_in_candidates_for_out(
        in_cands, 120.0, 2.0, needs_ramp=True
    )
    assert [c.time for c in kept] == [30.0, 90.0]
