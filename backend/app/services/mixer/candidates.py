"""Seam-candidate generation for planner v2.

The single biggest failure mode of the free-form (v1) LLM planner was
asking the model to *derive* seam timestamps from a sections array —
LLMs are unreliable at that kind of arithmetic, so plans routinely
violated headroom/vocal-safety rules and were silently replaced by the
deterministic fallback.

Planner v2 inverts the responsibility: this module pre-computes a small
menu of *valid-by-construction* seam candidates (downbeat-snapped,
inside the headroom budget, annotated with section role / energy /
vocal-safety), and the LLM only has to pick from the menu — e.g.
"A2 → B1". A wrong number becomes impossible; a wrong *choice* is at
worst a matter of taste.

Pure module: no DB, no storage, no settings. Everything arrives via
arguments so it can be unit-tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.mixer.types import AnalysisBundle

# Maximum crossfade length the planner may use; headroom is reserved for
# it when computing max_seam_time. Mirrors the deterministic planner.
MAX_CROSSFADE_BARS = 16
# Safety buffer (seconds) past the crossfade end. Absorbs stem-WAV drift
# vs `Song.duration_seconds` metadata (Demucs trims/pads, yt-dlp rounds).
SEAM_SAFETY_SECONDS = 5.0
# Cap the menu so the prompt stays small and the choice stays easy.
MAX_CANDIDATES = 6
# Two candidates closer than this are duplicates for mixing purposes.
DEDUP_SECONDS = 2.0
# B entry points past this fraction of the song are pointless — the
# listener would barely hear B before the *next* transition starts.
B_ENTRY_MAX_FRACTION = 0.5
# Per-queue `song_length` mode -> the FALLBACK fraction used when the
# analyzer can't identify a chorus. "full" plays tracks out; "half" means
# one chorus (mix out at the end of the first one). The last song is never
# an outgoing song, so it always plays to its natural end.
SONG_LENGTH_FRACTIONS = {"full": 1.0, "half": 0.5}
# Lower bound on an OUT point when the track length is capped. Capping the
# ceiling alone leaves short/coarse section lists offering "mix out at
# 0:00" as a menu entry; no DJ leaves a track in its first bars.
MIN_OUT_FRACTION = 0.2
# Half mode reserves room after B's entry for the incoming blend and the
# B-side tempo ramp, plus a small safety gap. The cap uses the EARLIEST
# OUT the next pair can choose, so it holds no matter which OUT that pair
# picks. Kept here (not in planner_v2) so the stitch guard and tests can
# share the same numbers.
HALF_MIN_BLEND_BARS = 2
HALF_MIN_RAMP_BARS = 4
HALF_SEAM_GAP_SECONDS = 1.0
# A vocal-safe entry must stay clear of vocals for this many bars after
# the candidate point for a hard cut to land cleanly.
VOCAL_SAFE_LOOKAHEAD_BARS = 2.0
# Sections with normalized energy at/above this read as a drop/chorus.
HIGH_ENERGY_THRESHOLD = 0.8
# ...and at/below this read as a breakdown/outro (breakdown_blend fuel).
LOW_ENERGY_THRESHOLD = 0.4
# Phrase lengths to try when building the phrase grid, in bars. Human DJs
# mix on 16/32-bar phrases; 16 fits typical 3+ minute dance tracks, 8 is
# the fallback for shorter songs where a 16-bar grid leaves too few
# boundaries to choose from.
PHRASE_BARS_CHOICES = (16, 8)
# A phrase-snapped time may overshoot the raw target by at most one
# phrase; anything past the headroom ceiling falls back to downbeat snap.
MIN_GRID_BOUNDARIES = 3


def max_seam_time(duration: float, bpm: float, time_signature: int) -> float:
    """Latest seam time that leaves room for a full-length crossfade.

    Returned in original-song seconds. Clamped to 0 if the song is
    shorter than the reserved tail.
    """
    if not bpm or not duration:
        return 0.0
    sec_per_bar = (60.0 / bpm) * time_signature
    return max(0.0, duration - MAX_CROSSFADE_BARS * sec_per_bar - SEAM_SAFETY_SECONDS)


def enrich_sections(sections: list[dict], energy_curve: list[float]) -> list[dict]:
    """Annotate each section with mean energy, normalized 0..1 over the song.

    The librosa analyzer's section `label`s are opaque cluster IDs
    (`section_1`…) that tell a planner nothing; a structure-aware
    detector (e.g. allin1) emits real roles (`chorus`, `verse`). We keep
    meaningful labels and drop the opaque ones. The energy curve is
    sampled at 1 Hz (index ≈ second); we average it per section and
    normalize to the song's hottest section so ~1.0 reads as a
    drop/chorus and low values as intro/breakdown/outro.
    """
    if not sections:
        return []
    n = len(energy_curve)
    raw: list[float] = []
    for s in sections:
        lo = int(s["start"])
        hi = max(lo + 1, int(round(s["end"])))
        window = energy_curve[lo:hi] if n else []
        raw.append(sum(window) / len(window) if window else 0.0)
    peak = max(raw) or 1.0
    out = []
    for s, e in zip(sections, raw):
        item = {
            "start": round(s["start"], 1),
            "end": round(s["end"], 1),
            "energy": round(e / peak, 2),
        }
        label = s.get("label")
        if label and not str(label).startswith("section_"):
            item["label"] = label
        out.append(item)
    return out


def _is_chorus_label(label: str | None) -> bool:
    """True for a functional chorus/drop label, excluding pre-chorus."""
    if not label:
        return False
    tokens = [t for t in re.split(r"[^a-z]+", str(label).lower()) if t]
    if "pre" in tokens:
        return False
    return "chorus" in tokens or "drop" in tokens


def first_chorus_section(
    sections: list[dict], energy_curve: list[float]
) -> dict | None:
    """The song's first chorus (or drop), or None when structure is opaque.

    The default librosa detector emits `section_N` cluster labels, so a
    label-only rule would never fire. Preference order:

      1. a functional label (`chorus` / `drop`, ignoring `pre-chorus`);
      2. the most energetic *repeated* cluster — librosa gives similar
         sections the same opaque label, and the chorus is usually the
         hot cluster that recurs;
      3. the first high-energy section (the existing drop/chorus proxy).

    Returns the enriched section dict (`start`, `end`, `energy`, and a
    functional `label` when the detector supplied one).
    """
    if not sections:
        return None
    enriched = enrich_sections(sections, energy_curve)
    if not enriched:
        return None

    # 1. Real labels (allin1 / any future functional detector).
    for s in enriched:
        if _is_chorus_label(s.get("label")):
            return s

    # A flat curve normalises to all-1.0 in enrich_sections, which would
    # make the intro look like a drop. Without labels, require real
    # dynamic range before trusting any energy inference.
    if not energy_curve:
        return None
    peak = max(energy_curve)
    mean = sum(energy_curve) / len(energy_curve)
    if peak <= 0 or peak - mean < 0.05:
        return None

    # 2. Opaque cluster labels: the hot cluster that repeats is the
    # chorus. `enrich_sections` drops these labels from the LLM view, but
    # the raw sections still carry them for this inference.
    groups: dict[str, list[dict]] = {}
    for raw, s in zip(sections, enriched):
        label = str(raw.get("label") or "")
        if label:
            groups.setdefault(label, []).append(s)
    repeated = [g for g in groups.values() if len(g) >= 2]
    if repeated:
        hot = max(
            repeated,
            key=lambda g: sum(x["energy"] for x in g) / len(g),
        )
        mean = sum(x["energy"] for x in hot) / len(hot)
        if mean >= 0.75:
            return hot[0]

    # 3. Energy proxy: the first convincingly hot section.
    for threshold in (HIGH_ENERGY_THRESHOLD, 0.7):
        for s in enriched:
            if s["energy"] >= threshold:
                return s
    return None


@dataclass(frozen=True)
class SeamCandidate:
    """One pre-validated seam point the LLM may choose."""

    id: str                # "A1", "B3", ...
    time: float            # seconds, original-song time, downbeat-snapped
    description: str       # human/LLM-readable role, e.g. "start of final section"
    energy: float          # normalized 0..1 section energy at this point
    vocal_safe: bool       # True if a hard cut here avoids chopping a word
    # Densities are 0.0-1.0 over the lookahead window, or None when the
    # underlying data (safety regions / stem envelopes) is missing —
    # None is omitted from the LLM dict rather than passed off as 0.0.
    vocal_density: float | None = None
    drum_density: float | None = None
    bass_density: float | None = None
    lyrics_preview: str | None = None

    def to_llm_dict(self) -> dict:
        d = {
            "id": self.id,
            "time": round(self.time, 2),
            "description": self.description,
            "energy": self.energy,
            "vocal_safe": self.vocal_safe,
        }
        for name in ("vocal_density", "drum_density", "bass_density"):
            value = getattr(self, name)
            if value is not None:
                d[name] = round(value, 2)
        if self.lyrics_preview:
            d["lyrics_preview"] = self.lyrics_preview
        return d


@dataclass(frozen=True)
class PairCandidates:
    out_candidates: list[SeamCandidate] = field(default_factory=list)
    in_candidates: list[SeamCandidate] = field(default_factory=list)

    def find(self, candidate_id: str) -> SeamCandidate | None:
        for c in (*self.out_candidates, *self.in_candidates):
            if c.id == candidate_id:
                return c
        return None


def _snap_to_downbeat(t: float, downbeats: list[float]) -> float:
    """First downbeat ≥ t, or the last downbeat if none qualifies."""
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def _snap_to_downbeat_at_or_before(t: float, downbeats: list[float]) -> float:
    """Latest downbeat ≤ t, or the first downbeat if none qualifies."""
    if not downbeats:
        return t
    best = None
    for d in downbeats:
        if d <= t:
            best = d
        else:
            break
    return best if best is not None else downbeats[0]


def phrase_grid(
    downbeats: list[float],
    sections: list[dict],
    bars_per_phrase: int | None = None,
) -> list[float]:
    """Times of musical phrase starts, derived from the downbeat grid.

    Human DJs mix on 8/16-bar phrase boundaries, not arbitrary downbeats
    — a seam on bar 3 of a phrase is on-grid but musically "off". Each
    section re-anchors the phrase counter at its first downbeat (song
    structure resets phrasing); within a section every Nth downbeat is a
    phrase start. With no section data the grid anchors at downbeat 0.

    When `bars_per_phrase` is None, tries PHRASE_BARS_CHOICES in order
    and keeps the first grid dense enough to be a real menu
    (>= MIN_GRID_BOUNDARIES boundaries) — long tracks get 16-bar
    phrasing, short ones fall back to 8.
    """
    if not downbeats:
        return []

    def _build(n_bars: int) -> list[float]:
        anchors: list[tuple[float, float]] = []  # (anchor_time, region_end)
        if sections:
            for i, s in enumerate(sections):
                end = sections[i + 1]["start"] if i + 1 < len(sections) else float("inf")
                anchors.append((float(s["start"]), float(end)))
        else:
            anchors.append((downbeats[0], float("inf")))

        grid: list[float] = []
        for start, end in anchors:
            # First downbeat at/after the anchor.
            idx = next((i for i, d in enumerate(downbeats) if d >= start), None)
            if idx is None:
                continue
            i = idx
            while i < len(downbeats) and downbeats[i] < end:
                if not grid or downbeats[i] > grid[-1]:
                    grid.append(downbeats[i])
                i += n_bars
        return grid

    if bars_per_phrase is not None:
        return _build(bars_per_phrase)
    grid: list[float] = []
    for n_bars in PHRASE_BARS_CHOICES:
        grid = _build(n_bars)
        if len(grid) >= MIN_GRID_BOUNDARIES:
            return grid
    return grid


def _snap_to_phrase(
    t: float,
    grid: list[float],
    downbeats: list[float],
    ceiling: float | None = None,
) -> float:
    """Next phrase boundary ≥ t; plain downbeat snap when the grid is
    empty, exhausted, or the boundary would overshoot the ceiling."""
    for g in grid:
        if g >= t:
            if ceiling is not None and g > ceiling:
                break
            return g
    return _snap_to_downbeat(t, downbeats)


def _snap_to_phrase_at_or_before(
    t: float, grid: list[float], downbeats: list[float]
) -> float:
    """Latest phrase boundary ≤ t; downbeat-at-or-before fallback."""
    best = None
    for g in grid:
        if g <= t:
            best = g
        else:
            break
    if best is not None:
        return best
    return _snap_to_downbeat_at_or_before(t, downbeats)


def _is_vocal_safe(
    t: float,
    safe_regions: list[dict],
    lookahead_seconds: float,
) -> bool:
    """True when [t, t + lookahead] sits inside one no-vocal span.

    `safe_regions` rows are `{"start", "end"}` (optionally with a
    `safe`/`reason` field from the vocal-safety service — when a `safe`
    key is present, only rows with safe=True count). With no region data
    at all we conservatively return False: "unknown" must not read as
    "safe" or hard cuts would chop words on un-transcribed songs.
    """
    if not safe_regions:
        return False
    for r in safe_regions:
        if "safe" in r and not r.get("safe"):
            continue
        if r["start"] <= t and (t + lookahead_seconds) <= r["end"]:
            return True
    return False


def _vocal_density(
    t: float,
    safe_regions: list[dict],
    lookahead_seconds: float,
) -> float | None:
    """Fraction of [t, t + lookahead] NOT covered by a safe (silent) region.

    Returns None when there is no safety data at all — "unknown", which
    to_llm_dict omits, matching _is_vocal_safe's conservative False for
    the same case rather than claiming the point is instrumental.
    """
    if not safe_regions or lookahead_seconds <= 0:
        return None

    safe_dur = 0.0
    end_t = t + lookahead_seconds
    for r in safe_regions:
        if "safe" in r and not r.get("safe"):
            continue
        overlap_start = max(t, r["start"])
        overlap_end = min(end_t, r["end"])
        if overlap_start < overlap_end:
            safe_dur += (overlap_end - overlap_start)
            
    return min(1.0, (lookahead_seconds - safe_dur) / lookahead_seconds)


def _stem_density(
    t: float, envelopes: dict | None, stem_name: str, lookahead_seconds: float
) -> float | None:
    """Average stem RMS over the lookahead window, scaled to 0..1.

    None (= "unknown", omitted from the LLM dict) when the envelope
    sidecar is missing or predates the multi-stem schema."""
    if not envelopes or stem_name not in envelopes or lookahead_seconds <= 0:
        return None
    frame_hz = envelopes.get("frame_hz", 10)
    rms_array = envelopes[stem_name].get("rms", [])
    if not rms_array:
        return None
    start_frame = int(t * frame_hz)
    end_frame = int((t + lookahead_seconds) * frame_hz)
    end_frame = max(start_frame + 1, min(end_frame, len(rms_array)))
    if start_frame >= len(rms_array):
        return 0.0
    window = rms_array[start_frame:end_frame]
    avg = sum(window) / len(window)
    # Scale: an RMS of 0.15 is generally "full" density for a single stem
    return min(1.0, avg / 0.15)


def _lyrics_preview(
    t: float, segments: list[dict] | None, lookahead_seconds: float, is_out: bool
) -> str | None:
    if not segments:
        return None
    start_t = t - lookahead_seconds if is_out else t
    end_t = t if is_out else t + lookahead_seconds
    
    words = []
    for seg in segments:
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                if start_t <= w["start"] and w["end"] <= end_t:
                    words.append(w["word"].strip())
        else:
            if start_t <= seg["start"] and seg["end"] <= end_t:
                words.append(seg["text"].strip())
    
    if not words:
        return None
    text = " ".join(words)
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _phrase_tag(t: float, grid: list[float]) -> str:
    """Suffix for candidate descriptions when `t` sits on the phrase grid,
    so the LLM can prefer phrase-aligned seams explicitly."""
    return " (phrase-aligned)" if any(abs(t - g) < 1e-6 for g in grid) else ""


def _dedup_and_cap(
    cands: list[SeamCandidate], *, keep_last: bool = False
) -> list[SeamCandidate]:
    cands = sorted(cands, key=lambda c: c.time)
    kept: list[SeamCandidate] = []
    for c in cands:
        if kept and abs(c.time - kept[-1].time) < DEDUP_SECONDS:
            continue
        kept.append(c)
    if len(kept) <= MAX_CANDIDATES:
        return kept
    if keep_last:
        # The "latest possible" fallback is appended last and is the whole
        # point of the capped modes; don't let a long section list crowd it
        # out of the menu.
        return kept[: MAX_CANDIDATES - 1] + [kept[-1]]
    return kept[:MAX_CANDIDATES]


def _reid(cands: list[SeamCandidate], prefix: str) -> list[SeamCandidate]:
    return [
        SeamCandidate(
            id=f"{prefix}{i + 1}",
            time=c.time,
            description=c.description,
            energy=c.energy,
            vocal_safe=c.vocal_safe,
            vocal_density=c.vocal_density,
            drum_density=c.drum_density,
            bass_density=c.bass_density,
            lyrics_preview=c.lyrics_preview,
        )
        for i, c in enumerate(cands)
    ]


def build_out_candidates(
    a: AnalysisBundle,
    energy_curve: list[float],
    safe_regions: list[dict],
    *,
    max_out_fraction: float = 1.0,
    one_chorus: bool = False,
) -> list[SeamCandidate]:
    """OUT points for the outgoing song A: late-song section starts that
    leave a full crossfade of headroom, downbeat-snapped.

    `max_out_fraction` caps how far into the track the mix may linger.
    1.0 (the default) plays A out. 0.5 is the fallback cap used when
    `one_chorus` is set but no chorus can be identified — the exit is
    re-derived from the *kept* part (the last three sections under the
    cap), so it lands on a section/phrase boundary when one exists. The
    "latest possible" candidate is the last phrase boundary or downbeat
    at/before the cap, so a capped mode always leaves a usable menu even
    when the cap falls between bars.

    `one_chorus` switches to the intended short-format rule: find the
    song's first chorus and make the ONLY OUT candidate its end. A
    typical track repeats its chorus, and a mix only needs one — the
    duration cap is just a fallback for songs whose structure is too
    opaque to find a chorus.
    """
    hard_ceiling = max_seam_time(a.duration, a.bpm, a.time_signature)
    if hard_ceiling <= 0:
        return []
    ceiling = hard_ceiling
    sec_per_bar = (60.0 / a.bpm) * a.time_signature if a.bpm else 0.0
    lookahead = VOCAL_SAFE_LOOKAHEAD_BARS * sec_per_bar
    sections = enrich_sections(a.sections, energy_curve)
    if max_out_fraction < 1.0:
        ceiling = min(hard_ceiling, max(0.0, a.duration * max_out_fraction))
        # Filtering, rather than only lowering `ceiling`, is what lets the
        # rest of this function work unchanged: every "late song" rule
        # below is relative to the list it is handed. The floor keeps the
        # shortened menu musical — without it a two-section track would
        # offer "mix out at 0:00".
        floor = a.duration * MIN_OUT_FRACTION
        sections = [s for s in sections if floor <= s["start"] <= ceiling]
    grid = phrase_grid(a.downbeats, a.sections)

    def _out_candidate(t: float, description: str, energy: float) -> SeamCandidate:
        return SeamCandidate(
            id="A?", time=t, description=description, energy=energy,
            vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
            vocal_density=_vocal_density(t, safe_regions, lookahead),
            drum_density=_stem_density(t, a.envelopes, "drums", lookahead),
            bass_density=_stem_density(t, a.envelopes, "bass", lookahead),
            lyrics_preview=_lyrics_preview(
                t, a.transcription_segments, lookahead, is_out=True
            ),
        )

    raw: list[SeamCandidate] = []
    n = len(sections)

    if one_chorus:
        chorus = first_chorus_section(a.sections, energy_curve)
        if chorus is not None:
            # Start the crossfade on the next downbeat so the whole chorus
            # is heard. If that overruns the headroom, use the last safe
            # downbeat (the song plays out).
            t = _snap_to_downbeat(float(chorus["end"]), a.downbeats)
            if t > hard_ceiling:
                t = _snap_to_downbeat_at_or_before(
                    hard_ceiling, a.downbeats
                )
            if 0.0 < t <= hard_ceiling:
                label = chorus.get("label")
                label_txt = (
                    f" ({label})"
                    if label and not str(label).startswith("section_")
                    else ""
                )
                return _reid([_out_candidate(
                    t,
                    "end of first chorus" + label_txt,
                    float(chorus["energy"]),
                )], "A")

    # The last high-energy section start ("the drop") and the final
    # low-energy section start ("the breakdown/outro") — double_drop and
    # breakdown_blend material. Appended FIRST: on a time collision with
    # the generic section candidates below, dedup keeps the first entry,
    # and these role labels tell the LLM strictly more.
    for s in reversed(sections):
        if s["energy"] >= HIGH_ENERGY_THRESHOLD:
            t = _snap_to_phrase(s["start"], grid, a.downbeats, ceiling)
            if t <= ceiling:
                raw.append(_out_candidate(
                    t,
                    "last drop/chorus (high energy)" + _phrase_tag(t, grid),
                    s["energy"],
                ))
            break
    for s in reversed(sections):
        if s["energy"] <= LOW_ENERGY_THRESHOLD:
            t = _snap_to_phrase(s["start"], grid, a.downbeats, ceiling)
            if t <= ceiling:
                raw.append(_out_candidate(
                    t,
                    "final breakdown (low energy)" + _phrase_tag(t, grid),
                    s["energy"],
                ))
            break

    for i, s in enumerate(sections):
        t = _snap_to_phrase(s["start"], grid, a.downbeats, ceiling)
        if t > ceiling:
            # Section starts too late to fit a crossfade — skip; the
            # "late as possible" fallback below covers the tail.
            continue
        # Only late-song sections are musical OUT points.
        if n >= 3 and i < n - 3:
            continue
        role = "final section" if i == n - 1 else (
            "second-to-last section" if i == n - 2 else "late section"
        )
        label = s.get("label")
        desc = (
            f"start of {role}" + (f" ({label})" if label else "")
            + _phrase_tag(t, grid)
        )
        raw.append(_out_candidate(t, desc, s["energy"]))

    # Always offer "as late as the headroom allows" — the v1 default.
    if max_out_fraction < 1.0:
        # Under a cap, the phrase grid's previous boundary can be far short
        # of it (e.g. 94 s when the cap is 120 s), while the next boundary
        # often overshoots. Use the later of the two safe anchors: the last
        # phrase boundary at/before the cap, or the last downbeat at/before
        # it. The latter always exists (unless the song is shorter than one
        # bar), so a non-bar-aligned duration can't produce an empty menu.
        late = max(
            _snap_to_phrase_at_or_before(ceiling, grid, a.downbeats),
            _snap_to_downbeat_at_or_before(ceiling, a.downbeats),
        )
        late_desc = "latest possible out point (capped)"
    else:
        late = _snap_to_phrase_at_or_before(ceiling, grid, a.downbeats)
        late_desc = "latest possible out point"
    if late <= ceiling:
        raw.append(_out_candidate(
            late,
            late_desc + _phrase_tag(late, grid),
            _energy_at(sections, late),
        ))

    return _reid(_dedup_and_cap(raw, keep_last=True), "A")


def build_in_candidates(
    b: AnalysisBundle,
    energy_curve: list[float],
    safe_regions: list[dict],
) -> list[SeamCandidate]:
    """IN points for the incoming song B: early-song moments — post-intro
    downbeat, first energy rise, first couple of section starts."""
    ceiling = min(
        max_seam_time(b.duration, b.bpm, b.time_signature),
        b.duration * B_ENTRY_MAX_FRACTION,
    )
    if ceiling <= 0:
        return []
    sec_per_bar = (60.0 / b.bpm) * b.time_signature if b.bpm else 0.0
    lookahead = VOCAL_SAFE_LOOKAHEAD_BARS * sec_per_bar
    sections = enrich_sections(b.sections, energy_curve)
    grid = phrase_grid(b.downbeats, b.sections)

    def _candidate(t: float, description: str, energy: float) -> SeamCandidate:
        return SeamCandidate(
            id="B?", time=t, description=description, energy=energy,
            vocal_safe=_is_vocal_safe(t, safe_regions, lookahead),
            vocal_density=_vocal_density(t, safe_regions, lookahead),
            drum_density=_stem_density(t, b.envelopes, "drums", lookahead),
            bass_density=_stem_density(t, b.envelopes, "bass", lookahead),
            lyrics_preview=_lyrics_preview(
                t, b.transcription_segments, lookahead, is_out=False
            ),
        )

    raw: list[SeamCandidate] = []

    # v1's default: first phrase boundary after the first section (skips
    # silent intros / count-ins).
    if sections:
        t = _snap_to_phrase(sections[0]["end"], grid, b.downbeats, ceiling)
        if t <= ceiling:
            raw.append(
                _candidate(
                    t,
                    "end of intro / first section" + _phrase_tag(t, grid),
                    _energy_at(sections, t),
                )
            )
    else:
        t = _snap_to_downbeat(0.0, b.downbeats)
        if t <= ceiling:
            raw.append(_candidate(t, "start of the song", 0.5))

    # First high-energy section start = "the first drop / chorus".
    for i, s in enumerate(sections):
        if s["energy"] >= HIGH_ENERGY_THRESHOLD:
            t = _snap_to_phrase(s["start"], grid, b.downbeats, ceiling)
            if t <= ceiling:
                label = s.get("label")
                desc = (
                    "first high-energy section (drop/chorus)"
                    + (f" ({label})" if label else "")
                    + _phrase_tag(t, grid)
                )
                raw.append(_candidate(t, desc, s["energy"]))
            break

    # Starts of sections 2 and 3 round out the early-song menu.
    for i, s in enumerate(sections[1:3], start=2):
        t = _snap_to_phrase(s["start"], grid, b.downbeats, ceiling)
        if t <= ceiling:
            label = s.get("label")
            desc = (
                f"start of section {i}" + (f" ({label})" if label else "")
                + _phrase_tag(t, grid)
            )
            raw.append(_candidate(t, desc, s["energy"]))

    if not raw:
        # Degenerate analysis — offer the first usable downbeat.
        t = _snap_to_downbeat(0.0, b.downbeats)
        if t <= ceiling:
            raw.append(_candidate(t, "start of the song", 0.5))

    return _reid(_dedup_and_cap(raw), "B")


def _energy_at(enriched_sections: list[dict], t: float) -> float:
    for s in enriched_sections:
        if s["start"] <= t < s["end"]:
            return s["energy"]
    return enriched_sections[-1]["energy"] if enriched_sections else 0.5


def cap_in_candidates_for_out(
    in_candidates: list[SeamCandidate],
    next_out_time: float | None,
    sec_per_bar_b: float,
    *,
    tempo_ratio: float = 1.0,
    needs_ramp: bool = False,
    min_blend_bars: int = HALF_MIN_BLEND_BARS,
    min_ramp_bars: int = HALF_MIN_RAMP_BARS,
    gap_seconds: float = HALF_SEAM_GAP_SECONDS,
) -> list[SeamCandidate]:
    """Keep entries that leave room for the incoming blend and B-side
    tempo ramp before the *earliest* OUT the next pair can choose.

    `next_out_time` must be the earliest candidate in song B's own OUT
    menu, not the cap. The next pair can legally pick any entry in that
    menu, so clearing the earliest one is what makes the constraint hold
    for every choice — no cross-pair state or ordering required.

    If nothing fits, keep the earliest entry. The expander then shortens
    the blend/ramp to the available room, and the stitch refuses any
    residual overlap instead of emitting a broken mix.
    """
    if not in_candidates or next_out_time is None:
        return list(in_candidates)
    reserve = (
        min_blend_bars * sec_per_bar_b * tempo_ratio
        + (min_ramp_bars * sec_per_bar_b if needs_ramp else 0.0)
        + gap_seconds
    )
    max_in = next_out_time - reserve
    kept = [c for c in in_candidates if c.time <= max_in]
    return kept or [in_candidates[0]]


def build_pair_candidates(
    a: AnalysisBundle,
    b: AnalysisBundle,
    a_energy_curve: list[float],
    b_energy_curve: list[float],
    a_safe_regions: list[dict],
    b_safe_regions: list[dict],
    *,
    out_fraction: float = 1.0,
    one_chorus: bool = False,
    next_out_time: float | None = None,
    tempo_ratio: float = 1.0,
    needs_ramp: bool = False,
) -> PairCandidates:
    in_candidates = build_in_candidates(b, b_energy_curve, b_safe_regions)
    if out_fraction < 1.0 and next_out_time is not None:
        spb_b = (60.0 / b.bpm) * b.time_signature if b.bpm else 0.0
        in_candidates = cap_in_candidates_for_out(
            in_candidates,
            next_out_time,
            spb_b,
            tempo_ratio=tempo_ratio,
            needs_ramp=needs_ramp,
        )
    return PairCandidates(
        out_candidates=build_out_candidates(
            a, a_energy_curve, a_safe_regions,
            max_out_fraction=out_fraction,
            one_chorus=one_chorus,
        ),
        in_candidates=in_candidates,
    )
