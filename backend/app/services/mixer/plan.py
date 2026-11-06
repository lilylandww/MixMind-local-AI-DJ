"""Hand-built mix-plan generator — the deterministic fallback for the LLM.

Takes two AnalysisBundles and emits a list of tool-call dicts matching
the spec's Mix Plan Schema. Pure function; no I/O, no DB.

Strategy:
1. Seam in A = last section start, clamped to "no more than 16 bars
   before A.duration", snapped to nearest downbeat ≥ that point.
2. Seam in B = first downbeat ≥ end of B's first section (skip silent
   intros / count-ins).
3. duration_bars = min(16, available outro / bar, available stretched
   intro / bar), floored at 4 with a warning when shorter.

Phase 7 explicitly does NOT pitch-shift. pyrubberband artifacts at
shifts > 2 semitones can sound worse than the harmonic dissonance
they're trying to fix, and small shifts are subtle enough that most
listeners won't notice the mismatch. Phase 9's LLM has the
`pitch_shift` and `temporary_pitch_shift` tools available and can
decide per-pair whether to use them; the hand-built generator stays
neutral and leaves keys as-is.

`compute_pitch_shift` stays in the module as a public utility so
Phase 9 (and anything else that wants to reason about smallest-shift
targets) doesn't have to duplicate the key-parsing logic.
"""

from __future__ import annotations

import logging

from app.services.mixer.types import AnalysisBundle, MixPlanJSON

logger = logging.getLogger(__name__)

# Pitch classes use sharps (matches services/analysis/key.py).
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

DEFAULT_DURATION_BARS = 16
MIN_DURATION_BARS = 4
LARGE_SHIFT_THRESHOLD = 2  # |δ| > 2 → WARN (locked: apply anyway)
# Bars over which B's pitch crossfades back to native. Kept short and
# decoupled from the (longer, 8-bar) tempo ramp so B doesn't linger in an
# audible dual-pitch blend.
PITCH_RETURN_BARS = 4


def _parse_key(key: str) -> tuple[int, bool]:
    """Return (pitch_class_index, is_minor). Accepts 'C', 'F#', 'Cm', 'F#m',
    plus flat aliases like 'Db' for forward compatibility (the analyzer
    only emits sharps, but the LLM in Phase 9 might emit flats)."""
    is_minor = key.endswith("m")
    root = key[:-1] if is_minor else key
    root = _FLAT_TO_SHARP.get(root, root)
    if root not in _PITCH_CLASSES:
        raise ValueError(f"unknown key: {key!r}")
    return _PITCH_CLASSES.index(root), is_minor


def compute_pitch_shift(a_key: str, b_key: str) -> int:
    """Smallest signed semitone shift to move B's tonic to A's tonic.

    Relative major/minor pairs (e.g. C major ↔ A minor, same Camelot
    position) are treated as compatible: δ=0.
    """
    a_root, a_minor = _parse_key(a_key)
    b_root, b_minor = _parse_key(b_key)

    # Smallest signed shift in [-6, 6].
    raw = a_root - b_root
    delta = ((raw + 6) % 12) - 6

    # Relative-major↔minor: same Camelot position, treat as compatible.
    # Relative minor of a major key is 3 semitones below (e.g. C major's
    # relative minor is Am, which is 9 above or -3 below). If mode flips
    # AND |delta| == 3, no shift needed.
    if a_minor != b_minor and abs(delta) == 3:
        return 0
    return delta


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


def build_pair_plan(
    a: AnalysisBundle,
    b: AnalysisBundle,
    *,
    max_out_fraction: float = 1.0,
) -> MixPlanJSON:
    """Build the tool-call list for the A → B transition.

    `max_out_fraction` < 1.0 caps the A seam for the short-format modes.
    It mirrors the v2 candidate menu: use the later of the last section
    start under the cap and the last downbeat under it, so a capped fallback
    still gets roughly half the track rather than an arbitrary early
    section. The IN/ramp budget is enforced by the v2 planner; this
    deterministic path keeps B's entry at the post-intro downbeat, which is
    early enough in practice.
    """
    sec_per_bar_a = (60.0 / a.bpm) * a.time_signature
    max_outro_seconds = DEFAULT_DURATION_BARS * sec_per_bar_a

    # A seam: last section start, clamped to "no more than 16 bars before end".
    last_section_start = a.sections[-1]["start"] if a.sections else 0.0
    if max_out_fraction < 1.0:
        cap = a.duration * max_out_fraction
        under_cap = [s["start"] for s in a.sections if s["start"] <= cap]
        cap_anchor = max(under_cap) if under_cap else cap
        cap_downbeat = _snap_to_downbeat_at_or_before(cap, a.downbeats)
        seam_a = _snap_to_downbeat_at_or_before(
            max(cap_anchor, cap_downbeat), a.downbeats
        )
    else:
        seam_a_raw = max(last_section_start, a.duration - max_outro_seconds)
        seam_a = _snap_to_downbeat(seam_a_raw, a.downbeats)

    # B seam: first downbeat ≥ end of first section.
    first_section_end = b.sections[0]["end"] if b.sections else 0.0
    seam_b = _snap_to_downbeat(first_section_end, b.downbeats)

    # Crossfade length: clamp to whatever's available in A's outro and
    # B's post-stretch intro. Stretch factor = b.bpm / a.bpm.
    stretch_factor = b.bpm / a.bpm
    available_a = a.duration - seam_a
    available_b_stretched = (b.duration - seam_b) * stretch_factor
    duration_bars = min(
        DEFAULT_DURATION_BARS,
        int(available_a / sec_per_bar_a),
        int(available_b_stretched / sec_per_bar_a),
    )
    if duration_bars < MIN_DURATION_BARS:
        logger.warning(
            "build_pair_plan: insufficient overlap (available_a=%.2fs, "
            "available_b_stretched=%.2fs); flooring duration_bars at %d",
            available_a, available_b_stretched, MIN_DURATION_BARS,
        )
        duration_bars = MIN_DURATION_BARS

    # Calculate pitch shift
    n_steps = compute_pitch_shift(a.key, b.key)

    plan: MixPlanJSON = [
        {
            "tool": "set_transition_window",
            "from_song_time_start": seam_a,
            "to_song_time_start": seam_b,
            "duration_bars": duration_bars,
        }
    ]

    # Calculate post-crossfade space for B
    sec_per_bar_b = (60.0 / b.bpm) * b.time_signature
    crossfade_end_b = seam_b + (duration_bars * sec_per_bar_b)
    # Tempo ramp length, kept equal to the pitch-return window so tempo and
    # pitch both settle to B's native values at the same moment.
    ramp_duration_bars = PITCH_RETURN_BARS
    ramp_end_b = crossfade_end_b + (ramp_duration_bars * sec_per_bar_b)

    can_ramp = b.duration >= ramp_end_b

    if not can_ramp:
        logger.warning(
            "build_pair_plan: song B too short to accommodate tempo/pitch ramp "
            "(duration=%.2fs, need=%.2fs). Falling back to permanent shifts.",
            b.duration, ramp_end_b
        )
        if n_steps != 0:
            # A permanent shift can't ramp back, so a big shift would detune B
            # for its entire body. Cap magnitude at ±2 semitones — past that
            # the artifacts/detune cost outweighs the key match.
            capped = max(-2, min(2, n_steps))
            if capped != n_steps:
                logger.warning(
                    "build_pair_plan: capping permanent pitch shift %d -> %d "
                    "(B too short to ramp back to its native key)",
                    n_steps, capped,
                )
            plan.append({
                "tool": "pitch_shift",
                "song": "B",
                "semitones": capped,
            })
    else:
        # Emit tempo ramp
        plan.append({
            "tool": "set_tempo_ramp",
            "song": "B",
            "start_time": crossfade_end_b,
            "end_time": ramp_end_b,
            "start_bpm": a.bpm,
            "end_bpm": b.bpm,
        })
        # Emit temporary pitch shift if needed
        if n_steps != 0:
            plan.append({
                "tool": "temporary_pitch_shift",
                "song": "B",
                "start_time": crossfade_end_b,
                "semitones": n_steps,
                "fade_in_bars": 0,
                "hold_bars": 0,
                "fade_out_bars": PITCH_RETURN_BARS,
            })

    for stem in ("vocals", "drums", "bass", "other"):
        plan.append({
            "tool": "crossfade_stem",
            "stem": stem,
            "from_song": "A",
            "to_song": "B",
            "start_bar": 0,
            "duration_bars": duration_bars,
            "curve": "equal_power",
        })
    return plan
