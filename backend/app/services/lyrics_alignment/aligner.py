"""Align an authoritative lyric text against Whisper's word timings.

Two reference sources reach here:

* **Genius text** (no timing) — the whole lyric body is matched globally
  against the Whisper transcript and words inherit Whisper's timestamps.
* **`.lrc` sidecar** (timed) — passed as ``line_times``. Each line carries
  its own timestamp, which lets us (a) constrain the match to the right
  window so repeated choruses can't steal each other's timings, (b) fit a
  time warp that absorbs the 0.5-2 s drift hand-made `.lrc` files almost
  always carry, and (c) fall back to the *author's* own timing where
  Whisper heard nothing.

Output entries look like::

    {"word", "start", "end", "confidence", "source"}

``source`` is the provenance of the *timestamp* (not of the text — the
text is always the authoritative reference):

======================  ====================================================
``whisper_match``       exact token match; timing from Whisper (fine, per
                        word, DTW-derived)
``whisper_substitution`` fuzzy/phonetic match; timing from Whisper, confidence
                        scaled down
``lrc_word``            timing from an enhanced ``<time>`` word tag
``lrc_line``            timing estimated from the line's own timestamp
``interpolated``        timing guessed between two anchored neighbours
``unmapped``            no timing at all (``start``/``end`` are None)
======================  ====================================================

``app.services.vocal_safety`` trusts the first three outright and requires
vocal-stem energy to corroborate the guessed ones.
"""
from __future__ import annotations

import bisect
import difflib
import re
import statistics
from typing import Any

from app.models.lyrics import LyricsAlignmentStatus
from app.services.lyrics.tokenizer import clean_word, is_latin_word, tokenize

# Assumed sung rate, used to place words when a `.lrc` line gives us a
# start but no per-word times. ~170 wpm. Slightly generous so a slow
# vocal line still covers all its words.
SECONDS_PER_WORD = 0.35

# How far a line's vocal may start before / spill past its own stamp when
# the fit is used to bound the interpolation window. The transcript is
# *partitioned* at the midpoints between consecutive lines (so windows
# never overlap); these only pad the range used to spread a line whose
# words Whisper missed entirely.
LINE_LEAD_SLACK = 1.5
LINE_TAIL_SLACK = 1.5

# Below this Theil-Sen slope the `.lrc` and the audio are simply at the
# same speed and only a constant offset is needed; outside 0.7-1.3 the
# slope estimate is unreliable (too few anchors) so fall back to a
# constant offset again.
_WARP_SLOPE_MIN = 0.98
_WARP_SLOPE_MAX = 1.02
_WARP_SLOPE_ABS_MIN = 0.7
_WARP_SLOPE_ABS_MAX = 1.3
_MIN_ANCHORS_FOR_SLOPE = 4


def _clean_word(w: str) -> str:
    """See ``app.services.lyrics.tokenizer.clean_word``.

    Kept as a module-level alias so the normaliser is one import away and
    older call sites keep working.
    """
    return clean_word(w)


_SOUNDEX_MAP = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}
# H and W are "transparent" in classical American Soundex — they don't
# count toward the code, but they also don't reset the dedupe state.
# Vowels (and Y) DO reset.
_SOUNDEX_TRANSPARENT = {"h", "w"}


def _soundex(word: str) -> str:
    """Classical American Soundex: first letter + up to 3 digit codes.

    English phonotactics only. Callers gate it with
    ``is_latin_word`` — the code is meaningless for CJK / Cyrillic /
    Arabic and would make unrelated words collide.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    first = w[0]
    digits: list[str] = []
    last_code = _SOUNDEX_MAP.get(first, "")
    for ch in w[1:]:
        code = _SOUNDEX_MAP.get(ch)
        if code is None:
            if ch in _SOUNDEX_TRANSPARENT:
                continue  # transparent — preserve last_code
            last_code = ""  # vowel/y — resets dedupe
            continue
        if code == last_code:
            continue
        digits.append(code)
        last_code = code
        if len(digits) == 3:
            break
    return first.upper() + "".join(digits).ljust(3, "0")


# Genius uses U+00D7 (×) or ASCII 'x' followed by a digit, optionally
# parenthesised. "(×4)", "(x4)" are the common shapes.
_REPEAT_RE = re.compile(r"\(\s*[x×]\s*(\d+)\s*\)", re.IGNORECASE)


def _expand_repeat_markers(text: str) -> str:
    """Expand ``(×N)`` markers: repeat the immediately preceding block
    (the contiguous run of non-blank lines ending at the marker line)
    so it appears N times total. Strips the marker."""
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        m = _REPEAT_RE.search(line)
        if not m:
            out.append(line)
            continue

        n = int(m.group(1))
        # Find the block we're repeating — everything appended since
        # the last blank line (or the doc start).
        block_end = len(out)
        block_start = block_end
        for k in range(block_end - 1, -1, -1):
            if out[k].strip() == "":
                block_start = k + 1
                break
            block_start = k
        block = out[block_start:block_end]

        # Block currently appears once. Add (n - 1) more copies.
        for _ in range(max(0, n - 1)):
            out.extend(block)

        # Preserve any text on the marker line beyond the marker itself.
        remainder = _REPEAT_RE.sub("", line).strip()
        if remainder:
            out.append(remainder)
    return "\n".join(out)


# --- reference / hypothesis tokenisation ------------------------------


def _reference_tokens(words: list[str]) -> list[dict[str, Any]]:
    return [{"word": w, "clean": _clean_word(w)} for w in words]


def _flatten_transcript(segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Whisper words -> alignment tokens.

    Both sides must be tokenised identically or the diff is meaningless.
    A Whisper "word" can hold several tokens (CJK arrives unspaced), so
    each one is split and its timestamp span divided proportionally.
    Single-token words keep their exact original boundaries so callers
    can rely on end-of-transcript timestamps surviving untouched.
    """
    out: list[dict[str, Any]] = []
    for seg in segments or []:
        seg_logprob = seg.get("avg_logprob")
        for w in seg.get("words") or []:
            raw = w.get("word") or ""
            if not raw:
                continue
            start, end = w.get("start"), w.get("end")
            if start is None or end is None:
                continue
            toks = tokenize(raw)
            if not toks:
                continue
            prob = w.get("probability")
            if len(toks) == 1:
                out.append({
                    "word": toks[0],
                    "clean": _clean_word(toks[0]),
                    "start": float(start),
                    "end": float(end),
                    "probability": prob,
                    "seg_logprob": seg_logprob,
                })
                continue
            span = max(float(end) - float(start), 1e-3)
            for i, t in enumerate(toks):
                out.append({
                    "word": t,
                    "clean": _clean_word(t),
                    "start": float(start) + span * i / len(toks),
                    "end": float(start) + span * (i + 1) / len(toks),
                    "probability": prob,
                    "seg_logprob": seg_logprob,
                })
    return out


def _phonetic_hit(ref_clean: str, hyp_clean: str) -> bool:
    """Soundex collision, but only where Soundex means anything."""
    if not (is_latin_word(ref_clean) and is_latin_word(hyp_clean)):
        return False
    a, b = _soundex(ref_clean), _soundex(hyp_clean)
    return bool(a) and a == b


def _align_opcodes(
    ref: list[dict[str, Any]], hyp: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """difflib match of reference tokens against transcript tokens.

    Returns one entry per reference token with Whisper's timing attached
    where a counterpart was found. Hypothesis-only tokens are dropped
    (Whisper ad-libs, backing "oh-oh"s) — the reference text is what the
    player and the planner show.
    """
    matcher = difflib.SequenceMatcher(
        None, [r["clean"] for r in ref], [h["clean"] for h in hyp]
    )

    aligned: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                h = hyp[j]
                aligned.append({
                    "word": ref[i]["word"],
                    "start": h["start"],
                    "end": h["end"],
                    "confidence": h["probability"] if h["probability"] is not None else 1.0,
                    "source": "whisper_match",
                    "_ref": ref[i],
                })
        elif tag == "replace":
            for idx, i in enumerate(range(i1, i2)):
                j = j1 + idx
                if j < j2:
                    h = hyp[j]
                    # Phonetic match → trustworthy timing despite wrong
                    # word ("there"/"their"). Non-phonetic → likely
                    # pairing two unrelated words; low conf.
                    hit = _phonetic_hit(ref[i]["clean"], h["clean"])
                    factor = 0.7 if hit else 0.4
                    prob = h["probability"] if h["probability"] is not None else 1.0
                    aligned.append({
                        "word": ref[i]["word"],
                        "start": h["start"],
                        "end": h["end"],
                        "confidence": prob * factor,
                        "source": "whisper_substitution",
                        "_ref": ref[i],
                        "_phonetic_hit": hit,
                    })
                else:
                    aligned.append({
                        "word": ref[i]["word"],
                        "start": None,
                        "end": None,
                        "confidence": 0.0,
                        "source": "unmapped",
                        "_ref": ref[i],
                    })
        elif tag == "delete":
            for i in range(i1, i2):
                aligned.append({
                    "word": ref[i]["word"],
                    "start": None,
                    "end": None,
                    "confidence": 0.0,
                    "source": "unmapped",
                    "_ref": ref[i],
                })
        elif tag == "insert":
            # Words in Whisper not in the reference. Drop them.
            pass
    return aligned


def _interpolate(
    aligned: list[dict[str, Any]],
    window: tuple[float, float] | None = None,
) -> None:
    """Fill in missing timestamps in place.

    Non-zero duration for interpolated words. When `window` is given (the
    `.lrc` path) a run with no anchor at all is spread across the window
    instead of being left untimed — the line's own timestamp already
    bounds it, so leaving it None would throw away information we have.
    """
    for i, a in enumerate(aligned):
        if a["start"] is not None:
            continue

        prev_idx: int | None = None
        next_idx: int | None = None
        for k in range(i - 1, -1, -1):
            if aligned[k]["start"] is not None:
                prev_idx = k
                break
        for k in range(i + 1, len(aligned)):
            if aligned[k]["start"] is not None:
                next_idx = k
                break

        if prev_idx is not None and next_idx is not None:
            prev_end = aligned[prev_idx]["end"]
            next_start = aligned[next_idx]["start"]
            gap_span = max(next_start - prev_end, 1e-3)
            slot = (i - prev_idx) / (next_idx - prev_idx)
            per_word = gap_span / (next_idx - prev_idx)
            t_center = prev_end + slot * gap_span
            a["start"] = t_center
            a["end"] = t_center + per_word
            a["source"] = "interpolated"
        elif prev_idx is not None:
            prev_end = aligned[prev_idx]["end"]
            a["start"] = prev_end + 0.05
            a["end"] = prev_end + 0.15
            a["source"] = "interpolated"
        elif next_idx is not None:
            next_start = aligned[next_idx]["start"]
            a["start"] = max(0.0, next_start - 0.15)
            a["end"] = max(0.0, next_start - 0.05)
            a["source"] = "interpolated"
        elif window is not None:
            lo, hi = window
            # Whole line unmatched — the `.lrc` still tells us the vocal
            # lives inside this window, so spread the words through it.
            missing = [k for k in range(len(aligned)) if aligned[k]["start"] is None]
            span = max(hi - lo, SECONDS_PER_WORD * len(missing))
            per_word = span / max(len(missing), 1)
            for rank, k in enumerate(missing):
                t = lo + rank * per_word
                aligned[k]["start"] = t
                aligned[k]["end"] = t + per_word
                aligned[k]["source"] = "lrc_line"
            return
        else:
            # No anchor on either side — leave timestamps None.
            # Downstream consumers (vocal_safety) must skip these.
            a["source"] = "interpolated"
            # a["start"] and a["end"] remain None.


def _score(aligned: list[dict[str, Any]]) -> tuple[float, LyricsAlignmentStatus]:
    # Weight matches 1.0, phonetic subs 0.5, other subs 0.1.
    # Interpolations contribute 0 but count toward the denominator so
    # alignments with many gaps are correctly flagged low_quality.
    matches = phonetic_subs = other_subs = interpolated_count = 0
    for a in aligned:
        src = a["source"]
        if src in ("whisper_match", "lrc_word"):
            matches += 1
        elif src == "whisper_substitution":
            if a.get("_phonetic_hit"):
                phonetic_subs += 1
            else:
                other_subs += 1
        elif src in ("interpolated", "lrc_line"):
            interpolated_count += 1

    denom = matches + phonetic_subs + other_subs + interpolated_count
    quality = (
        (matches + 0.5 * phonetic_subs + 0.1 * other_subs) / denom
        if denom else 0.0
    )
    status = (
        LyricsAlignmentStatus.low_quality
        if quality < 0.3
        else LyricsAlignmentStatus.success
    )
    return quality, status


def _finish(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    # Score first: _phonetic_hit is what distinguishes a 0.5-weight
    # phonetic substitution from a 0.1-weight one.
    quality, status = _score(aligned)
    for a in aligned:
        a.pop("_phonetic_hit", None)
        a.pop("_ref", None)
    return {
        "aligned_words": aligned,
        "alignment_quality": quality,
        "alignment_status": status,
    }


# --- Genius (untimed) path --------------------------------------------


def _align_free(hyp: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    clean_text = re.sub(r"\[.*?\]", "", text)
    clean_text = _expand_repeat_markers(clean_text)
    ref = _reference_tokens(tokenize(clean_text))
    aligned = _align_opcodes(ref, hyp)
    _interpolate(aligned)
    return aligned


# --- `.lrc` (timed) path ----------------------------------------------


def _line_reference(line: dict[str, Any]) -> list[dict[str, Any]]:
    """Reference tokens for one `.lrc` line, each with a nominal time.

    Enhanced ``<time>`` word tags win when present. Otherwise words are
    packed from the line's own start at ``SECONDS_PER_WORD`` rather than
    spread to the next line — the gap after a line is usually an
    instrumental, and smearing words across it would claim vocal energy
    that isn't there.
    """
    words = line.get("words") or []
    if words:
        out: list[dict[str, Any]] = []
        for i, w in enumerate(words):
            toks = tokenize(w.get("text") or "")
            if not toks:
                continue
            start = w.get("start")
            nxt = next(
                (words[j].get("start") for j in range(i + 1, len(words))
                 if words[j].get("start") is not None),
                None,
            )
            if start is None or nxt is None or nxt <= start:
                times = [start] * len(toks)
            else:
                step = (nxt - start) / len(toks)
                times = [start + k * step for k in range(len(toks))]
            for t, ts in zip(toks, times):
                out.append({"word": t, "clean": _clean_word(t), "nominal": ts, "fine": True})
        return out

    toks = tokenize(line.get("text") or "")
    start = line.get("start")
    if not toks:
        return []
    if start is None:
        return [{"word": t, "clean": _clean_word(t), "nominal": None, "fine": False} for t in toks]
    end = line.get("end")
    span = min(SECONDS_PER_WORD * len(toks), max((end or start) - start, SECONDS_PER_WORD))
    step = span / len(toks)
    return [
        {"word": t, "clean": _clean_word(t), "nominal": start + k * step, "fine": False}
        for k, t in enumerate(toks)
    ]


def _estimate_warp(pairs: list[tuple[float, float]]):
    """Fit ``lrc_time -> audio_time`` from matched anchor pairs.

    Hand-made `.lrc` files are routinely shifted by a second or two, and
    some are stretched (recorded from a different master). A Theil-Sen
    slope — the median of all pairwise slopes — absorbs both and shrugs
    off the bogus anchors a fuzzy match will occasionally produce.

    Returns ``(offset, slope)`` so the caller can report what it applied.
    """
    if not pairs:
        return 0.0, 1.0

    if len(pairs) >= _MIN_ANCHORS_FOR_SLOPE:
        xs = [p[0] for p in pairs]
        if max(xs) - min(xs) > 5.0:
            slopes = [
                (pairs[j][1] - pairs[i][1]) / (pairs[j][0] - pairs[i][0])
                for i in range(len(pairs))
                for j in range(i + 1, len(pairs))
                if abs(pairs[j][0] - pairs[i][0]) > 1e-3
            ]
            if slopes:
                slope = statistics.median(slopes)
                plausible = (
                    _WARP_SLOPE_ABS_MIN <= slope <= _WARP_SLOPE_ABS_MAX
                )
                drifted = not (_WARP_SLOPE_MIN <= slope <= _WARP_SLOPE_MAX)
                if plausible and drifted:
                    offset = statistics.median(y - slope * x for x, y in pairs)
                    return offset, slope

    offset = statistics.median(y - x for x, y in pairs)
    return offset, 1.0


def _align_with_lines(
    hyp: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    warp_params: tuple[float, float] | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    """Constrained alignment using `.lrc` line timestamps.

    A global free match is what breaks on a chorus sung three times: the
    diff happily binds the second occurrence's words to the third
    occurrence's Whisper words. Bounding each line to its own time window
    makes that impossible.

    Timing provenance per token, best available:
    Whisper (fine, per-word) -> `.lrc` word tag -> the line's own stamp.

    `warp_params` supplies a previously fitted ``(offset, slope)`` instead
    of re-estimating one, so re-runs stay stable.
    """
    prepared: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda d: d.get("start") is None):
        ref = _line_reference(line)
        if ref:
            prepared.append({"ref": ref, "line": line})
    if not prepared:
        return [], 0.0, 1.0

    if warp_params is not None:
        offset, slope = warp_params
    else:
        # Pass 1: a cheap global match, used only to fit the time warp.
        flat_ref = [r for p in prepared for r in p["ref"]]
        warp_pairs = [
            (r["nominal"], a["start"])
            for r, a in zip(flat_ref, _align_opcodes(flat_ref, hyp))
            if a["source"] == "whisper_match" and r.get("nominal") is not None
        ]
        offset, slope = _estimate_warp(warp_pairs)

    def warp(t: float) -> float:
        return offset + slope * t

    aligned: list[dict[str, Any]] = []
    timed = [p for p in prepared if p["line"].get("start") is not None]
    untimed = [p for p in prepared if p["line"].get("start") is None]

    # Partition, don't window. A sliding window still lets a chorus's
    # words reach back into the previous chorus; giving every transcript
    # word to exactly one line (nearest warped line start, boundaries at
    # the midpoints) makes cross-contamination impossible. Words that
    # spill past a boundary are cheap to lose here — the `.lrc`'s own
    # stamp is the fallback for anything unmatched.
    warped = [warp(p["line"]["start"]) for p in timed]
    bounds = [
        (warped[i] + warped[i + 1]) / 2.0 for i in range(len(warped) - 1)
    ]
    slices: list[list[dict[str, Any]]] = [[] for _ in timed]
    for h in hyp:
        slices[bisect.bisect_left(bounds, h["start"])].append(h)

    for entry, scoped in zip(timed, slices):
        ref = entry["ref"]
        line = entry["line"]
        start = line.get("start")
        nxt = line.get("end")
        expected = SECONDS_PER_WORD * len(ref) + LINE_TAIL_SLACK
        if nxt is not None:
            expected = min(expected, max(nxt - start, SECONDS_PER_WORD))
        window = (
            warp(start) - LINE_LEAD_SLACK,
            warp(start) + expected + LINE_TAIL_SLACK,
        )
        _merge_line(ref, scoped, window, aligned)

    # Lines with no stamp at all can't be partitioned; match them against
    # the whole transcript. `Lyrics.line_timestamps` never carries these
    # (an unsynced `.lrc` yields `[]` and the caller uses `reference_text`
    # instead), so this only matters for hand-built payloads.
    for entry in untimed:
        _merge_line(entry["ref"], hyp, None, aligned)

    return aligned, offset, slope


def _merge_line(
    ref: list[dict[str, Any]],
    scoped: list[dict[str, Any]],
    window: tuple[float, float] | None,
    out: list[dict[str, Any]],
) -> None:
    """Align one line's reference tokens against its transcript slice.

    Timing provenance per token, best available: Whisper (fine, per-word)
    -> `.lrc` word tag -> the line's own stamp.
    """
    line_aligned = _align_opcodes(ref, scoped)

    # Fall back to the `.lrc`'s own timing wherever Whisper had nothing —
    # an author's word stamp beats a guess.
    for a in line_aligned:
        if a["start"] is not None:
            continue
        nominal = a["_ref"].get("nominal")
        if nominal is None:
            continue
        a["start"] = nominal
        a["end"] = nominal + SECONDS_PER_WORD
        a["source"] = "lrc_word" if a["_ref"].get("fine") else "lrc_line"
        a["confidence"] = 1.0 if a["_ref"].get("fine") else 0.3

    # Give every timed word a plausible end: the next word's start within
    # the line beats a flat constant.
    for a in line_aligned:
        if a["start"] is None or a["end"] is not None:
            continue
        a["end"] = a["start"] + SECONDS_PER_WORD
    for i in range(len(line_aligned) - 1):
        a, b = line_aligned[i], line_aligned[i + 1]
        if (
            a["start"] is not None
            and b["start"] is not None
            and b["start"] > a["start"]
        ):
            a["end"] = b["start"]

    _interpolate(line_aligned, window=window)
    out.extend(line_aligned)


# --- public API -------------------------------------------------------


def align_lyrics(
    transcription_segments: list[dict[str, Any]],
    reference_text: str,
    *,
    line_times: list[dict[str, Any]] | None = None,
    alignment_offset: float | None = None,
    alignment_slope: float | None = None,
) -> dict[str, Any]:
    """Align authoritative lyric text with Whisper transcription timestamps.

    Pass `line_times` (the persisted `Lyrics.line_timestamps` payload) when
    a timed `.lrc` source exists — it produces markedly better word
    boundaries on repeats and lets us recover timing for words Whisper
    missed entirely. `reference_text` is then only a fallback.

    Returns a dict with aligned_words, alignment_quality, and
    alignment_status — plus, on the timed path, the `offset` and `slope`
    of the fitted time warp so the caller can persist the drift
    correction it discovered.

    `alignment_offset` / `alignment_slope` supply a previously fitted warp
    instead of re-estimating one, which keeps re-runs stable.
    """
    hyp = _flatten_transcript(transcription_segments)
    warp_params: tuple[float, float] | None = None
    if alignment_offset is not None:
        warp_params = (float(alignment_offset), float(alignment_slope or 1.0))

    if line_times:
        aligned, offset, slope = _align_with_lines(hyp, line_times, warp_params)
        if not aligned and reference_text:
            aligned = _align_free(hyp, reference_text)
        result = _finish(aligned)
        result["offset"] = offset
        result["slope"] = slope
        return result

    if not reference_text:
        return {
            "aligned_words": [],
            "alignment_quality": 0.0,
            "alignment_status": LyricsAlignmentStatus.error,
        }
    return _finish(_align_free(hyp, reference_text))
