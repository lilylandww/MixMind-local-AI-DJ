"""Forced alignment: reuse known lyric text instead of recognising it.

Whisper *recognises* speech. When we already hold the true lyric text —
from a `.lrc` sidecar, or embedded lyrics — recognition is the wrong
problem. Forced alignment solves the easier one: given the text and the
audio, where does each word land? Benefits:

* **The text cannot be misheard.** No "there"/"their", no hallucinated
  ``[applause]``, no words dropped on a sparse vocal.
* **It is much cheaper.** No autoregressive decode — one forward pass.
* **It is language-agnostic** where the aligner is, rather than where
  Whisper's acoustic model happens to be strong.

Backend is Meta's MMS forced aligner (``torchaudio.pipelines.MMS_FA``):
a single CTC model covering 1000+ languages with no per-language setup —
the reason it beats WhisperX's per-language wav2vec2 checkpoints and the
Montreal Forced Aligner (which needs a pronunciation lexicon per
language) for a multilingual library.

Caveat worth knowing
--------------------
MMS_FA aligns over a per-language *character* inventory. Latin-script
languages work out of the box; other scripts need a romaniser, which
only some torchaudio builds expose (``bundle.get_to_phone()``). When a
character isn't in the inventory we raise :class:`ForcedAlignUnavailable`
rather than emit nonsense timings — the caller then falls back to
Whisper, whose per-word DTW timestamps constrained by the `.lrc` line
times (see :mod:`app.services.lyrics_alignment.aligner`) are still a good
answer for those languages.

Everything torch-backed lives behind lazy imports so this module imports
(and its pure helpers test) on hosts with no PyTorch at all.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Sequence

from app.services.lyrics.tokenizer import clean_word

logger = logging.getLogger(__name__)

MODEL_NAME = "mms-fa"
TARGET_SAMPLE_RATE = 16_000


class ForcedAlignUnavailable(RuntimeError):
    """Raised instead of returning bad timings.

    Covers: torchaudio missing, MMS_FA weights missing, text outside the
    model's character inventory, or any decode failure. Callers fall back
    to Whisper.
    """


# --- pure helpers (no torch) ------------------------------------------


def assign_word_times(
    words: Sequence[str],
    char_spans: Sequence[tuple[float, float]],
    counts: Sequence[int],
) -> list[dict[str, Any]]:
    """Distribute per-character spans back over the reference words.

    `char_spans` holds one ``(start, end)`` per token character, in text
    order; `counts` is how many token characters each word contributed.
    A word's span runs from its first character's start to its last
    character's end. A word that contributed nothing (pure punctuation,
    or dropped by the romaniser) gets ``None`` timings rather than
    borrowing a neighbour's.

    This grouping is the part worth testing: everything upstream is
    torchaudio's business, and getting the slicing wrong would silently
    shift every word boundary the DJ cut safety depends on.
    """
    out: list[dict[str, Any]] = []
    cursor = 0
    for word, count in zip(words, counts):
        if count <= 0:
            out.append({"word": word, "start": None, "end": None})
            continue
        chunk = char_spans[cursor : cursor + count]
        cursor += count
        if len(chunk) < count:
            out.append({"word": word, "start": None, "end": None})
            continue
        out.append({
            "word": word,
            "start": float(chunk[0][0]),
            "end": float(chunk[-1][1]),
        })
    return out


def merge_token_spans(
    items: Sequence[Any],
    *,
    frame_rate: float,
    frame_count: int,
) -> list[tuple[float, float]]:
    """Turn frame-level alignment items into per-character ``(start, end)``.

    torchaudio's ``forced_align`` reports a start frame per token; a
    token's extent therefore runs to the next token's start (the last one
    to the end of the emission). Accepts both tuple-shaped items
    ``(token, start, score)`` and attribute-shaped ones (``.start``), so
    this survives the API drift between torchaudio releases.
    """
    starts: list[float] = []
    for it in items:
        if hasattr(it, "start"):
            starts.append(float(it.start))
        elif isinstance(it, (tuple, list)) and len(it) >= 2:
            starts.append(float(it[1]))
        else:
            raise ForcedAlignUnavailable(
                f"unrecognised alignment item shape: {it!r}"
            )

    spans: list[tuple[float, float]] = []
    for i, start in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else float(frame_count)
        spans.append((start / frame_rate, max(nxt, start) / frame_rate))
    return spans


def segments_from_words(
    words: Sequence[dict[str, Any]],
    *,
    max_segment_seconds: float = 30.0,
    break_gap: float = 1.5,
) -> list[dict[str, Any]]:
    """Group placed words into Whisper-shaped segments.

    Forced alignment has no notion of a "segment" — it only knows word
    boundaries — but `Transcription.segments` is the wire format every
    consumer reads, so we synthesise them the way Whisper would have:
    break on a long silence and roll over at ~30 s.

    Decode confidences are all None: there was no hypothesis to be unsure
    about, because the text is known. Consumers must treat None as
    passing (vocal_safety does).
    """
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        segments.append({
            "start": float(current[0]["start"]),
            "end": float(current[-1]["end"]),
            "text": "".join(_display(w["word"]) for w in current),
            "words": [
                {
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "word": _display(w["word"]),
                    "probability": None,
                }
                for w in current
            ],
            "avg_logprob": None,
            "no_speech_prob": None,
            "compression_ratio": None,
            "temperature": None,
        })
        current.clear()

    for w in words:
        if current:
            gap = float(w["start"]) - float(current[-1]["end"])
            span = float(w["end"]) - float(current[0]["start"])
            if gap > break_gap or span > max_segment_seconds:
                flush()
        current.append(w)
    flush()
    return segments


def _display(word: str) -> str:
    """Leading space, matching mlx-whisper's word convention.

    `TranscriptionService` documents that re-joining segments by
    concatenation must produce fluent text, and its words keep the
    leading space. Do the same here so a forced-aligned row is
    indistinguishable downstream.
    """
    return f" {word}" if word else word


# --- torch-backed adapter ---------------------------------------------


def _load_bundle():
    try:
        import torchaudio
        from torchaudio.pipelines import MMS_FA
    except ImportError as exc:  # pragma: no cover - needs torchaudio
        raise ForcedAlignUnavailable(f"torchaudio not installed: {exc}") from exc
    return torchaudio, MMS_FA


def _romaniser(bundle):
    """The optional text -> model-alphabet mapper, or None.

    Only some torchaudio builds ship it. Absent means non-Latin text will
    hit the inventory check below and fall back to Whisper — the right
    outcome rather than a wrong one.
    """
    getter = getattr(bundle, "get_to_phone", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception as exc:  # pragma: no cover - version dependent
        logger.warning("MMS_FA romaniser unavailable: %s", exc)
        return None


def _char_table(bundle) -> dict:
    get_dict = bundle.get_dict
    if "star" in _params_of(get_dict):
        return get_dict(star=None)
    return get_dict()


def _params_of(fn) -> dict:
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}


def _tokenise_words(
    words: Sequence[str], bundle
) -> tuple[list[int], list[int]]:
    """Map reference words onto MMS_FA token ids.

    Returns ``(ids, chars_per_word)`. Each word is normalised on its own
    (and romanised on its own if a romaniser exists) so `chars_per_word`
    stays an exact partition of `ids` and the spans map back to the
    caller's word list whatever the script.

    Raises :class:`ForcedAlignUnavailable` if any character falls outside
    the model's inventory — we would rather use Whisper than invent
    timings.
    """
    romanise = _romaniser(bundle)
    table = _char_table(bundle)

    ids: list[int] = []
    counts: list[int] = []
    for word in words:
        form = clean_word(word)
        if romanise is not None and form:
            try:
                form = str(romanise(form))
            except Exception as exc:  # pragma: no cover - version dependent
                raise ForcedAlignUnavailable(
                    f"romanisation failed for {word!r}: {exc}"
                ) from exc
        n = 0
        for ch in form:
            key = ch.casefold()
            if key not in table:
                raise ForcedAlignUnavailable(
                    f"MMS_FA has no token for {ch!r} — this build's character "
                    "inventory does not cover the lyric's script"
                )
            ids.append(int(table[key]))
            n += 1
        counts.append(n)
    return ids, counts


def align_words(
    words: Sequence[str],
    wav_path: Path,
    *,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Align `words` to the audio in `wav_path`, returning word timings.

    Output entries match one element of ``Transcription.segments[*].words``
    — ``{"word", "start", "end", "probability"}`` with `probability` None,
    since there was no hypothesis to be unsure about. Words that could not
    be placed are omitted; the lyrics aligner re-injects their text with
    `.lrc` timing downstream.

    `language` is accepted for interface symmetry but MMS_FA picks its
    inventory from the text's characters.

    Raises :class:`ForcedAlignUnavailable` on any gap in the toolchain so
    the caller can fall back to Whisper instead of persisting bad timing.
    """
    words = [w for w in words if w and clean_word(w)]
    if not words:
        return []

    torchaudio, bundle = _load_bundle()
    ids, counts = _tokenise_words(words, bundle)
    if not ids:
        return []

    try:
        model = bundle.get_model(with_star=False)
    except TypeError:  # pragma: no cover - older signature
        model = bundle.get_model()

    try:
        waveform, sample_rate = torchaudio.load(str(wav_path))
    except Exception as exc:
        raise ForcedAlignUnavailable(f"could not load {wav_path}: {exc}") from exc

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )
    duration = waveform.shape[-1] / TARGET_SAMPLE_RATE

    try:
        import torch

        with torch.no_grad():
            emission, _ = model(waveform)
        frames = int(emission.shape[-1])
        if frames <= 0 or duration <= 0:
            raise ForcedAlignUnavailable("empty emission or zero-length audio")

        tokens = torch.tensor([ids], dtype=torch.int32)
        alignment, _ = torchaudio.functional.forced_align(
            emission, tokens, blank=0
        )
    except ForcedAlignUnavailable:
        raise
    except Exception as exc:  # pragma: no cover - needs weights
        raise ForcedAlignUnavailable(f"MMS_FA alignment failed: {exc}") from exc

    items = alignment[0] if alignment and isinstance(alignment[0], list) else alignment
    char_spans = merge_token_spans(
        list(items), frame_rate=frames / duration, frame_count=frames
    )
    placed = assign_word_times(words, char_spans, counts)
    dropped = sum(1 for a in placed if a["start"] is None)
    if dropped:
        logger.warning(
            "forced_align: %d/%d words could not be placed in %s",
            dropped, len(placed), wav_path,
        )
    return [
        {"word": a["word"], "start": a["start"], "end": a["end"], "probability": None}
        for a in placed
        if a["start"] is not None
    ]
