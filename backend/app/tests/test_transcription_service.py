"""Unit tests for TranscriptionService.

mlx-whisper.transcribe is patched at module scope so the tests never
download the ~3 GB MLX weights or run actual inference. The thing the
service layer owns is the raw -> wire-format normalization (segment shape,
word entries, language, duration heuristic), which is verifiable against
a stubbed return value.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.services.transcription import TranscriptionService
from app.services.transcription.service import (
    WHISPER_OPTIONS,
    build_options,
    default_model_repo,
)


def _stub_raw(
    segments: list[dict] | None = None,
    language: str | None = "en",
) -> dict:
    return {
        "language": language,
        "segments": segments or [],
        "text": "".join(s.get("text", "") for s in (segments or [])),
    }


def test_transcribe_normalizes_segments_and_words():
    raw = _stub_raw(
        segments=[
            {
                "start": 0.0,
                "end": 1.5,
                "text": " hello world",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": " hello"},
                    {"start": 0.6, "end": 1.5, "word": " world"},
                ],
            },
            {
                "start": 1.6,
                "end": 3.0,
                "text": " more text",
                "words": [
                    {"start": 1.6, "end": 3.0, "word": " more text"},
                ],
            },
        ],
    )
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    assert result.language == "en"
    assert len(result.segments) == 2
    # Core timing + text shape on the first segment.
    seg = result.segments[0]
    assert seg["start"] == 0.0
    assert seg["end"] == 1.5
    assert seg["text"] == " hello world"
    assert seg["words"] == [
        # Confidence fields are None when the stub omits them (see the
        # dedicated test below for the present-case).
        {"start": 0.0, "end": 0.5, "word": " hello", "probability": None},
        {"start": 0.6, "end": 1.5, "word": " world", "probability": None},
    ]
    # Duration is the last segment's end timestamp.
    assert result.duration_seconds == 3.0


def test_transcribe_preserves_confidence_fields():
    """mlx-whisper populates per-word probability and per-segment
    avg_logprob / no_speech_prob / compression_ratio / temperature on
    every real decode. We persist them on the segments JSONB so the
    planned vocal-safety logic (Phase 7+) doesn't have to re-transcribe.
    """
    raw = _stub_raw(
        segments=[
            {
                "start": 0.0,
                "end": 1.5,
                "text": " hello",
                "avg_logprob": -0.32,
                "no_speech_prob": 0.04,
                "compression_ratio": 1.8,
                "temperature": 0.0,
                "words": [
                    {
                        "start": 0.0,
                        "end": 1.5,
                        "word": " hello",
                        "probability": 0.94,
                    },
                ],
            },
        ],
    )
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    seg = result.segments[0]
    assert seg["avg_logprob"] == -0.32
    assert seg["no_speech_prob"] == 0.04
    assert seg["compression_ratio"] == 1.8
    assert seg["temperature"] == 0.0
    assert seg["words"][0]["probability"] == 0.94


def test_transcribe_confidence_fields_default_to_none_when_missing():
    """A stubbed mlx-whisper return that omits the confidence keys must
    still produce well-formed segments — the JSONB column is consumed by
    code that expects the keys to be present with None as the absent
    marker."""
    raw = _stub_raw(
        segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": " hi",
                "words": [{"start": 0.0, "end": 1.0, "word": " hi"}],
            }
        ]
    )
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    seg = result.segments[0]
    assert seg["avg_logprob"] is None
    assert seg["no_speech_prob"] is None
    assert seg["compression_ratio"] is None
    assert seg["temperature"] is None
    assert seg["words"][0]["probability"] is None


def test_transcribe_handles_missing_words_key():
    """mlx-whisper sometimes returns segments with no `words` key when
    word_timestamps fails on a chunk — the normalizer should treat that as
    an empty word list, not error."""
    raw = _stub_raw(
        segments=[
            {"start": 0.0, "end": 2.0, "text": " hello"},
        ]
    )
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    assert result.segments[0]["words"] == []
    assert result.duration_seconds == 2.0


def test_transcribe_empty_result_zero_duration():
    raw = _stub_raw(segments=[], language=None)
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=raw,
    ):
        result = svc.transcribe(Path("/fake/vocals.wav"))

    assert result.segments == []
    assert result.duration_seconds == 0.0
    assert result.language is None


def test_transcribe_auto_detects_language_by_default():
    """Multilingual queues must not be pinned to one language. A forced
    "en" decodes Japanese vocals as English gibberish."""
    svc = TranscriptionService()
    assert svc.language is None
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=_stub_raw(language="ja"),
    ) as mock_transcribe:
        result = svc.transcribe(Path("/fake/vocals.wav"))

    _, kwargs = mock_transcribe.call_args
    assert kwargs.get("language") is None
    assert result.language == "ja"


def test_transcribe_honours_explicit_language():
    """A per-song hint (e.g. the script of its .lrc) pins the decode."""
    svc = TranscriptionService(language="ja")
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=_stub_raw(language="ja"),
    ) as mock_transcribe:
        svc.transcribe(Path("/fake/vocals.wav"))

    _, kwargs = mock_transcribe.call_args
    assert kwargs.get("language") == "ja"


def test_model_name_and_repo_follow_settings():
    assert default_model_repo("large-v3") == "mlx-community/whisper-large-v3-mlx"
    svc = TranscriptionService(model_name="medium")
    assert svc.model_name == "medium"
    assert svc.model_repo == "mlx-community/whisper-medium-mlx"
    # An explicit repo wins over the derived name.
    svc = TranscriptionService(
        model_name="large-v3", model_repo="mlx-community/whisper-large-v3-bf16-mlx"
    )
    assert svc.model_repo == "mlx-community/whisper-large-v3-bf16-mlx"


def test_build_options_leaves_base_set_untouched():
    """build_options copies — mutating one call must not leak language
    into the next (a module-level dict would)."""
    assert build_options("ja")["language"] == "ja"
    assert build_options()["language"] is None
    assert build_options("")["language"] is None
    assert WHISPER_OPTIONS["language"] is None


def test_transcribe_passes_full_override_set():
    """Every override from WHISPER_OPTIONS must flow through to mlx-whisper.

    These are the tuned values for music + Demucs-leaked vocals — a
    silent regression would either re-introduce the "Thank you"
    hallucination loop, eat real lyric repetition, or hallucinate
    "(music)" / "[applause]" tags. The test pins every knob so a
    one-line edit to WHISPER_OPTIONS gets flagged."""
    svc = TranscriptionService()
    with patch(
        "app.services.transcription.service.mlx_whisper.transcribe",
        return_value=_stub_raw(),
    ) as mock_transcribe:
        svc.transcribe(Path("/fake/vocals.wav"))

    mock_transcribe.assert_called_once()
    _, kwargs = mock_transcribe.call_args
    assert "mlx-community" in kwargs.get("path_or_hf_repo", "")

    # Hallucination / quality guards
    assert kwargs.get("compression_ratio_threshold") == 3.0
    assert kwargs.get("logprob_threshold") == -1.2
    assert kwargs.get("no_speech_threshold") == 0.45
    assert kwargs.get("hallucination_silence_threshold") == 1.5
    assert kwargs.get("condition_on_previous_text") is False
    assert kwargs.get("temperature") == (0.0, 0.2)

    # Content / formatting
    assert "vocals" in (kwargs.get("initial_prompt") or "").lower()
    assert kwargs.get("word_timestamps") is True
    assert kwargs.get("clip_timestamps") == "0"

    # Decoder options
    # Language is auto-detected by default — pinning "en" here would
    # transcribe every non-English vocal in a multilingual queue as
    # English gibberish. See the dedicated language tests below.
    assert kwargs.get("language") is None
    assert kwargs.get("task") == "transcribe"
    # beam_size is intentionally NOT passed — mlx-whisper hasn't
    # implemented beam search yet and raises NotImplementedError if it's
    # set. Greedy decode at temperature 0.0 + best_of at higher temps.
    assert "beam_size" not in kwargs
    assert kwargs.get("best_of") == 3
