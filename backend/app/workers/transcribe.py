from __future__ import annotations

import logging
import uuid

from sqlalchemy import select, update

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import (
    Song,
    SongStatus,
    Stems,
    Transcription,
    TranscriptionStatus,
)
from app.services.storage import get_storage
from app.workers import celery_app

# Duplicated from app.services.transcription so this module can be imported
# without pulling in mlx_whisper (which is macOS/MLX-only — the slim Linux
# worker image on the droplet doesn't ship it, since heavy transcription
# goes via Modal). Kept in sync with Settings.whisper_model rather than
# hardcoded, so `WHISPER_MODEL` actually means something.
DEFAULT_MODEL_NAME = settings.whisper_model or "large-v3"

# Bind TranscriptionService at module top *if* the local stack is installed.
# On macOS dev / native worker this lets unit tests `patch(
# "app.workers.transcribe.TranscriptionService")` like they always have.
# On a slim Linux container without mlx_whisper, the import fails — we set
# it to None and the runtime takes the Modal branch instead.
try:
    from app.services.transcription import TranscriptionService  # noqa: F401
except ImportError:
    TranscriptionService = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Statuses we can transition into `transcribing`. Mirrors the API gate so
# manual re-transcribe and recovery from a failed run both work. `analyzed`
# is the post-separation resting state (Phase 5 bounces back from
# `separating` to `analyzed`); `ready` and `failed` allow re-runs.
CLAIMABLE_STATUSES = (
    SongStatus.analyzed,
    SongStatus.ready,
    SongStatus.failed,
)


def _delete_existing(db, song_uuid: uuid.UUID) -> None:
    existing = (
        db.query(Transcription)
        .filter(Transcription.song_id == song_uuid)
        .one_or_none()
    )
    if existing is not None:
        db.delete(existing)
        db.flush()


@celery_app.task(name="app.workers.transcribe.transcribe_song")
def transcribe_song(song_id: str) -> str | None:
    """Run Whisper over the song's vocal stem and persist segments.

    Idempotent under concurrent dispatch via the same atomic-claim pattern
    as analyze_song / separate_stems: the analyzed|ready|failed ->
    transcribing transition is a single UPDATE; losers log and return.

    Transitions: analyzed/ready/failed -> transcribing -> ready (success
    OR skipped_instrumental) | failed (on error).

    `ready` is now the terminal Song status — Phase 6 is the gate that
    promotes a song from `analyzed` to `ready`. Skipped-instrumental songs
    are still treated as `ready` (no Whisper data, but the pipeline doesn't
    require it for downstream phases).

    Vocal path comes from the Stems row's `vocals_path` — that's also the
    natural guardrail against running transcription before separation has
    completed.
    """
    song_uuid = uuid.UUID(song_id)
    storage = get_storage()
    threshold = settings.whisper_vocal_rms_threshold

    with SessionLocal() as db:
        song = db.get(Song, song_uuid)
        if song is None:
            logger.warning(
                "transcribe_song: song %s not found, skipping", song_id
            )
            return None

        claim = db.execute(
            update(Song)
            .where(Song.id == song_uuid)
            .where(Song.status.in_(CLAIMABLE_STATUSES))
            .values(status=SongStatus.transcribing, error_text=None)
        )
        db.commit()

        if claim.rowcount == 0:
            db.refresh(song)
            logger.info(
                "transcribe_song: %s already %s, skipping duplicate dispatch",
                song_id,
                song.status.value,
            )
            return None

        stems = db.scalar(select(Stems).where(Stems.song_id == song_uuid))
        if stems is None or stems.vocals_path is None:
            # Chain ordering violated: separate_stems must run first. Don't
            # write a Transcription row (no useful context to record) — flip
            # the song to failed and bail.
            logger.error(
                "transcribe_song: no stems/vocals_path for %s; "
                "separate_stems must run first",
                song_id,
            )
            song = db.get(Song, song_uuid)
            assert song is not None
            song.status = SongStatus.failed
            song.error_text = (
                "transcription failed: no separated vocal stem "
                "(separation must run first)"
            )
            db.commit()
            return None

        vocal_rms = float(stems.vocal_rms or 0.0)
        vocals_key = stems.vocals_path
        video_id = song.youtube_video_id

    # Skip-if-instrumental decision. Threshold lives in settings so the
    # user can tune it without code changes.
    if vocal_rms < threshold:
        logger.info(
            "transcribe_song: %s vocal_rms=%.4f < %.4f, skipping Whisper",
            song_id,
            vocal_rms,
            threshold,
        )
        with SessionLocal() as db:
            _delete_existing(db, song_uuid)
            row = Transcription(
                song_id=song_uuid,
                model_name=DEFAULT_MODEL_NAME,
                status=TranscriptionStatus.skipped_instrumental,
                language=None,
                segments=[],
                vocal_rms_threshold=threshold,
                vocal_rms_observed=vocal_rms,
                duration_seconds=None,
            )
            db.add(row)
            song = db.get(Song, song_uuid)
            assert song is not None
            song.status = SongStatus.ready
            db.commit()
            _maybe_dispatch_stitch(song_uuid, db)

        return str(song_uuid)

    import tempfile
    import asyncio
    from pathlib import Path

    use_modal = bool(settings.modal_token_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        vocals_path = Path(tmpdir) / "vocals.wav"

        try:
            # Prefer forced alignment whenever we already hold the true
            # lyric text: recognition is the wrong problem, and it can't
            # mishear the words. Placed before the Whisper branch so it
            # works for the Modal path too (it needs a local copy of the
            # stem, which is cheap next to a GPU decode).
            forced = None
            if settings.forced_align_enabled:
                asyncio.run(storage.download_file(vocals_key, vocals_path))
                forced = _forced_align(song_uuid, vocals_path)

            if forced is not None:
                segments, language, duration_seconds, model_name = forced
            elif use_modal:
                import modal

                fn = modal.Function.from_name(
                    "ai-dj-gpu-workers", "run_transcription"
                )
                result = fn.remote(
                    vocals_key,
                    video_id,
                    settings.s3_endpoint_url,
                    settings.s3_bucket_name,
                    settings.s3_access_key,
                    settings.s3_secret_key,
                    settings.s3_region_name,
                    settings.whisper_model,
                    _language_hint(song_uuid),
                )
                # Modal returns segments inline (already JSON-shaped).
                segments = result["segments"]
                language = result["language"]
                duration_seconds = result["duration"]
                model_name = result["model_name"]
            else:
                if TranscriptionService is None:
                    raise RuntimeError(
                        "transcribe_song: local Whisper unavailable "
                        "(mlx_whisper not installed) and MODAL_TOKEN_ID unset"
                    )
                if not vocals_path.exists():
                    asyncio.run(storage.download_file(vocals_key, vocals_path))
                # A per-song hint from the .lrc script beats audio-only
                # detection for unspaced scripts; None = auto-detect.
                service = TranscriptionService(
                    language=_language_hint(song_uuid)
                )
                result = service.transcribe(vocals_path)
                language = result.language
                segments = result.segments
                duration_seconds = result.duration_seconds
                model_name = service.model_name

        except Exception as exc:
            logger.exception("transcription failed for song %s", song_id)
            with SessionLocal() as db:
                _delete_existing(db, song_uuid)
                row = Transcription(
                    song_id=song_uuid,
                    model_name=DEFAULT_MODEL_NAME,
                    status=TranscriptionStatus.error,
                    language=None,
                    segments=[],
                    vocal_rms_threshold=threshold,
                    vocal_rms_observed=vocal_rms,
                    duration_seconds=None,
                )
                db.add(row)
                song = db.get(Song, song_uuid)
                if song is not None:
                    song.status = SongStatus.failed
                    song.error_text = f"transcription failed: {exc}"[:1000]
                db.commit()
            raise

    with SessionLocal() as db:
        _delete_existing(db, song_uuid)
        row = Transcription(
            song_id=song_uuid,
            model_name=model_name,
            status=TranscriptionStatus.success,
            language=language,
            segments=segments,
            vocal_rms_threshold=threshold,
            vocal_rms_observed=vocal_rms,
            duration_seconds=duration_seconds,
        )
        db.add(row)
        song = db.get(Song, song_uuid)
        assert song is not None
        song.status = SongStatus.ready
        db.commit()
        _maybe_dispatch_stitch(song_uuid, db)

    from app.workers.align_lyrics import align_lyrics_task
    align_lyrics_task.delay(str(song_uuid))

    return str(song_uuid)


def _lyrics_row(song_uuid: uuid.UUID):
    from app.models import Lyrics

    with SessionLocal() as db:
        return db.scalar(select(Lyrics).where(Lyrics.song_id == song_uuid))


def _language_hint(song_uuid: uuid.UUID) -> str | None:
    """Whisper language pin from the lyric source, if it has one.

    An `.lrc` written in kana / Hangul / bare ideographs pins the decode
    far better than audio-only detection on a Demucs vocal stem, where
    the detector is easily fooled by sparse or reverbed vocals. Latin
    lyrics give no signal and we auto-detect as usual.
    """
    row = _lyrics_row(song_uuid)
    return (row.language or None) if row is not None else None


def _forced_align(song_uuid: uuid.UUID, vocals_path):
    """Build segments from known lyric text instead of recognising it.

    Returns ``(segments, language, duration_seconds, model_name)`` or
    None to fall back to Whisper — no text to align against, or the
    aligner's toolchain can't cover the script (see
    app.services.lyrics_alignment.forced_align).
    """
    from app.services.lyrics.tokenizer import tokenize
    from app.services.lyrics_alignment.forced_align import (
        MODEL_NAME,
        ForcedAlignUnavailable,
        align_words,
        segments_from_words,
    )

    row = _lyrics_row(song_uuid)
    if row is None or not (row.text or "").strip():
        logger.info(
            "transcribe_song: %s has no known lyric text; using Whisper", song_id=song_uuid
        )
        return None

    words = tokenize(row.text)
    if not words:
        return None

    try:
        placed = align_words(words, vocals_path, language=row.language)
    except ForcedAlignUnavailable as exc:
        logger.info(
            "transcribe_song: %s forced-align unavailable (%s); falling back to Whisper",
            song_uuid, exc,
        )
        return None

    if not placed:
        logger.info(
            "transcribe_song: %s forced-align placed no words; falling back to Whisper",
            song_uuid,
        )
        return None

    segments = segments_from_words(placed)
    duration = float(segments[-1]["end"]) if segments else 0.0
    logger.info(
        "transcribe_song: %s aligned %d known words to audio (%d segments)",
        song_uuid, len(placed), len(segments),
    )
    return segments, row.language, duration, MODEL_NAME


def _maybe_dispatch_stitch(song_uuid: uuid.UUID, db) -> None:
    """Phase 10 eager stitch: when this song reaching `ready` was the last
    one its locked queue was waiting on, fire the render+stitch chord so the
    continuous mix renders during the Processing state. Best-effort — a
    failure here must never fail the transcription that just succeeded.
    """
    try:
        from app.workers.auto_stitch import maybe_dispatch_stitch_for_song

        maybe_dispatch_stitch_for_song(song_uuid, db)
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "transcribe_song: eager stitch dispatch failed for %s", song_uuid
        )
