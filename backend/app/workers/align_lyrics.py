import logging
import uuid
from typing import Any

from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import select

from app.workers import celery_app
from app.core.db import SessionLocal
from app.models.lyrics import Lyrics, LyricsAlignmentStatus, LyricsFetchStatus
from app.models.transcription import Transcription, TranscriptionStatus
from app.services.lyrics_alignment.aligner import align_lyrics

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.align_lyrics.align_lyrics_task", bind=True, max_retries=5)
def align_lyrics_task(self: Any, song_id: uuid.UUID | str) -> str | None:
    if isinstance(song_id, str):
        song_id = uuid.UUID(song_id)

    # Exponential backoff for "row not ready yet" retries — Genius
    # fetch can take 30-90s on a cold cache; fixed 10s gives up too soon.
    countdowns = [10, 20, 40, 80, 120]
    countdown_for_retry = countdowns[min(self.request.retries, len(countdowns) - 1)]

    def _mark_error_and_exit(reason: str) -> str | None:
        with SessionLocal() as db:
            row = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
            if row is not None:
                row.alignment_status = LyricsAlignmentStatus.error
                db.commit()
        logger.warning("align_lyrics: %s gave up — %s", song_id, reason)
        return str(song_id)

    with SessionLocal() as db:
        lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
        if not lyrics:
            try:
                raise self.retry(countdown=countdown_for_retry)
            except MaxRetriesExceededError:
                return _mark_error_and_exit("lyrics row never created")

        if lyrics.fetch_status == LyricsFetchStatus.not_attempted:
            try:
                raise self.retry(countdown=countdown_for_retry)
            except MaxRetriesExceededError:
                return _mark_error_and_exit("fetch_lyrics never completed")

        if lyrics.fetch_status == LyricsFetchStatus.not_found:
            # No Genius match exists; alignment isn't possible.
            logger.info(
                "align_lyrics: %s has no Genius match; marking whisper_only",
                song_id,
            )
            lyrics.alignment_status = LyricsAlignmentStatus.whisper_only
            db.commit()
            return str(song_id)

        if lyrics.fetch_status != LyricsFetchStatus.success:
            # fetch_status == error — Genius call failed earlier.
            logger.info(
                "align_lyrics: %s fetch_status=%s, marking error",
                song_id, lyrics.fetch_status,
            )
            lyrics.alignment_status = LyricsAlignmentStatus.error
            db.commit()
            return str(song_id)

        transcription = db.scalar(
            select(Transcription).where(Transcription.song_id == song_id)
        )
        if not transcription:
            try:
                raise self.retry(countdown=countdown_for_retry)
            except MaxRetriesExceededError:
                return _mark_error_and_exit("transcription never created")

        if transcription.status == TranscriptionStatus.skipped_instrumental:
            logger.info(
                "align_lyrics: %s is instrumental; marking whisper_only",
                song_id,
            )
            lyrics.alignment_status = LyricsAlignmentStatus.whisper_only
            db.commit()
            return str(song_id)
        if transcription.status != TranscriptionStatus.success:
            logger.info(
                "align_lyrics: %s transcription=%s, marking error",
                song_id, transcription.status,
            )
            lyrics.alignment_status = LyricsAlignmentStatus.error
            db.commit()
            return str(song_id)

        if lyrics.alignment_status in (
            LyricsAlignmentStatus.success,
            LyricsAlignmentStatus.low_quality,
            LyricsAlignmentStatus.whisper_only,
        ):
            return str(song_id)

        try:
            result = align_lyrics(
                transcription.segments,
                lyrics.text or "",
                line_times=lyrics.line_timestamps,
            )
            lyrics.aligned_words = result["aligned_words"]
            lyrics.alignment_quality = result["alignment_quality"]
            lyrics.alignment_status = result["alignment_status"]
            # Persist the fitted .lrc time warp (None on the untimed
            # Genius path) so a re-run reuses it and the UI can show how
            # far the sidecar had drifted.
            lyrics.alignment_offset = result.get("offset")
            lyrics.alignment_slope = result.get("slope")
            logger.info(
                "align_lyrics: %s aligned (%s), quality=%.2f",
                song_id,
                "lrc-timed" if lyrics.line_timestamps else "free",
                lyrics.alignment_quality,
            )
        except Exception as exc:
            logger.exception("align_lyrics: %s aligner raised", song_id)
            lyrics.alignment_status = LyricsAlignmentStatus.error
            db.commit()
            try:
                raise self.retry(exc=exc, countdown=30)
            except MaxRetriesExceededError:
                return _mark_error_and_exit(f"aligner kept raising: {exc}")

        db.commit()

    return str(song_id)
