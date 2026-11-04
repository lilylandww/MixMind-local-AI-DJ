import logging
import uuid
from typing import Any

from asgiref.sync import async_to_sync
from sqlalchemy import select

from app.workers import celery_app
from app.core.db import SessionLocal
from app.models.lyrics import Lyrics, LyricsFetchStatus, LyricsSource
from app.models.song import Song
from app.services.lyrics.genius import fetch_lyrics as genius_fetch

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.fetch_lyrics.fetch_lyrics", bind=True, max_retries=3)
def fetch_lyrics_task(self: Any, song_id: uuid.UUID | str) -> str | None:
    if isinstance(song_id, str):
        song_id = uuid.UUID(song_id)

    with SessionLocal() as db:
        song = db.scalar(select(Song).where(Song.id == song_id))
        if not song:
            logger.error(f"fetch_lyrics: song {song_id} not found")
            return None

        from sqlalchemy.exc import IntegrityError

        lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
        if not lyrics:
            lyrics = Lyrics(song_id=song_id)
            db.add(lyrics)
            try:
                db.commit()
            except IntegrityError:
                # A concurrent worker beat us to it. Rollback and
                # re-read instead of crashing.
                db.rollback()
                lyrics = db.scalar(
                    select(Lyrics).where(Lyrics.song_id == song_id)
                )
                if lyrics is None:
                    # Shouldn't happen — unique constraint either holds
                    # a row or doesn't. Defensive log + retry.
                    logger.error(
                        "fetch_lyrics: %s IntegrityError but row not found",
                        song_id,
                    )
                    raise self.retry(countdown=30)


        if lyrics.source == LyricsSource.lrc and lyrics.text:
            # A local sidecar already holds the authoritative text for
            # this exact recording. Genius would be a downgrade — and it
            # must never overwrite a hand-supplied .lrc.
            logger.info(f"fetch_lyrics: {song_id} has .lrc lyrics, skipping Genius")
            if lyrics.fetch_status != LyricsFetchStatus.success:
                lyrics.fetch_status = LyricsFetchStatus.success
                db.commit()
            return str(song_id)

        if lyrics.fetch_status == LyricsFetchStatus.success:
            logger.info(f"fetch_lyrics: {song_id} already has lyrics, skipping")
            return str(song_id)

        try:
            result = async_to_sync(genius_fetch)(song.title, song.artist)
            if result:
                genius_id, text = result
                lyrics.genius_id = genius_id
                lyrics.text = text
                lyrics.fetch_status = LyricsFetchStatus.success
                logger.info(f"fetch_lyrics: {song_id} fetched successfully")
            else:
                lyrics.fetch_status = LyricsFetchStatus.not_found
                logger.info(f"fetch_lyrics: {song_id} not found on Genius")
        except Exception as e:
            lyrics.fetch_status = LyricsFetchStatus.error
            logger.error(f"fetch_lyrics: error for {song_id}: {e}", exc_info=True)
            db.commit()
            raise self.retry(exc=e, countdown=60)

        db.commit()

    return str(song_id)
