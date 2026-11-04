"""Persist a lyric source against a song.

The one rule that matters: **a local `.lrc` beats Genius.** It belongs to
this exact recording, it is already timed, and it needs no network round
trip — so once we have one we never let a later Genius scrape overwrite
it (see :mod:`app.workers.fetch_lyrics`).
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.lyrics import (
    Lyrics,
    LyricsAlignmentStatus,
    LyricsFetchStatus,
    LyricsSource,
)
from app.services.lyrics.lrc import LrcData
from app.services.lyrics.tokenizer import script_hint

logger = logging.getLogger(__name__)


def _get_or_create(db: Session, song_id: uuid.UUID) -> Lyrics:
    row = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
    if row is not None:
        return row
    row = Lyrics(song_id=song_id)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        # A concurrent fetch_lyrics beat us to the insert.
        db.rollback()
        row = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
        if row is None:
            raise
    return row


def attach_lrc(db: Session, song_id: uuid.UUID, data: LrcData) -> Lyrics | None:
    """Store parsed `.lrc` content as the song's lyric source.

    Overwrites any existing text: a hand-supplied sidecar is a deliberate
    act, and it is more likely right than whatever Genius guessed. That
    also invalidates any alignment done against the old text.

    Returns None (and stores nothing) when the file carries no lyric text
    at all — metadata-only `.lrc` files exist and shouldn't blank out
    lyrics we already have.
    """
    text = data.text.strip()
    if not text:
        logger.info("attach_lrc: no lyric text for song %s; ignoring sidecar", song_id)
        return None

    row = _get_or_create(db, song_id)
    if row.source == LyricsSource.lrc and row.text == text:
        # Already attached (re-importing an unchanged folder). Don't
        # reset the alignment — that would force a pointless re-run.
        logger.debug("attach_lrc: song %s already has this .lrc; keeping", song_id)
        return row

    row.source = LyricsSource.lrc
    row.text = text
    row.line_timestamps = data.to_json() or None
    # Script hint only where it's unambiguous (kana -> ja, Hangul -> ko,
    # bare ideographs -> zh, ...). Latin text gives nothing, and a wrong
    # hint is worse than none.
    row.language = script_hint(text)
    row.fetch_status = LyricsFetchStatus.success
    # The text changed, so any alignment against the old text is stale.
    row.aligned_words = None
    row.alignment_quality = None
    row.alignment_offset = None
    row.alignment_slope = None
    row.alignment_status = LyricsAlignmentStatus.not_attempted
    db.commit()
    logger.info(
        "attach_lrc: song %s <- %d lines (%s), language hint=%s",
        song_id,
        len(row.line_timestamps or []),
        "word-timed" if data.word_level else ("line-timed" if data.timed else "unsynced"),
        row.language,
    )
    return row
