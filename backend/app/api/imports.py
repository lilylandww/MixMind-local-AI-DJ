from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.models import Song, SongStatus
from app.schemas import SongRead
from app.services.imports import (
    AUDIO_EXTENSIONS,
    NormalizeError,
    content_digest,
    find_sidecar_lrc,
    guess_metadata,
    normalize_to_wav,
    probe_duration,
    read_lrc,
)
from app.services.lyrics.ingest import attach_lrc
from app.services.lyrics.lrc import LrcData, parse_lrc_bytes
from app.services.storage import get_storage
from app.workers import PRI_ANALYZE
from app.workers.analyze import analyze_song

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/imports", tags=["imports"])

CHUNK = 1024 * 1024


def _storage_key(song_key: str) -> str:
    # Same layout as the YouTube path so eviction / delete / range-streaming
    # need no new code: audio/<key>.wav
    return f"audio/{song_key}.wav"


def _dedupe(db: Session, song_key: str) -> Song | None:
    return db.scalar(select(Song).where(Song.youtube_video_id == song_key))


def _read_sidecar(path: Path) -> LrcData | None:
    """Parse the `.lrc` next to `path`, if there is one. Never raises."""
    sidecar = find_sidecar_lrc(path)
    if sidecar is None:
        return None
    try:
        return read_lrc(sidecar)
    except Exception:  # pragma: no cover - defensive
        logger.warning("import: unreadable sidecar %s", sidecar, exc_info=True)
        return None


def _maybe_attach_lrc(db: Session, song: Song, lrc_data: LrcData | None) -> None:
    """Best-effort sidecar ingest.

    Lyrics are strictly optional — a bad or unreadable `.lrc` must never
    turn a successful audio import into a failure.
    """
    if lrc_data is None:
        return
    try:
        attach_lrc(db, song.id, lrc_data)
    except Exception:  # pragma: no cover - defensive
        logger.exception("import: failed to attach .lrc for song %s", song.id)


async def _ingest_local_path(
    src: Path,
    *,
    song_key: str,
    title: str,
    artist: str | None,
    db: Session,
    process_now: bool = False,
    lrc_data: LrcData | None = None,
) -> tuple[Song, bool]:
    """Normalise `src` and persist a `downloaded` Song for it.

    Returns (song, created). `created=False` means an identical upload was
    already in the library and nothing new was written. `lrc_data`, when
    given, is attached as the song's lyric source either way — re-importing
    a folder is how a sidecar shows up after the fact.
    Idempotent on `song_key`, which is derived from the file's content hash.
    """
    existing = _dedupe(db, song_key)
    if existing is not None:
        _maybe_attach_lrc(db, existing, lrc_data)
        return existing, False

    storage = get_storage()
    # get_storage() is @lru_cache'd and its factory may hand back S3Storage,
    # so tests override `app.api.imports.get_storage` rather than settings.
    key = _storage_key(song_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = Path(tmpdir) / "normalized.wav"
        try:
            await run_in_threadpool(normalize_to_wav, src, wav)
            duration = await run_in_threadpool(probe_duration, wav)
        except NormalizeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"could not decode audio: {exc}",
            ) from exc
        await storage.upload_file(wav, key)

    song = Song(
        youtube_video_id=song_key,
        title=title,
        artist=artist,
        duration_seconds=duration,
        thumbnail_url=None,
        audio_path=key,
        status=SongStatus.downloaded,
        pipeline_requested=process_now,
    )
    db.add(song)
    try:
        db.commit()
    except IntegrityError:
        # Two identical uploads raced. The other one won; drop our blob and
        # return the winner so the caller still gets a usable song.
        db.rollback()
        winner = _dedupe(db, song_key)
        if winner is None:
            raise
        try:
            await storage.delete(key)
        except Exception:
            logger.warning("import: failed to clean orphan blob %r", key, exc_info=True)
        return winner, False
    db.refresh(song)

    _maybe_attach_lrc(db, song, lrc_data)

    if process_now:
        analyze_song.apply_async(args=[str(song.id)], priority=PRI_ANALYZE)

    return song, True


@router.post("/audio", response_model=SongRead, status_code=status.HTTP_201_CREATED)
async def import_audio(
    # `response` first: it has no default, so it must precede the Form params.
    # Injecting it lets us downgrade 201 -> 200 on a dedupe hit.
    response: Response,
    file: UploadFile = File(...),
    lrc_file: UploadFile | None = File(default=None),
    title: str | None = Form(default=None),
    artist: str | None = Form(default=None),
    process_now: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> SongRead:
    """Import one audio file from the user's machine.

    Multipart form fields: `file` (required), `lrc_file` (optional `.lrc`
    lyric sidecar), `title`, `artist`, `process_now`. Returns the created
    Song (201), or the existing Song for an identical re-upload (200).

    `lrc_file` is the browser-upload analogue of the sidecar the folder
    importer picks up automatically: a single-file upload has no real
    directory for us to look in, so the client has to send it along.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type {suffix or '(none)'}; "
                   f"expected one of {sorted(AUDIO_EXTENSIONS)}",
        )

    lrc_data: LrcData | None = None
    if lrc_file is not None:
        raw = await lrc_file.read()
        if raw.strip():
            lrc_data = parse_lrc_bytes(raw)

    max_bytes = int(settings.music_max_upload_mb * 1024 * 1024)

    h = hashlib.sha256()
    received = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / f"upload{suffix}"
        with src.open("wb") as out:
            while chunk := await file.read(CHUNK):
                received += len(chunk)
                if received > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"file exceeds {settings.music_max_upload_mb} MB limit",
                    )
                h.update(chunk)
                out.write(chunk)
        if received == 0:
            raise HTTPException(status_code=400, detail="empty upload")

        song_key = f"local-{h.hexdigest()[:16]}"
        guessed_title, guessed_artist = guess_metadata(file.filename or "")
        song, created = await _ingest_local_path(
            src,
            song_key=song_key,
            title=(title or "").strip() or guessed_title,
            artist=(artist or "").strip() or guessed_artist,
            db=db,
            process_now=process_now,
            lrc_data=lrc_data,
        )

    if not created and response is not None:
        response.status_code = status.HTTP_200_OK
    sr = SongRead.model_validate(song)
    sr.has_stems = False
    sr.has_transcription = False
    return sr


class FolderImportRequest(BaseModel):
    path: str                      # relative to settings.import_root
    recurse: bool = True
    process_now: bool = False


@router.post("/folder", response_model=list[SongRead])
async def import_folder(
    payload: FolderImportRequest, db: Session = Depends(get_db)
) -> list[SongRead]:
    """Import every decodable audio file under a directory on the server.

    Only enabled when settings.import_root is configured (bind-mount your
    music folder into the backend container). `payload.path` is resolved
    relative to that root and may not escape it. Files already in the
    library (same content hash) come back unchanged instead of re-adding.

    A `.lrc` sitting next to an audio file with the same stem is picked up
    automatically and stored as that song's lyric source — see
    `app.services.lyrics.ingest.attach_lrc`. That is the main reason to
    use this endpoint over per-file upload: a library that already has
    sidecar lyrics gets them with no extra work, and they beat Genius.
    """
    root = settings.import_root
    if not root or str(root) in ("", "."):
        raise HTTPException(status_code=400, detail="IMPORT_ROOT not configured")
    root = Path(root).resolve()
    target = (root / payload.path).resolve()
    # Containment check: refuse anything that escapes the configured root.
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="path escapes IMPORT_ROOT")
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="directory not found")

    files = sorted(
        p for p in (target.rglob("*") if payload.recurse else target.glob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    out: list[SongRead] = []
    for path in files:
        digest = content_digest([path.read_bytes()])  # stream if files get huge
        title, artist = guess_metadata(path.name)
        lrc_data = _read_sidecar(path)
        song, _ = await _ingest_local_path(
            path, song_key=f"local-{digest[:16]}",
            title=title, artist=artist, db=db, process_now=payload.process_now,
            lrc_data=lrc_data,
        )
        sr = SongRead.model_validate(song)
        sr.has_stems, sr.has_transcription = False, False
        out.append(sr)
    return out
