from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.song import SongStatus


class SearchResultSchema(BaseModel):
    youtube_video_id: str
    title: str
    artist: str | None = None
    duration_seconds: float
    thumbnail_url: str | None = None


class SongCreate(BaseModel):
    # YouTube video ids are exactly 11 characters from this alphabet. This
    # is not cosmetic: the id is interpolated into the download URL and
    # into storage keys (`audio/{id}.wav`, `stems/{id}/...`), so an
    # unconstrained string is an unbounded download trigger and a path
    # -shaping primitive. Local imports manage their own `local-<hash>`
    # keys and never go through this schema.
    youtube_video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    title: str
    artist: str | None = None
    duration_seconds: float = Field(ge=0)
    thumbnail_url: str | None = None


class SongRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    youtube_video_id: str
    title: str
    artist: str | None
    duration_seconds: float
    thumbnail_url: str | None
    audio_path: str | None
    status: SongStatus
    # Reason the song last failed (null otherwise). Surfaced in the
    # Processing view so a failure isn't a silent stall.
    error_text: str | None = None
    created_at: datetime
    updated_at: datetime
    # Derived from related rows so the Library + Player can render a
    # consistent "separated" / "transcribed" badge without per-song N+1
    # fetches. Default False for backwards compat — endpoints populate
    # the real value with one exists() per row.
    has_stems: bool = False
    has_transcription: bool = False
