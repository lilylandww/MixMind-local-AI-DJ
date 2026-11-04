import enum
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, ForeignKey, String, Enum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base

if TYPE_CHECKING:
    from app.models.song import Song


class LyricsFetchStatus(str, enum.Enum):
    not_attempted = "not_attempted"
    success = "success"
    not_found = "not_found"
    error = "error"


class LyricsAlignmentStatus(str, enum.Enum):
    not_attempted = "not_attempted"
    success = "success"
    whisper_only = "whisper_only"
    low_quality = "low_quality"
    error = "error"


class LyricsSource(str, enum.Enum):
    """Where `Lyrics.text` / `line_timestamps` came from.

    `lrc` beats `genius` whenever both exist: a sidecar belongs to this
    exact recording, is already timed, and needs no network round trip.
    """

    genius = "genius"
    lrc = "lrc"
    embedded = "embedded"


class Lyrics(Base):
    __tablename__ = "lyrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    song_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    genius_id: Mapped[int | None] = mapped_column(nullable=True)
    text: Mapped[str | None] = mapped_column(String, nullable=True)
    fetch_status: Mapped[LyricsFetchStatus] = mapped_column(
        Enum(LyricsFetchStatus, name="lyricsfetchstatus"),
        nullable=False,
        default=LyricsFetchStatus.not_attempted,
    )

    source: Mapped[LyricsSource] = mapped_column(
        Enum(LyricsSource, name="lyricssource"),
        nullable=False,
        default=LyricsSource.genius,
    )
    # Persisted LrcData.to_json(): [{"start", "end", "text", "words"}, ...].
    # Null for untimed sources (Genius). Presence of this is what switches
    # the aligner onto its line-constrained path.
    line_timestamps: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True
    )
    # Script-derived Whisper language hint (e.g. "ja"), set when the lyric
    # text can pin the language better than audio-only detection can.
    language: Mapped[str | None] = mapped_column(String, nullable=True)

    aligned_words: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    alignment_status: Mapped[LyricsAlignmentStatus] = mapped_column(
        Enum(LyricsAlignmentStatus, name="lyricsalignmentstatus"),
        nullable=False,
        default=LyricsAlignmentStatus.not_attempted,
    )
    alignment_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Time warp fitted while aligning a timed .lrc: audio_time =
    # alignment_offset + alignment_slope * lrc_time. Persists the drift
    # correction so re-runs stay stable and the UI can show it.
    alignment_offset: Mapped[float | None] = mapped_column(Float, nullable=True)
    alignment_slope: Mapped[float | None] = mapped_column(Float, nullable=True)

    song: Mapped["Song"] = relationship("Song")
