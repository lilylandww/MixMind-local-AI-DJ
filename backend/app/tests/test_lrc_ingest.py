"""`.lrc` sidecar discovery, ingest, and the forced-align segment builder.

The aligner's use of `line_timestamps` is covered in
`test_lyrics_aligner_lrc.py`; this covers getting a sidecar off disk and
into the `Lyrics` row without damaging anything already there.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.main import app
from app.models import (
    Lyrics,
    LyricsAlignmentStatus,
    LyricsFetchStatus,
    LyricsSource,
    Song,
    SongStatus,
)
from app.services.imports.lrc import find_sidecar_lrc, read_lrc
from app.services.lyrics.ingest import attach_lrc
from app.services.lyrics.lrc import parse_lrc
from app.services.lyrics_alignment.forced_align import segments_from_words

ffmpeg_missing = shutil.which("ffmpeg") is None
needs_ffmpeg = pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg not installed")


class TestFindSidecar:
    def test_same_stem_same_directory(self, tmp_path: Path):
        audio = tmp_path / "track.mp3"
        audio.write_bytes(b"x")
        lrc = tmp_path / "track.lrc"
        lrc.write_text("[00:01.00]hi", encoding="utf-8")
        assert find_sidecar_lrc(audio) == lrc

    def test_uppercase_extension(self, tmp_path: Path):
        # Windows libraries ship .LRC.
        audio = tmp_path / "track.mp3"
        audio.write_bytes(b"x")
        lrc = tmp_path / "track.LRC"
        lrc.write_text("[00:01.00]hi", encoding="utf-8")
        assert find_sidecar_lrc(audio) == lrc

    def test_stem_match_falls_back_to_case_insensitive(self, tmp_path: Path):
        audio = tmp_path / "Track.mp3"
        audio.write_bytes(b"x")
        lrc = tmp_path / "track.lrc"
        lrc.write_text("[00:01.00]hi", encoding="utf-8")
        assert find_sidecar_lrc(audio) == lrc

    def test_exact_stem_wins_over_the_folded_one(self, tmp_path: Path):
        audio = tmp_path / "Track.mp3"
        audio.write_bytes(b"x")
        exact = tmp_path / "Track.lrc"
        exact.write_text("[00:01.00]exact", encoding="utf-8")
        folded = tmp_path / "track.lrc"
        folded.write_text("[00:01.00]folded", encoding="utf-8")
        assert find_sidecar_lrc(audio) == exact

    def test_different_stem_is_not_a_match(self, tmp_path: Path):
        audio = tmp_path / "track.mp3"
        audio.write_bytes(b"x")
        (tmp_path / "other.lrc").write_text("[00:01.00]hi", encoding="utf-8")
        assert find_sidecar_lrc(audio) is None

    def test_other_sidecar_types_are_ignored(self, tmp_path: Path):
        audio = tmp_path / "track.mp3"
        audio.write_bytes(b"x")
        (tmp_path / "track.txt").write_text("hi", encoding="utf-8")
        (tmp_path / "track.srt").write_text("hi", encoding="utf-8")
        assert find_sidecar_lrc(audio) is None

    def test_missing_sidecar_is_not_an_error(self, tmp_path: Path):
        audio = tmp_path / "track.mp3"
        audio.write_bytes(b"x")
        assert find_sidecar_lrc(audio) is None

    def test_does_not_walk_parent_directories(self, tmp_path: Path):
        (tmp_path / "track.lrc").write_text("[00:01.00]hi", encoding="utf-8")
        nested = tmp_path / "album"
        nested.mkdir()
        audio = nested / "track.mp3"
        audio.write_bytes(b"x")
        assert find_sidecar_lrc(audio) is None

    def test_directory_named_like_a_sidecar_is_skipped(self, tmp_path: Path):
        audio = tmp_path / "track.mp3"
        audio.write_bytes(b"x")
        (tmp_path / "track.lrc").mkdir()
        assert find_sidecar_lrc(audio) is None


class TestReadLrc:
    def test_reads_gb18030_sidecar(self, tmp_path: Path):
        p = tmp_path / "track.lrc"
        p.write_bytes("[00:01.00]我爱北京\n".encode("gb18030"))
        assert read_lrc(p).text == "我爱北京"


@pytest.fixture
def song(db_session: Session) -> Song:
    s = Song(
        youtube_video_id=f"lrc-{uuid.uuid4().hex[:8]}",
        title="Test Song",
        artist="Artist",
        duration_seconds=60.0,
        audio_path="audio/x.wav",
        status=SongStatus.downloaded,
    )
    db_session.add(s)
    db_session.commit()
    return s


class TestAttachLrc:
    def test_stores_text_timestamps_and_language_hint(
        self, db_session: Session, song: Song
    ):
        data = parse_lrc("[ar:誰]\n[00:01.00]東京の夜\n[00:05.00]静かな光\n")
        row = attach_lrc(db_session, song.id, data)

        assert row.source == LyricsSource.lrc
        assert row.text == "東京の夜\n静かな光"
        assert row.fetch_status.value == "success"
        assert [ln["start"] for ln in row.line_timestamps] == [1.0, 5.0]
        assert row.line_timestamps[0]["end"] == 5.0
        # Kana pins Japanese far better than audio-only detection would.
        assert row.language == "ja"

    def test_latin_text_sets_no_language_hint(self, db_session: Session, song: Song):
        row = attach_lrc(db_session, song.id, parse_lrc("[00:01.00]one more time\n"))
        assert row.language is None

    def test_invalidates_a_stale_alignment(self, db_session: Session, song: Song):
        row = attach_lrc(db_session, song.id, parse_lrc("[00:01.00]old text\n"))
        row.aligned_words = [{"word": "old", "start": 0.0, "end": 1.0}]
        row.alignment_status = LyricsAlignmentStatus.success
        row.alignment_quality = 0.9
        row.alignment_offset = -1.5
        row.alignment_slope = 1.02
        db_session.commit()

        attach_lrc(db_session, song.id, parse_lrc("[00:01.00]new text\n"))
        db_session.refresh(row)
        assert row.text == "new text"
        assert row.aligned_words is None
        assert row.alignment_quality is None
        assert row.alignment_offset is None
        assert row.alignment_slope is None
        assert row.alignment_status == LyricsAlignmentStatus.not_attempted

    def test_reimporting_the_same_file_keeps_the_alignment(
        self, db_session: Session, song: Song
    ):
        row = attach_lrc(db_session, song.id, parse_lrc("[00:01.00]same\n"))
        row.aligned_words = [{"word": "same", "start": 0.0, "end": 1.0}]
        row.alignment_status = LyricsAlignmentStatus.success
        db_session.commit()

        attach_lrc(db_session, song.id, parse_lrc("[00:01.00]same\n"))
        db_session.refresh(row)
        assert row.alignment_status == LyricsAlignmentStatus.success
        assert row.aligned_words is not None

    def test_metadata_only_sidecar_never_blanks_existing_lyrics(
        self, db_session: Session, song: Song
    ):
        row = attach_lrc(db_session, song.id, parse_lrc("[00:01.00]real lyrics\n"))
        assert row.text == "real lyrics"

        assert attach_lrc(db_session, song.id, parse_lrc("[ti:Only A Title]\n")) is None
        db_session.refresh(row)
        assert row.text == "real lyrics"

    def test_genius_row_is_promoted_to_lrc(self, db_session: Session, song: Song):
        row = Lyrics(song_id=song.id, text="scraped", fetch_status=LyricsFetchStatus.success)
        db_session.add(row)
        db_session.commit()

        updated = attach_lrc(db_session, song.id, parse_lrc("[00:01.00]sidecar wins\n"))
        assert updated.source == LyricsSource.lrc
        assert updated.text == "sidecar wins"

    def test_creates_only_one_row_per_song(self, db_session: Session, song: Song):
        attach_lrc(db_session, song.id, parse_lrc("[00:01.00]first\n"))
        attach_lrc(db_session, song.id, parse_lrc("[00:01.00]second\n"))
        rows = db_session.scalars(
            select(Lyrics).where(Lyrics.song_id == song.id)
        ).all()
        assert len(rows) == 1
        assert rows[0].text == "second"


class TestSegmentsFromWords:
    def test_breaks_on_a_long_gap(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.3},
            {"word": "b", "start": 0.4, "end": 0.7},
            {"word": "c", "start": 5.0, "end": 5.3},
        ]
        segs = segments_from_words(words, break_gap=1.5)
        assert len(segs) == 2
        assert segs[0]["start"] == 0.0 and segs[0]["end"] == 0.7
        assert segs[1]["start"] == 5.0

    def test_breaks_at_the_max_span(self):
        words = [
            {"word": str(i), "start": float(i), "end": float(i) + 0.5}
            for i in range(40)
        ]
        segs = segments_from_words(words, max_segment_seconds=30.0, break_gap=9.0)
        assert len(segs) > 1

    def test_uses_mlx_leading_space_convention(self):
        # TranscriptionService documents that concatenating segment words
        # yields fluent text, so words keep their leading space.
        segs = segments_from_words([
            {"word": "hello", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.4, "end": 0.7},
        ])
        assert segs[0]["text"] == " hello world"
        assert [w["word"] for w in segs[0]["words"]] == [" hello", " world"]

    def test_confidences_are_none_not_zero(self):
        # None = "no hypothesis to be unsure about". A 0 would be read as
        # "certainly wrong" and filtered out by vocal_safety.
        segs = segments_from_words([{"word": "a", "start": 0.0, "end": 0.3}])
        assert segs[0]["words"][0]["probability"] is None
        assert segs[0]["avg_logprob"] is None
        assert segs[0]["no_speech_prob"] is None

    def test_empty_input(self):
        assert segments_from_words([]) == []


class _FakeStorage:
    async def upload_file(self, src, key):
        return key

    async def download_file(self, key, dest):
        return dest

    async def delete(self, key):
        return None


class TestImportIngestsSidecar:
    @needs_ffmpeg
    def test_upload_with_lrc_file_stores_the_lyrics(self, db_session: Session, tmp_path: Path):
        audio = tmp_path / "song.wav"
        sf.write(audio, [0.0] * 22050, 22050)
        lrc = tmp_path / "song.lrc"
        lrc.write_text("[00:01.00]sidecar line\n", encoding="utf-8")

        def override_db():
            yield db_session

        app.dependency_overrides[get_db] = override_db
        try:
            with patch("app.api.imports.get_storage", return_value=_FakeStorage()):
                client = TestClient(app)
                with audio.open("rb") as fh, lrc.open("rb") as lh:
                    r = client.post(
                        "/api/imports/audio",
                        files={
                            "file": ("song.wav", fh, "audio/wav"),
                            "lrc_file": ("song.lrc", lh, "text/plain"),
                        },
                    )
        finally:
            app.dependency_overrides.clear()

        assert r.status_code == 201, r.text
        song_id = uuid.UUID(r.json()["id"])
        row = db_session.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
        assert row is not None
        assert row.source == LyricsSource.lrc
        assert row.text == "sidecar line"

    @needs_ffmpeg
    def test_upload_without_lrc_still_works(self, db_session: Session, tmp_path: Path):
        audio = tmp_path / "song.wav"
        sf.write(audio, [0.0] * 22050, 22050)

        def override_db():
            yield db_session

        app.dependency_overrides[get_db] = override_db
        try:
            with patch("app.api.imports.get_storage", return_value=_FakeStorage()):
                client = TestClient(app)
                with audio.open("rb") as fh:
                    r = client.post(
                        "/api/imports/audio",
                        files={"file": ("song.wav", fh, "audio/wav")},
                    )
        finally:
            app.dependency_overrides.clear()

        assert r.status_code == 201, r.text
