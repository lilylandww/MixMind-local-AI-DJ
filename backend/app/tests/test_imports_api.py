from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.main import app
from app.models import Song
from app.services.storage import LocalFilesystemStorage

ffmpeg_missing = shutil.which("ffmpeg") is None
needs_ffmpeg = pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg not installed")


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    p = tmp_path / "song.wav"
    sf.write(p, [0.0] * 22050, 22050)  # 1 s mono 22.05 kHz
    return p


@pytest.fixture
def tmp_storage(tmp_path: Path) -> LocalFilesystemStorage:
    storage = LocalFilesystemStorage(tmp_path / "storage")
    # The import endpoint writes via app.api.imports.get_storage while the
    # audio-streaming endpoint reads via app.api.songs.get_storage — patch
    # both so both routers see the same root.
    with patch("app.api.imports.get_storage", return_value=storage), \
            patch("app.api.songs.get_storage", return_value=storage):
        yield storage


def _client(db_session: Session) -> TestClient:
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _post_import(
    client: TestClient,
    path: Path,
    filename: str | None = None,
    **form_fields: str,
):
    data = {k: v for k, v in form_fields.items()}
    with path.open("rb") as fh:
        return client.post(
            "/api/imports/audio",
            files={"file": (filename or path.name, fh, "audio/wav")},
            data=data,
        )


def test_import_creates_downloaded_song(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    r = _post_import(client, sample_wav)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "downloaded"
    assert body["youtube_video_id"].startswith("local-")
    assert body["audio_path"] == f"audio/{body['youtube_video_id']}.wav"
    assert body["has_stems"] is False
    assert body["has_transcription"] is False
    assert body["title"] == "song"  # guessed from the filename
    # Library-added song must not auto-pipeline (mirrors POST /api/songs).
    assert body["pipeline_requested"] is False if "pipeline_requested" in body else True
    row = db_session.get(Song, body["id"])
    assert row is not None
    assert row.pipeline_requested is False
    # duration ≈ 1.0 s (mono 22.05 kHz source, 22050 frames)
    assert body["duration_seconds"] == pytest.approx(1.0, abs=0.05)
    # blob landed in storage under the same key layout as the YouTube path
    assert tmp_storage._path(body["audio_path"]).exists()


@needs_ffmpeg
def test_import_normalizes_to_stereo_44k(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    r = _post_import(client, sample_wav)
    assert r.status_code == 201, r.text
    stored = tmp_storage._path(r.json()["audio_path"])
    info = sf.info(str(stored))
    assert info.samplerate == 44100
    assert info.channels == 2


@needs_ffmpeg
def test_reimport_same_bytes_dedupes(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    r1 = _post_import(client, sample_wav)
    assert r1.status_code == 201
    r2 = _post_import(client, sample_wav)
    assert r2.status_code == 200
    assert r2.json()["id"] == r1.json()["id"]
    # no second blob appeared in storage
    import asyncio

    local_blobs = [
        key for key, _ in asyncio.run(tmp_storage.list_objects("audio/"))
        if key.startswith("audio/local-")
    ]
    assert local_blobs == [r1.json()["audio_path"]]


def test_import_rejects_unknown_extension(db_session: Session, tmp_path: Path):
    client = _client(db_session)
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"hello")
    r = _post_import(client, notes)
    assert r.status_code == 400
    assert "unsupported file type" in r.json()["detail"]


@needs_ffmpeg
def test_import_rejects_corrupt_audio(
    db_session: Session, tmp_path: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio")
    r = _post_import(client, bad)
    assert r.status_code == 400
    assert "could not decode audio" in r.json()["detail"]


@needs_ffmpeg
def test_import_rejects_oversized(
    db_session: Session,
    sample_wav: Path,
    tmp_storage: LocalFilesystemStorage,
    monkeypatch: pytest.MonkeyPatch,
):
    # 0.0001 MB ≈ 105 bytes — a real WAV blows past it on the first chunk.
    monkeypatch.setattr(settings, "music_max_upload_mb", 0.0001)
    client = _client(db_session)
    r = _post_import(client, sample_wav)
    assert r.status_code == 413
    assert "exceeds" in r.json()["detail"]


@needs_ffmpeg
def test_import_title_from_filename(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    r = _post_import(client, sample_wav, filename="01 - Daft Punk - One More Time.mp3")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["title"] == "One More Time"
    assert body["artist"] == "Daft Punk"


@needs_ffmpeg
def test_explicit_title_artist_win_over_filename(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    r = _post_import(
        client, sample_wav, filename="01 - Daft Punk - One More Time.mp3",
        title="Given Title", artist="Given Artist",
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["title"] == "Given Title"
    assert body["artist"] == "Given Artist"


@needs_ffmpeg
def test_process_now_dispatches_analyze(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    from app.workers import PRI_ANALYZE

    client = _client(db_session)
    with patch("app.api.imports.analyze_song.apply_async") as dispatch:
        r = _post_import(client, sample_wav, process_now="true")
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    dispatch.assert_called_once_with(args=[sid], priority=PRI_ANALYZE)
    row = db_session.get(Song, sid)
    assert row is not None
    assert row.pipeline_requested is True


def test_process_now_default_does_not_dispatch(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    with patch("app.api.imports.analyze_song.apply_async") as dispatch:
        r = _post_import(client, sample_wav)
    assert r.status_code == 201, r.text
    dispatch.assert_not_called()


@needs_ffmpeg
def test_imported_song_queues_and_advances_pipeline(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    """The real integration contract: an imported song behaves exactly like
    a YouTube one once queued — POST /api/queues + add item flips
    pipeline_requested and dispatches analyze (download is skipped)."""
    from app.workers import PRI_ANALYZE

    client = _client(db_session)
    r = _post_import(client, sample_wav)
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    q = client.post("/api/queues")
    assert q.status_code == 201, q.text
    queue_id = q.json()["id"]
    with patch("app.api.queues.analyze_song.apply_async") as dispatch:
        add = client.post(f"/api/queues/{queue_id}/items", json={"song_id": sid})
    assert add.status_code == 201, add.text
    dispatch.assert_called_once_with(args=[sid], priority=PRI_ANALYZE)
    row = db_session.get(Song, sid)
    assert row is not None
    assert row.pipeline_requested is True


@needs_ffmpeg
def test_imported_audio_streams_with_range(
    db_session: Session, sample_wav: Path, tmp_storage: LocalFilesystemStorage
):
    client = _client(db_session)
    r = _post_import(client, sample_wav)
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    full = client.get(f"/api/songs/{sid}/audio")
    assert full.status_code == 200
    assert full.headers["content-type"] == "audio/wav"
    assert len(full.content) > 100

    partial = client.get(
        f"/api/songs/{sid}/audio", headers={"Range": "bytes=0-99"}
    )
    assert partial.status_code == 206
    assert len(partial.content) == 100
