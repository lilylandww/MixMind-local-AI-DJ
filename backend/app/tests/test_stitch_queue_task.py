"""stitch_queue Celery task tests. Builds a locked queue with rendered
MixPlan WAVs, mocks storage so dummy stereo audio stands in for the
real per-pair transitions, then runs stitch_queue end-to-end and
asserts on QueueRender's terminal state."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
    SongStatus,
)


# --- Helpers ------------------------------------------------------------


def _make_analysis(song_id, bpm: float = 120.0, key: str = "C") -> Analysis:
    return Analysis(
        song_id=song_id,
        bpm=bpm,
        key=key,
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * 0.5 for i in range(360)],
        downbeats=[i * 2.0 for i in range(90)],
        sections=[
            {"start": 0.0, "end": 30.0, "label": "intro"},
            {"start": 30.0, "end": 150.0, "label": "body"},
            {"start": 150.0, "end": 180.0, "label": "outro"},
        ],
        energy_curve=[0.5] * 180,
        vocal_segments=[],
    )


def _seed_locked_queue(
    n_songs: int = 3,
    mix_plan_status: MixPlanStatus = MixPlanStatus.ready,
    include_tempo_ramp: bool = False,
    include_queue_render: bool = True,
) -> str:
    """Build N-song locked queue with N-1 MixPlans and (optionally) a
    pending QueueRender row. Returns the queue id."""
    with SessionLocal() as db:
        q = Queue(locked=True)
        db.add(q)
        db.flush()

        songs = []
        for i in range(n_songs):
            s = Song(
                youtube_video_id=f"sq-{i}-{uuid.uuid4().hex}",
                title=f"Song {i}",
                duration_seconds=180.0,
                audio_path=f"audio/{i}.wav",
                status=SongStatus.ready,
            )
            db.add(s)
            songs.append(s)
        db.flush()

        for i, s in enumerate(songs):
            db.add(_make_analysis(s.id, bpm=120.0 + i * 5))
            db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
        db.flush()

        for i in range(n_songs - 1):
            plan_json: list[dict] = [{
                "tool": "set_transition_window",
                "from_song_time_start": 150.0,
                "to_song_time_start": 30.0,
                "duration_bars": 16,
            }]
            if include_tempo_ramp:
                plan_json.append({
                    "tool": "set_tempo_ramp",
                    "song": "B",
                    "start_time": 62.0,
                    "end_time": 78.0,
                    "start_bpm": 120.0,
                    "end_bpm": 120.0 + (i + 1) * 5,
                })
            db.add(MixPlan(
                queue_id=q.id,
                from_song_id=songs[i].id,
                to_song_id=songs[i + 1].id,
                status=mix_plan_status,
                rendered_audio_path=(
                    f"mixes/mp{i}.wav"
                    if mix_plan_status == MixPlanStatus.ready
                    else None
                ),
                plan_json=plan_json,
            ))

        if include_queue_render:
            db.add(QueueRender(queue_id=q.id, status=QueueRenderStatus.pending))
        db.commit()
        return str(q.id)


def _storage_writing_stereo_wav(samples_per_pair: int = 44100) -> AsyncMock:
    """Storage mock whose download_file writes a fixed-length stereo
    WAV at the destination path. AsyncMock so call assertions work."""
    storage = AsyncMock()

    async def _download_side(_key, dest):
        sf.write(str(dest), np.zeros((samples_per_pair, 2), dtype=np.float32), 44100)

    storage.download_file = AsyncMock(side_effect=_download_side)
    storage.upload_file = AsyncMock(return_value="/abs/out")
    return storage


@pytest.fixture(autouse=True)
def _no_live_broker():
    """Keep these tests off the real Redis broker.

    `stitch_queue` dispatches `enforce_cache_budget` fire-and-forget. Against
    a live broker a running dev worker consumes it, and the evictor then
    deletes songs/renders underneath this test's fixtures — surfacing as
    ObjectDeletedError or a TRUNCATE deadlock against the next test.
    Dispatch behaviour is covered in test_auto_stitch.py, not here.
    """
    with patch("app.workers.stitch_queue.celery_app.send_task") as send_task:
        yield send_task


# --- Tests --------------------------------------------------------------


@pytest.fixture
def locked_queue_with_mixes():
    yield _seed_locked_queue(n_songs=3, include_tempo_ramp=True)


def test_stitch_queue_happy_path(locked_queue_with_mixes):
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(locked_queue_with_mixes)

    assert res == locked_queue_with_mixes
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(locked_queue_with_mixes)
        ))
        assert qr.status == QueueRenderStatus.ready
        assert qr.rendered_audio_path == f"queue_mixes/{locked_queue_with_mixes}.m4a"
        assert qr.error_text is None
        # Phase 10: the player timeline is persisted alongside the M4A.
        tl = qr.timeline
        assert tl is not None
        assert len(tl["songs"]) == 3
        assert len(tl["transitions"]) == 2
        for t in tl["transitions"]:
            assert t["from_song_id"] and t["to_song_id"]
            assert t["label"]  # human-readable technique
            assert t["end"] >= t["start"]


def test_stitch_queue_atomic_claim_skips_when_already_rendering():
    queue_id = _seed_locked_queue(n_songs=2)
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        qr.status = QueueRenderStatus.rendering
        db.commit()

    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res is None
    storage.download_file.assert_not_called()
    storage.upload_file.assert_not_called()


def test_stitch_queue_fails_when_only_one_song():
    queue_id = _seed_locked_queue(n_songs=1)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res is None
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        assert qr.status == QueueRenderStatus.failed
        assert "at least 2 songs" in (qr.error_text or "")


def test_stitch_queue_fails_when_mix_plan_not_ready():
    queue_id = _seed_locked_queue(n_songs=2, mix_plan_status=MixPlanStatus.pending)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res is None
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        assert qr.status == QueueRenderStatus.failed
        assert "MixPlan" in (qr.error_text or "")


def test_stitch_queue_warns_when_no_queue_render_row():
    """Queue without an explicit QueueRender row → no-op, no crash."""
    queue_id = _seed_locked_queue(n_songs=2, include_queue_render=False)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)
    assert res is None


def test_stitch_queue_persists_m4a_key_and_uploads_from_the_encoded_file():
    queue_id = _seed_locked_queue(n_songs=3)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        stitch_queue(queue_id)

    expected_key = f"queue_mixes/{queue_id}.m4a"
    assert storage.upload_file.call_count == 1
    call = storage.upload_file.call_args
    assert Path(call.args[0]).suffix == ".m4a"  # a path, not a bytes blob
    assert call.args[1] == expected_key


def test_stitch_queue_never_buffers_the_whole_mix_in_memory():
    """Finding #8: the mix used to be written to a full PCM WAV, encoded,
    then read back as bytes for the upload — three full-size copies on top
    of the concatenate spike. It now streams PCM into ffmpeg and uploads
    from the encoded file's path."""
    queue_id = _seed_locked_queue(n_songs=3)
    storage = _storage_writing_stereo_wav()
    observed = {}

    async def _upload(src, key):
        path = Path(src)
        observed["existed"] = path.exists()
        observed["suffix"] = path.suffix
        return key

    storage.upload_file = AsyncMock(side_effect=_upload)
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        stitch_queue(queue_id)

    # Nothing may materialise the mix as a single bytes object.
    assert storage.write.call_count == 0
    assert storage.upload_file.call_count == 1
    # And the upload happened while the encoded file still existed.
    assert observed == {"existed": True, "suffix": ".m4a"}


def test_iter_pcm_blocks_slices_without_copying_the_whole_array():
    from app.workers.stitch_queue import _iter_pcm_blocks

    audio = np.arange(10, dtype=np.float32).reshape(5, 2)
    blocks = list(_iter_pcm_blocks(audio, block_frames=2))

    assert len(blocks) == 3  # 5 frames at 2 per block
    assert sum(len(b) for b in blocks) == audio.nbytes
    restored = np.frombuffer(b"".join(blocks), dtype=np.float32).reshape(5, 2)
    assert np.array_equal(restored, audio)


def test_iter_pcm_blocks_promotes_mono_to_one_channel():
    from app.workers.stitch_queue import _iter_pcm_blocks

    audio = np.zeros(4, dtype=np.float32)
    blocks = list(_iter_pcm_blocks(audio, block_frames=2))

    assert len(blocks) == 2
    assert sum(len(b) for b in blocks) == 4 * np.dtype(np.float32).itemsize


def test_stitch_queue_handles_tempo_ramp_in_plan_json():
    queue_id = _seed_locked_queue(n_songs=3, include_tempo_ramp=True)
    storage = _storage_writing_stereo_wav()
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(queue_id)

    assert res == queue_id
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(
            QueueRender.queue_id == uuid.UUID(queue_id)
        ))
        assert qr.status == QueueRenderStatus.ready


def _set_half_and_plans(queue_id: str, pair0: dict, pair1: dict) -> None:
    """Rewrite the two seeded plans for a 3-song capped-mode scenario."""
    with SessionLocal() as db:
        qid = uuid.UUID(queue_id)
        q = db.get(Queue, qid)
        q.song_length = "half"
        items = db.scalars(
            select(QueueItem)
            .where(QueueItem.queue_id == qid)
            .order_by(QueueItem.position)
        ).all()
        plans = db.scalars(
            select(MixPlan).where(MixPlan.queue_id == qid)
        ).all()
        by_from = {p.from_song_id: p for p in plans}
        by_from[items[0].song_id].plan_json = [pair0]
        by_from[items[1].song_id].plan_json = [pair1]
        db.commit()


def _cleanup_seed(queue_id: str) -> None:
    with SessionLocal() as db:
        qid = uuid.UUID(queue_id)
        items = db.scalars(
            select(QueueItem).where(QueueItem.queue_id == qid)
        ).all()
        song_ids = [it.song_id for it in items]
        q = db.get(Queue, qid)
        if q is not None:
            db.delete(q)
        db.flush()
        for sid in song_ids:
            song = db.get(Song, sid)
            if song is not None:
                db.delete(song)
        db.commit()


def test_stitch_queue_capped_mode_fails_loudly_on_overlap():
    """A late IN + early OUT used to clamp T_orig to 89 s and collapse the
    middle song to a zero-length timeline span. In a capped mode the
    stitcher must fail the render with a clear error instead."""
    queue_id = _seed_locked_queue(n_songs=3, include_tempo_ramp=False)
    try:
        _set_half_and_plans(
            queue_id,
            pair0={
                "tool": "set_transition_window",
                "from_song_time_start": 30.0,
                "to_song_time_start": 90.0,
                "duration_bars": 16,
            },
            pair1={
                "tool": "set_transition_window",
                "from_song_time_start": 90.0,
                "to_song_time_start": 30.0,
                "duration_bars": 8,
            },
        )
        storage = _storage_writing_stereo_wav(samples_per_pair=120 * 44100)
        with patch("app.workers.stitch_queue.get_storage", return_value=storage):
            from app.workers.stitch_queue import stitch_queue
            result = stitch_queue(queue_id)

        assert result is None
        with SessionLocal() as db:
            qr = db.scalar(select(QueueRender).where(
                QueueRender.queue_id == uuid.UUID(queue_id)
            ))
            assert qr.status == QueueRenderStatus.failed
            assert "overlap" in (qr.error_text or "").lower()
    finally:
        _cleanup_seed(queue_id)


def test_stitch_queue_capped_mode_keeps_middle_song_when_seams_clear():
    """The same late IN is fine when the next pair exits after its first
    chorus (150 s): the middle song keeps a real, non-zero span."""
    queue_id = _seed_locked_queue(n_songs=3, include_tempo_ramp=False)
    try:
        _set_half_and_plans(
            queue_id,
            pair0={
                "tool": "set_transition_window",
                "from_song_time_start": 30.0,
                "to_song_time_start": 90.0,
                "duration_bars": 16,
            },
            pair1={
                "tool": "set_transition_window",
                "from_song_time_start": 150.0,
                "to_song_time_start": 30.0,
                "duration_bars": 8,
            },
        )
        storage = _storage_writing_stereo_wav(samples_per_pair=160 * 44100)
        with patch("app.workers.stitch_queue.get_storage", return_value=storage):
            from app.workers.stitch_queue import stitch_queue
            result = stitch_queue(queue_id)

        assert result == queue_id
        with SessionLocal() as db:
            qr = db.scalar(select(QueueRender).where(
                QueueRender.queue_id == uuid.UUID(queue_id)
            ))
            assert qr.status == QueueRenderStatus.ready
            middle = qr.timeline["songs"][1]
            assert middle["end"] - middle["start"] > 5.0
    finally:
        _cleanup_seed(queue_id)
