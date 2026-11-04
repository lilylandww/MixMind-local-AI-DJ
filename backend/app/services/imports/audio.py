"""Audio-file ingest helpers: normalisation, hashing, metadata guessing.

Kept DB-free and storage-free so the API layer owns persistence and the
comment stays true: everything in here can be tested with plain paths.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

# Containers ffmpeg can decode and that users realistically have.
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma"}
)

# 44.1 kHz stereo PCM — what Demucs/Whisper/librosa all want, so every
# downstream stage sees a uniform input no matter what the user dropped in.
TARGET_SAMPLE_RATE = 44_100
TARGET_CHANNELS = 2

_LEADING_TRACK_NO = re.compile(r"^\s*\d{1,3}\s*[-._)\]]\s*")


class NormalizeError(RuntimeError):
    """ffmpeg could not decode the file (corrupt, DRM, not audio)."""


def content_digest(chunks: Iterable[bytes]) -> str:
    """sha256 of a byte-chunk iterable, hex. Streamed, so we never hold the
    whole upload in memory."""
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def guess_metadata(filename: str) -> tuple[str, str | None]:
    """Best-effort (title, artist) from a filename.

    "01 - Daft Punk - One More Time.mp3" -> ("One More Time", "Daft Punk")
    "One More Time.mp3"                  -> ("One More Time", None)
    """
    stem = Path(filename).stem
    stem = _LEADING_TRACK_NO.sub("", stem)
    stem = stem.replace("_", " ").strip()
    if " - " in stem:
        head, _, tail = stem.rpartition(" - ")
        if head and tail:
            return tail.strip(), head.strip()
    return stem or "Untitled", None


def normalize_to_wav(src: Path, dest: Path) -> Path:
    """Transcode `src` to stereo 44.1 kHz s16 WAV at `dest`.

    Raises NormalizeError if ffmpeg is missing or cannot decode the input.
    """
    if shutil.which("ffmpeg") is None:
        raise NormalizeError("ffmpeg not found on PATH")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-map", "0:a:0", "-vn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        msg = (proc.stderr or "ffmpeg produced no output").strip().splitlines()
        raise NormalizeError(msg[-1] if msg else "ffmpeg failed")
    return dest


def probe_duration(wav_path: Path) -> float:
    """Duration in seconds of a normalised WAV (soundfile is a dep)."""
    import soundfile as sf

    info = sf.info(str(wav_path))
    if info.samplerate <= 0:
        raise NormalizeError("invalid sample rate")
    return round(info.frames / info.samplerate, 3)
