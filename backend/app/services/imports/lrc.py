"""Locate and read `.lrc` sidecars next to an imported audio file.

Path/IO only — persistence lives in
:mod:`app.services.lyrics.ingest`, keeping this importable in contexts
with no database.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.services.lyrics.lrc import LrcData, decode_lrc_bytes, parse_lrc

logger = logging.getLogger(__name__)

LRC_EXTENSION = ".lrc"


def find_sidecar_lrc(audio_path: Path) -> Path | None:
    """Locate a `.lrc` sitting next to `audio_path` with the same stem.

    Matches ``track.mp3`` -> ``track.lrc``, with a case-insensitive
    extension comparison (Windows libraries ship ``.LRC``). Returns None
    rather than raising: a missing sidecar must never fail an import.

    Only the file's own directory is consulted — walking parents would
    start guessing at unrelated lyrics in a big library folder.
    """
    try:
        parent = audio_path.parent
        stem = audio_path.stem
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        entries = list(parent.iterdir())
    except OSError as exc:
        logger.debug("find_sidecar_lrc: cannot list %s: %s", parent, exc)
        return None

    matches = [
        p for p in entries
        if p.is_file()
        and p.suffix.lower() == LRC_EXTENSION
        and p.stem == stem
    ]
    if not matches:
        # Tolerate `Track.mp3` -> `track.LRC`: libraries get renamed by
        # one side or the other. Deterministic pick so repeat imports
        # agree.
        matches = [
            p for p in entries
            if p.is_file()
            and p.suffix.lower() == LRC_EXTENSION
            and p.stem.casefold() == stem.casefold()
        ]
    return sorted(matches)[0] if matches else None


def read_lrc(path: Path) -> LrcData:
    """Read and parse a sidecar, tolerating the encodings real files use."""
    return parse_lrc(decode_lrc_bytes(path.read_bytes()))
