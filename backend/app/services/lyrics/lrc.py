"""Parse `.lrc` lyric sidecar files.

An `.lrc` is a strictly better lyric source than a Genius scrape for our
purposes: it is local (no rate limit, no scraping), it is usually the
*correct* text for the exact recording, it works for tracks Genius has
never heard of, and — unlike Genius — **it carries timestamps**.

Formats handled
---------------
Standard line-level::

    [ti:Song] [ar:Artist] [al:Album] [by:Author] [offset:+500]
    [00:12.00] first line of the verse
    [00:17.20][01:45.20] a chorus line that repeats

Enhanced / word-level (A2 extension) — the tag precedes each word::

    [00:12.00]<00:12.00>first <00:12.40>line

Also tolerated: ``[hh:mm:ss.frac]`` extended stamps, unsynced files with
no timestamps at all (still useful — the text can be aligned to Whisper),
CRLF, UTF-8 BOM, and non-UTF-8 encodings (``.lrc`` files from CJK sources
are frequently GB18030 — see :func:`decode_lrc_bytes`).

Ambiguity note
--------------
A bare three-part stamp ``[03:45:00]`` could be ``mm:ss:hundredths``
(old Winamp style) or ``hh:mm:ss``. It is read per file: a file that uses
explicit fractional stamps anywhere is clearly on the extended
``hh:mm:ss.frac`` family, one that never does is read as ``mm:ss:ff``.
The aligner's offset/drift correction (see
``app.services.lyrics_alignment.aligner``) absorbs residual error here,
and callers can force either reading via ``colon_hundredths``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- timestamp shapes -------------------------------------------------
# Groups: (1) leading field, (2) middle field, (3) trailing field,
# (4) explicit fraction. Field semantics depend on the arity — see
# _read_time.
_TIME_CORE = r"(?:(\d{1,2}):)?(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?"
_BRACKET_TIME = re.compile(r"\[(?:" + _TIME_CORE + r")\]")
_ANGLE_TIME = re.compile(r"<(?:" + _TIME_CORE + r")>")
_META_TAG = re.compile(r"^\[([A-Za-z][A-Za-z0-9_-]*):(.*)\]\s*$")
_SECTION_HEADER = re.compile(r"\[[^\[\]]*\]")

_ENCODINGS = (
    "utf-8-sig", "utf-8", "gb18030", "big5", "shift_jis", "euc-kr", "latin-1",
)


def decode_lrc_bytes(data: bytes) -> str:
    """Decode raw `.lrc` bytes, tolerating the encodings real files use.

    `.lrc` files from Chinese/Japanese/Korean sources are frequently
    GB18030 / Big5 / Shift_JIS rather than UTF-8, and a multilingual
    playlist will hit those. Try the likely candidates in order and take
    the first strict decode that succeeds; `latin-1` is the always-works
    last resort (mojibake beats a crash).
    """
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("latin-1", errors="replace")


def _frac_ms(frac: str | None) -> int:
    """`.x` -> tenths, `.xx` -> hundredths, `.xxx` -> milliseconds."""
    return int((frac or "0").ljust(3, "0")[:3])


def _read_time(match: re.Match[str], *, colon_hundredths: bool) -> float:
    g1, g2, g3, g4 = match.groups()
    if g1 is not None and g4 is None and colon_hundredths:
        # Bare three-part stamp on the old [mm:ss:ff] family: the third
        # field counts hundredths of a second, not seconds.
        return int(g1) * 60 + int(g2) + _frac_ms(g3) / 1000.0
    return (
        int(g1 or 0) * 3600 + int(g2) * 60 + int(g3) + _frac_ms(g4) / 1000.0
    )


def _file_uses_fractions(text: str) -> bool:
    """True if any stamp in the file carries an explicit fraction.

    Used to disambiguate bare three-part stamps: a file that writes
    `[00:12.34]` everywhere is using the extended `[hh:mm:ss.frac]`
    family, so `[01:02:03]` is 1h02m03s. A file with no fractions at all
    is far more likely the old `[mm:ss:ff]` style.
    """
    for m in _BRACKET_TIME.finditer(text):
        if m.group(4) is not None:
            return True
    for m in _ANGLE_TIME.finditer(text):
        if m.group(4) is not None:
            return True
    return False


@dataclass
class LrcWord:
    """One enhanced-format word. `start` is None for text that precedes
    the first `<time>` tag on its line (untimed lead-in)."""

    start: float | None
    text: str

    def to_json(self) -> dict:
        return {"start": self.start, "text": self.text}


@dataclass
class LrcLine:
    start: float | None
    text: str
    words: list[LrcWord] = field(default_factory=list)

    def to_json(self, end: float | None = None) -> dict:
        return {
            "start": self.start,
            "end": end,
            "text": self.text,
            "words": [w.to_json() for w in self.words] or None,
        }


@dataclass
class LrcData:
    lines: list[LrcLine]
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    author: str | None = None
    offset_ms: int = 0

    @property
    def timed(self) -> bool:
        return any(ln.start is not None for ln in self.lines)

    @property
    def text(self) -> str:
        """Plain lyric text — what the aligner wants as its reference."""
        return "\n".join(ln.text for ln in self.lines if ln.text.strip())

    @property
    def word_level(self) -> bool:
        """True when enhanced `<time>` tags give per-word anchors."""
        return any(ln.words for ln in self.lines)

    def timed_lines(self) -> list[LrcLine]:
        """Lines with a usable start time, in time order.

        Blank timed lines are kept — a timestamped empty line is an
        authored instrumental gap, which is exactly the signal
        `vocal_safe_regions` wants.
        """
        return sorted(
            (ln for ln in self.lines if ln.start is not None),
            key=lambda ln: ln.start or 0.0,
        )

    def to_json(self) -> list[dict]:
        """Shape persisted into `Lyrics.line_timestamps`.

        Only timed lines are emitted; `end` is the next line's start (or
        None on the tail). An entirely unsynced `.lrc` yields `[]` and the
        caller should fall back to `LrcData.text`.
        """
        timed = self.timed_lines()
        out: list[dict] = []
        for i, ln in enumerate(timed):
            nxt = timed[i + 1].start if i + 1 < len(timed) else None
            out.append(ln.to_json(end=nxt))
        return out


def _parse_words(
    remainder: str, *, colon_hundredths: bool
) -> list[LrcWord]:
    """Split enhanced-format line text into `<time>`-anchored words.

    A `<time>` tag opens a span that runs until the next tag; the span's
    text may hold several whitespace words, which are then spread evenly
    across the span so every token gets its own timestamp. Text before
    the first tag (untimed lead-in) keeps `start=None`.
    """
    spans: list[list] = []  # [start|None, text]
    pos = 0
    saw_tag = False
    for m in _ANGLE_TIME.finditer(remainder):
        saw_tag = True
        chunk = remainder[pos : m.start()].strip()
        if chunk:
            if spans:
                spans[-1][1] = f"{spans[-1][1]} {chunk}".strip()
            else:
                spans.append([None, chunk])
        spans.append([_read_time(m, colon_hundredths=colon_hundredths), ""])
        pos = m.end()
    tail = remainder[pos:].strip()
    if tail:
        if spans:
            spans[-1][1] = f"{spans[-1][1]} {tail}".strip()
        else:
            spans.append([None, tail])

    if not saw_tag:
        # Plain line-level .lrc — timing is on the line, and splitting its
        # text into untimed pseudo-words would only duplicate `text`.
        return []

    out: list[LrcWord] = []
    for i, (start, text) in enumerate(spans):
        chunk_words = text.split()
        if not chunk_words:
            continue
        if start is None:
            out.extend(LrcWord(None, w) for w in chunk_words)
            continue
        end = next(
            (spans[j][0] for j in range(i + 1, len(spans)) if spans[j][0] is not None),
            None,
        )
        if end is None or end <= start:
            out.extend(LrcWord(start, w) for w in chunk_words)
            continue
        step = (end - start) / len(chunk_words)
        for k, w in enumerate(chunk_words):
            out.append(LrcWord(start + k * step, w))
    return out


def parse_lrc(
    text: str,
    *,
    colon_hundredths: bool | None = None,
) -> LrcData:
    """Parse `.lrc` source text into timed lines + metadata.

    `colon_hundredths` forces the bare `[mm:ss:ff]` reading (True) or the
    `hh:mm:ss` reading (False); `None` (default) infers it per file.
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if colon_hundredths is None:
        colon_hundredths = not _file_uses_fractions(text)

    data = LrcData(lines=[])
    untimed: list[str] = []

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue

        meta = _META_TAG.match(line)
        if meta and not _BRACKET_TIME.match(line):
            key, value = meta.group(1).lower(), meta.group(2).strip()
            if key == "ti":
                data.title = value
            elif key == "ar":
                data.artist = value
            elif key == "al":
                data.album = value
            elif key == "by":
                data.author = value
            elif key == "offset":
                try:
                    data.offset_ms = int(float(value))
                except ValueError:
                    pass
            # length / re / ve / tool tags carry nothing we need.
            continue

        stamps: list[float] = []
        pos = 0
        while pos < len(line):
            m = _BRACKET_TIME.match(line, pos)
            if not m:
                break
            stamps.append(_read_time(m, colon_hundredths=colon_hundredths))
            pos = m.end()

        # Drop [Chorus]-style section headers, matching what the Genius
        # path strips before tokenising.
        remainder = _SECTION_HEADER.sub("", line[pos:]).strip()
        words = _parse_words(remainder, colon_hundredths=colon_hundredths)

        if not stamps:
            # No leading timestamp. Enhanced <time> tags still time it at
            # word level; otherwise it is plain unsynced lyric text (still
            # valuable — see LrcData.text).
            if any(w.start is not None for w in words):
                first = next(w.start for w in words if w.start is not None)
                data.lines.append(
                    LrcLine(start=first, text=remainder, words=words)
                )
            elif remainder:
                untimed.append(remainder)
            continue

        # A line can carry several stamps ("[00:21][01:45] chorus") —
        # emit one entry per stamp so repeats land at every occurrence.
        for t in stamps:
            data.lines.append(
                LrcLine(start=t, text=remainder, words=list(words))
            )

    for chunk in untimed:
        data.lines.append(LrcLine(start=None, text=chunk))

    _apply_offset(data)
    return data


def _apply_offset(data: LrcData) -> None:
    """Fold `[offset:...]` into the timestamps.

    Convention (the common one): a positive offset makes the lyrics
    appear *earlier*, so it is subtracted from each stamp. Getting this
    backwards is not fatal — the aligner re-estimates a global offset
    against Whisper anchors and corrects any residual sign error.
    """
    if not data.offset_ms:
        return
    shift = data.offset_ms / 1000.0
    for ln in data.lines:
        if ln.start is not None:
            ln.start = max(0.0, ln.start - shift)
        for w in ln.words:
            if w.start is not None:
                w.start = max(0.0, w.start - shift)


def parse_lrc_bytes(data: bytes, **kwargs) -> LrcData:
    """Convenience: decode then parse."""
    return parse_lrc(decode_lrc_bytes(data), **kwargs)
