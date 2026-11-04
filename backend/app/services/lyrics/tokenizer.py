"""Script-aware tokenisation + normalisation for lyric matching.

The aligner compares an *authoritative* lyric text against Whisper's
transcript word-for-word. That comparison is only meaningful if both
sides are tokenised the same way and normalised without destroying the
script the lyrics are written in.

Two problems this module solves:

1. **Normalisation used to be Latin-only.** ``re.sub(r"[^a-z0-9]", ...)``
   turned every CJK / Hangul / Arabic word into ``""`` (so *all* words
   looked identical and ``difflib`` aligned garbage) and mangled accented
   Latin (``café`` -> ``caf``). ``clean_word`` now folds accents off Latin
   text (NFKD + combining-mark strip) while preserving non-Latin letters.

2. **Whitespace tokenisation is wrong for unspaced scripts.** Japanese,
   Chinese, Korean, Thai, Lao, Khmer and Burmese do not separate words
   with spaces, so a whole line arrives as one giant "word" and per-word
   timing is impossible. ``tokenize`` splits those scripts at *character*
   granularity instead, which is the standard trick for aligning CJK —
   each character then gets its own timestamp slice.

Nothing here does real word segmentation (MeCab / pkuseg / pythainlp).
Character granularity is enough for what the alignment is used for:
word/character *boundaries* for vocal-safe cut regions.
"""
from __future__ import annotations

import re
import unicodedata

# Scripts written without word separators. For these, tokenise per
# character rather than per whitespace-delimited chunk.
_UNSPACED_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),   # Thai
    (0x0E80, 0x0EFF),   # Lao
    (0x1000, 0x109F),   # Myanmar
    (0x1100, 0x11FF),   # Hangul jamo
    (0x1780, 0x17FF),   # Khmer
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3130, 0x318F),   # Hangul compatibility jamo
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xA960, 0xA97F),   # Hangul jamo ext A
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0xFF66, 0xFF9F),   # Halfwidth katakana
)

_LATIN_RE = re.compile(r"^[a-z]+$")


def is_unspaced_script(ch: str) -> bool:
    """True when `ch` belongs to a script written without word spaces."""
    if not ch:
        return False
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _UNSPACED_RANGES)


def clean_word(word: str) -> str:
    """Normalise a token for fuzzy comparison.

    Folds accented Latin to ASCII (``café`` -> ``cafe``), casefolds, and
    keeps every alphanumeric character in *any* script. Punctuation and
    separators are dropped so ``"hey!"`` == ``"hey"`` == ``"«hey»"``.

    The recompose step (NFKD -> drop marks -> NFC) matters: NFKD alone
    decomposes Hangul syllables into jamo (``사`` -> ``ᄉ`` + ``ᅡ``), which
    would silently break Korean matching the way the old ASCII filter
    broke everything. Stripping the combining marks first keeps the
    Latin accent fold while NFC restores Hangul composition.

    Deliberately not "keep word characters": that keeps underscores (so
    ``snake_case`` would collide with ``snakecase``) and keeps the
    combining marks NFKD just decomposed.
    """
    decomposed = unicodedata.normalize("NFKD", word)
    without_marks = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    recomposed = unicodedata.normalize("NFC", without_marks)
    return "".join(ch for ch in recomposed.casefold() if ch.isalnum())


def is_latin_word(word: str) -> bool:
    """True when `word` is a plain ASCII-letters word.

    Gate for the Soundex phonetic fallback, which encodes *English*
    phonotactics and is meaningless for anything else. Callers should
    pass the ``clean_word`` form.
    """
    return bool(word) and bool(_LATIN_RE.match(word))


def tokenize(text: str) -> list[str]:
    """Split lyric text into alignment tokens.

    Whitespace-delimited words stay whole; characters in unspaced scripts
    become individual tokens. Bracketed section headers (``[Chorus]``) and
    ``(x3)`` repeat markers are *not* touched here — those are source
    specific and handled by the caller.
    """
    out: list[str] = []
    for chunk in text.split():
        buf: list[str] = []
        for ch in chunk:
            if is_unspaced_script(ch):
                if buf:
                    out.append("".join(buf))
                    buf = []
                out.append(ch)
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
    return out


def script_hint(text: str) -> str | None:
    """Coarse script-based language guess, used as a Whisper hint.

    Returns an ISO-639-1 code Whisper understands, or None when the text
    gives no useful signal (pure Latin / mixed / empty). Latin is never
    reported — the script alone can't separate English from French from
    Vietnamese, and a wrong guess there is worse than none.

    Priority rather than majority, because Japanese is written as a kanji
    + kana mix: a naive "dominant script" count reads most Japanese lines
    as Chinese. Kana is used by Japanese *only*, Hangul by Korean only,
    and bare ideographs are the Chinese default.
    """
    kana = hangul = ideograph = 0
    indic: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF or 0xFF66 <= cp <= 0xFF9F:
            kana += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            hangul += 1
        elif (
            0x3400 <= cp <= 0x4DBF
            or 0x4E00 <= cp <= 0x9FFF
            or 0xF900 <= cp <= 0xFAFF
        ):
            ideograph += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            indic["th"] = indic.get("th", 0) + 1
        elif 0x0E80 <= cp <= 0x0EFF:
            indic["lo"] = indic.get("lo", 0) + 1
        elif 0x1000 <= cp <= 0x109F:
            indic["my"] = indic.get("my", 0) + 1
        elif 0x1780 <= cp <= 0x17FF:
            indic["km"] = indic.get("km", 0) + 1

    if kana:
        return "ja"
    if hangul:
        return "ko"
    if indic:
        return max(indic, key=indic.get)  # type: ignore[arg-type]
    if ideograph:
        return "zh"
    return None
