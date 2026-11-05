from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp

from app.core.config import settings

logger = logging.getLogger(__name__)

# Anything with a URL scheme, or a bare `www.` host. Deliberately does not
# try to match bare domains: "Mr. Brightside" is a song, not a hostname.
_URL_LIKE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://|^www\.", re.IGNORECASE)


def _cookies_opt() -> dict[str, Any]:
    """Yield a `cookiefile` opt iff settings.yt_dlp_cookies_file points at
    a real, non-empty file. Returning {} on miss lets the caller spread it
    into any opts dict without conditionals. We treat empty files as
    "not configured" because the docker bind-mount on the droplet
    pre-creates an empty placeholder so the mount works even before the
    user has dropped their cookies in."""
    path = settings.yt_dlp_cookies_file
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        logger.warning("yt_dlp_cookies_file=%s not a file; ignoring", path)
        return {}
    if p.stat().st_size == 0:
        logger.warning("yt_dlp_cookies_file=%s is empty; ignoring", path)
        return {}
    return {"cookiefile": path}


@dataclass(frozen=True)
class SearchResult:
    youtube_video_id: str
    title: str
    artist: str | None
    duration_seconds: float
    thumbnail_url: str | None


class YouTubeDownloadError(RuntimeError):
    pass


class SearchQueryRejected(ValueError):
    """The query isn't a search term (it looks like a URL).

    Raised instead of handing it to yt-dlp: `extract_info` fetches a URL as
    itself, so on an unauthenticated host the search box doubles as an SSRF
    primitive pointed at the cloud metadata address and the compose
    network. Surfaced as HTTP 400.
    """


class YouTubeService:
    """Thin wrapper around yt-dlp for search + best-audio download.

    No DB awareness; callers are responsible for persisting the result.
    """

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        term = (query or "").strip()
        if not term:
            raise SearchQueryRejected("empty search query")
        if _URL_LIKE.search(term):
            raise SearchQueryRejected(
                "search takes keywords, not URLs"
            )

        # Always address the search extractor explicitly. `default_search`
        # only kicks in when yt-dlp does *not* recognise the input as a URL,
        # so relying on it leaves the fetch-this-URL branch reachable.
        search_term = f"ytsearch{limit}:{term}"
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            **_cookies_opt(),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_term, download=False)

        entries = (info or {}).get("entries") or []
        results: list[SearchResult] = []
        for entry in entries[:limit]:
            if not entry or not entry.get("id"):
                continue
            results.append(_entry_to_result(entry))
        return results

    def download(self, video_id: str, dest_path: Path) -> None:
        """Download `video_id` as WAV to `dest_path`. Overwrites existing file."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # yt-dlp's --extract-audio postprocessor writes to outtmpl with the new
        # extension. We give it a template without an extension to control the
        # final filename precisely.
        out_template = str(dest_path.with_suffix(""))
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            # Lenient format chain: prefer pure-audio m4a, then any bestaudio
            # with a real audio codec, then any best with audio. The plain
            # "bestaudio/best" we used before fails on videos where YouTube
            # only serves DASH/HLS adaptive formats — common when cookies
            # carry Premium- or region-specific entitlements.
            "format": (
                "bestaudio[ext=m4a]/bestaudio[acodec!=none]/"
                "best[acodec!=none]/bestaudio/best"
            ),
            "outtmpl": out_template + ".%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "overwrites": True,
            **_cookies_opt(),
        }
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:
            raise YouTubeDownloadError(str(exc)) from exc

        if not dest_path.exists():
            raise YouTubeDownloadError(
                f"yt-dlp completed but {dest_path} was not produced"
            )


def _entry_to_result(entry: dict[str, Any]) -> SearchResult:
    duration = entry.get("duration")
    return SearchResult(
        youtube_video_id=entry["id"],
        title=entry.get("title") or entry["id"],
        artist=entry.get("artist") or entry.get("uploader"),
        duration_seconds=float(duration) if duration is not None else 0.0,
        thumbnail_url=_pick_thumbnail(entry),
    )


def _pick_thumbnail(entry: dict[str, Any]) -> str | None:
    thumbs = entry.get("thumbnails") or []
    if thumbs:
        # last thumbnail is typically highest resolution
        return thumbs[-1].get("url")
    return entry.get("thumbnail")
