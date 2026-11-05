from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.deps import get_youtube_service
from app.main import app
from app.services.youtube import SearchResult, YouTubeService


def _override(results):
    svc = MagicMock()
    svc.search.return_value = results
    app.dependency_overrides[get_youtube_service] = lambda: svc
    return svc


def teardown_function():
    app.dependency_overrides.clear()


def test_search_endpoint_returns_results():
    _override(
        [
            SearchResult("abc", "T1", "A1", 100.0, "https://t"),
            SearchResult("def", "T2", None, 200.0, None),
        ]
    )
    client = TestClient(app)
    r = client.get("/api/search", params={"q": "daft punk", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body[0] == {
        "youtube_video_id": "abc",
        "title": "T1",
        "artist": "A1",
        "duration_seconds": 100.0,
        "thumbnail_url": "https://t",
    }
    assert body[1]["artist"] is None
    assert body[1]["thumbnail_url"] is None


def test_search_endpoint_rejects_empty_query():
    _override([])
    client = TestClient(app)
    r = client.get("/api/search", params={"q": ""})
    assert r.status_code == 422


def test_search_endpoint_clamps_limit():
    _override([])
    client = TestClient(app)
    r = client.get("/api/search", params={"q": "x", "limit": 100})
    assert r.status_code == 422


def test_search_endpoint_forwards_limit():
    svc = _override([])
    client = TestClient(app)
    client.get("/api/search", params={"q": "x", "limit": 7})
    svc.search.assert_called_once_with("x", limit=7)


def test_search_endpoint_rejects_urls():
    """A URL used to reach yt-dlp, which fetches it as itself — an SSRF
    primitive pointed at the cloud metadata address and the compose
    network. Uses the real service so the guard itself is exercised."""
    app.dependency_overrides[get_youtube_service] = lambda: YouTubeService()
    client = TestClient(app)
    r = client.get("/api/search", params={"q": "http://169.254.169.254/"})
    assert r.status_code == 400
    assert "URL" in r.json()["detail"]

