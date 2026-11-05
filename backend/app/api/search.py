from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_youtube_service
from app.schemas import SearchResultSchema
from app.services.youtube import YouTubeService
from app.services.youtube.service import SearchQueryRejected

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=list[SearchResultSchema])
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=20),
    yt: YouTubeService = Depends(get_youtube_service),
) -> list[SearchResultSchema]:
    try:
        results = yt.search(q, limit=limit)
    except SearchQueryRejected as exc:
        # A URL in the search box would be fetched as-is by yt-dlp. Reject
        # it explicitly rather than quietly searching for the URL string.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return [SearchResultSchema(**r.__dict__) for r in results]
