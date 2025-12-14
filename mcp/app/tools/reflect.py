"""memory_reflect tool for enqueuing background reflection."""

from uuid import UUID

from app.services import reflection_job_service


async def memory_reflect(
    user_id: str = "default",
    focus: str | None = None,
    max_insights: int = 5,
    force: bool = False,
) -> dict:
    """Enqueue reflection to synthesize episodic memories into semantic/procedural insights.

    Args:
        user_id: User identifier (default: "default").
        focus: Optional focus area to guide reflection (semantic filter).
        max_insights: Maximum number of insights to generate (default: 5).
        force: Force reflection even if triggers not met (default: False). Queued jobs run with force=True to avoid re-checking triggers.

    Returns:
        dict with queue status and job_id for polling.

    Example:
        >>> memory_reflect(focus="error handling patterns", force=True)
        {
            "status": "queued",
            "job_id": "...",
            "queued_at": "2025-02-14T22:17:00Z"
        }
    """
    try:
        uid = UUID(user_id)
    except ValueError:
        import hashlib
        uid = UUID(hashlib.md5(user_id.encode()).hexdigest())

    return await reflection_job_service.enqueue(
        user_id=uid,
        focus=focus,
        max_insights=max_insights,
        force=force,
        skip_if_active=True,
    )
