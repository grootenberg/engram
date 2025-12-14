"""memory_reflect tool for synthesizing episodic memories into semantic insights."""

from uuid import UUID

from app.db import get_session
from app.services import reflection_service


async def memory_reflect(
    user_id: str = "default",
    focus: str | None = None,
    max_insights: int = 5,
    force: bool = False,
) -> dict:
    """Synthesize episodic memories into semantic insights.

    Reflection is triggered automatically based on:
    - Accumulated importance >= 150
    - Observation count >= 100
    - Hours since last reflection >= 24

    Use force=True to trigger reflection regardless of thresholds.

    Args:
        user_id: User identifier (default: "default").
        focus: Optional focus area to guide reflection (semantic filter).
        max_insights: Maximum number of insights to generate (default: 5).
        force: Force reflection even if triggers not met (default: False).

    Returns:
        dict with:
            - status: "completed", "skipped", or "error"
            - reason: Why reflection was skipped (if applicable)
            - insights_created: Number of new semantic memories created
            - insights: List of generated insights with citations
            - memories_analyzed: Number of episodic memories analyzed

    Example:
        >>> memory_reflect(focus="error handling patterns", force=True)
        {
            "status": "completed",
            "insights_created": 3,
            "insights": [
                {
                    "content": "Always validate input before processing...",
                    "importance": 8,
                    "citations": ["mem-1", "mem-2"]
                }
            ],
            "memories_analyzed": 25
        }
    """
    try:
        uid = UUID(user_id)
    except ValueError:
        import hashlib
        uid = UUID(hashlib.md5(user_id.encode()).hexdigest())

    async with get_session() as session:
        result = await reflection_service.reflect(
            session=session,
            user_id=uid,
            focus=focus,
            max_insights=max_insights,
            force=force,
        )
        return result
