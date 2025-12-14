"""memory_feedback tool for providing feedback signals."""

from uuid import UUID

from sqlalchemy import select

from app.db import get_session
from app.models.memory import Memory


async def memory_feedback(
    memory_id: str,
    helpful: bool,
    reason: str | None = None,
) -> dict:
    """Provide helpful/harmful feedback on a memory.

    Feedback adjusts the effective importance of memories over time.
    Effective importance = base_importance + (helpful_count - harmful_count) * 0.5

    Args:
        memory_id: UUID of the memory to provide feedback on.
        helpful: True if the memory was helpful, False if harmful.
        reason: Optional reason for the feedback.

    Returns:
        dict with status, updated counts, and effective importance.

    Example:
        >>> memory_feedback(
        ...     memory_id="abc-123-def",
        ...     helpful=True,
        ...     reason="This insight helped me avoid a bug"
        ... )
        {"status": "updated", "helpful_count": 3, "harmful_count": 0, "effective_importance": 8.5}
    """
    try:
        mid = UUID(memory_id)
    except ValueError:
        return {
            "status": "error",
            "reason": f"Invalid memory_id format: {memory_id}",
        }

    async with get_session() as session:
        result = await session.execute(
            select(Memory).where(Memory.id == mid)
        )
        memory = result.scalar_one_or_none()

        if memory is None:
            return {
                "status": "error",
                "reason": f"Memory not found: {memory_id}",
            }

        # Update feedback counts
        if helpful:
            memory.helpful_count += 1
        else:
            memory.harmful_count += 1

        # Store reason in metadata if provided
        if reason:
            if memory.extra_metadata is None:
                memory.extra_metadata = {}
            feedback_log = memory.extra_metadata.get("feedback_log", [])
            feedback_log.append({
                "helpful": helpful,
                "reason": reason,
            })
            memory.extra_metadata["feedback_log"] = feedback_log

        await session.flush()

        return {
            "status": "updated",
            "memory_id": str(memory.id),
            "helpful_count": memory.helpful_count,
            "harmful_count": memory.harmful_count,
            "effective_importance": memory.effective_importance,
        }
