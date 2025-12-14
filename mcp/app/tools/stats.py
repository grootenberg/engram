"""memory_stats tool for system health and debugging."""

from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.memory import Memory, MemoryType
from app.models.reflection_job import ReflectionJob, ReflectionJobStatus
from app.models.reflection_state import ReflectionState


async def memory_stats(
    user_id: str = "default",
    scope: str = "summary",
) -> dict:
    """Get memory system statistics and health metrics.

    Args:
        user_id: User identifier (default: "default").
        scope: Level of detail. Options:
            - "summary": High-level counts and averages
            - "detailed": Breakdown by type and section
            - "reflection": Reflection trigger state and history

    Returns:
        dict with statistics based on scope.

    Example:
        >>> memory_stats(scope="summary")
        {
            "total_memories": 150,
            "by_type": {"episodic": 120, "semantic": 30},
            "avg_importance": 6.2,
            "recent_24h": 15
        }
    """
    try:
        uid = UUID(user_id)
    except ValueError:
        import hashlib
        uid = UUID(hashlib.md5(user_id.encode()).hexdigest())

    async with get_session() as session:
        if scope == "summary":
            return await _get_summary_stats(session, uid)
        elif scope == "detailed":
            return await _get_detailed_stats(session, uid)
        elif scope == "reflection":
            return await _get_reflection_stats(session, uid)
        else:
            return {
                "status": "error",
                "reason": f"Unknown scope: {scope}. Use summary, detailed, or reflection.",
            }


async def _get_summary_stats(session: AsyncSession, user_id: UUID) -> dict:
    """Get high-level summary statistics."""
    # Total count
    total_result = await session.execute(
        select(func.count()).select_from(Memory).where(Memory.user_id == user_id)
    )
    total = total_result.scalar() or 0

    # Count by type
    type_result = await session.execute(
        select(Memory.memory_type, func.count())
        .where(Memory.user_id == user_id)
        .group_by(Memory.memory_type)
    )
    by_type = {row[0].value: row[1] for row in type_result}

    # Average importance
    avg_result = await session.execute(
        select(func.avg(Memory.importance_score)).where(Memory.user_id == user_id)
    )
    avg_importance = avg_result.scalar() or 0.0

    # Recent 24h count
    recent_result = await session.execute(
        text("""
            SELECT COUNT(*)
            FROM memories
            WHERE user_id = :user_id
              AND created_at > NOW() - INTERVAL '24 hours'
        """),
        {"user_id": str(user_id)},
    )
    recent_24h = recent_result.scalar() or 0

    # Synthetic count
    synthetic_result = await session.execute(
        select(func.count())
        .select_from(Memory)
        .where(Memory.user_id == user_id, Memory.is_synthetic == True)  # noqa: E712
    )
    synthetic_count = synthetic_result.scalar() or 0

    return {
        "total_memories": total,
        "by_type": by_type,
        "synthetic_count": synthetic_count,
        "avg_importance": round(avg_importance, 2),
        "recent_24h": recent_24h,
    }


async def _get_detailed_stats(session: AsyncSession, user_id: UUID) -> dict:
    """Get detailed breakdown statistics."""
    summary = await _get_summary_stats(session, user_id)

    # By observation type
    obs_result = await session.execute(
        select(Memory.observation_type, func.count())
        .where(Memory.user_id == user_id, Memory.observation_type.isnot(None))
        .group_by(Memory.observation_type)
    )
    by_observation_type = {
        row[0].value if row[0] else "unknown": row[1] for row in obs_result
    }

    # By section
    section_result = await session.execute(
        select(Memory.section, func.count())
        .where(Memory.user_id == user_id, Memory.section.isnot(None))
        .group_by(Memory.section)
    )
    by_section = {row[0].value if row[0] else "unknown": row[1] for row in section_result}

    # Feedback statistics
    feedback_result = await session.execute(
        select(
            func.sum(Memory.helpful_count),
            func.sum(Memory.harmful_count),
        ).where(Memory.user_id == user_id)
    )
    feedback_row = feedback_result.first()
    total_helpful = feedback_row[0] or 0
    total_harmful = feedback_row[1] or 0

    # Top importance memories
    top_result = await session.execute(
        select(Memory.id, Memory.content, Memory.importance_score)
        .where(Memory.user_id == user_id)
        .order_by(Memory.importance_score.desc())
        .limit(5)
    )
    top_memories = [
        {"id": str(row.id), "content": row.content[:100], "importance": row.importance_score}
        for row in top_result
    ]

    return {
        **summary,
        "by_observation_type": by_observation_type,
        "by_section": by_section,
        "feedback": {
            "total_helpful": total_helpful,
            "total_harmful": total_harmful,
        },
        "top_importance_memories": top_memories,
    }


async def _get_reflection_stats(session: AsyncSession, user_id: UUID) -> dict:
    """Get reflection trigger state, history, and job queue status."""
    # Get reflection state
    result = await session.execute(
        select(ReflectionState).where(ReflectionState.user_id == user_id)
    )
    state = result.scalar_one_or_none()

    if state is None:
        return {
            "status": "no_reflection_state",
            "message": "No reflection state found for user. No observations have been recorded yet.",
        }

    should_reflect, reason = state.should_reflect()

    # Count semantic memories (results of reflection)
    semantic_result = await session.execute(
        select(func.count())
        .select_from(Memory)
        .where(
            Memory.user_id == user_id,
            Memory.memory_type == MemoryType.SEMANTIC,
            Memory.is_synthetic == True,  # noqa: E712
        )
    )
    semantic_count = semantic_result.scalar() or 0

    # Job queue stats
    queued_result = await session.execute(
        select(func.count())
        .select_from(ReflectionJob)
        .where(
            ReflectionJob.user_id == user_id,
            ReflectionJob.status == ReflectionJobStatus.QUEUED,
        )
    )
    queued_jobs = queued_result.scalar() or 0

    running_result = await session.execute(
        select(func.count())
        .select_from(ReflectionJob)
        .where(
            ReflectionJob.user_id == user_id,
            ReflectionJob.status == ReflectionJobStatus.RUNNING,
        )
    )
    running_jobs = running_result.scalar() or 0

    last_job_result = await session.execute(
        select(ReflectionJob)
        .where(
            ReflectionJob.user_id == user_id,
            ReflectionJob.status == ReflectionJobStatus.COMPLETED,
        )
        .order_by(ReflectionJob.completed_at.desc())
        .limit(1)
    )
    last_job = last_job_result.scalar_one_or_none()

    return {
        "reflection_state": state.to_dict(),
        "should_reflect": should_reflect,
        "trigger_reason": reason,
        "total_synthetic_insights": semantic_count,
        "jobs": {
            "queued": queued_jobs,
            "running": running_jobs,
            "last_completed": {
                "job_id": str(last_job.id),
                "completed_at": last_job.completed_at.isoformat() if last_job else None,
                "insights_created": last_job.insights_created if last_job else None,
                "procedures_created": last_job.procedures_created if last_job else None,
                "memories_analyzed": last_job.memories_analyzed if last_job else None,
            }
            if last_job
            else None,
        },
    }
