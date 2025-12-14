"""memory_reflect_status tool for polling background reflection jobs."""

from uuid import UUID

from app.services import reflection_job_service


async def memory_reflect_status(job_id: str, user_id: str = "default") -> dict:
    """Get the status of a reflection job."""
    try:
        jid = UUID(job_id)
    except ValueError:
        return {"status": "error", "reason": "invalid job_id"}

    status = await reflection_job_service.get_status(jid)
    # Optional user_id check (defensive)
    if status.get("status") != "not_found" and status.get("user_id") and status["user_id"] != user_id:
        status["warning"] = "job belongs to different user_id"
    return status
