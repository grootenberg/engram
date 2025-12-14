"""Background job management for reflection."""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.config import settings
from app.models import ReflectionJob, ReflectionJobStatus
from app.services.reflection import reflection_service


class ReflectionJobService:
    """Manage enqueueing and processing reflection jobs."""

    def __init__(self, poll_interval_seconds: float = settings.engram_reflection_poll_interval_seconds):
        self._worker_started = False
        self._poll_interval_seconds = poll_interval_seconds
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        user_id: UUID,
        focus: str | None = None,
        max_insights: int = 5,
        force: bool = False,
        session: AsyncSession | None = None,
        skip_if_active: bool = True,
    ) -> dict:
        """Create a reflection job. Returns job metadata."""
        if session:
            return await self._enqueue_with_session(
                session=session,
                user_id=user_id,
                focus=focus,
                max_insights=max_insights,
                force=force,
                skip_if_active=skip_if_active,
            )

        async with get_session() as session_ctx:
            return await self._enqueue_with_session(
                session=session_ctx,
                user_id=user_id,
                focus=focus,
                max_insights=max_insights,
                force=force,
                skip_if_active=skip_if_active,
            )

    async def _enqueue_with_session(
        self,
        session: AsyncSession,
        user_id: UUID,
        focus: str | None,
        max_insights: int,
        force: bool,
        skip_if_active: bool,
    ) -> dict:
        # Ensure only one active job per user if requested
        if skip_if_active:
            existing = await session.execute(
                select(ReflectionJob)
                .where(
                    ReflectionJob.user_id == user_id,
                    ReflectionJob.status.in_(
                        [ReflectionJobStatus.QUEUED, ReflectionJobStatus.RUNNING]
                    ),
                )
                .order_by(ReflectionJob.created_at)
            )
            row = existing.first()
            if row:
                job = row[0]
                return {
                    "status": "already_running",
                    "job_id": str(job.id),
                    "queued_at": job.created_at.isoformat(),
                }

        job = ReflectionJob(
            user_id=user_id,
            focus=focus,
            max_insights=max(1, max_insights),
            force=force,
            status=ReflectionJobStatus.QUEUED,
        )
        session.add(job)
        await session.flush()

        # Start background worker lazily
        await self.ensure_worker()

        return {
            "status": "queued",
            "job_id": str(job.id),
            "queued_at": job.created_at.isoformat(),
        }

    async def enqueue_if_triggered(
        self,
        session: AsyncSession,
        user_id: UUID,
        focus: str | None = None,
        max_insights: int = 5,
    ) -> Optional[dict]:
        """If triggers are met and no active job, enqueue one."""
        should, reason = await reflection_service.should_reflect(session, user_id)
        if not should:
            return None
        return await self._enqueue_with_session(
            session=session,
            user_id=user_id,
            focus=focus,
            max_insights=max_insights,
            force=True,
            skip_if_active=True,
        )

    async def get_status(self, job_id: UUID) -> dict:
        """Get job status by ID."""
        async with get_session() as session:
            job = await session.get(ReflectionJob, job_id)
            if not job:
                return {"status": "not_found"}
            return self._serialize_job(job)

    def _serialize_job(self, job: ReflectionJob) -> dict:
        return {
            "status": job.status.value,
            "job_id": str(job.id),
            "user_id": str(job.user_id),
            "focus": job.focus,
            "max_insights": job.max_insights,
            "force": job.force,
            "queued_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "insights_created": job.insights_created,
            "procedures_created": job.procedures_created,
            "memories_analyzed": job.memories_analyzed,
            "error_message": job.error_message,
            "result": job.result_payload,
        }

    async def ensure_worker(self) -> None:
        """Start background worker once."""
        if self._worker_started:
            return
        async with self._lock:
            if self._worker_started:
                return
            loop = asyncio.get_running_loop()
            loop.create_task(self._worker_loop(), name="engram-reflection-worker")
            self._worker_started = True

    async def _worker_loop(self) -> None:
        """Continuously process queued reflection jobs."""
        while True:
            try:
                processed = await self._process_one_job()
                if not processed:
                    await asyncio.sleep(self._poll_interval_seconds)
            except Exception:
                # Backoff on unexpected errors
                await asyncio.sleep(self._poll_interval_seconds)

    async def _process_one_job(self) -> bool:
        """Fetch and process a single job. Returns True if a job was processed."""
        async with get_session() as session:
            job_row = await session.execute(
                select(ReflectionJob)
                .where(ReflectionJob.status == ReflectionJobStatus.QUEUED)
                .order_by(ReflectionJob.created_at)
                .limit(1)
            )
            job = job_row.scalar_one_or_none()
            if not job:
                return False

            # Mark running
            job.status = ReflectionJobStatus.RUNNING
            job.started_at = datetime.utcnow()
            await session.flush()

            # Run reflection
            try:
                result = await reflection_service.reflect(
                    session=session,
                    user_id=job.user_id,
                    focus=job.focus,
                    max_insights=job.max_insights,
                    force=True,  # queued jobs bypass trigger checks
                )
                job.result_payload = result
                job.insights_created = result.get("insights_created")
                job.procedures_created = result.get("procedures_created")
                job.memories_analyzed = result.get("memories_analyzed")
                job.status = ReflectionJobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.error_message = None
            except Exception as exc:  # noqa: BLE001
                job.status = ReflectionJobStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.error_message = str(exc)

            await session.flush()

        return True


# Global singleton
reflection_job_service = ReflectionJobService()
