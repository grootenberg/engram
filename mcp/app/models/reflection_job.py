"""Reflection job model for background reflection processing."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class ReflectionJobStatus(str, Enum):
    """States for reflection jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReflectionJob(SQLModel, table=True):
    """Background reflection job record."""

    __tablename__ = "reflection_jobs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(index=True)

    status: ReflectionJobStatus = Field(default=ReflectionJobStatus.QUEUED, index=True)
    focus: str | None = Field(default=None)
    max_insights: int = Field(default=5, ge=1)
    force: bool = Field(default=False)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    insights_created: int | None = Field(default=None)
    procedures_created: int | None = Field(default=None)
    memories_analyzed: int | None = Field(default=None)

    error_message: str | None = Field(default=None)
    result_payload: dict | None = Field(default=None, sa_column=Column(JSON))
