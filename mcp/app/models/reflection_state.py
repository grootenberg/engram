"""Reflection state model for tracking trigger conditions."""

from datetime import datetime
from uuid import UUID

from sqlmodel import Field, SQLModel


class ReflectionState(SQLModel, table=True):
    """Tracks reflection trigger state per user."""

    __tablename__ = "reflection_state"

    user_id: UUID = Field(primary_key=True)
    accumulated_importance: float = Field(default=0.0)
    observations_since: int = Field(default=0)
    last_reflection_at: datetime = Field(default_factory=datetime.utcnow)

    def should_reflect(
        self,
        importance_threshold: float = 150.0,
        observation_threshold: int = 100,
        time_threshold_hours: int = 24,
    ) -> tuple[bool, str]:
        """
        Check if reflection should be triggered.

        Returns:
            Tuple of (should_reflect, reason)
        """
        hours_since = (datetime.utcnow() - self.last_reflection_at).total_seconds() / 3600

        if self.accumulated_importance >= importance_threshold:
            return True, f"accumulated_importance ({self.accumulated_importance:.1f}) >= threshold ({importance_threshold})"

        if self.observations_since >= observation_threshold:
            return True, f"observations_since ({self.observations_since}) >= threshold ({observation_threshold})"

        if hours_since >= time_threshold_hours:
            return True, f"hours_since_last ({hours_since:.1f}) >= threshold ({time_threshold_hours})"

        return False, "no trigger conditions met"

    def reset(self) -> None:
        """Reset trigger state after reflection."""
        self.accumulated_importance = 0.0
        self.observations_since = 0
        self.last_reflection_at = datetime.utcnow()

    def accumulate(self, importance: float) -> None:
        """Accumulate importance from a new observation."""
        self.accumulated_importance += importance
        self.observations_since += 1

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        hours_since = (datetime.utcnow() - self.last_reflection_at).total_seconds() / 3600
        return {
            "accumulated_importance": self.accumulated_importance,
            "observations_since": self.observations_since,
            "hours_since_last": round(hours_since, 2),
            "last_reflection_at": self.last_reflection_at.isoformat(),
        }
