"""Memory data model for engram."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class MemoryType(str, Enum):
    """Types of memory in the engram system."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class ObservationType(str, Enum):
    """Types of observations for importance heuristics."""

    ERROR = "error"
    INSTRUCTION = "instruction"
    DECISION = "decision"
    CODE_CHANGE = "code_change"
    INSIGHT = "insight"
    TEST_RESULT = "test_result"
    GENERAL = "general"
    TOOL_OUTPUT = "tool_output"


class SectionType(str, Enum):
    """ACE-style sections for organization."""

    STRATEGIES = "strategies"
    SNIPPETS = "snippets"
    PITFALLS = "pitfalls"
    CONTEXT = "context"
    PREFERENCES = "preferences"


class Memory(SQLModel, table=True):
    """Memory model representing an observation or insight in the engram system."""

    __tablename__ = "memories"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(index=True)

    # Content
    content: str = Field(max_length=10000)
    embedding: list[float] | None = Field(
        default=None, sa_column=Column(Vector(1536))
    )

    # Classification
    memory_type: MemoryType = Field(default=MemoryType.EPISODIC)
    observation_type: ObservationType | None = Field(default=None)
    section: SectionType | None = Field(default=None)

    # Scoring
    importance_score: float = Field(ge=1.0, le=10.0, default=5.0)
    helpful_count: int = Field(default=0)
    harmful_count: int = Field(default=0)

    # Provenance
    is_synthetic: bool = Field(default=False)
    citations: list[str] | None = Field(default=None, sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)

    # Extensibility
    extra_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    @property
    def effective_importance(self) -> float:
        """Calculate effective importance including feedback adjustment."""
        adjustment = (self.helpful_count - self.harmful_count) * 0.5
        return max(1.0, min(10.0, self.importance_score + adjustment))

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary for API responses."""
        return {
            "id": str(self.id),
            "content": self.content,
            "memory_type": self.memory_type.value,
            "observation_type": self.observation_type.value if self.observation_type else None,
            "section": self.section.value if self.section else None,
            "importance_score": self.importance_score,
            "effective_importance": self.effective_importance,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "is_synthetic": self.is_synthetic,
            "citations": self.citations,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "metadata": self.extra_metadata,
        }
