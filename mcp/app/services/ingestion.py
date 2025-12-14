"""Ingestion service for engram memory observations."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_importance_heuristic, settings
from app.models.memory import Memory, MemoryType, ObservationType, SectionType
from app.models.reflection_state import ReflectionState
from app.services.embedding import embedding_service


class IngestionService:
    """Service for ingesting observations into the memory system."""

    async def ingest(
        self,
        session: AsyncSession,
        user_id: UUID,
        content: str,
        observation_type: str = "general",
        importance: float | None = None,
        section: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Ingest a new observation into the memory stream.

        Uses delta updates: checks for duplicates via cosine similarity
        before inserting. Append-only semantics.

        Args:
            session: Database session.
            user_id: User identifier.
            content: Observation content.
            observation_type: Type of observation for importance heuristics.
            importance: Override importance score (1-10).
            section: ACE-style section classification.
            metadata: Optional extra metadata.

        Returns:
            Dict with ingestion result (created, deduplicated, or error).
        """
        # Generate embedding
        embedding = await embedding_service.embed(content)

        # Check for duplicates using cosine similarity
        duplicate = await self._find_duplicate(session, user_id, embedding)

        if duplicate:
            return {
                "status": "deduplicated",
                "message": "Similar memory already exists",
                "existing_id": str(duplicate.id),
                "similarity": duplicate.similarity,
            }

        # Calculate importance
        if importance is None:
            importance = get_importance_heuristic(observation_type)
        importance = max(1.0, min(10.0, importance))

        # Parse enums
        obs_type = None
        if observation_type:
            try:
                obs_type = ObservationType(observation_type)
            except ValueError:
                pass

        section_type = None
        if section:
            try:
                section_type = SectionType(section)
            except ValueError:
                pass

        # Create memory
        memory = Memory(
            user_id=user_id,
            content=content,
            embedding=embedding,
            memory_type=MemoryType.EPISODIC,
            observation_type=obs_type,
            section=section_type,
            importance_score=importance,
            extra_metadata=metadata,
        )

        session.add(memory)

        # Update reflection state
        await self._update_reflection_state(session, user_id, importance)

        await session.flush()

        return {
            "status": "created",
            "memory_id": str(memory.id),
            "importance_score": memory.importance_score,
            "memory_type": memory.memory_type.value,
        }

    async def _find_duplicate(
        self,
        session: AsyncSession,
        user_id: UUID,
        embedding: list[float],
    ) -> Memory | None:
        """Find a duplicate memory using cosine similarity.

        Args:
            session: Database session.
            user_id: User identifier.
            embedding: Embedding vector to compare.

        Returns:
            Existing memory if a duplicate is found, None otherwise.
        """
        threshold = settings.engram_similarity_threshold

        # Use pgvector's cosine distance (1 - similarity)
        # Only check recent memories (last 7 days) for performance
        # Note: Using \:vector to escape the cast from being interpreted as a bind param
        query = text(r"""
            SELECT id, content, 1 - (embedding <=> :embedding\:\:vector) as similarity
            FROM memories
            WHERE user_id = :user_id
              AND embedding IS NOT NULL
              AND created_at > NOW() - INTERVAL '7 days'
              AND 1 - (embedding <=> :embedding\:\:vector) >= :threshold
            ORDER BY similarity DESC
            LIMIT 1
        """)

        result = await session.execute(
            query,
            {
                "user_id": str(user_id),
                "embedding": str(embedding),
                "threshold": threshold,
            },
        )
        row = result.first()

        if row:
            # Return a mock object with the needed attributes
            class DuplicateResult:
                id = row.id
                similarity = row.similarity

            return DuplicateResult()

        return None

    async def _update_reflection_state(
        self,
        session: AsyncSession,
        user_id: UUID,
        importance: float,
    ) -> None:
        """Update the reflection trigger state for a user.

        Args:
            session: Database session.
            user_id: User identifier.
            importance: Importance score to accumulate.
        """
        # Get or create reflection state
        result = await session.execute(
            select(ReflectionState).where(ReflectionState.user_id == user_id)
        )
        state = result.scalar_one_or_none()

        if state is None:
            state = ReflectionState(user_id=user_id)
            session.add(state)

        state.accumulate(importance)


# Global singleton instance
ingestion_service = IngestionService()
