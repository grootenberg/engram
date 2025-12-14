"""Ingestion service for engram memory observations."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_importance_heuristic, settings
from app.models.memory import Memory, MemoryType, ObservationType, SectionType
from app.models.reflection_state import ReflectionState
from app.services.embedding import embedding_service
from app.services.reflection_jobs import reflection_job_service


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
            helpful_count = await self._reinforce_duplicate(session, duplicate.id)
            pruned = await self._compact_stale_memories(session, user_id)
            await session.flush()
            return {
                "status": "deduplicated",
                "message": "Similar memory already exists",
                "existing_id": str(duplicate.id),
                "similarity": duplicate.similarity,
                "helpful_count": helpful_count,
                "compacted": pruned,
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

        # Run periodic compaction to forget stale, low-importance memories
        pruned = await self._compact_stale_memories(session, user_id)

        await session.flush()

        return {
            "status": "created",
            "memory_id": str(memory.id),
            "importance_score": memory.importance_score,
            "memory_type": memory.memory_type.value,
            "compacted": pruned,
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

        # Auto-enqueue reflection when triggers are met (force to bypass further checks)
        should_reflect, _ = state.should_reflect(
            importance_threshold=settings.engram_importance_threshold,
            observation_threshold=settings.engram_observation_threshold,
            time_threshold_hours=settings.engram_time_threshold_hours,
        )
        if should_reflect:
            await reflection_job_service.enqueue(
                session=session,
                user_id=user_id,
                focus=None,
                max_insights=settings.engram_default_reflection_insights,
                force=True,
                skip_if_active=True,
            )

    async def _reinforce_duplicate(
        self,
        session: AsyncSession,
        memory_id: UUID,
    ) -> int | None:
        """Increment helpful count when a duplicate is observed."""
        result = await session.execute(
            text("""
                UPDATE memories
                SET helpful_count = helpful_count + 1,
                    last_accessed_at = NOW()
                WHERE id = :memory_id
                RETURNING helpful_count
            """),
            {"memory_id": str(memory_id)},
        )
        row = result.first()
        return row.helpful_count if row else None

    async def _compact_stale_memories(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> int:
        """Prune stale, low-importance episodic memories."""
        ttl_days = settings.engram_compaction_ttl_days
        importance_threshold = settings.engram_compaction_importance_threshold
        batch_limit = settings.engram_compaction_batch_limit

        if ttl_days <= 0 or batch_limit <= 0:
            return 0

        query = text("""
            WITH candidates AS (
                SELECT id
                FROM memories
                WHERE user_id = :user_id
                  AND memory_type = 'EPISODIC'
                  AND is_synthetic = FALSE
                  AND (importance_score + (helpful_count - harmful_count) * 0.5) < :importance_threshold
                  AND created_at < NOW() - (:ttl_days || ' days')::interval
                ORDER BY created_at
                LIMIT :limit
            )
            DELETE FROM memories m
            USING candidates c
            WHERE m.id = c.id
            RETURNING m.id
        """)

        result = await session.execute(
            query,
            {
                "user_id": str(user_id),
                "importance_threshold": importance_threshold,
                "ttl_days": str(ttl_days),
                "limit": batch_limit,
            },
        )
        deleted = result.fetchall()
        return len(deleted)


# Global singleton instance
ingestion_service = IngestionService()
