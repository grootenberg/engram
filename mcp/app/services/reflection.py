"""Reflection service for synthesizing episodic memories into semantic insights."""

from datetime import datetime
from uuid import UUID

from anthropic import AsyncAnthropic
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.memory import Memory, MemoryType, ObservationType
from app.models.reflection_state import ReflectionState
from app.services.embedding import embedding_service


REFLECTION_SYSTEM_PROMPT = """You are a memory synthesis system. Your task is to analyze episodic observations and extract durable insights that will be useful for future interactions.

Guidelines:
1. Look for patterns, lessons learned, and reusable knowledge
2. Focus on insights that transcend specific contexts
3. Each insight should be self-contained and actionable
4. Cite the source observations by their IDs
5. Assign an importance score (1-10) based on how broadly applicable the insight is

Output format (JSON array):
[
  {
    "insight": "Clear, actionable insight statement",
    "importance": 7,
    "citations": ["memory_id_1", "memory_id_2"]
  }
]

Only output valid JSON. No markdown, no explanations."""


class ReflectionService:
    """Service for reflecting on episodic memories to generate semantic insights."""

    def __init__(self):
        self._client: AsyncAnthropic | None = None

    @property
    def client(self) -> AsyncAnthropic:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def should_reflect(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> tuple[bool, str]:
        """Check if reflection should be triggered for a user.

        Args:
            session: Database session.
            user_id: User identifier.

        Returns:
            Tuple of (should_reflect, reason).
        """
        result = await session.execute(
            select(ReflectionState).where(ReflectionState.user_id == user_id)
        )
        state = result.scalar_one_or_none()

        if state is None:
            return False, "no reflection state found"

        return state.should_reflect(
            importance_threshold=settings.engram_importance_threshold,
            observation_threshold=settings.engram_observation_threshold,
            time_threshold_hours=settings.engram_time_threshold_hours,
        )

    async def reflect(
        self,
        session: AsyncSession,
        user_id: UUID,
        focus: str | None = None,
        max_insights: int = 5,
        force: bool = False,
    ) -> dict:
        """Perform reflection to synthesize episodic memories into semantic insights.

        Args:
            session: Database session.
            user_id: User identifier.
            focus: Optional focus area to guide reflection.
            max_insights: Maximum number of insights to generate.
            force: Force reflection even if triggers not met.

        Returns:
            Dict with reflection results.
        """
        # Check if reflection should be triggered
        if not force:
            should, reason = await self.should_reflect(session, user_id)
            if not should:
                return {
                    "status": "skipped",
                    "reason": reason,
                }

        # Get recent episodic memories for reflection
        memories = await self._get_reflection_candidates(session, user_id, focus)

        if len(memories) < 3:
            return {
                "status": "skipped",
                "reason": f"insufficient episodic memories ({len(memories)} found, need at least 3)",
            }

        # Format memories for LLM
        memories_text = self._format_memories_for_reflection(memories)

        # Generate insights using LLM
        prompt = f"""Analyze these episodic observations and extract {max_insights} key insights:

{memories_text}

{f"Focus area: {focus}" if focus else ""}

Extract patterns, lessons learned, and reusable knowledge. Output as JSON array."""

        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        try:
            import json
            insights_data = json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return {
                "status": "error",
                "reason": "failed to parse LLM response",
            }

        # Create semantic memories from insights
        created_insights = []
        for insight_data in insights_data[:max_insights]:
            insight_text = insight_data.get("insight", "")
            importance = insight_data.get("importance", 7)
            citations = insight_data.get("citations", [])

            if not insight_text:
                continue

            # Generate embedding for insight
            embedding = await embedding_service.embed(insight_text)

            # Create semantic memory
            memory = Memory(
                user_id=user_id,
                content=insight_text,
                embedding=embedding,
                memory_type=MemoryType.SEMANTIC,
                observation_type=ObservationType.INSIGHT,
                importance_score=max(1.0, min(10.0, importance)),
                is_synthetic=True,
                citations=citations,
            )
            session.add(memory)
            created_insights.append({
                "content": insight_text,
                "importance": memory.importance_score,
                "citations": citations,
            })

        # Reset reflection state
        await self._reset_reflection_state(session, user_id)

        await session.flush()

        return {
            "status": "completed",
            "insights_created": len(created_insights),
            "insights": created_insights,
            "memories_analyzed": len(memories),
        }

    async def _get_reflection_candidates(
        self,
        session: AsyncSession,
        user_id: UUID,
        focus: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        """Get episodic memories for reflection.

        Args:
            session: Database session.
            user_id: User identifier.
            focus: Optional focus area to filter memories.
            limit: Maximum memories to retrieve.

        Returns:
            List of Memory objects.
        """
        if focus:
            # Use semantic search if focus is provided
            focus_embedding = await embedding_service.embed(focus)
            query = text(r"""
                SELECT id, content, observation_type, importance_score, created_at
                FROM memories
                WHERE user_id = :user_id
                  AND memory_type = 'EPISODIC'
                  AND embedding IS NOT NULL
                ORDER BY 1 - (embedding <=> :embedding\:\:vector) DESC
                LIMIT :limit
            """)
            result = await session.execute(
                query,
                {
                    "user_id": str(user_id),
                    "embedding": str(focus_embedding),
                    "limit": limit,
                },
            )
        else:
            # Get recent high-importance memories
            query = text("""
                SELECT id, content, observation_type, importance_score, created_at
                FROM memories
                WHERE user_id = :user_id
                  AND memory_type = 'EPISODIC'
                ORDER BY importance_score DESC, created_at DESC
                LIMIT :limit
            """)
            result = await session.execute(
                query,
                {"user_id": str(user_id), "limit": limit},
            )

        return result.fetchall()

    def _format_memories_for_reflection(self, memories) -> str:
        """Format memories for the reflection prompt.

        Args:
            memories: List of memory rows.

        Returns:
            Formatted string for LLM prompt.
        """
        lines = []
        for mem in memories:
            obs_type = mem.observation_type or "general"
            lines.append(
                f"[{mem.id}] ({obs_type}, importance={mem.importance_score}): {mem.content}"
            )
        return "\n".join(lines)

    async def _reset_reflection_state(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> None:
        """Reset the reflection state after successful reflection.

        Args:
            session: Database session.
            user_id: User identifier.
        """
        result = await session.execute(
            select(ReflectionState).where(ReflectionState.user_id == user_id)
        )
        state = result.scalar_one_or_none()

        if state:
            state.reset()


# Global singleton instance
reflection_service = ReflectionService()
