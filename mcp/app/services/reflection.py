"""Reflection service for synthesizing episodic memories into semantic and procedural knowledge."""

import json
from datetime import datetime
from uuid import UUID

from anthropic import AsyncAnthropic
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.memory import Memory, MemoryType, ObservationType
from app.models.reflection_state import ReflectionState
from app.services.embedding import embedding_service


REFLECTION_SYSTEM_PROMPT = """You are a memory synthesis system. Given a reflection question and supporting episodic observations, extract durable, self-contained insights and any procedural workflows worth reusing.

Guidelines:
1. Each insight should generalize beyond the exact context and be actionable
2. Cite the source observations by their IDs for every item
3. Assign an importance score (1-10) based on how broadly applicable the item is
4. When a repeatable workflow emerges, emit a procedural memory with a short title and 3-7 concise steps

Output format (JSON object):
{
  "insights": [
    {
      "insight": "Clear, actionable insight statement",
      "importance": 7,
      "citations": ["memory_id_1", "memory_id_2"]
    }
  ],
  "procedures": [
    {
      "title": "Short workflow name",
      "steps": ["Step 1", "Step 2"],
      "importance": 7,
      "citations": ["memory_id_3"]
    }
  ]
}

Return an empty array for sections with no content. Only output valid JSON. No markdown, no explanations."""

QUESTION_SYSTEM_PROMPT = """You are planning a reflection over episodic observations. Propose high-level questions that would surface the most important themes, decisions, and lessons.

Return a JSON array of concise questions. Avoid yes/no questions. Only output JSON."""


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
        """Perform reflection to synthesize episodic memories into semantic and procedural knowledge.

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

        # Stage 1: generate reflection questions
        questions = await self._generate_questions(memories, focus=focus)
        if not questions:
            fallback = focus or "What are the most important learnings from recent work?"
            questions = [fallback]

        # Stage 2: per-question insight extraction - collect all outputs first
        all_outputs: list[tuple[str, dict]] = []  # (question, outputs)

        for idx, question in enumerate(questions):
            if max_insights > 0:
                total_collected = sum(
                    len(o.get("insights", [])) + len(o.get("procedures", []))
                    for _, o in all_outputs
                )
                if total_collected >= max_insights:
                    break

                remaining_slots = max(max_insights - total_collected, 1)
                remaining_questions = len(questions) - idx
                per_question_limit = remaining_slots
                if remaining_questions > 1:
                    per_question_limit = max(
                        1,
                        (per_question_limit + remaining_questions - 1) // remaining_questions,
                    )
            else:
                per_question_limit = 5

            relevant_memories = await self._get_memories_for_question(
                session=session,
                user_id=user_id,
                question=question,
                limit=12,
            )

            if not relevant_memories:
                continue

            outputs = await self._extract_insights_for_question(
                question=question,
                memories=relevant_memories,
                max_insights=per_question_limit or max_insights,
            )

            all_outputs.append((question, outputs))

        # Collect all texts to embed in a single batch
        texts_to_embed: list[str] = []
        embed_metadata: list[tuple[str, int, dict]] = []  # (type, index_in_type, data)

        pending_insights: list[dict] = []
        pending_procedures: list[dict] = []

        for question, outputs in all_outputs:
            for insight_data in outputs.get("insights", []):
                if max_insights > 0 and (len(pending_insights) + len(pending_procedures)) >= max_insights:
                    break

                insight_text = (insight_data.get("insight") or "").strip()
                if not insight_text:
                    continue

                importance = max(1.0, min(10.0, insight_data.get("importance", 7)))
                citations = insight_data.get("citations", [])

                texts_to_embed.append(insight_text)
                embed_metadata.append(("insight", len(pending_insights), insight_data))
                pending_insights.append({
                    "content": insight_text,
                    "importance": importance,
                    "citations": citations,
                    "question": question,
                })

            for proc_data in outputs.get("procedures", []):
                if max_insights > 0 and (len(pending_insights) + len(pending_procedures)) >= max_insights:
                    break

                title = (proc_data.get("title") or "").strip()
                steps = [s.strip() for s in proc_data.get("steps", []) if isinstance(s, str) and s.strip()]
                if not title or not steps:
                    continue

                importance = max(1.0, min(10.0, proc_data.get("importance", 7)))
                citations = proc_data.get("citations", [])
                content = self._format_procedural_content(title, steps)

                texts_to_embed.append(content)
                embed_metadata.append(("procedure", len(pending_procedures), proc_data))
                pending_procedures.append({
                    "title": title,
                    "steps": steps,
                    "importance": importance,
                    "citations": citations,
                    "question": question,
                    "_content": content,
                })

        # Batch embed all texts at once
        embeddings = await embedding_service.embed_batch(texts_to_embed) if texts_to_embed else []

        # Create Memory objects with pre-computed embeddings
        created_insights: list[dict] = []
        created_procedures: list[dict] = []

        for idx, (embed_type, type_idx, _) in enumerate(embed_metadata):
            embedding = embeddings[idx] if idx < len(embeddings) else None
            if embedding is None:
                continue

            if embed_type == "insight":
                data = pending_insights[type_idx]
                memory = Memory(
                    user_id=user_id,
                    content=data["content"],
                    embedding=embedding,
                    memory_type=MemoryType.SEMANTIC,
                    observation_type=ObservationType.INSIGHT,
                    importance_score=data["importance"],
                    is_synthetic=True,
                    citations=data["citations"],
                )
                session.add(memory)
                created_insights.append({
                    "content": data["content"],
                    "importance": data["importance"],
                    "citations": data["citations"],
                    "question": data["question"],
                })
            else:  # procedure
                data = pending_procedures[type_idx]
                memory = Memory(
                    user_id=user_id,
                    content=data["_content"],
                    embedding=embedding,
                    memory_type=MemoryType.PROCEDURAL,
                    observation_type=ObservationType.INSIGHT,
                    importance_score=data["importance"],
                    is_synthetic=True,
                    citations=data["citations"],
                )
                session.add(memory)
                created_procedures.append({
                    "title": data["title"],
                    "steps": data["steps"],
                    "importance": data["importance"],
                    "citations": data["citations"],
                    "question": data["question"],
                })

        # Reset reflection state
        await self._reset_reflection_state(session, user_id)

        await session.flush()

        return {
            "status": "completed",
            "questions": questions,
            "insights_created": len(created_insights),
            "procedures_created": len(created_procedures),
            "insights": created_insights,
            "procedures": created_procedures,
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
                SELECT id, content, observation_type, importance_score, created_at, last_accessed_at
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
                SELECT id, content, observation_type, importance_score, created_at, last_accessed_at
                FROM memories
                WHERE user_id = :user_id
                  AND memory_type = 'EPISODIC'
                ORDER BY importance_score DESC, last_accessed_at DESC, created_at DESC
                LIMIT :limit
            """)
            result = await session.execute(
                query,
                {"user_id": str(user_id), "limit": limit},
            )

        return result.fetchall()

    async def _get_memories_for_question(
        self,
        session: AsyncSession,
        user_id: UUID,
        question: str,
        limit: int = 12,
    ):
        """Retrieve episodic memories most relevant to a reflection question."""
        question_embedding = await embedding_service.embed(question)
        query = text(r"""
            SELECT id, content, observation_type, importance_score, created_at, last_accessed_at
            FROM memories
            WHERE user_id = :user_id
              AND memory_type = 'EPISODIC'
              AND embedding IS NOT NULL
            ORDER BY 1 - (embedding <=> :embedding\:\:vector) DESC, importance_score DESC
            LIMIT :limit
        """)
        result = await session.execute(
            query,
            {
                "user_id": str(user_id),
                "embedding": str(question_embedding),
                "limit": limit,
            },
        )
        return result.fetchall()

    async def _generate_questions(
        self,
        memories,
        focus: str | None = None,
        max_questions: int = 3,
    ) -> list[str]:
        """Generate high-level reflection questions from episodic memories."""
        memories_text = self._format_memories_for_prompt(memories)
        prompt = f"""Given the episodic observations below, propose up to {max_questions} high-level reflection questions that would surface the most important themes, decisions, and lessons.

{f"Prioritize the focus area: {focus}" if focus else ""}

Observations:
{memories_text}

Return only the JSON array of questions."""

        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=QUESTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            questions = json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return []

        if isinstance(questions, str):
            questions = [questions]

        if not isinstance(questions, list):
            return []

        cleaned = []
        for q in questions:
            if isinstance(q, str):
                q = q.strip()
                if q:
                    cleaned.append(q)
        return cleaned[:max_questions]

    async def _extract_insights_for_question(
        self,
        question: str,
        memories,
        max_insights: int,
    ) -> dict:
        """Run the LLM to extract insights and procedures for a question."""
        memories_text = self._format_memories_for_prompt(memories)
        prompt = f"""Reflection question: {question}

Observations:
{memories_text}

Extract up to {max_insights} insights and any repeatable procedures that answer the question. Return JSON with 'insights' and 'procedures' arrays."""

        response = await self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return {"insights": [], "procedures": []}

        if isinstance(data, list):
            # Backward compatibility with prior prompt shape
            return {"insights": data[:max_insights], "procedures": []}

        if not isinstance(data, dict):
            return {"insights": [], "procedures": []}

        return {
            "insights": data.get("insights", []) or [],
            "procedures": data.get("procedures", []) or [],
        }

    def _format_memories_for_prompt(self, memories) -> str:
        """Format memories for reflection and question prompts."""
        lines = []
        for mem in memories:
            obs_type = mem.observation_type or "general"
            lines.append(
                f"[{mem.id}] ({obs_type}, importance={mem.importance_score}): {mem.content}"
            )
        return "\n".join(lines)

    def _format_procedural_content(self, title: str, steps: list[str]) -> str:
        """Convert procedural JSON data into a stored content string."""
        formatted_steps = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
        return f"{title}\n{formatted_steps}"

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
