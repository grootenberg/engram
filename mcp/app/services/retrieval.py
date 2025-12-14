"""Retrieval service for engram with three-factor scoring."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.embedding import embedding_service


class RetrievalService:
    """Service for retrieving memories using three-factor scoring.

    Scoring formula:
        score = α * recency + β * importance + γ * relevance

    Where:
        - recency: Exponential decay based on hours since last access
        - importance: Effective importance (base + feedback adjustment)
        - relevance: Cosine similarity to query embedding
        - Each factor is min-max normalized across the candidate set
    """

    async def retrieve(
        self,
        session: AsyncSession,
        user_id: UUID,
        query: str,
        limit: int = 10,
        memory_types: list[str] | None = None,
        min_importance: float = 0.0,
        include_synthetic: bool = True,
        recency_weight: float = 0.33,
        importance_weight: float = 0.33,
        relevance_weight: float = 0.33,
    ) -> dict:
        """Retrieve memories using three-factor scoring.

        Args:
            session: Database session.
            user_id: User identifier.
            query: Search query text.
            limit: Maximum number of results.
            memory_types: Filter by memory types (episodic, semantic, procedural).
            min_importance: Minimum importance score filter.
            include_synthetic: Whether to include synthetic (reflected) memories.
            recency_weight: Weight for recency factor (α).
            importance_weight: Weight for importance factor (β).
            relevance_weight: Weight for relevance factor (γ).

        Returns:
            Dict with retrieved memories and scoring metadata.
        """
        # Normalize weights to sum to 1
        total_weight = recency_weight + importance_weight + relevance_weight
        if total_weight > 0:
            recency_weight /= total_weight
            importance_weight /= total_weight
            relevance_weight /= total_weight

        # Generate query embedding
        query_embedding = await embedding_service.embed(query)

        # Build memory type filter
        type_filter = ""
        if memory_types:
            types_str = ", ".join(f"'{t.upper()}'" for t in memory_types)
            type_filter = f"AND memory_type::text IN ({types_str})"

        # Build synthetic filter
        synthetic_filter = "" if include_synthetic else "AND is_synthetic = FALSE"

        # Three-factor scoring query
        # recency = decay ^ hours_since last access (normalized)
        # importance = effective_importance / 10 (normalized then min-maxed)
        # relevance = cosine_similarity (min-maxed)
        decay = settings.engram_recency_decay

        query_sql = text(rf"""
            WITH base AS (
                SELECT
                    id,
                    content,
                    memory_type,
                    observation_type,
                    section,
                    importance_score,
                    helpful_count,
                    harmful_count,
                    is_synthetic,
                    citations,
                    created_at,
                    last_accessed_at,
                    extra_metadata,
                    -- Recency score: exponential decay using last access time
                    POWER(:decay, EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed_at, created_at))) / 3600) as recency_raw,
                    -- Effective importance normalized to 0-1 before min-max
                    LEAST(1.0, GREATEST(0.0,
                        (importance_score + (helpful_count - harmful_count) * 0.5) / 10.0
                    )) as importance_raw,
                    -- Relevance score: cosine similarity
                    1 - (embedding <=> :query_embedding\:\:vector) as relevance_score
                FROM memories
                WHERE user_id = :user_id
                  AND embedding IS NOT NULL
                  AND importance_score >= :min_importance
                  {type_filter}
                  {synthetic_filter}
            ),
            stats AS (
                SELECT
                    MIN(recency_raw) AS recency_min,
                    MAX(recency_raw) AS recency_max,
                    MIN(importance_raw) AS importance_min,
                    MAX(importance_raw) AS importance_max,
                    MIN(relevance_score) AS relevance_min,
                    MAX(relevance_score) AS relevance_max
                FROM base
            ),
            normalized AS (
                SELECT
                    base.*,
                    CASE
                        WHEN stats.recency_max > stats.recency_min
                            THEN (base.recency_raw - stats.recency_min) / NULLIF(stats.recency_max - stats.recency_min, 0)
                        ELSE base.recency_raw
                    END AS recency_score,
                    CASE
                        WHEN stats.importance_max > stats.importance_min
                            THEN (base.importance_raw - stats.importance_min) / NULLIF(stats.importance_max - stats.importance_min, 0)
                        ELSE base.importance_raw
                    END AS importance_score_normalized,
                    CASE
                        WHEN stats.relevance_max > stats.relevance_min
                            THEN (base.relevance_score - stats.relevance_min) / NULLIF(stats.relevance_max - stats.relevance_min, 0)
                        ELSE base.relevance_score
                    END AS relevance_score_normalized
                FROM base
                CROSS JOIN stats
            )
            SELECT
                *,
                (
                    :recency_weight * recency_score +
                    :importance_weight * importance_score_normalized +
                    :relevance_weight * relevance_score_normalized
                ) as combined_score
            FROM normalized
            WHERE relevance_score > 0.1  -- Minimum relevance threshold
            ORDER BY combined_score DESC
            LIMIT :limit
        """)

        result = await session.execute(
            query_sql,
            {
                "user_id": str(user_id),
                "query_embedding": str(query_embedding),
                "min_importance": min_importance,
                "decay": decay,
                "recency_weight": recency_weight,
                "importance_weight": importance_weight,
                "relevance_weight": relevance_weight,
                "limit": limit,
            },
        )

        rows = result.fetchall()

        # Update last_accessed_at for retrieved memories
        if rows:
            memory_ids = [str(row.id) for row in rows]
            await session.execute(
                text(r"""
                    UPDATE memories
                    SET last_accessed_at = NOW()
                    WHERE id = ANY(:ids\:\:uuid[])
                """),
                {"ids": memory_ids},
            )

        # Format results
        memories = []
        for row in rows:
            memories.append({
                "id": str(row.id),
                "content": row.content,
                "memory_type": row.memory_type,
                "observation_type": row.observation_type,
                "section": row.section,
                "importance_score": row.importance_score,
                "effective_importance": row.importance_score + (row.helpful_count - row.harmful_count) * 0.5,
                "is_synthetic": row.is_synthetic,
                "citations": row.citations,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "metadata": row.extra_metadata,
                "scores": {
                    "combined": round(row.combined_score, 4),
                    "recency": round(row.recency_score, 4),
                    "recency_raw": round(row.recency_raw, 4),
                    "importance": round(row.importance_score_normalized, 4),
                    "importance_raw": round(row.importance_raw, 4),
                    "relevance": round(row.relevance_score_normalized, 4),
                    "relevance_raw": round(row.relevance_score, 4),
                },
            })

        return {
            "memories": memories,
            "count": len(memories),
            "weights": {
                "recency": round(recency_weight, 2),
                "importance": round(importance_weight, 2),
                "relevance": round(relevance_weight, 2),
            },
        }


# Global singleton instance
retrieval_service = RetrievalService()
