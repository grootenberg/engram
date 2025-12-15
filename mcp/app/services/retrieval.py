"""Retrieval service for engram with three-factor scoring."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.embedding import embedding_service


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse ISO datetime string to datetime object."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


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
        observation_types: list[str] | None = None,
        sections: list[str] | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        min_helpful_count: int = 0,
        max_harmful_count: int | None = None,
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
            observation_types: Filter by observation types (error, instruction, etc.).
            sections: Filter by ACE sections (strategies, snippets, pitfalls, etc.).
            created_after: Only include memories created after this ISO datetime.
            created_before: Only include memories created before this ISO datetime.
            min_helpful_count: Minimum helpful feedback count (default: 0).
            max_harmful_count: Maximum harmful feedback count (default: None).

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

        # Build observation type filter
        obs_type_filter = ""
        if observation_types:
            obs_types_str = ", ".join(f"'{t.lower()}'" for t in observation_types)
            obs_type_filter = f"AND observation_type IN ({obs_types_str})"

        # Build section filter
        section_filter = ""
        if sections:
            sections_str = ", ".join(f"'{s.lower()}'" for s in sections)
            section_filter = f"AND section IN ({sections_str})"

        # Build time range filters
        created_after_dt = _parse_datetime(created_after)
        created_before_dt = _parse_datetime(created_before)
        time_filter = ""
        if created_after_dt:
            time_filter += f"AND created_at >= '{created_after_dt.isoformat()}'"
        if created_before_dt:
            time_filter += f" AND created_at <= '{created_before_dt.isoformat()}'"

        # Build feedback filters
        feedback_filter = f"AND helpful_count >= {min_helpful_count}"
        if max_harmful_count is not None:
            feedback_filter += f" AND harmful_count <= {max_harmful_count}"

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
                  {obs_type_filter}
                  {section_filter}
                  {time_filter}
                  {feedback_filter}
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


    async def retrieve_debug(
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
        observation_types: list[str] | None = None,
        sections: list[str] | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        min_helpful_count: int = 0,
        max_harmful_count: int | None = None,
    ) -> dict:
        """Debug version of retrieve with detailed scoring breakdown.

        Returns comprehensive information about how memories are scored
        and filtered, useful for understanding retrieval behavior.
        """
        # Normalize weights to sum to 1
        total_weight = recency_weight + importance_weight + relevance_weight
        if total_weight > 0:
            recency_weight /= total_weight
            importance_weight /= total_weight
            relevance_weight /= total_weight

        # Generate query embedding
        query_embedding = await embedding_service.embed(query)

        # Build filters (same as retrieve)
        type_filter = ""
        if memory_types:
            types_str = ", ".join(f"'{t.upper()}'" for t in memory_types)
            type_filter = f"AND memory_type::text IN ({types_str})"

        synthetic_filter = "" if include_synthetic else "AND is_synthetic = FALSE"

        obs_type_filter = ""
        if observation_types:
            obs_types_str = ", ".join(f"'{t.lower()}'" for t in observation_types)
            obs_type_filter = f"AND observation_type IN ({obs_types_str})"

        section_filter = ""
        if sections:
            sections_str = ", ".join(f"'{s.lower()}'" for s in sections)
            section_filter = f"AND section IN ({sections_str})"

        created_after_dt = _parse_datetime(created_after)
        created_before_dt = _parse_datetime(created_before)
        time_filter = ""
        if created_after_dt:
            time_filter += f"AND created_at >= '{created_after_dt.isoformat()}'"
        if created_before_dt:
            time_filter += f" AND created_at <= '{created_before_dt.isoformat()}'"

        feedback_filter = f"AND helpful_count >= {min_helpful_count}"
        if max_harmful_count is not None:
            feedback_filter += f" AND harmful_count <= {max_harmful_count}"

        decay = settings.engram_recency_decay

        # Count total candidates before filters
        count_query = text("""
            SELECT COUNT(*) as cnt
            FROM memories
            WHERE user_id = :user_id
              AND embedding IS NOT NULL
        """)
        count_result = await session.execute(count_query, {"user_id": str(user_id)})
        candidates_before = count_result.scalar() or 0

        # Debug query that returns ALL matching rows (including those below threshold)
        debug_query = text(rf"""
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
                    POWER(:decay, EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed_at, created_at))) / 3600) as recency_raw,
                    LEAST(1.0, GREATEST(0.0,
                        (importance_score + (helpful_count - harmful_count) * 0.5) / 10.0
                    )) as importance_raw,
                    1 - (embedding <=> :query_embedding\:\:vector) as relevance_score
                FROM memories
                WHERE user_id = :user_id
                  AND embedding IS NOT NULL
                  AND importance_score >= :min_importance
                  {type_filter}
                  {synthetic_filter}
                  {obs_type_filter}
                  {section_filter}
                  {time_filter}
                  {feedback_filter}
            ),
            stats AS (
                SELECT
                    MIN(recency_raw) AS recency_min,
                    MAX(recency_raw) AS recency_max,
                    MIN(importance_raw) AS importance_min,
                    MAX(importance_raw) AS importance_max,
                    MIN(relevance_score) AS relevance_min,
                    MAX(relevance_score) AS relevance_max,
                    COUNT(*) AS total_candidates
                FROM base
            ),
            normalized AS (
                SELECT
                    base.*,
                    stats.recency_min,
                    stats.recency_max,
                    stats.importance_min,
                    stats.importance_max,
                    stats.relevance_min,
                    stats.relevance_max,
                    stats.total_candidates,
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
                ) as combined_score,
                CASE WHEN relevance_score <= 0.1 THEN 'relevance_below_threshold' ELSE NULL END as filter_reason
            FROM normalized
            ORDER BY combined_score DESC
            LIMIT :limit + 5
        """)

        result = await session.execute(
            debug_query,
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

        # Separate passing and filtered results
        passing_results = []
        filtered_examples = []
        score_ranges = None

        for row in rows:
            if score_ranges is None and hasattr(row, 'total_candidates'):
                score_ranges = {
                    "recency": {"min": round(row.recency_min or 0, 4), "max": round(row.recency_max or 0, 4)},
                    "importance": {"min": round(row.importance_min or 0, 4), "max": round(row.importance_max or 0, 4)},
                    "relevance": {"min": round(row.relevance_min or 0, 4), "max": round(row.relevance_max or 0, 4)},
                }

            result_data = {
                "memory_id": str(row.id),
                "content_preview": row.content[:100] + "..." if len(row.content) > 100 else row.content,
                "memory_type": row.memory_type,
                "observation_type": row.observation_type,
                "raw_scores": {
                    "recency": round(row.recency_raw, 4),
                    "importance": round(row.importance_raw, 4),
                    "relevance": round(row.relevance_score, 4),
                },
                "normalized_scores": {
                    "recency": round(row.recency_score or 0, 4),
                    "importance": round(row.importance_score_normalized or 0, 4),
                    "relevance": round(row.relevance_score_normalized or 0, 4),
                },
                "combined_score": round(row.combined_score or 0, 4),
                "hours_since_access": round(
                    (datetime.now() - (row.last_accessed_at or row.created_at)).total_seconds() / 3600, 2
                ) if row.last_accessed_at or row.created_at else None,
                "effective_importance": row.importance_score + (row.helpful_count - row.harmful_count) * 0.5,
                "cosine_similarity": round(row.relevance_score, 4),
            }

            if row.filter_reason:
                result_data["filter_reason"] = f"relevance_score {round(row.relevance_score, 4)} <= threshold 0.1"
                if len(filtered_examples) < 3:
                    filtered_examples.append(result_data)
            else:
                if len(passing_results) < limit:
                    passing_results.append(result_data)

        return {
            "query": query,
            "query_embedding_preview": [round(x, 6) for x in query_embedding[:5]],
            "candidates_before_filters": candidates_before,
            "candidates_after_filters": rows[0].total_candidates if rows and hasattr(rows[0], 'total_candidates') else 0,
            "weights_used": {
                "recency": round(recency_weight, 2),
                "importance": round(importance_weight, 2),
                "relevance": round(relevance_weight, 2),
            },
            "score_ranges": score_ranges or {
                "recency": {"min": 0, "max": 0},
                "importance": {"min": 0, "max": 0},
                "relevance": {"min": 0, "max": 0},
            },
            "results": passing_results,
            "filtered_out_examples": filtered_examples,
        }


# Global singleton instance
retrieval_service = RetrievalService()
