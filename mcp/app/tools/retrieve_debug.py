"""memory_retrieve_debug tool for debugging retrieval scoring."""

from uuid import UUID

from app.db import get_session
from app.services.retrieval import retrieval_service


async def memory_retrieve_debug(
    query: str,
    user_id: str = "default",
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
    """Debug retrieval scoring for a query.

    Returns detailed breakdown of how each memory was scored,
    including raw scores before normalization, filter effects,
    and score ranges across candidates.

    Args:
        query: Search query text for semantic matching.
        user_id: User identifier (default: "default").
        limit: Maximum number of results (default: 10).
        memory_types: Filter by memory types. Options: episodic, semantic, procedural.
        min_importance: Minimum importance score filter (default: 0.0).
        include_synthetic: Include reflected/synthesized memories (default: True).
        recency_weight: Weight for recency factor α (default: 0.33).
        importance_weight: Weight for importance factor β (default: 0.33).
        relevance_weight: Weight for relevance factor γ (default: 0.33).
        observation_types: Filter by observation types.
        sections: Filter by ACE sections.
        created_after: Only include memories created after this ISO datetime.
        created_before: Only include memories created before this ISO datetime.
        min_helpful_count: Minimum helpful feedback count (default: 0).
        max_harmful_count: Maximum harmful feedback count. If None, no limit.

    Returns:
        dict with:
            - query: Original query
            - query_embedding_preview: First 5 dimensions of query embedding
            - candidates_before_filters: Total candidates matching user_id
            - candidates_after_filters: Candidates after applying all filters
            - weights_used: Normalized weights
            - score_ranges: Min/max for each factor across candidates
            - results: Detailed scoring for each result
            - filtered_out_examples: Examples of filtered memories with reasons
    """
    try:
        uid = UUID(user_id)
    except ValueError:
        import hashlib
        uid = UUID(hashlib.md5(user_id.encode()).hexdigest())

    async with get_session() as session:
        result = await retrieval_service.retrieve_debug(
            session=session,
            user_id=uid,
            query=query,
            limit=limit,
            memory_types=memory_types,
            min_importance=min_importance,
            include_synthetic=include_synthetic,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            relevance_weight=relevance_weight,
            observation_types=observation_types,
            sections=sections,
            created_after=created_after,
            created_before=created_before,
            min_helpful_count=min_helpful_count,
            max_harmful_count=max_harmful_count,
        )
        return result
