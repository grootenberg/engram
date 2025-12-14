"""memory_retrieve tool for semantic memory search."""

from uuid import UUID

from app.db import get_session
from app.services import retrieval_service


async def memory_retrieve(
    query: str,
    user_id: str = "default",
    limit: int = 10,
    memory_types: list[str] | None = None,
    min_importance: float = 0.0,
    include_synthetic: bool = True,
    recency_weight: float = 0.33,
    importance_weight: float = 0.33,
    relevance_weight: float = 0.33,
) -> dict:
    """Retrieve memories using three-factor scoring.

    Combines recency, importance, and semantic relevance to rank results.
    Scoring formula: score = α*recency + β*importance + γ*relevance

    Args:
        query: Search query text for semantic matching.
        user_id: User identifier (default: "default").
        limit: Maximum number of results (default: 10).
        memory_types: Filter by memory types. Options: episodic, semantic, procedural.
            If None, returns all types.
        min_importance: Minimum importance score filter (default: 0.0).
        include_synthetic: Include reflected/synthesized memories (default: True).
        recency_weight: Weight for recency factor α (default: 0.33).
        importance_weight: Weight for importance factor β (default: 0.33).
        relevance_weight: Weight for relevance factor γ (default: 0.33).

    Returns:
        dict with:
            - memories: List of matching memories with scores
            - count: Number of results
            - weights: Normalized weights used for scoring

    Example:
        >>> memory_retrieve(
        ...     query="user code preferences",
        ...     memory_types=["semantic", "episodic"],
        ...     relevance_weight=0.5,
        ...     importance_weight=0.3,
        ...     recency_weight=0.2
        ... )
        {"memories": [...], "count": 5, "weights": {...}}
    """
    try:
        uid = UUID(user_id)
    except ValueError:
        import hashlib
        uid = UUID(hashlib.md5(user_id.encode()).hexdigest())

    async with get_session() as session:
        result = await retrieval_service.retrieve(
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
        )
        return result
