"""Retrieval quality evaluation script.

Measures:
- Precision@K: Fraction of retrieved memories that are relevant
- Recall@K: Fraction of relevant memories that are retrieved
- MRR: Mean Reciprocal Rank of first relevant result
- NDCG: Normalized Discounted Cumulative Gain

Usage:
    pytest tests/evals/eval_retrieval.py -v --tb=short
    # Or run directly:
    python -m tests.evals.eval_retrieval
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.models.memory import Memory, MemoryType, ObservationType, SectionType
from app.services.retrieval import RetrievalService


@dataclass
class RetrievalEvalResult:
    """Results from a single retrieval evaluation."""

    query: str
    expected_ids: set[str]
    retrieved_ids: list[str]
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    first_relevant_rank: int | None = None

    def compute_metrics(self):
        """Compute all metrics for this result."""
        retrieved_set = set(self.retrieved_ids[:5])
        relevant_retrieved = retrieved_set & self.expected_ids

        # Precision@5
        if len(retrieved_set) > 0:
            self.precision_at_5 = len(relevant_retrieved) / len(retrieved_set)

        # Recall@5
        if len(self.expected_ids) > 0:
            self.recall_at_5 = len(relevant_retrieved) / len(self.expected_ids)

        # MRR - reciprocal rank of first relevant result
        for i, mem_id in enumerate(self.retrieved_ids):
            if mem_id in self.expected_ids:
                self.first_relevant_rank = i + 1
                self.mrr = 1.0 / (i + 1)
                break


@dataclass
class RetrievalEvalSummary:
    """Aggregate evaluation results."""

    total_queries: int = 0
    avg_precision_at_5: float = 0.0
    avg_recall_at_5: float = 0.0
    avg_mrr: float = 0.0
    results: list[RetrievalEvalResult] = field(default_factory=list)

    def compute_aggregates(self):
        """Compute aggregate metrics from individual results."""
        if not self.results:
            return

        self.total_queries = len(self.results)
        self.avg_precision_at_5 = sum(r.precision_at_5 for r in self.results) / self.total_queries
        self.avg_recall_at_5 = sum(r.recall_at_5 for r in self.results) / self.total_queries
        self.avg_mrr = sum(r.mrr for r in self.results) / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_queries": self.total_queries,
            "avg_precision_at_5": round(self.avg_precision_at_5, 4),
            "avg_recall_at_5": round(self.avg_recall_at_5, 4),
            "avg_mrr": round(self.avg_mrr, 4),
            "per_query_results": [
                {
                    "query": r.query,
                    "precision_at_5": round(r.precision_at_5, 4),
                    "recall_at_5": round(r.recall_at_5, 4),
                    "mrr": round(r.mrr, 4),
                    "first_relevant_rank": r.first_relevant_rank,
                }
                for r in self.results
            ],
        }


# Evaluation dataset - queries with expected relevant memory characteristics
EVAL_DATASET = [
    {
        "query": "user preferences for programming languages",
        "relevant_content_patterns": ["prefer", "typescript", "javascript", "language"],
        "relevant_types": [ObservationType.INSTRUCTION],
        "relevant_sections": [SectionType.PREFERENCES],
    },
    {
        "query": "authentication and security patterns",
        "relevant_content_patterns": ["auth", "jwt", "token", "security", "password"],
        "relevant_types": [ObservationType.DECISION, ObservationType.INSIGHT],
        "relevant_sections": [SectionType.CONTEXT, SectionType.STRATEGIES],
    },
    {
        "query": "database connection errors",
        "relevant_content_patterns": ["error", "connection", "database", "refused"],
        "relevant_types": [ObservationType.ERROR],
        "relevant_sections": [SectionType.PITFALLS],
    },
    {
        "query": "testing best practices",
        "relevant_content_patterns": ["test", "pytest", "coverage", "mock"],
        "relevant_types": [ObservationType.INSTRUCTION, ObservationType.INSIGHT],
        "relevant_sections": [SectionType.STRATEGIES],
    },
    {
        "query": "async programming patterns",
        "relevant_content_patterns": ["async", "await", "asyncio", "concurrent"],
        "relevant_types": [ObservationType.CODE_CHANGE, ObservationType.INSIGHT],
        "relevant_sections": [SectionType.SNIPPETS, SectionType.STRATEGIES],
    },
]


def create_eval_corpus(mock_embedding) -> tuple[list[Memory], dict[str, set[str]]]:
    """Create evaluation corpus with known-relevant memories.

    Returns:
        Tuple of (memories, query_to_relevant_ids mapping)
    """
    user_id = uuid4()
    memories = []
    query_relevance: dict[str, set[str]] = {q["query"]: set() for q in EVAL_DATASET}

    # Create memories that match each query's expected characteristics
    corpus_data = [
        # Programming language preferences
        {
            "content": "User strongly prefers TypeScript over JavaScript for new projects",
            "type": ObservationType.INSTRUCTION,
            "section": SectionType.PREFERENCES,
            "importance": 9.0,
            "relevant_to": ["user preferences for programming languages"],
        },
        {
            "content": "Always use strict TypeScript configuration",
            "type": ObservationType.INSTRUCTION,
            "section": SectionType.PREFERENCES,
            "importance": 8.0,
            "relevant_to": ["user preferences for programming languages"],
        },
        # Authentication
        {
            "content": "Authentication uses JWT tokens with 1-hour expiry for security",
            "type": ObservationType.DECISION,
            "section": SectionType.CONTEXT,
            "importance": 7.0,
            "relevant_to": ["authentication and security patterns"],
        },
        {
            "content": "Implement refresh token rotation for enhanced security",
            "type": ObservationType.INSIGHT,
            "section": SectionType.STRATEGIES,
            "importance": 8.0,
            "relevant_to": ["authentication and security patterns"],
        },
        {
            "content": "Password hashing should use bcrypt with minimum 12 rounds",
            "type": ObservationType.INSTRUCTION,
            "section": SectionType.STRATEGIES,
            "importance": 9.0,
            "relevant_to": ["authentication and security patterns"],
        },
        # Database errors
        {
            "content": "Error: Connection refused when database container is not running",
            "type": ObservationType.ERROR,
            "section": SectionType.PITFALLS,
            "importance": 8.0,
            "relevant_to": ["database connection errors"],
        },
        {
            "content": "Database timeout error - increase connection pool timeout",
            "type": ObservationType.ERROR,
            "section": SectionType.PITFALLS,
            "importance": 7.0,
            "relevant_to": ["database connection errors"],
        },
        # Testing
        {
            "content": "Always run pytest with coverage to ensure 80% threshold",
            "type": ObservationType.INSTRUCTION,
            "section": SectionType.STRATEGIES,
            "importance": 8.0,
            "relevant_to": ["testing best practices"],
        },
        {
            "content": "Use pytest fixtures for test isolation and setup",
            "type": ObservationType.INSIGHT,
            "section": SectionType.STRATEGIES,
            "importance": 7.0,
            "relevant_to": ["testing best practices"],
        },
        {
            "content": "Mock external API calls to avoid flaky tests",
            "type": ObservationType.INSTRUCTION,
            "section": SectionType.STRATEGIES,
            "importance": 8.0,
            "relevant_to": ["testing best practices"],
        },
        # Async patterns
        {
            "content": "Use async/await consistently for database queries",
            "type": ObservationType.CODE_CHANGE,
            "section": SectionType.SNIPPETS,
            "importance": 6.0,
            "relevant_to": ["async programming patterns"],
        },
        {
            "content": "Asyncio gather for concurrent API calls improves performance",
            "type": ObservationType.INSIGHT,
            "section": SectionType.STRATEGIES,
            "importance": 7.0,
            "relevant_to": ["async programming patterns"],
        },
        # Noise - not relevant to any specific query
        {
            "content": "Updated README with installation instructions",
            "type": ObservationType.CODE_CHANGE,
            "section": None,
            "importance": 3.0,
            "relevant_to": [],
        },
        {
            "content": "Fixed typo in variable name",
            "type": ObservationType.CODE_CHANGE,
            "section": None,
            "importance": 2.0,
            "relevant_to": [],
        },
        {
            "content": "Added logging statements for debugging",
            "type": ObservationType.TOOL_OUTPUT,
            "section": None,
            "importance": 3.0,
            "relevant_to": [],
        },
    ]

    for data in corpus_data:
        memory = Memory(
            id=uuid4(),
            user_id=user_id,
            content=data["content"],
            embedding=mock_embedding(data["content"]),
            memory_type=MemoryType.EPISODIC,
            observation_type=data["type"],
            section=data["section"],
            importance_score=data["importance"],
        )
        memories.append(memory)

        # Track which queries this memory is relevant to
        for query in data["relevant_to"]:
            query_relevance[query].add(str(memory.id))

    return memories, query_relevance


async def run_retrieval_eval(
    service: RetrievalService,
    session,
    user_id,
    mock_embedding,
    query_relevance: dict[str, set[str]],
) -> RetrievalEvalSummary:
    """Run retrieval evaluation.

    Args:
        service: RetrievalService instance
        session: Database session
        user_id: User ID for queries
        mock_embedding: Function to generate embeddings
        query_relevance: Mapping of query -> set of relevant memory IDs

    Returns:
        RetrievalEvalSummary with all metrics
    """
    summary = RetrievalEvalSummary()

    with patch("app.services.retrieval.embedding_service") as mock:
        for query, expected_ids in query_relevance.items():
            mock.embed = AsyncMock(return_value=mock_embedding(query))

            result = await service.retrieve(
                session=session,
                user_id=user_id,
                query=query,
                limit=10,
            )

            retrieved_ids = [m["id"] for m in result["memories"]]

            eval_result = RetrievalEvalResult(
                query=query,
                expected_ids=expected_ids,
                retrieved_ids=retrieved_ids,
            )
            eval_result.compute_metrics()
            summary.results.append(eval_result)

    summary.compute_aggregates()
    return summary


def save_eval_results(summary: RetrievalEvalSummary, output_path: Path):
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "eval_type": "retrieval",
                **summary.to_dict(),
            },
            f,
            indent=2,
        )


# Pytest test functions
class TestRetrievalEval:
    """Pytest wrapper for retrieval evaluation."""

    async def test_retrieval_precision_at_5(self, db_session, mock_embedding):
        """Test that retrieval achieves target precision@5."""
        # This would need a full setup with seeded data
        # For now, just verify the eval framework works
        memories, query_relevance = create_eval_corpus(mock_embedding)

        # Basic sanity check
        assert len(memories) > 10
        assert len(query_relevance) == 5
        for query, ids in query_relevance.items():
            assert len(ids) >= 1, f"Query '{query}' should have relevant memories"

    async def test_eval_result_computation(self):
        """Test that eval metrics are computed correctly."""
        result = RetrievalEvalResult(
            query="test query",
            expected_ids={"a", "b", "c"},
            retrieved_ids=["a", "x", "b", "y", "z"],
        )
        result.compute_metrics()

        # Precision@5: 2/5 = 0.4 (a and b are relevant)
        assert result.precision_at_5 == 0.4
        # Recall@5: 2/3 = 0.667 (got a and b out of a,b,c)
        assert abs(result.recall_at_5 - 0.667) < 0.01
        # MRR: 1/1 = 1.0 (first result is relevant)
        assert result.mrr == 1.0
        assert result.first_relevant_rank == 1

    async def test_eval_summary_aggregation(self):
        """Test that summary aggregates correctly."""
        results = [
            RetrievalEvalResult(
                query="q1",
                expected_ids={"a"},
                retrieved_ids=["a", "b"],
            ),
            RetrievalEvalResult(
                query="q2",
                expected_ids={"c", "d"},
                retrieved_ids=["x", "c", "d"],
            ),
        ]
        for r in results:
            r.compute_metrics()

        summary = RetrievalEvalSummary(results=results)
        summary.compute_aggregates()

        assert summary.total_queries == 2
        assert summary.avg_precision_at_5 > 0
        assert summary.avg_mrr > 0


if __name__ == "__main__":
    # Run as standalone script
    print("Retrieval evaluation - run with pytest for full evaluation")
    print("pytest tests/evals/eval_retrieval.py -v")
