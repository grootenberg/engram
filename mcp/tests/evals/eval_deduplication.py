"""Deduplication quality evaluation script.

Measures:
- True Positive Rate: Correctly identified duplicates
- False Positive Rate: Non-duplicates incorrectly flagged
- False Negative Rate: Duplicates missed
- Threshold analysis: Performance at different similarity thresholds

Usage:
    pytest tests/evals/eval_deduplication.py -v --tb=short
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.services.ingestion import IngestionService


@dataclass
class DedupTestCase:
    """A single deduplication test case."""

    text_a: str
    text_b: str
    should_dedup: bool
    actual_deduped: bool | None = None
    similarity: float | None = None

    @property
    def is_correct(self) -> bool:
        """Whether the deduplication decision was correct."""
        return self.actual_deduped == self.should_dedup

    @property
    def is_true_positive(self) -> bool:
        """True duplicate correctly identified."""
        return self.should_dedup and self.actual_deduped

    @property
    def is_true_negative(self) -> bool:
        """Non-duplicate correctly allowed."""
        return not self.should_dedup and not self.actual_deduped

    @property
    def is_false_positive(self) -> bool:
        """Non-duplicate incorrectly blocked."""
        return not self.should_dedup and self.actual_deduped

    @property
    def is_false_negative(self) -> bool:
        """Duplicate incorrectly allowed."""
        return self.should_dedup and not self.actual_deduped


@dataclass
class DedupEvalSummary:
    """Aggregate deduplication evaluation results."""

    total_cases: int = 0
    accuracy: float = 0.0
    true_positive_rate: float = 0.0  # Sensitivity/Recall
    true_negative_rate: float = 0.0  # Specificity
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    test_cases: list[DedupTestCase] = field(default_factory=list)

    def compute_metrics(self):
        """Compute aggregate metrics from test cases."""
        if not self.test_cases:
            return

        self.total_cases = len(self.test_cases)

        # Count outcomes
        tp = sum(1 for tc in self.test_cases if tc.is_true_positive)
        tn = sum(1 for tc in self.test_cases if tc.is_true_negative)
        fp = sum(1 for tc in self.test_cases if tc.is_false_positive)
        fn = sum(1 for tc in self.test_cases if tc.is_false_negative)

        total_positive = tp + fn  # Actually duplicates
        total_negative = tn + fp  # Actually non-duplicates
        predicted_positive = tp + fp

        # Accuracy
        self.accuracy = (tp + tn) / self.total_cases if self.total_cases > 0 else 0

        # Sensitivity / True Positive Rate
        self.true_positive_rate = tp / total_positive if total_positive > 0 else 0

        # Specificity / True Negative Rate
        self.true_negative_rate = tn / total_negative if total_negative > 0 else 0

        # False rates
        self.false_positive_rate = fp / total_negative if total_negative > 0 else 0
        self.false_negative_rate = fn / total_positive if total_positive > 0 else 0

        # Precision
        self.precision = tp / predicted_positive if predicted_positive > 0 else 0

        # F1 Score
        if self.precision + self.true_positive_rate > 0:
            self.f1_score = 2 * (self.precision * self.true_positive_rate) / (
                self.precision + self.true_positive_rate
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_cases": self.total_cases,
            "accuracy": round(self.accuracy, 4),
            "true_positive_rate": round(self.true_positive_rate, 4),
            "true_negative_rate": round(self.true_negative_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "precision": round(self.precision, 4),
            "f1_score": round(self.f1_score, 4),
            "per_case_results": [
                {
                    "text_a": tc.text_a[:50] + "..." if len(tc.text_a) > 50 else tc.text_a,
                    "text_b": tc.text_b[:50] + "..." if len(tc.text_b) > 50 else tc.text_b,
                    "expected": tc.should_dedup,
                    "actual": tc.actual_deduped,
                    "correct": tc.is_correct,
                    "similarity": round(tc.similarity, 4) if tc.similarity else None,
                }
                for tc in self.test_cases
            ],
        }


# Deduplication test dataset
DEDUP_TEST_CASES = [
    # Should deduplicate - near-identical content
    DedupTestCase(
        text_a="User prefers TypeScript over JavaScript for new projects",
        text_b="User prefers TypeScript instead of JavaScript for new projects",
        should_dedup=True,
    ),
    DedupTestCase(
        text_a="Always run tests before committing code",
        text_b="Always run the tests before you commit code",
        should_dedup=True,
    ),
    DedupTestCase(
        text_a="Error: Connection refused when database is not running",
        text_b="Error: Connection refused - database is not running",
        should_dedup=True,
    ),
    DedupTestCase(
        text_a="Authentication uses JWT tokens",
        text_b="Auth uses JWT tokens",
        should_dedup=True,
    ),
    DedupTestCase(
        text_a="Use async/await for database queries",
        text_b="Use async await for database queries",
        should_dedup=True,
    ),
    # Should NOT deduplicate - different meaning
    DedupTestCase(
        text_a="User prefers TypeScript over JavaScript",
        text_b="User prefers Python over JavaScript",
        should_dedup=False,
    ),
    DedupTestCase(
        text_a="Error: Connection refused when database is not running",
        text_b="Error: Authentication failed for user postgres",
        should_dedup=False,
    ),
    DedupTestCase(
        text_a="Always run tests before committing",
        text_b="Never skip code review before merging",
        should_dedup=False,
    ),
    DedupTestCase(
        text_a="PostgreSQL for production database",
        text_b="Redis for caching layer",
        should_dedup=False,
    ),
    DedupTestCase(
        text_a="Use FastAPI for REST endpoints",
        text_b="Use GraphQL for flexible queries",
        should_dedup=False,
    ),
    # Edge cases - similar topic, different details
    DedupTestCase(
        text_a="JWT tokens expire after 1 hour",
        text_b="JWT tokens expire after 24 hours",
        should_dedup=False,  # Different expiry = different info
    ),
    DedupTestCase(
        text_a="Database connection pool size is 10",
        text_b="Database connection pool size is 50",
        should_dedup=False,  # Different config value
    ),
    DedupTestCase(
        text_a="Use pytest for unit tests",
        text_b="Use pytest for integration tests",
        should_dedup=False,  # Different test types
    ),
    # Edge cases - subset/superset
    DedupTestCase(
        text_a="Always validate user input",
        text_b="Always validate user input before processing to prevent injection attacks",
        should_dedup=False,  # Second has more detail
    ),
    DedupTestCase(
        text_a="Error handling is important",
        text_b="Proper error handling with try-catch blocks is important for reliability",
        should_dedup=False,  # Second is more specific
    ),
]


async def run_dedup_eval_with_real_embeddings(
    service: IngestionService,
    session,
    user_id,
) -> DedupEvalSummary:
    """Run deduplication evaluation with real OpenAI embeddings.

    Requires OPENAI_API_KEY environment variable.
    """
    summary = DedupEvalSummary()

    for test_case in DEDUP_TEST_CASES:
        # Ingest first text
        result_a = await service.ingest(
            session=session,
            user_id=user_id,
            content=test_case.text_a,
        )

        await session.commit()

        # Try to ingest second text
        result_b = await service.ingest(
            session=session,
            user_id=user_id,
            content=test_case.text_b,
        )

        # Record outcome
        test_case.actual_deduped = result_b["status"] == "deduplicated"
        if test_case.actual_deduped and "similarity" in result_b:
            test_case.similarity = result_b["similarity"]

        summary.test_cases.append(test_case)

        # Clean up for next test (delete both memories)
        await session.rollback()

    summary.compute_metrics()
    return summary


async def run_dedup_eval_with_mock_embeddings(
    mock_embedding,
) -> DedupEvalSummary:
    """Run deduplication evaluation with mock embeddings.

    Uses deterministic hash-based embeddings for reproducibility.
    Note: Mock embeddings may not reflect real semantic similarity.
    """
    summary = DedupEvalSummary()

    for test_case in DEDUP_TEST_CASES:
        emb_a = mock_embedding(test_case.text_a)
        emb_b = mock_embedding(test_case.text_b)

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = sum(a * a for a in emb_a) ** 0.5
        norm_b = sum(b * b for b in emb_b) ** 0.5
        similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

        test_case.similarity = similarity
        test_case.actual_deduped = similarity >= 0.90  # Default threshold

        summary.test_cases.append(test_case)

    summary.compute_metrics()
    return summary


def analyze_threshold_sweep(
    mock_embedding,
    thresholds: list[float] = None,
) -> dict[float, DedupEvalSummary]:
    """Analyze deduplication performance at different thresholds.

    Returns mapping of threshold -> evaluation summary.
    """
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]

    results = {}

    for threshold in thresholds:
        summary = DedupEvalSummary()

        for test_case in DEDUP_TEST_CASES:
            emb_a = mock_embedding(test_case.text_a)
            emb_b = mock_embedding(test_case.text_b)

            dot_product = sum(a * b for a, b in zip(emb_a, emb_b))
            norm_a = sum(a * a for a in emb_a) ** 0.5
            norm_b = sum(b * b for b in emb_b) ** 0.5
            similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

            tc_copy = DedupTestCase(
                text_a=test_case.text_a,
                text_b=test_case.text_b,
                should_dedup=test_case.should_dedup,
                similarity=similarity,
                actual_deduped=similarity >= threshold,
            )
            summary.test_cases.append(tc_copy)

        summary.compute_metrics()
        results[threshold] = summary

    return results


def save_eval_results(summary: DedupEvalSummary, output_path: Path):
    """Save evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "eval_type": "deduplication",
                **summary.to_dict(),
            },
            f,
            indent=2,
        )


# Pytest test functions
class TestDedupEval:
    """Pytest wrapper for deduplication evaluation."""

    async def test_dedup_accuracy_with_mock(self, mock_embedding):
        """Test deduplication accuracy with mock embeddings."""
        summary = await run_dedup_eval_with_mock_embeddings(mock_embedding)

        print(f"\nDeduplication Eval Results (Mock Embeddings):")
        print(f"  Accuracy: {summary.accuracy:.2%}")
        print(f"  True Positive Rate: {summary.true_positive_rate:.2%}")
        print(f"  False Positive Rate: {summary.false_positive_rate:.2%}")
        print(f"  F1 Score: {summary.f1_score:.2%}")

        # Note: Mock embeddings won't achieve good accuracy since they're
        # based on hash, not semantic similarity. This test just verifies
        # the eval framework works.
        assert summary.total_cases == len(DEDUP_TEST_CASES)

    def test_threshold_sweep(self, mock_embedding):
        """Test deduplication at different thresholds."""
        results = analyze_threshold_sweep(mock_embedding)

        print("\nThreshold Sweep Results (Mock Embeddings):")
        for threshold, summary in sorted(results.items()):
            print(
                f"  {threshold:.2f}: Acc={summary.accuracy:.2%}, "
                f"TPR={summary.true_positive_rate:.2%}, "
                f"FPR={summary.false_positive_rate:.2%}"
            )

        # Verify we got results for all thresholds
        assert len(results) == 6

    def test_dedup_test_case_properties(self):
        """Test DedupTestCase classification properties."""
        # True positive
        tc_tp = DedupTestCase("a", "b", should_dedup=True, actual_deduped=True)
        assert tc_tp.is_true_positive
        assert tc_tp.is_correct

        # False positive
        tc_fp = DedupTestCase("a", "b", should_dedup=False, actual_deduped=True)
        assert tc_fp.is_false_positive
        assert not tc_fp.is_correct

        # True negative
        tc_tn = DedupTestCase("a", "b", should_dedup=False, actual_deduped=False)
        assert tc_tn.is_true_negative
        assert tc_tn.is_correct

        # False negative
        tc_fn = DedupTestCase("a", "b", should_dedup=True, actual_deduped=False)
        assert tc_fn.is_false_negative
        assert not tc_fn.is_correct

    def test_summary_metric_computation(self):
        """Test that summary metrics are computed correctly."""
        test_cases = [
            DedupTestCase("a", "b", should_dedup=True, actual_deduped=True),  # TP
            DedupTestCase("c", "d", should_dedup=True, actual_deduped=False),  # FN
            DedupTestCase("e", "f", should_dedup=False, actual_deduped=False),  # TN
            DedupTestCase("g", "h", should_dedup=False, actual_deduped=True),  # FP
        ]

        summary = DedupEvalSummary(test_cases=test_cases)
        summary.compute_metrics()

        # TP=1, TN=1, FP=1, FN=1
        assert summary.total_cases == 4
        assert summary.accuracy == 0.5  # 2/4
        assert summary.true_positive_rate == 0.5  # 1/2 actual positives
        assert summary.true_negative_rate == 0.5  # 1/2 actual negatives
        assert summary.precision == 0.5  # 1/2 predicted positives


if __name__ == "__main__":
    print("Deduplication evaluation - run with pytest for full evaluation")
    print("pytest tests/evals/eval_deduplication.py -v")
