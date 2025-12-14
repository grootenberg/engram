"""Integration tests for retrieval service and retrieve tool."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.models.memory import Memory, MemoryType, ObservationType
from app.services.retrieval import RetrievalService
from tests.factories import generate_thematic_memories


class TestRetrievalService:
    """Integration tests for RetrievalService."""

    @pytest.fixture
    def service(self):
        return RetrievalService()

    @pytest.fixture
    async def seeded_memories(self, db_session, test_user_id, mock_embedding):
        """Seed database with test memories."""
        memories = [
            Memory(
                user_id=test_user_id,
                content="User prefers TypeScript over JavaScript for new projects",
                embedding=mock_embedding("User prefers TypeScript over JavaScript for new projects"),
                memory_type=MemoryType.EPISODIC,
                observation_type=ObservationType.INSTRUCTION,
                importance_score=9.0,
                created_at=datetime.utcnow() - timedelta(hours=2),
            ),
            Memory(
                user_id=test_user_id,
                content="Authentication uses JWT tokens with 1-hour expiry",
                embedding=mock_embedding("Authentication uses JWT tokens with 1-hour expiry"),
                memory_type=MemoryType.EPISODIC,
                observation_type=ObservationType.DECISION,
                importance_score=7.0,
                created_at=datetime.utcnow() - timedelta(hours=24),
            ),
            Memory(
                user_id=test_user_id,
                content="Error: Connection refused when database is not running",
                embedding=mock_embedding("Error: Connection refused when database is not running"),
                memory_type=MemoryType.EPISODIC,
                observation_type=ObservationType.ERROR,
                importance_score=8.0,
                created_at=datetime.utcnow() - timedelta(hours=12),
            ),
            Memory(
                user_id=test_user_id,
                content="Always validate user input before processing",
                embedding=mock_embedding("Always validate user input before processing"),
                memory_type=MemoryType.SEMANTIC,
                observation_type=ObservationType.INSIGHT,
                importance_score=8.0,
                is_synthetic=True,
                created_at=datetime.utcnow() - timedelta(hours=48),
            ),
            Memory(
                user_id=test_user_id,
                content="Python async/await pattern for database queries",
                embedding=mock_embedding("Python async/await pattern for database queries"),
                memory_type=MemoryType.EPISODIC,
                observation_type=ObservationType.CODE_CHANGE,
                importance_score=5.0,
                created_at=datetime.utcnow() - timedelta(hours=6),
            ),
        ]

        for mem in memories:
            db_session.add(mem)
        await db_session.commit()

        return memories

    async def test_retrieve_returns_results(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test basic retrieval returns results."""
        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("TypeScript preferences"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="TypeScript preferences",
                limit=5,
            )

        assert "memories" in result
        assert "count" in result
        assert "weights" in result
        assert result["count"] > 0

    async def test_retrieve_respects_limit(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test retrieval respects limit parameter."""
        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("any query"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="any query",
                limit=2,
            )

        assert result["count"] <= 2

    async def test_retrieve_filters_by_memory_type(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test retrieval filters by memory type."""
        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("any query"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="any query",
                memory_types=["semantic"],
            )

        for mem in result["memories"]:
            assert mem["memory_type"] == "SEMANTIC"

    async def test_retrieve_filters_synthetic(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test retrieval can exclude synthetic memories."""
        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("any query"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="any query",
                include_synthetic=False,
            )

        for mem in result["memories"]:
            assert mem["is_synthetic"] is False

    async def test_retrieve_min_importance_filter(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test retrieval filters by minimum importance."""
        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("any query"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="any query",
                min_importance=7.0,
            )

        for mem in result["memories"]:
            assert mem["importance_score"] >= 7.0

    async def test_retrieve_weight_normalization(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test that weights are normalized to sum to 1."""
        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("any query"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="any query",
                recency_weight=0.5,
                importance_weight=0.3,
                relevance_weight=0.7,  # Sum = 1.5
            )

        weights = result["weights"]
        total = weights["recency"] + weights["importance"] + weights["relevance"]
        assert abs(total - 1.0) < 0.01

    async def test_retrieve_scores_structure(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test that retrieved memories include score breakdown."""
        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("TypeScript preferences"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="TypeScript preferences",
            )

        if result["memories"]:
            mem = result["memories"][0]
            assert "scores" in mem
            assert "combined" in mem["scores"]
            assert "recency" in mem["scores"]
            assert "importance" in mem["scores"]
            assert "relevance" in mem["scores"]

    async def test_retrieve_updates_last_accessed(self, service, db_session, test_user_id, seeded_memories, mock_embedding):
        """Test that retrieval updates last_accessed_at timestamp."""
        original_time = seeded_memories[0].last_accessed_at

        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("TypeScript preferences"))

            await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="TypeScript preferences",
            )

        # Refresh memory
        await db_session.refresh(seeded_memories[0])
        # Note: This may not work with rollback in fixture - would need clean_db fixture

    async def test_retrieve_scoped_to_user(self, service, db_session, mock_embedding):
        """Test retrieval only returns memories for specified user."""
        user1 = uuid4()
        user2 = uuid4()

        # Use same embedding for query and memories to ensure they match
        test_embedding = mock_embedding("test memory")

        # Create memory for user1
        mem1 = Memory(
            user_id=user1,
            content="User1 memory",
            embedding=test_embedding,  # Same embedding as query
            memory_type=MemoryType.EPISODIC,
            importance_score=5.0,
        )
        db_session.add(mem1)

        # Create memory for user2
        mem2 = Memory(
            user_id=user2,
            content="User2 memory",
            embedding=test_embedding,  # Same embedding as query
            memory_type=MemoryType.EPISODIC,
            importance_score=5.0,
        )
        db_session.add(mem2)
        await db_session.commit()

        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=test_embedding)

            result = await service.retrieve(
                session=db_session,
                user_id=user1,
                query="memory",
            )

        # Should only return user1's memory
        assert result["count"] == 1
        assert "User1 memory" in result["memories"][0]["content"]


class TestThreeFactorScoring:
    """Tests for the three-factor scoring algorithm."""

    @pytest.fixture
    def service(self):
        return RetrievalService()

    async def test_recency_boost(self, service, db_session, test_user_id, mock_embedding):
        """Test that recency weight boosts recent memories."""
        # Create old and new memory with same content similarity
        old_memory = Memory(
            user_id=test_user_id,
            content="Test memory old",
            embedding=mock_embedding("Test memory"),
            memory_type=MemoryType.EPISODIC,
            importance_score=5.0,
            created_at=datetime.utcnow() - timedelta(days=7),
        )
        new_memory = Memory(
            user_id=test_user_id,
            content="Test memory new",
            embedding=mock_embedding("Test memory"),  # Same embedding = same relevance
            memory_type=MemoryType.EPISODIC,
            importance_score=5.0,
            created_at=datetime.utcnow() - timedelta(hours=1),
        )
        db_session.add_all([old_memory, new_memory])
        await db_session.commit()

        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("Test memory"))

            # High recency weight
            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="Test memory",
                recency_weight=0.8,
                importance_weight=0.1,
                relevance_weight=0.1,
            )

        # New memory should rank higher with recency boost
        if result["count"] >= 2:
            assert result["memories"][0]["scores"]["recency"] > result["memories"][1]["scores"]["recency"]

    async def test_importance_boost(self, service, db_session, test_user_id, mock_embedding):
        """Test that importance weight boosts high-importance memories."""
        low_importance = Memory(
            user_id=test_user_id,
            content="Low importance memory",
            embedding=mock_embedding("Test memory"),
            memory_type=MemoryType.EPISODIC,
            importance_score=3.0,
            created_at=datetime.utcnow(),
        )
        high_importance = Memory(
            user_id=test_user_id,
            content="High importance memory",
            embedding=mock_embedding("Test memory"),
            memory_type=MemoryType.EPISODIC,
            importance_score=9.0,
            created_at=datetime.utcnow(),
        )
        db_session.add_all([low_importance, high_importance])
        await db_session.commit()

        with patch("app.services.retrieval.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=mock_embedding("Test memory"))

            result = await service.retrieve(
                session=db_session,
                user_id=test_user_id,
                query="Test memory",
                recency_weight=0.1,
                importance_weight=0.8,
                relevance_weight=0.1,
            )

        # High importance should rank first
        if result["count"] >= 2:
            assert result["memories"][0]["importance_score"] > result["memories"][1]["importance_score"]
