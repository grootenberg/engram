"""Integration tests for ingestion service and observe tool."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.memory import Memory, MemoryType, ObservationType
from app.models.reflection_state import ReflectionState
from app.services.ingestion import IngestionService


class TestIngestionService:
    """Integration tests for IngestionService."""

    @pytest.fixture
    def service(self):
        """Create ingestion service instance."""
        return IngestionService()

    @pytest.fixture
    def mock_embed(self, mock_embedding):
        """Patch embedding service with mock."""
        with patch("app.services.ingestion.embedding_service") as mock:
            mock.embed = AsyncMock(side_effect=lambda text: mock_embedding(text))
            yield mock

    async def test_ingest_creates_memory(self, service, db_session, test_user_id, mock_embed):
        """Test basic ingestion creates a memory record."""
        result = await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="User prefers TypeScript over JavaScript",
            observation_type="instruction",
        )

        assert result["status"] == "created"
        assert "memory_id" in result
        assert result["importance_score"] == 10.0  # instruction default

        # Verify in database
        memories = (await db_session.execute(
            select(Memory).where(Memory.user_id == test_user_id)
        )).scalars().all()
        assert len(memories) == 1
        assert memories[0].content == "User prefers TypeScript over JavaScript"
        assert memories[0].memory_type == MemoryType.EPISODIC

    async def test_ingest_with_custom_importance(self, service, db_session, test_user_id, mock_embed):
        """Test ingestion with custom importance override."""
        result = await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="Some observation",
            observation_type="general",
            importance=8.5,
        )

        assert result["status"] == "created"
        assert result["importance_score"] == 8.5

    async def test_ingest_with_section(self, service, db_session, test_user_id, mock_embed):
        """Test ingestion with section classification."""
        result = await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="Always use async/await",
            observation_type="instruction",
            section="strategies",
        )

        assert result["status"] == "created"

        memory = (await db_session.execute(
            select(Memory).where(Memory.id == result["memory_id"])
        )).scalar_one()
        assert memory.section.value == "strategies"

    async def test_ingest_with_metadata(self, service, db_session, test_user_id, mock_embed):
        """Test ingestion with extra metadata."""
        metadata = {"source": "conversation", "session_id": "abc123"}
        result = await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="Some observation",
            metadata=metadata,
        )

        assert result["status"] == "created"

        memory = (await db_session.execute(
            select(Memory).where(Memory.id == result["memory_id"])
        )).scalar_one()
        assert memory.extra_metadata == metadata

    async def test_ingest_updates_reflection_state(self, service, db_session, test_user_id, mock_embed):
        """Test ingestion updates reflection trigger state."""
        # Ingest first observation
        await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="First observation",
            observation_type="instruction",  # importance 10
        )

        state = (await db_session.execute(
            select(ReflectionState).where(ReflectionState.user_id == test_user_id)
        )).scalar_one()

        assert state.accumulated_importance == 10.0
        assert state.observations_since == 1

        # Ingest second observation
        await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="Second observation",
            observation_type="error",  # importance 9
        )

        await db_session.refresh(state)
        assert state.accumulated_importance == 19.0
        assert state.observations_since == 2

    async def test_ingest_importance_clamping(self, service, db_session, test_user_id, mock_embed):
        """Test importance is clamped to valid range."""
        result_high = await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="High importance",
            importance=15.0,  # Should be clamped to 10
        )
        assert result_high["importance_score"] == 10.0

        result_low = await service.ingest(
            session=db_session,
            user_id=test_user_id,
            content="Low importance",
            importance=-5.0,  # Should be clamped to 1
        )
        assert result_low["importance_score"] == 1.0


class TestDeduplication:
    """Tests for deduplication behavior."""

    @pytest.fixture
    def service(self):
        return IngestionService()

    async def test_dedup_rejects_similar_content(self, service, db_session, test_user_id, mock_embedding):
        """Test that very similar content is deduplicated."""
        # This test requires real embeddings to work properly
        # For unit testing, we use a mock that returns identical embeddings for similar text

        # Create a mock that returns same embedding for similar texts
        base_embedding = mock_embedding("User prefers TypeScript")

        with patch("app.services.ingestion.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=base_embedding)

            # First ingestion
            result1 = await service.ingest(
                session=db_session,
                user_id=test_user_id,
                content="User prefers TypeScript over JavaScript",
            )
            assert result1["status"] == "created"

            # Commit to make visible for duplicate check
            await db_session.commit()

            # Second ingestion with same embedding (simulating identical content)
            result2 = await service.ingest(
                session=db_session,
                user_id=test_user_id,
                content="User prefers TypeScript instead of JavaScript",  # Very similar
            )

            # Should be deduplicated since mock returns same embedding
            assert result2["status"] == "deduplicated"
            assert "existing_id" in result2

    async def test_dedup_allows_different_content(self, service, db_session, test_user_id, mock_embedding):
        """Test that different content is not deduplicated."""
        with patch("app.services.ingestion.embedding_service") as mock:
            # Return different embeddings for different texts
            mock.embed = AsyncMock(side_effect=lambda text: mock_embedding(text))

            result1 = await service.ingest(
                session=db_session,
                user_id=test_user_id,
                content="User prefers TypeScript over JavaScript",
            )
            assert result1["status"] == "created"

            await db_session.commit()

            result2 = await service.ingest(
                session=db_session,
                user_id=test_user_id,
                content="Database uses PostgreSQL with pgvector",
            )
            # Different content should not be deduplicated
            assert result2["status"] == "created"

    async def test_dedup_scoped_to_user(self, service, db_session, mock_embedding):
        """Test deduplication is scoped to user."""
        user1 = uuid4()
        user2 = uuid4()

        base_embedding = mock_embedding("Same content")

        with patch("app.services.ingestion.embedding_service") as mock:
            mock.embed = AsyncMock(return_value=base_embedding)

            result1 = await service.ingest(
                session=db_session,
                user_id=user1,
                content="Same content for both users",
            )
            assert result1["status"] == "created"

            await db_session.commit()

            # Same content but different user - should NOT be deduplicated
            result2 = await service.ingest(
                session=db_session,
                user_id=user2,
                content="Same content for both users",
            )
            assert result2["status"] == "created"
