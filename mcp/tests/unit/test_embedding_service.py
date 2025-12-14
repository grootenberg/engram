"""Unit tests for embedding service with VCR cassettes."""

import os
from pathlib import Path

import pytest

from app.services.embedding import EmbeddingService

# Check if we have cassettes or real API key for recording
CASSETTES_DIR = Path(__file__).parent.parent / "fixtures" / "cassettes"


def has_real_openai_key() -> bool:
    """Check if a real OpenAI API key is available."""
    key = os.getenv("OPENAI_API_KEY")
    return bool(key) and key != "test-openai-key"


def cassette_exists(name: str) -> bool:
    """Check if a VCR cassette exists."""
    return (CASSETTES_DIR / f"{name}.yaml").exists()


class TestEmbeddingService:
    """Tests for the EmbeddingService."""

    @pytest.fixture
    def service(self):
        """Create embedding service instance."""
        return EmbeddingService()

    async def test_embed_batch_empty(self, service):
        """Test embedding empty list returns empty list."""
        embeddings = await service.embed_batch([])
        assert embeddings == []

    def test_client_lazy_initialization(self, service):
        """Test OpenAI client is lazily initialized."""
        assert service._client is None
        _ = service.client
        assert service._client is not None


class TestEmbeddingServiceMocked:
    """Tests using mock embeddings (no API calls)."""

    async def test_mock_embedding_deterministic(self, mock_embedding):
        """Test mock embedding produces consistent results."""
        text = "Test content"
        emb1 = mock_embedding(text)
        emb2 = mock_embedding(text)

        assert emb1 == emb2

    async def test_mock_embedding_different_texts(self, mock_embedding):
        """Test different texts produce different embeddings."""
        emb1 = mock_embedding("First text")
        emb2 = mock_embedding("Second text")

        assert emb1 != emb2

    async def test_mock_embedding_dimension(self, mock_embedding):
        """Test mock embedding has correct dimension."""
        emb = mock_embedding("Any text")
        assert len(emb) == 1536

    async def test_mock_embedding_normalized(self, mock_embedding):
        """Test mock embedding is unit normalized."""
        emb = mock_embedding("Any text")
        norm = sum(x**2 for x in emb) ** 0.5
        assert abs(norm - 1.0) < 0.001
