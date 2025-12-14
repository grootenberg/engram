"""Embedding generation service for engram."""

from openai import AsyncOpenAI

from app.config import settings


class EmbeddingService:
    """Service for generating text embeddings using OpenAI."""

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        response = await self.client.embeddings.create(
            model=settings.engram_embedding_model,
            input=text,
            dimensions=settings.engram_embedding_dimensions,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        response = await self.client.embeddings.create(
            model=settings.engram_embedding_model,
            input=texts,
            dimensions=settings.engram_embedding_dimensions,
        )
        # Return embeddings in the same order as input texts
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# Global singleton instance
embedding_service = EmbeddingService()
