"""Configuration management for engram MCP server."""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Engram configuration settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:birdseye@localhost:5432/engram"

    # Embedding
    openai_api_key: str = ""
    engram_embedding_model: str = "text-embedding-3-small"
    engram_embedding_dimensions: int = 1536

    # Reflection LLM
    anthropic_api_key: str = ""

    # Retrieval parameters
    engram_recency_decay: float = 0.995  # Per hour, ~14-day half-life
    engram_default_limit: int = 10

    # Reflection triggers
    engram_importance_threshold: float = 150.0
    engram_observation_threshold: int = 100
    engram_time_threshold_hours: int = 24

    # Deduplication
    engram_similarity_threshold: float = 0.90

    # Importance heuristics
    importance_error: float = 9.0
    importance_instruction: float = 10.0
    importance_decision: float = 8.0
    importance_code_change: float = 7.0
    importance_insight: float = 7.0
    importance_test_result: float = 6.0
    importance_general: float = 5.0
    importance_tool_output: float = 3.0


# Global settings instance
settings = Settings()


def get_importance_heuristic(observation_type: str) -> float:
    """Get importance score based on observation type."""
    heuristics = {
        "error": settings.importance_error,
        "instruction": settings.importance_instruction,
        "decision": settings.importance_decision,
        "code_change": settings.importance_code_change,
        "insight": settings.importance_insight,
        "test_result": settings.importance_test_result,
        "general": settings.importance_general,
        "tool_output": settings.importance_tool_output,
    }
    return heuristics.get(observation_type, settings.importance_general)
