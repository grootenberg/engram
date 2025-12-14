"""Factory Boy factories for generating test data."""

import random
from datetime import datetime, timedelta
from uuid import uuid4

import factory
from faker import Faker

from app.models.memory import Memory, MemoryType, ObservationType, SectionType

fake = Faker()


class MemoryFactory(factory.Factory):
    """Factory for generating Memory instances."""

    class Meta:
        model = Memory

    id = factory.LazyFunction(uuid4)
    user_id = factory.LazyFunction(uuid4)
    content = factory.LazyFunction(lambda: fake.paragraph(nb_sentences=2))
    embedding = None  # Set separately with mock_embedding fixture
    memory_type = factory.LazyFunction(lambda: random.choice(list(MemoryType)))
    observation_type = factory.LazyFunction(
        lambda: random.choice(list(ObservationType)) if random.random() > 0.2 else None
    )
    section = factory.LazyFunction(
        lambda: random.choice(list(SectionType)) if random.random() > 0.3 else None
    )
    importance_score = factory.LazyFunction(lambda: round(random.uniform(3.0, 9.0), 1))
    helpful_count = factory.LazyFunction(lambda: random.randint(0, 5))
    harmful_count = factory.LazyFunction(lambda: random.randint(0, 2))
    is_synthetic = factory.LazyAttribute(lambda o: o.memory_type == MemoryType.SEMANTIC)
    citations = None
    created_at = factory.LazyFunction(
        lambda: datetime.utcnow() - timedelta(hours=random.randint(0, 168))
    )
    last_accessed_at = factory.LazyAttribute(lambda o: o.created_at)
    extra_metadata = None


class EpisodicMemoryFactory(MemoryFactory):
    """Factory for episodic memories."""

    memory_type = MemoryType.EPISODIC
    is_synthetic = False


class SemanticMemoryFactory(MemoryFactory):
    """Factory for semantic (synthesized) memories."""

    memory_type = MemoryType.SEMANTIC
    is_synthetic = True
    observation_type = ObservationType.INSIGHT
    citations = factory.LazyFunction(lambda: [str(uuid4()) for _ in range(random.randint(2, 4))])


class ErrorMemoryFactory(EpisodicMemoryFactory):
    """Factory for error observations."""

    observation_type = ObservationType.ERROR
    importance_score = factory.LazyFunction(lambda: round(random.uniform(7.0, 9.0), 1))
    section = SectionType.PITFALLS
    content = factory.LazyFunction(
        lambda: f"Error: {fake.sentence()} - {fake.catch_phrase()}"
    )


class InstructionMemoryFactory(EpisodicMemoryFactory):
    """Factory for instruction observations."""

    observation_type = ObservationType.INSTRUCTION
    importance_score = factory.LazyFunction(lambda: round(random.uniform(8.0, 10.0), 1))
    section = SectionType.PREFERENCES
    content = factory.LazyFunction(
        lambda: f"User prefers {fake.word()} over {fake.word()} for {fake.bs()}"
    )


class DecisionMemoryFactory(EpisodicMemoryFactory):
    """Factory for decision observations."""

    observation_type = ObservationType.DECISION
    importance_score = factory.LazyFunction(lambda: round(random.uniform(6.0, 8.0), 1))
    section = SectionType.CONTEXT
    content = factory.LazyFunction(
        lambda: f"Decided to use {fake.word()} because {fake.sentence()}"
    )


class CodeChangeMemoryFactory(EpisodicMemoryFactory):
    """Factory for code change observations."""

    observation_type = ObservationType.CODE_CHANGE
    importance_score = factory.LazyFunction(lambda: round(random.uniform(5.0, 7.0), 1))
    section = SectionType.SNIPPETS
    content = factory.LazyFunction(
        lambda: f"Modified {fake.file_path()} to {fake.sentence()}"
    )


# Predefined test scenarios for eval datasets
RETRIEVAL_EVAL_QUERIES = [
    {
        "query": "error handling patterns",
        "expected_types": [ObservationType.ERROR, ObservationType.INSIGHT],
        "expected_sections": [SectionType.PITFALLS, SectionType.STRATEGIES],
    },
    {
        "query": "user preferences and settings",
        "expected_types": [ObservationType.INSTRUCTION],
        "expected_sections": [SectionType.PREFERENCES],
    },
    {
        "query": "authentication and security",
        "expected_types": [ObservationType.DECISION, ObservationType.INSIGHT],
        "expected_sections": [SectionType.CONTEXT, SectionType.STRATEGIES],
    },
    {
        "query": "testing best practices",
        "expected_types": [ObservationType.INSTRUCTION, ObservationType.INSIGHT],
        "expected_sections": [SectionType.STRATEGIES],
    },
    {
        "query": "database configuration",
        "expected_types": [ObservationType.DECISION, ObservationType.CODE_CHANGE],
        "expected_sections": [SectionType.CONTEXT, SectionType.SNIPPETS],
    },
]


DEDUP_EVAL_PAIRS = [
    # Should deduplicate (high similarity)
    {
        "text_a": "User prefers TypeScript over JavaScript for new projects",
        "text_b": "User prefers TypeScript instead of JavaScript for new projects",
        "should_dedup": True,
    },
    {
        "text_a": "Always run tests before committing code",
        "text_b": "Always run the tests before you commit code",
        "should_dedup": True,
    },
    # Should NOT deduplicate (different meaning)
    {
        "text_a": "User prefers TypeScript over JavaScript",
        "text_b": "User prefers Python over JavaScript",
        "should_dedup": False,
    },
    {
        "text_a": "Error: Connection refused when database is not running",
        "text_b": "Error: Authentication failed for user postgres",
        "should_dedup": False,
    },
    # Edge cases
    {
        "text_a": "The authentication module uses JWT tokens",
        "text_b": "Authentication uses JWT tokens with 1-hour expiry",
        "should_dedup": False,  # Different detail level
    },
]


def generate_eval_corpus(mock_embedding, user_id, n_memories: int = 100) -> list[Memory]:
    """Generate a corpus of memories for evaluation.

    Args:
        mock_embedding: Function to generate deterministic embeddings.
        user_id: User ID for all memories.
        n_memories: Number of memories to generate.

    Returns:
        List of Memory instances with embeddings.
    """
    memories = []

    # Mix of different types
    factories = [
        (EpisodicMemoryFactory, 0.4),
        (ErrorMemoryFactory, 0.15),
        (InstructionMemoryFactory, 0.15),
        (DecisionMemoryFactory, 0.1),
        (CodeChangeMemoryFactory, 0.1),
        (SemanticMemoryFactory, 0.1),
    ]

    for _ in range(n_memories):
        # Select factory based on weights
        rand = random.random()
        cumulative = 0
        for factory_cls, weight in factories:
            cumulative += weight
            if rand < cumulative:
                memory = factory_cls(user_id=user_id)
                memory.embedding = mock_embedding(memory.content)
                memories.append(memory)
                break

    return memories


def generate_thematic_memories(mock_embedding, user_id, theme: str, n: int = 10) -> list[Memory]:
    """Generate memories around a specific theme for retrieval testing.

    Args:
        mock_embedding: Function to generate deterministic embeddings.
        user_id: User ID for all memories.
        theme: Theme like "authentication", "testing", "errors".
        n: Number of memories to generate.

    Returns:
        List of Memory instances related to the theme.
    """
    theme_templates = {
        "authentication": [
            "Implemented JWT authentication with refresh tokens",
            "User session expires after 1 hour of inactivity",
            "OAuth2 flow requires redirect URI configuration",
            "Password hashing uses bcrypt with 12 rounds",
            "MFA enabled for admin accounts",
        ],
        "testing": [
            "Unit tests should cover edge cases",
            "Integration tests run against test database",
            "Use pytest fixtures for test isolation",
            "Mock external API calls in tests",
            "Coverage threshold set to 80%",
        ],
        "errors": [
            "ConnectionError when database is unreachable",
            "TimeoutError for long-running queries",
            "ValidationError for invalid input data",
            "AuthenticationError when token expires",
            "RateLimitError when exceeding API limits",
        ],
        "database": [
            "PostgreSQL with pgvector for embeddings",
            "Connection pool size set to 10",
            "Alembic for migration management",
            "Index on user_id for fast lookups",
            "JSONB for flexible metadata storage",
        ],
    }

    templates = theme_templates.get(theme, theme_templates["testing"])
    memories = []

    for i in range(n):
        template = templates[i % len(templates)]
        memory = EpisodicMemoryFactory(
            user_id=user_id,
            content=f"{template} - {fake.sentence()}",
        )
        memory.embedding = mock_embedding(memory.content)
        memories.append(memory)

    return memories
