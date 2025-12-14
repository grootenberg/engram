"""Pytest configuration and fixtures for engram tests."""

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

# Set test environment variables BEFORE importing app modules
# This ensures the Settings singleton loads with test values
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:test@localhost:5434/engram_test"
TEST_DATABASE_URL_SYNC = "postgresql://postgres:test@localhost:5434/engram_test"
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.memory import Memory, MemoryType, ObservationType, SectionType
from app.models.reflection_state import ReflectionState

# Path to cassettes
CASSETTES_DIR = Path(__file__).parent / "fixtures" / "cassettes"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def docker_compose_file():
    """Path to test docker-compose file."""
    return Path(__file__).parent.parent / "docker-compose.test.yml"


@pytest.fixture(scope="session")
def postgres_container(docker_compose_file):
    """Start postgres container for tests, run migrations, then tear down."""
    # Check if container is already running
    result = subprocess.run(
        ["docker", "exec", "engram-postgres-test", "pg_isready", "-U", "postgres"],
        capture_output=True,
    )
    container_was_running = result.returncode == 0

    if container_was_running:
        # Container already running, assume migrations are done
        yield
        return

    # Start container
    subprocess.run(
        ["docker", "compose", "-f", str(docker_compose_file), "up", "-d", "--wait"],
        check=True,
        capture_output=True,
    )

    # Wait for postgres to be ready
    max_retries = 30
    for i in range(max_retries):
        result = subprocess.run(
            ["docker", "exec", "engram-postgres-test", "pg_isready", "-U", "postgres"],
            capture_output=True,
        )
        if result.returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Postgres container failed to start")

    # Enable pgvector extension
    subprocess.run(
        [
            "docker", "exec", "engram-postgres-test",
            "psql", "-U", "postgres", "-d", "engram_test",
            "-c", "CREATE EXTENSION IF NOT EXISTS vector;"
        ],
        check=True,
        capture_output=True,
    )

    # Run migrations
    env = os.environ.copy()
    env["DATABASE_URL"] = TEST_DATABASE_URL
    subprocess.run(
        [".venv/bin/alembic", "upgrade", "head"],
        check=True,
        capture_output=True,
        cwd=Path(__file__).parent.parent,
        env=env,
    )

    yield

    # Tear down container
    subprocess.run(
        ["docker", "compose", "-f", str(docker_compose_file), "down", "-v"],
        check=True,
        capture_output=True,
    )


@pytest.fixture
async def db_engine(postgres_container):
    """Create async engine for test database."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async session with auto-cleanup for test isolation.

    Tests can call commit() freely. Tables are truncated after each test.
    """
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session

    # Clean up after test
    async with db_engine.begin() as conn:
        await conn.execute(text("TRUNCATE memories, reflection_state CASCADE"))


@pytest.fixture
async def clean_db(db_engine):
    """Clean all data from tables before test (for integration tests that need commits)."""
    async with db_engine.begin() as conn:
        await conn.execute(text("TRUNCATE memories, reflection_state CASCADE"))
    yield
    # Clean up after test too
    async with db_engine.begin() as conn:
        await conn.execute(text("TRUNCATE memories, reflection_state CASCADE"))


@pytest.fixture
def test_user_id():
    """Generate a unique user ID for test isolation."""
    return uuid4()


@pytest.fixture
def sample_memory(test_user_id) -> Memory:
    """Create a sample memory for testing."""
    return Memory(
        user_id=test_user_id,
        content="User prefers TypeScript over JavaScript for new projects",
        memory_type=MemoryType.EPISODIC,
        observation_type=ObservationType.INSTRUCTION,
        section=SectionType.PREFERENCES,
        importance_score=8.0,
    )


@pytest.fixture
def sample_memories(test_user_id) -> list[Memory]:
    """Create a batch of sample memories for testing."""
    return [
        Memory(
            user_id=test_user_id,
            content="Always run tests before committing code",
            memory_type=MemoryType.EPISODIC,
            observation_type=ObservationType.INSTRUCTION,
            importance_score=9.0,
            section=SectionType.STRATEGIES,
        ),
        Memory(
            user_id=test_user_id,
            content="The authentication module uses JWT tokens with 1-hour expiry",
            memory_type=MemoryType.EPISODIC,
            observation_type=ObservationType.DECISION,
            importance_score=7.0,
            section=SectionType.CONTEXT,
        ),
        Memory(
            user_id=test_user_id,
            content="Error: Connection refused when database is not running",
            memory_type=MemoryType.EPISODIC,
            observation_type=ObservationType.ERROR,
            importance_score=8.0,
            section=SectionType.PITFALLS,
        ),
        Memory(
            user_id=test_user_id,
            content="Use async/await consistently throughout the codebase",
            memory_type=MemoryType.SEMANTIC,
            observation_type=ObservationType.INSIGHT,
            importance_score=7.0,
            is_synthetic=True,
        ),
        Memory(
            user_id=test_user_id,
            content="pytest-asyncio requires asyncio_mode = 'auto' in pyproject.toml",
            memory_type=MemoryType.EPISODIC,
            observation_type=ObservationType.CODE_CHANGE,
            importance_score=5.0,
            section=SectionType.SNIPPETS,
        ),
    ]


@pytest.fixture
def mock_embedding():
    """Generate a deterministic mock embedding vector."""
    import hashlib

    def _mock_embed(text: str) -> list[float]:
        # Generate deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Create 1536-dim vector from hash (repeating pattern)
        embedding = []
        for i in range(1536):
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1] range
            embedding.append((hash_bytes[byte_idx] - 128) / 128)
        # Normalize to unit vector
        norm = sum(x**2 for x in embedding) ** 0.5
        return [x / norm for x in embedding]

    return _mock_embed


@pytest.fixture
def vcr_config():
    """VCR configuration for recording HTTP interactions."""
    return {
        "cassette_library_dir": str(CASSETTES_DIR),
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path"],
        "filter_headers": [
            "authorization",
            "x-api-key",
            "anthropic-version",
            "openai-organization",
        ],
        "filter_post_data_parameters": ["api_key"],
        "decode_compressed_response": True,
    }


# Note: Test environment variables are set at module import time (top of file)
# to ensure Settings singleton loads with test values before any app imports.
