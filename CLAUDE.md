# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engram is a portable memory graph for Claude Code interactions. It provides RAG-backed semantic search, observation ingestion, and episodic-to-semantic reflection via a FastMCP server.

## Architecture

The project has two main components:

1. **MCP Server** (`mcp/`): FastMCP backend providing memory storage and retrieval
   - `server.py`: Entry point registering MCP tools (observe, retrieve, feedback, reflect, stats)
   - `app/services/`: Core business logic (embedding, ingestion, retrieval, reflection)
   - `app/tools/`: MCP tool implementations wrapping services
   - `app/models/`: SQLModel definitions for Memory and ReflectionState
   - Uses PostgreSQL + pgvector for vector similarity search

2. **Claude Code Plugin** (`plugins/engram/`): Plugin manifest, agents, skills, and hooks
   - `plugin.json`: Manifest requiring the MCP server
   - `agents/memory-manager.md`: Orchestrator agent for memory operations
   - `skills/`: Retrieval, reflection, and curation skill definitions
   - `hooks/hooks.json`: PostToolUse hooks for automatic observation capture

## Common Commands

All commands run from the `mcp/` directory:

```bash
# Start PostgreSQL + pgvector
docker-compose up -d

# Install dependencies
uv sync

# Run database migrations
uv run alembic upgrade head

# Run all tests (requires docker-compose.test.yml container)
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_models.py

# Run specific test function
uv run pytest tests/integration/test_retrieval.py::test_retrieve_basic -v

# Run with coverage
uv run pytest --cov=app --cov-report=term-missing

# Start the MCP server directly (stdio transport)
uv run python server.py
```

## Environment Variables

Create `mcp/.env` with:
- `DATABASE_URL`: PostgreSQL connection string (required)
- `OPENAI_API_KEY`: For text-embedding-3-small embeddings (required)
- `ANTHROPIC_API_KEY`: For reflection synthesis (required)

## Testing

- Unit tests in `tests/unit/` mock external dependencies
- Integration tests in `tests/integration/` use `docker-compose.test.yml` (port 5434)
- Evals in `tests/evals/` test retrieval quality and deduplication
- VCR cassettes in `tests/fixtures/cassettes/` record API responses
- Tests auto-start container if not running; truncate tables between tests

## Key Concepts

**Three-Factor Scoring**: Retrieval ranks by `α*recency + β*importance + γ*relevance`
- Recency: `0.995^hours_since_access` (~14 day half-life, uses `last_accessed_at`)
- Importance: `(base + (helpful - harmful) * 0.5) / 10`
- Relevance: Cosine similarity to query embedding
- All three factors are min-max normalized across the candidate set before weighting

**Memory Types**:
- Episodic: Raw observations (is_synthetic=false)
- Semantic: Synthesized insights from reflection (is_synthetic=true)
- Procedural: Workflow-style memories generated when reflection surfaces repeatable steps

**Observation Types**: error (9), instruction (10), decision (8), code_change (7), insight (7), test_result (6), general (5), tool_output (3) - numbers are default importance scores

**Deduplication**: Observations with >90% cosine similarity are deduplicated (helpful_count incremented, `last_accessed_at` refreshed)

**Forgetting/Compaction**: Episodic memories below an importance threshold are pruned after a TTL (defaults: 90 days, importance <4, batch limit 500). Configure via `ENGRAM_COMPACTION_*` settings or disable by setting TTL to 0.

**Reflection**: Two-stage reflection runs in the background (enqueue via `reflect`, poll via `reflect_status`), generates guiding questions, retrieves targeted episodic memories per question, and emits semantic insights and optional procedural workflows with citations.

**Reflection Triggers**: importance_sum >= 150 OR obs_count >= 100 OR hours >= 24
