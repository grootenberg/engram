# engram

A portable memory graph for Claude Code interactions. Engram provides semantic search, observation ingestion, and episodic-to-semantic reflection via a RAG-backed memory system.

## Overview

Engram enables Claude Code to maintain persistent memory across sessions through:

- **Observation Ingestion** - Store valuable insights, decisions, errors, and patterns
- **Semantic Retrieval** - Three-factor scoring combining recency, importance, and relevance
- **Episodic-to-Semantic Reflection** - Synthesize raw observations into lasting insights
- **Quality Curation** - Feedback loops to calibrate memory importance over time

## Project Structure

```
engram/
├── .claude-plugin/          # Plugin marketplace manifest
│   └── marketplace.json
├── mcp/                     # FastMCP server (memory backend)
│   ├── app/                 # Application code
│   ├── server.py            # MCP server entry point
│   └── docker-compose.yml   # PostgreSQL + pgvector
└── plugins/
    └── engram/              # Claude Code plugin
        ├── plugin.json      # Plugin manifest
        ├── agents/          # Memory manager agent
        ├── skills/          # Retrieval, reflection, curation skills
        └── hooks/           # Automatic memory capture hooks
```

## Quick Start

### 1. Start the Database

```bash
cd mcp
docker-compose up -d
```

### 2. Install Dependencies

```bash
cd mcp
uv sync
```

### 3. Run Database Migrations

```bash
cd mcp
uv run alembic upgrade head
```

### 4. Configure Environment

Create `mcp/.env` with your API keys:

```env
OPENAI_API_KEY=sk-...          # Required for embeddings
ANTHROPIC_API_KEY=sk-ant-...   # Required for reflection
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5442/engram
```

### 5. Install the Plugin

Add this repository to your Claude Code plugins. The plugin will automatically start the MCP server when needed.

## MCP Tools

| Tool | Description |
|------|-------------|
| `observe` | Ingest observations with automatic deduplication |
| `retrieve` | Search memories with three-factor scoring |
| `feedback` | Provide helpful/harmful signals on memories |
| `reflect` | Enqueue two-stage reflection (background) that synthesizes semantic insights and procedural workflows |
| `reflect_status` | Poll the status/result of a queued reflection job |
| `stats` | Get memory system statistics and health metrics |

## Memory Types

- **Episodic** - Raw observations from sessions
- **Semantic** - Synthesized insights from reflection
- **Procedural** - Workflow and how-to knowledge

## Observation Types

| Type | Default Importance |
|------|-------------------|
| instruction | 10 |
| error | 9 |
| decision | 8 |
| code_change | 7 |
| insight | 7 |
| test_result | 6 |
| general | 5 |
| tool_output | 3 |

## Documentation

- [MCP Server Installation & Usage](mcp/README.md)
- [Memory Manager Agent](plugins/engram/agents/memory-manager.md)
- [Retrieval Skill](plugins/engram/skills/memory-retrieval/SKILL.md)
- [Reflection Skill](plugins/engram/skills/memory-reflection/SKILL.md)
- [Curation Skill](plugins/engram/skills/memory-curation/SKILL.md)

## License

See [LICENSE](LICENSE) for details.
