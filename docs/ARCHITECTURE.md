# Engram Architecture

> System architecture for engram - a portable memory graph for Claude Code interactions with RAG-backed semantic search and ingestion via FastMCP.

**Version:** 0.2.0
**Last Updated:** 2025-12-09
**Status:** Implementation Phase

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [MCP Tools](#mcp-tools)
4. [Component Architecture](#component-architecture)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Memory Lifecycle](#memory-lifecycle)
7. [Plugin Architecture](#plugin-architecture)
8. [Deployment Architecture](#deployment-architecture)
9. [Data Model](#data-model)

---

## System Overview

Engram is a memory system for Claude Code that enables:

- **Persistent Memory**: Remember insights across sessions
- **Semantic Search**: Find relevant past context using RAG
- **Self-Improvement**: Learn from feedback to improve retrieval
- **Reflection**: Synthesize raw observations into higher-level insights
- **Proactive Capture**: Automatically observe significant tool results

### Design Principles

| Principle | Description | Research Source |
|-----------|-------------|-----------------|
| **Append, Don't Overwrite** | Never delete memories; promote via reflection | All three papers |
| **Three-Factor Retrieval** | Score by recency + importance + relevance | Generative Agents |
| **Delta Updates** | Incremental changes prevent context collapse | ACE |
| **Modular Memory** | Separate episodic/semantic/procedural | CoALA |
| **Feedback-Driven** | Helpful/harmful counters improve over time | ACE |

### v1 Scope

**Included:**
- Episodic memory (append-only observations)
- Semantic memory (reflection-generated insights)
- Three-factor retrieval scoring
- Feedback loop for importance adjustment
- PostToolUse hooks for proactive capture

**Deferred to v2:**
- Session management (working memory lifecycle)
- Session-scoped retrieval
- Procedural memory extraction

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLAUDE CODE                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Claude Code Session                          │    │
│  │                                                                      │    │
│  │   User ──▶ Claude ──▶ Tool Calls ──▶ Results ──▶ Response           │    │
│  │                           │              │                           │    │
│  └───────────────────────────┼──────────────┼───────────────────────────┘    │
│                              │              │                                │
│  ┌───────────────────────────┼──────────────┼───────────────────────────┐    │
│  │                    ENGRAM PLUGIN         │                            │    │
│  │                           │              │                            │    │
│  │   ┌───────────────────────┼──────────────┼─────────────────────────┐ │    │
│  │   │                       ▼              │                         │ │    │
│  │   │  ┌─────────────────────────────┐    │ PostToolUse Hook        │ │    │
│  │   │  │      memory-manager         │◀───┘ (Proactive Capture)     │ │    │
│  │   │  │      (Orchestrator)         │                               │ │    │
│  │   │  └────────────┬────────────────┘                               │ │    │
│  │   │               │                                                 │ │    │
│  │   │   ┌───────────┼───────────┐                                    │ │    │
│  │   │   ▼           ▼           ▼                                    │ │    │
│  │   │ ┌─────────┐ ┌─────────┐ ┌─────────┐                           │ │    │
│  │   │ │retrieval│ │reflection│ │curation │  Skills                  │ │    │
│  │   │ │  skill  │ │  skill  │ │  skill  │                           │ │    │
│  │   │ └─────────┘ └─────────┘ └─────────┘                           │ │    │
│  │   │                                                                 │ │    │
│  │   └─────────────────────────────────────────────────────────────────┘ │    │
│  │                              │                                        │    │
│  └──────────────────────────────┼────────────────────────────────────────┘    │
│                                 │                                             │
└─────────────────────────────────┼─────────────────────────────────────────────┘
                                  │
                                  │    MCP Protocol (stdio)
                                  │
┌─────────────────────────────────┼─────────────────────────────────────────────┐
│                                 ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                       MEMOCC SERVER (FastMCP)                         │    │
│  │                                                                       │    │
│  │   ┌─────────────────────────────────────────────────────────────┐    │    │
│  │   │                      MCP TOOLS                               │    │    │
│  │   │                                                              │    │    │
│  │   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │    │
│  │   │  │ memory   │ │ memory   │ │ memory   │ │ memory   │       │    │    │
│  │   │  │ _observe │ │_retrieve │ │ _reflect │ │_feedback │       │    │    │
│  │   │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │    │    │
│  │   │       │            │            │            │              │    │    │
│  │   │       │            │   ┌──────────┐          │              │    │    │
│  │   │       │            │   │ memory   │          │              │    │    │
│  │   │       │            │   │ _stats   │          │              │    │    │
│  │   │       │            │   └────┬─────┘          │              │    │    │
│  │   └───────┼────────────┼────────┼────────────────┼──────────────┘    │    │
│  │           │            │        │                │                   │    │
│  │   ┌───────┼────────────┼────────┼────────────────┼──────────────┐    │    │
│  │   │       ▼            ▼        ▼                ▼              │    │    │
│  │   │  ┌─────────────────────────────────────────────────────┐   │    │    │
│  │   │  │                  CORE SERVICES                       │   │    │    │
│  │   │  │                                                      │   │    │    │
│  │   │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │    │    │
│  │   │  │  │ Embedding  │  │ Retrieval  │  │ Reflection │    │   │    │    │
│  │   │  │  │  Service   │  │  Engine    │  │   Engine   │    │   │    │    │
│  │   │  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │   │    │    │
│  │   │  │        │               │               │            │   │    │    │
│  │   │  │  ┌────────────┐  ┌────────────┐                    │   │    │    │
│  │   │  │  │ Ingestion  │  │  Curation  │                    │   │    │    │
│  │   │  │  │  Pipeline  │  │  Service   │                    │   │    │    │
│  │   │  │  └─────┬──────┘  └─────┬──────┘                    │   │    │    │
│  │   │  │        │               │                            │   │    │    │
│  │   │  └────────┼───────────────┼────────────────────────────┘   │    │    │
│  │   │           │               │                                │    │    │
│  │   │   ┌───────┴───────────────┴───────────────────────┐       │    │    │
│  │   │   │              MEMORY STORE                      │       │    │    │
│  │   │   │                                                │       │    │    │
│  │   │   │  ┌──────────────────┐  ┌──────────────────┐   │       │    │    │
│  │   │   │  │ Episodic Memory  │  │ Semantic Memory  │   │       │    │    │
│  │   │   │  │   (Postgres +    │  │   (Postgres +    │   │       │    │    │
│  │   │   │  │    pgvector)     │  │    pgvector)     │   │       │    │    │
│  │   │   │  └──────────────────┘  └──────────────────┘   │       │    │    │
│  │   │   │                                                │       │    │    │
│  │   │   └────────────────────────────────────────────────┘       │    │    │
│  │   │                                                             │    │    │
│  │   │                    DATA LAYER                               │    │    │
│  │   └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│                            MEMOCC                                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## MCP Tools

### Tool Overview

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `memory_observe` | Append-only observation ingestion | content, observation_type, importance, section |
| `memory_retrieve` | Three-factor scored retrieval | query, limit, memory_types, weights |
| `memory_reflect` | Episodic → semantic synthesis | focus, max_insights, force |
| `memory_feedback` | Helpful/harmful signals | memory_id, helpful, reason |
| `memory_stats` | Debugging/inspection | scope (summary/detailed/reflection) |

### Tool Signatures

```python
@mcp.tool()
async def memory_observe(
    content: str,                          # Observation text (1-10000 chars)
    observation_type: str = "general",     # error|instruction|code_change|decision|test_result|tool_output|insight|general
    importance: float | None = None,       # Override auto-computed (1.0-10.0)
    section: str | None = None,            # strategies|snippets|pitfalls|context|preferences
    metadata: dict | None = None,          # Additional key-value pairs
) -> dict:
    """
    Ingest observation into memory stream (append-only with dedup).

    Returns:
        {
            "id": str,                     # UUID of created/matched memory
            "action": "created" | "deduplicated",
            "importance": float,           # Final importance score
            "similarity": float | None,    # If deduplicated
            "reflection_pending": bool     # Whether threshold is met
        }
    """

@mcp.tool()
async def memory_retrieve(
    query: str,                            # Search query (1-1000 chars)
    limit: int = 10,                       # Max results (1-100)
    memory_types: list[str] | None = None, # episodic|semantic|procedural
    min_importance: float = 0.0,           # Filter threshold
    include_synthetic: bool = True,        # Include reflections
    recency_weight: float = 0.33,          # Weight factors
    importance_weight: float = 0.33,
    relevance_weight: float = 0.33,
) -> dict:
    """
    Three-factor scored retrieval (recency + importance + relevance).

    Returns:
        {
            "memories": [...],             # Ranked results
            "query_embedding_time_ms": float,
            "retrieval_time_ms": float,
            "total_candidates": int
        }
    """

@mcp.tool()
async def memory_reflect(
    focus: str | None = None,              # Topic focus (optional)
    max_insights: int = 5,                 # Max insights (1-20)
    force: bool = False,                   # Override trigger check
) -> dict:
    """
    Synthesize episodic memories into semantic insights.

    Triggers: importance_sum >= 150 OR obs_count >= 100 OR hours >= 24

    Returns:
        {
            "triggered": bool,
            "reason": str,
            "insights": [...],             # New semantic memories
            "questions_generated": [str],
            "memories_processed": int,
            "trigger_state": {...}
        }
    """

@mcp.tool()
async def memory_feedback(
    memory_id: str,                        # UUID of memory
    helpful: bool,                         # Usefulness signal
    reason: str | None = None,             # Optional explanation
) -> dict:
    """
    Provide helpful/harmful feedback on a memory.

    Effect: effective_importance = base + (helpful - harmful) * 0.5

    Returns:
        {
            "id": str,
            "helpful_count": int,
            "harmful_count": int,
            "base_importance": float,
            "effective_importance": float
        }
    """

@mcp.tool()
async def memory_stats(
    scope: str = "summary",                # summary|detailed|reflection
) -> dict:
    """
    Get memory system statistics and health metrics.

    Returns vary by scope - see implementation for details.
    """
```

---

## Component Architecture

### Memory System Components (CoALA-Inspired, v1 Simplified)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENGRAM MEMORY SYSTEM (v1)                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        LONG-TERM MEMORY                                 │ │
│  │                                                                         │ │
│  │  ┌───────────────────────────────┐  ┌───────────────────────────────┐  │ │
│  │  │      EPISODIC MEMORY          │  │      SEMANTIC MEMORY          │  │ │
│  │  │      (Event Stream)           │  │      (Knowledge Base)         │  │ │
│  │  │                               │  │                               │  │ │
│  │  │  • Raw observations           │  │  • Synthesized insights       │  │ │
│  │  │  • Tool results               │  │  • Patterns & preferences     │  │ │
│  │  │  • Errors & decisions         │  │  • Reflections with citations │  │ │
│  │  │                               │  │                               │  │ │
│  │  │  Retention: 30 days + decay   │  │  Retention: Permanent         │  │ │
│  │  │  is_synthetic: FALSE          │  │  is_synthetic: TRUE           │  │ │
│  │  │                               │  │                               │  │ │
│  │  └──────────────┬────────────────┘  └──────────────▲────────────────┘  │ │
│  │                 │                                   │                   │ │
│  │                 │          reflection               │                   │ │
│  │                 └───────────────────────────────────┘                   │ │
│  │                                                                         │ │
│  │  Storage: Postgres + pgvector                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      WORKING MEMORY (v2 - Deferred)                     │ │
│  │                                                                         │ │
│  │  Session-scoped ephemeral memory with auto-promotion on session end.   │ │
│  │  Not implemented in v1 - observations go directly to episodic.         │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Core Services Architecture (ACE-Inspired)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE SERVICES                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      INGESTION PIPELINE                                  ││
│  │                                                                          ││
│  │   Observation ──▶ ┌──────────────┐ ──▶ ┌──────────────┐ ──▶ Memory     ││
│  │                   │  GENERATOR   │     │   CURATOR    │     Store       ││
│  │                   │              │     │              │                  ││
│  │                   │ • Classify   │     │ • Dedup      │                  ││
│  │                   │ • Score      │     │ (≥0.90 sim)  │                  ││
│  │                   │ • Embed      │     │ • Delta      │                  ││
│  │                   └──────────────┘     └──────────────┘                  ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      RETRIEVAL ENGINE                                    ││
│  │                                                                          ││
│  │   Query ──▶ ┌────────────┐ ──▶ ┌────────────┐ ──▶ ┌────────────┐       ││
│  │             │   EMBED    │     │   SCORE    │     │   RANK     │       ││
│  │             │            │     │            │     │            │       ││
│  │             │ • Encode   │     │ • Recency  │     │ • Normalize│       ││
│  │             │   query    │     │ • Import.  │     │ • Top-K    │       ││
│  │             │            │     │ • Relevance│     │ • Format   │       ││
│  │             └────────────┘     └────────────┘     └────────────┘       ││
│  │                                                                          ││
│  │   Score = α(recency) + β(importance) + γ(relevance)                     ││
│  │                                                                          ││
│  │   recency = 0.995 ^ hours_since_access                                  ││
│  │   importance = (base + (helpful - harmful) * 0.5) / 10                  ││
│  │   relevance = 1 - cosine_distance(query, memory)                        ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      REFLECTION ENGINE                                   ││
│  │                                                                          ││
│  │   Trigger ──▶ ┌──────────────┐ ──▶ ┌──────────────┐ ──▶ Semantic       ││
│  │   Check       │  QUESTION    │     │   INSIGHT    │     Memory          ││
│  │               │  GENERATOR   │     │  EXTRACTOR   │                     ││
│  │               │              │     │              │                     ││
│  │               │ • Retrieve   │     │ • Retrieve   │                     ││
│  │               │   recent     │     │   relevant   │                     ││
│  │               │ • Generate   │     │ • Synthesize │                     ││
│  │               │   questions  │     │ • Cite       │                     ││
│  │               └──────────────┘     └──────────────┘                     ││
│  │                                                                          ││
│  │   Triggers: importance_sum > 150 OR obs_count > 100 OR hours > 24       ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### Memory Observe Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY OBSERVE FLOW                                   │
│                                                                              │
│   ┌──────────────┐                                                          │
│   │ Observation  │                                                          │
│   │ (text, type) │                                                          │
│   └──────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 1. CLASSIFY & SCORE                                               │     │
│   │                                                                   │     │
│   │    importance = HEURISTICS[observation_type]                      │     │
│   │    ┌────────────────────────────────────────┐                    │     │
│   │    │ error: 9        instruction: 10       │                    │     │
│   │    │ decision: 8     code_change: 7        │                    │     │
│   │    │ insight: 7      test_result: 6        │                    │     │
│   │    │ general: 5      tool_output: 3        │                    │     │
│   │    └────────────────────────────────────────┘                    │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 2. GENERATE EMBEDDING                                             │     │
│   │                                                                   │     │
│   │    embedding = embed(observation)  # text-embedding-3-small       │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 3. CHECK FOR DUPLICATES                                           │     │
│   │                                                                   │     │
│   │    similar = SELECT * FROM memories                               │     │
│   │              WHERE 1 - (embedding <=> $new) >= 0.90               │     │
│   │              AND memory_type = 'episodic'                         │     │
│   │              LIMIT 1                                              │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                       │
│                    │                               │                       │
│              similar found?                  no duplicate                  │
│                    │                               │                       │
│                    ▼                               ▼                       │
│   ┌────────────────────────────┐   ┌────────────────────────────┐         │
│   │ 4a. UPDATE EXISTING        │   │ 4b. CREATE NEW             │         │
│   │                            │   │                            │         │
│   │ UPDATE memories            │   │ INSERT INTO memories       │         │
│   │ SET helpful_count += 1     │   │ (content, embedding,       │         │
│   │ WHERE id = $similar_id     │   │  importance, type, ...)    │         │
│   │                            │   │                            │         │
│   │ Return: action="deduplicated" │ │ Return: action="created"   │         │
│   └────────────────────────────┘   └────────────────────────────┘         │
│                    │                               │                       │
│                    └───────────────┬───────────────┘                       │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 5. UPDATE REFLECTION TRIGGER STATE                                │     │
│   │                                                                   │     │
│   │    accumulated_importance += importance                           │     │
│   │    observations_since_last += 1                                   │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Retrieve Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY RETRIEVE FLOW                                 │
│                                                                              │
│   ┌──────────┐                                                              │
│   │  User    │                                                              │
│   │  Query   │                                                              │
│   └────┬─────┘                                                              │
│        │                                                                     │
│        ▼                                                                     │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 1. EMBED QUERY                                                    │     │
│   │    query_embedding = embed(query)                                 │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 2. RETRIEVE CANDIDATES                                            │     │
│   │    SELECT * FROM memories                                         │     │
│   │    WHERE user_id = $1                                             │     │
│   │    ORDER BY embedding <=> $query_embedding                        │     │
│   │    LIMIT 100                                                      │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 3. COMPUTE THREE-FACTOR SCORES                                    │     │
│   │                                                                   │     │
│   │    For each memory:                                               │     │
│   │    ┌─────────────────────────────────────────────────────────┐   │     │
│   │    │ recency = 0.995 ^ hours_since_access                    │   │     │
│   │    │ importance = (base + (helpful - harmful) * 0.5) / 10    │   │     │
│   │    │ relevance = 1 - cosine_distance(query, memory)          │   │     │
│   │    └─────────────────────────────────────────────────────────┘   │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 4. NORMALIZE & COMBINE                                            │     │
│   │                                                                   │     │
│   │    norm(x) = (x - min) / (max - min)                             │     │
│   │    score = w_r*norm(recency) + w_i*norm(importance)              │     │
│   │          + w_v*norm(relevance)                                   │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 5. RANK & RETURN TOP-K                                            │     │
│   │                                                                   │     │
│   │    results = sorted(memories, key=score, reverse=True)[:k]       │     │
│   │    update last_accessed_at for returned memories                  │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│                            ┌──────────────┐                                 │
│                            │   Results    │                                 │
│                            │   (Top-K)    │                                 │
│                            └──────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Reflection Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REFLECTION FLOW                                     │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ TRIGGER CHECK                                                     │     │
│   │                                                                   │     │
│   │ should_reflect = (                                                │     │
│   │     accumulated_importance >= 150 OR                              │     │
│   │     observations_since_last >= 100 OR                             │     │
│   │     hours_since_last_reflection >= 24 OR                          │     │
│   │     force_flag == True                                            │     │
│   │ )                                                                 │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                              if should_reflect                              │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 1. RETRIEVE RECENT EPISODIC MEMORIES                              │     │
│   │                                                                   │     │
│   │    recent = SELECT * FROM memories                                │     │
│   │             WHERE memory_type = 'episodic'                        │     │
│   │             AND is_synthetic = FALSE                              │     │
│   │             ORDER BY created_at DESC                              │     │
│   │             LIMIT 100                                             │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 2. GENERATE REFLECTION QUESTIONS (LLM)                            │     │
│   │                                                                   │     │
│   │    "Given these observations, what are 3 high-level               │     │
│   │     questions about patterns, learnings, or insights?"            │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 3. FOR EACH QUESTION: RETRIEVE & SYNTHESIZE                       │     │
│   │                                                                   │     │
│   │    for question in questions:                                     │     │
│   │        relevant = memory_retrieve(question, limit=50)             │     │
│   │                                                                   │     │
│   │        insights = LLM: "Based on these memories, provide          │     │
│   │                   up to 5 insights with citations."               │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 4. STORE INSIGHTS AS SEMANTIC MEMORIES                            │     │
│   │                                                                   │     │
│   │    for insight in insights:                                       │     │
│   │        INSERT INTO memories (                                     │     │
│   │            content = insight.text,                                │     │
│   │            memory_type = 'semantic',                              │     │
│   │            is_synthetic = TRUE,                                   │     │
│   │            citations = insight.cited_memory_ids,                  │     │
│   │            importance_score = 8                                   │     │
│   │        )                                                          │     │
│   └────────────────────────────────┬─────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │ 5. RESET TRIGGER STATE                                            │     │
│   │                                                                   │     │
│   │    accumulated_importance = 0                                     │     │
│   │    observations_since_last = 0                                    │     │
│   │    last_reflection_at = NOW()                                     │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY LIFECYCLE (v1)                                │
│                                                                              │
│   ┌───────────────────────┐                                                 │
│   │ Observation Arrives   │                                                 │
│   │ (explicit or hook)    │                                                 │
│   └───────────┬───────────┘                                                 │
│               │                                                              │
│               ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    EPISODIC MEMORY                               │      │
│   │                   (Event Stream)                                 │      │
│   │                                                                  │      │
│   │   Lifetime: 30 days + exponential decay                          │      │
│   │   Storage: Postgres + pgvector                                   │      │
│   │   Purpose: Raw experience log                                    │      │
│   │                                                                  │      │
│   │   ┌─────────────────────────────────────────────────────────┐   │      │
│   │   │ Recency Score over Time                                  │   │      │
│   │   │                                                          │   │      │
│   │   │   1.0 ┤******                                           │   │      │
│   │   │   0.8 ┤      ****                                       │   │      │
│   │   │   0.6 ┤          ****                                   │   │      │
│   │   │   0.4 ┤              ****                               │   │      │
│   │   │   0.2 ┤                  ********                       │   │      │
│   │   │   0.0 ┼──────────────────────────────▶                  │   │      │
│   │   │       0    7    14   21   28  days                      │   │      │
│   │   │                                                          │   │      │
│   │   │   decay = 0.995^hours (~14 day half-life)               │   │      │
│   │   └─────────────────────────────────────────────────────────┘   │      │
│   └─────────────────────────┬───────────────────────────────────────┘      │
│                             │                                               │
│                             │ reflection: synthesize insights               │
│                             │                                               │
│                             ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    SEMANTIC MEMORY                               │      │
│   │                   (Knowledge Base)                               │      │
│   │                                                                  │      │
│   │   Lifetime: Permanent (no decay)                                 │      │
│   │   Storage: Postgres + pgvector                                   │      │
│   │   Purpose: Distilled insights, patterns, preferences             │      │
│   │                                                                  │      │
│   │   Properties:                                                    │      │
│   │   • is_synthetic = TRUE (generated by reflection)                │      │
│   │   • citations = [episodic memory IDs]                           │      │
│   │   • Higher base importance (8+)                                  │      │
│   │   • No recency decay (always accessible)                         │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                  FEEDBACK LOOP                                   │      │
│   │                                                                  │      │
│   │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │      │
│   │   │   Memory     │────▶│   Used in    │────▶│   Feedback   │   │      │
│   │   │  Retrieved   │     │   Response   │     │   Provided   │   │      │
│   │   └──────────────┘     └──────────────┘     └───────┬──────┘   │      │
│   │                                                      │          │      │
│   │                         ┌────────────────────────────┘          │      │
│   │                         │                                       │      │
│   │                         ▼                                       │      │
│   │   ┌─────────────────────────────────────────────────────────┐  │      │
│   │   │ helpful_count += 1  OR  harmful_count += 1              │  │      │
│   │   │                                                          │  │      │
│   │   │ Adjusted importance = base + (helpful - harmful) * 0.5   │  │      │
│   │   └─────────────────────────────────────────────────────────┘  │      │
│   │                                                                  │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Plugin Architecture

### Plugin Structure

```
claude-marketplace/plugins/engram/
├── plugin.json                    # Plugin manifest v0.2.0
├── agents/
│   └── memory-manager.md          # Single orchestrator agent
├── skills/
│   ├── memory-retrieval/
│   │   └── SKILL.md               # Query formulation patterns
│   ├── memory-reflection/
│   │   └── SKILL.md               # Synthesis procedures
│   └── memory-curation/
│       └── SKILL.md               # Maintenance operations
└── hooks/
    └── hooks.json                 # PostToolUse observation capture
```

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PLUGIN ARCHITECTURE                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    memory-manager (Orchestrator)                     │   │
│  │                                                                      │   │
│  │   Responsibilities:                                                  │   │
│  │   • Retrieval orchestration (formulate queries, assemble context)   │   │
│  │   • Ingestion decisions (evaluate importance, classify)             │   │
│  │   • Reflection triggering (recognize synthesis opportunities)       │   │
│  │   • Curation oversight (manage dedup, GC, recalibration)           │   │
│  │                                                                      │   │
│  │   MCP Tools: memory_observe, memory_retrieve, memory_reflect,       │   │
│  │              memory_feedback, memory_stats                          │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                            │
│               ┌─────────────────┼─────────────────┐                         │
│               │                 │                 │                         │
│               ▼                 ▼                 ▼                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │   Retrieval    │  │   Reflection   │  │   Curation     │                │
│  │    Skill       │  │    Skill       │  │    Skill       │                │
│  │                │  │                │  │                │                │
│  │ • Query        │  │ • Trigger      │  │ • Dedup        │                │
│  │   formulation  │  │   detection    │  │   thresholds   │                │
│  │ • Context      │  │ • Question     │  │ • GC criteria  │                │
│  │   assembly     │  │   generation   │  │ • Importance   │                │
│  │ • Scoring      │  │ • Insight      │  │   calibration  │                │
│  │   interpretation│ │   extraction   │  │                │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### PostToolUse Hooks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROACTIVE OBSERVATION CAPTURE                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PostToolUse Hook                                  │   │
│  │                                                                      │   │
│  │   Matcher: Write|Edit                                               │   │
│  │   ┌──────────────────────────────────────────────────────────┐      │   │
│  │   │ Prompt: Analyze file modification for memory capture.     │      │   │
│  │   │                                                           │      │   │
│  │   │ If significant (architectural decision, bug fix, pattern):│      │   │
│  │   │   {"capture": true, "observation": "...", "type": "...",  │      │   │
│  │   │    "importance": <5-9>}                                   │      │   │
│  │   │ Else:                                                     │      │   │
│  │   │   {"capture": false}                                      │      │   │
│  │   └──────────────────────────────────────────────────────────┘      │   │
│  │                                                                      │   │
│  │   Matcher: Bash                                                     │   │
│  │   ┌──────────────────────────────────────────────────────────┐      │   │
│  │   │ Prompt: Analyze command for memory capture.               │      │   │
│  │   │                                                           │      │   │
│  │   │ Capture: errors (9), test results (6), significant (5+)  │      │   │
│  │   │   {"capture": true, "observation": "...", "type": "...",  │      │   │
│  │   │    "importance": <3-9>}                                   │      │   │
│  │   │ Else:                                                     │      │   │
│  │   │   {"capture": false}                                      │      │   │
│  │   └──────────────────────────────────────────────────────────┘      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Hook output triggers memory-manager agent to call memory_observe           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Local Development

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LOCAL DEVELOPMENT SETUP                                 │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        Developer Machine                             │  │
│   │                                                                      │  │
│   │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │  │
│   │   │  Claude      │     │   Engram     │     │   MemoCC     │       │  │
│   │   │  Code CLI    │────▶│   Plugin     │────▶│   Server     │       │  │
│   │   │              │     │              │     │  (FastMCP)   │       │  │
│   │   │  ~/.claude/  │     │ marketplace/ │     │  stdio       │       │  │
│   │   └──────────────┘     └──────────────┘     └───────┬──────┘       │  │
│   │                                                      │              │  │
│   │   ┌─────────────────────────────────────────────────┼──────────┐   │  │
│   │   │                    Docker Compose                │          │   │  │
│   │   │                                                  │          │   │  │
│   │   │   ┌──────────────┐                              │          │   │  │
│   │   │   │  PostgreSQL  │◀─────────────────────────────┘          │   │  │
│   │   │   │  + pgvector  │                                          │   │  │
│   │   │   │  port 5432   │                                          │   │  │
│   │   │   └──────────────┘                                          │   │  │
│   │   │                                                              │   │  │
│   │   └──────────────────────────────────────────────────────────────┘   │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   Configuration:                                                             │
│   • project .mcp.json or ~/.claude.json                                     │
│   • MCP server: stdio transport                                             │
│   • Database: localhost:5432                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA MODEL (v1)                                      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                           memories                                   │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ id                UUID PRIMARY KEY                          │   │  │
│   │   │ user_id           UUID NOT NULL                             │   │  │
│   │   │                                                              │   │  │
│   │   │ -- Content                                                   │   │  │
│   │   │ content           TEXT NOT NULL                              │   │  │
│   │   │ embedding         VECTOR(1536)                               │   │  │
│   │   │                                                              │   │  │
│   │   │ -- Classification                                            │   │  │
│   │   │ memory_type       VARCHAR(20)  -- episodic/semantic          │   │  │
│   │   │ observation_type  VARCHAR(50)  -- error/instruction/code/etc │   │  │
│   │   │ section           VARCHAR(50)  -- strategies/snippets/etc    │   │  │
│   │   │                                                              │   │  │
│   │   │ -- Scoring                                                   │   │  │
│   │   │ importance_score  FLOAT CHECK (1 <= x <= 10)                │   │  │
│   │   │ helpful_count     INT DEFAULT 0                              │   │  │
│   │   │ harmful_count     INT DEFAULT 0                              │   │  │
│   │   │                                                              │   │  │
│   │   │ -- Provenance                                                │   │  │
│   │   │ is_synthetic      BOOLEAN DEFAULT FALSE                      │   │  │
│   │   │ citations         JSONB  -- [memory_id, ...]                 │   │  │
│   │   │                                                              │   │  │
│   │   │ -- Timestamps                                                │   │  │
│   │   │ created_at        TIMESTAMPTZ DEFAULT NOW()                  │   │  │
│   │   │ last_accessed_at  TIMESTAMPTZ DEFAULT NOW()                  │   │  │
│   │   │                                                              │   │  │
│   │   │ -- Extensibility                                             │   │  │
│   │   │ metadata          JSONB                                      │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   └────────────────────────────────┬────────────────────────────────────┘  │
│                                    │                                        │
│                                    │ user_id references                     │
│                                    │                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      reflection_state                                │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ user_id               UUID PRIMARY KEY                      │   │  │
│   │   │ accumulated_importance FLOAT DEFAULT 0                       │   │  │
│   │   │ observations_since    INT DEFAULT 0                          │   │  │
│   │   │ last_reflection_at    TIMESTAMPTZ DEFAULT NOW()              │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   INDEXES:                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ idx_memories_user_time     ON memories(user_id, created_at DESC)    │  │
│   │ idx_memories_type          ON memories(memory_type)                 │  │
│   │ idx_memories_accessed      ON memories(last_accessed_at DESC)       │  │
│   │ idx_memories_embedding     ON memories USING ivfflat                │  │
│   │                            (embedding vector_cosine_ops)            │  │
│   │                            WITH (lists = 100)                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SQL Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- memories table
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    memory_type VARCHAR(20) NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    observation_type VARCHAR(50),
    section VARCHAR(50),
    importance_score FLOAT NOT NULL CHECK (importance_score BETWEEN 1 AND 10),
    helpful_count INT DEFAULT 0,
    harmful_count INT DEFAULT 0,
    is_synthetic BOOLEAN DEFAULT FALSE,
    citations JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW()
);

-- reflection_state table
CREATE TABLE reflection_state (
    user_id UUID PRIMARY KEY,
    accumulated_importance FLOAT DEFAULT 0,
    observations_since INT DEFAULT 0,
    last_reflection_at TIMESTAMPTZ DEFAULT NOW()
);

-- indexes
CREATE INDEX idx_memories_user_time ON memories(user_id, created_at DESC);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_accessed ON memories(last_accessed_at DESC);
CREATE INDEX idx_memories_embedding ON memories
    USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);
```

---

## Appendix: File Structure

```
memocc/
├── docs/
│   ├── RESEARCH_SYNTHESIS.md      # Executive summary
│   ├── RESEARCH_DEEP_DIVE.md      # Comprehensive analysis
│   └── ARCHITECTURE.md            # This document
│
├── app/
│   ├── __init__.py
│   ├── server.py                  # FastMCP server entry point
│   ├── config.py                  # Configuration management
│   │
│   ├── tools/                     # MCP tool implementations
│   │   ├── __init__.py
│   │   ├── observe.py             # memory_observe tool
│   │   ├── retrieve.py            # memory_retrieve tool
│   │   ├── reflect.py             # memory_reflect tool
│   │   ├── feedback.py            # memory_feedback tool
│   │   └── stats.py               # memory_stats tool
│   │
│   ├── services/                  # Core business logic
│   │   ├── __init__.py
│   │   ├── embedding.py           # Embedding generation
│   │   ├── retrieval.py           # Three-factor retrieval engine
│   │   ├── ingestion.py           # Delta ingestion pipeline
│   │   └── reflection.py          # Reflection engine
│   │
│   ├── models/                    # Data models
│   │   ├── __init__.py
│   │   ├── memory.py              # Memory SQLModel
│   │   └── reflection_state.py    # Trigger state tracking
│   │
│   └── db/                        # Database layer
│       ├── __init__.py
│       ├── connection.py          # Database connection
│       └── migrations/            # Alembic migrations
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_ingestion.py
│   └── test_reflection.py
│
├── pyproject.toml
├── docker-compose.yml
└── README.md

claude-marketplace/plugins/engram/
├── plugin.json                    # Plugin manifest v0.2.0
├── agents/
│   └── memory-manager.md          # Single orchestrator agent
├── skills/
│   ├── memory-retrieval/
│   │   └── SKILL.md
│   ├── memory-reflection/
│   │   └── SKILL.md
│   └── memory-curation/
│       └── SKILL.md
└── hooks/
    └── hooks.json                 # PostToolUse observation capture
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENGRAM_RECENCY_DECAY` | 0.995 | Exponential decay per hour |
| `ENGRAM_IMPORTANCE_THRESHOLD` | 150 | Sum triggering reflection |
| `ENGRAM_OBSERVATION_THRESHOLD` | 100 | Count triggering reflection |
| `ENGRAM_TIME_THRESHOLD_HOURS` | 24 | Hours triggering reflection |
| `ENGRAM_SIMILARITY_THRESHOLD` | 0.90 | Cosine similarity for dedup |
| `ENGRAM_EMBEDDING_MODEL` | text-embedding-3-small | OpenAI model |
| `ENGRAM_EMBEDDING_DIMENSIONS` | 1536 | Vector dimensions |
| `OPENAI_API_KEY` | - | Required for embeddings |
| `ANTHROPIC_API_KEY` | - | Required for reflection LLM |
| `DATABASE_URL` | - | Postgres connection string |

---

## References

- [RESEARCH_SYNTHESIS.md](./RESEARCH_SYNTHESIS.md) - Executive summary of research findings
- [RESEARCH_DEEP_DIVE.md](./RESEARCH_DEEP_DIVE.md) - Comprehensive analysis of source papers
- CoALA: Cognitive Architectures for Language Agents (Sumers et al., 2024)
- ACE: Agentic Context Engineering (Zhang et al., 2025)
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)
