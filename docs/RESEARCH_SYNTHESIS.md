# Engram Research Synthesis

> Research grounding for building engram - a portable memory graph for Claude Code interactions with RAG-backed semantic search and ingestion via FastMCP.

**Papers Analyzed:**
- CoALA: Cognitive Architectures for Language Agents (Sumers et al., 2024)
- ACE: Agentic Context Engineering (Zhang et al., 2025)
- Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)

**Date:** 2025-12-09

---

## Executive Summary

Three complementary papers provide the theoretical and empirical foundation for engram:

| Paper | Type | Key Contribution |
|-------|------|------------------|
| **CoALA** | Theoretical Framework | Memory taxonomy (working/episodic/semantic/procedural), decision cycle |
| **ACE** | System + Empirical | Ingestion architecture (Generator/Reflector/Curator), delta updates |
| **Generative Agents** | System + Empirical | Retrieval scoring formula, reflection triggering mechanism |

---

## High-Confidence Findings

These principles are supported across multiple papers:

### 1. Multi-Factor Retrieval Scoring
**Confidence:** High

Pure semantic similarity is insufficient. Implement:
```
score = α*recency + β*importance + γ*relevance
```
Where all three factors are normalized to [0,1].

**Supporting Evidence:**
- Generative Agents: Empirical validation of three-factor scoring
- CoALA: Theoretical framework for retrieval actions
- ACE: Implicit via relevance + metadata tracking

### 2. Incremental Delta Updates Over Full Replacement
**Confidence:** High

Never overwrite memory wholesale. Each ingestion generates a delta that merges deterministically with existing memory.

**Supporting Evidence:**
- ACE: Explicit delta architecture (86.9% lower latency)
- CoALA: Learn module appends to memory
- Generative Agents: Append-only stream

### 3. Metadata-Rich Memory Nodes
**Confidence:** High

Every memory node needs:
- Timestamp (recency calculation)
- Importance score (retrieval weighting)
- Embeddings (semantic search)
- Citations/provenance (trust, debugging)
- Type classification (routing, filtering)

### 4. Reflection/Synthesis as Separate Process
**Confidence:** High

Don't synthesize during ingestion. Ingestion writes raw observations; reflection runs periodically to extract higher-order insights.

**Supporting Evidence:**
- Generative Agents: Explicit reflection mechanism
- ACE: Reflector + Curator roles
- CoALA: Learn module as separate action

---

## Key Trade-offs

### 1. Append-Only Stream vs Structured Graph

| Option | Pros | Cons |
|--------|------|------|
| **Stream** (Generative Agents) | Simple, preserves history, natural recency | Grows unbounded, harder to query |
| **Hybrid** (CoALA) | Clean separation, optimized storage | Complex routing, potential info loss |

**Recommendation:** Hybrid with stream foundation. Start append-only (episodic), promote to semantic/procedural via reflection.

### 2. Importance Scoring

| Option | Pros | Cons |
|--------|------|------|
| **LLM-scored** | Contextually aware | Expensive, inconsistent |
| **Rule-based** | Fast, predictable | May miss nuance |
| **Feedback-driven** (ACE) | Self-improving | Cold start problem |

**Recommendation:** Start rule-based, migrate to feedback-driven.

Default heuristics:
- Errors: 9
- User instructions: 10
- Code changes: 7
- Tool outputs: 3

### 3. Reflection Trigger Strategy

| Option | Pros | Cons |
|--------|------|------|
| **Threshold** (importance > 150) | Adaptive | Hard to tune |
| **Time-based** (every N observations) | Predictable | May waste compute |
| **On-demand** | Full control | May never happen |

**Recommendation:** Hybrid - time-based (every 100 observations) + explicit command.

### 4. Deduplication Strategy

| Option | Pros | Cons |
|--------|------|------|
| **Embedding similarity** (ACE) | Catches semantic duplicates | Embedding costs, false positives |
| **Exact match** | Fast, deterministic | Misses semantic duplicates |
| **None** | Simplest | Storage bloat |

**Recommendation:** Embedding similarity with high threshold (0.90) to minimize false positives.

---

## Core Algorithms

### Retrieval Scoring (Generative Agents)

```python
def retrieve_memories(query: str, memories: list, k: int = 10) -> list:
    current_time = get_current_time()
    query_embedding = embed(query)

    scored = []
    for memory in memories:
        # Recency: exponential decay (14-day half-life)
        hours_since = (current_time - memory.last_access).total_seconds() / 3600
        recency = 0.995 ** hours_since

        # Importance: pre-computed or feedback-adjusted
        importance = (memory.importance + (memory.helpful - memory.harmful) * 0.5) / 10

        # Relevance: cosine similarity
        relevance = cosine_similarity(query_embedding, memory.embedding)

        # Combined score (equal weights)
        score = (recency + importance + relevance) / 3
        scored.append((memory, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
```

### Reflection Triggering (Generative Agents)

```python
IMPORTANCE_THRESHOLD = 150

def should_reflect(memories: list, last_reflection_time: datetime) -> bool:
    recent_importance = sum(
        m.importance for m in memories
        if m.timestamp > last_reflection_time
    )
    return recent_importance >= IMPORTANCE_THRESHOLD

def reflect(memories: list) -> list[Insight]:
    # Step 1: Generate questions from recent memories
    recent = get_recent_memories(memories, n=100)
    questions = llm_generate_questions(recent, n=3)

    # Step 2: For each question, retrieve and synthesize
    insights = []
    for question in questions:
        relevant = retrieve_memories(question, memories, k=100)
        new_insights = llm_extract_insights(question, relevant, n=5)
        insights.extend(new_insights)

    return insights
```

### Delta Ingestion (ACE)

```python
SIMILARITY_THRESHOLD = 0.90

def generate_delta(insight: Insight, context: list[Bullet]) -> Delta:
    insight_embedding = embed(insight.content)

    # Check for similar existing bullet
    for bullet in context:
        similarity = cosine_similarity(insight_embedding, bullet.embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            # Update existing
            if insight.helpful:
                bullet.helpful_count += 1
            else:
                bullet.harmful_count += 1
            return Delta(updated=[bullet])

    # Create new bullet
    new_bullet = Bullet(
        id=generate_id(),
        content=insight.content,
        embedding=insight_embedding,
        helpful_count=1 if insight.helpful else 0,
        harmful_count=0 if insight.helpful else 1
    )
    return Delta(new=[new_bullet])
```

---

## Implementation Roadmap

### Phase 1: MVP - Episodic Stream + Retrieval (Weeks 1-2)
- Postgres schema with pgvector
- FastMCP tools: `memory_search`, `memory_ingest`
- Three-factor scoring
- Claude Code plugin: `memory-search` agent

### Phase 2: Feedback + Deduplication (Weeks 3-4)
- Helpful/harmful counters
- `memory_feedback` tool
- Async deduplication job

### Phase 3: Reflection + Semantic Memory (Weeks 5-6)
- Periodic synthesis (episodic → semantic)
- `memory_reflect` tool
- Citation tracking

### Phase 4: Working Memory + Sessions (Week 7)
- Ephemeral session-level memory
- Session boundaries
- Auto-promotion on session end

### Phase 5: Procedural Memory (Week 8+)
- Tool call patterns
- Code snippet extraction

---

## Data Model

### Postgres Schema

```sql
CREATE TABLE observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    importance FLOAT NOT NULL CHECK (importance BETWEEN 0 AND 10),
    memory_type VARCHAR(20) NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    observation_type VARCHAR(50),
    helpful_count INT DEFAULT 0,
    harmful_count INT DEFAULT 0,
    citations JSONB,
    is_synthetic BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_observations_user_time ON observations(user_id, timestamp DESC);
CREATE INDEX idx_observations_session ON observations(session_id);
CREATE INDEX idx_observations_type ON observations(memory_type);
CREATE INDEX idx_observations_embedding ON observations USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);
```

### FastMCP Tool Signatures

```python
@mcp.tool()
async def memory_search(
    query: str,
    limit: int = 10,
    session_only: bool = False,
    memory_types: list[str] = None
) -> list[dict]:
    """Search memory using three-factor scoring (recency + importance + relevance)"""

@mcp.tool()
async def memory_ingest(
    observation: str,
    observation_type: str = "general",
    importance: float = None
) -> dict:
    """Ingest new observation into memory stream"""

@mcp.tool()
async def memory_feedback(
    observation_id: str,
    helpful: bool
) -> dict:
    """Provide feedback on memory usefulness"""

@mcp.tool()
async def memory_reflect(
    focus: str = None,
    max_insights: int = 5
) -> list[dict]:
    """Trigger reflection to synthesize insights from recent memories"""
```

---

## Tunable Parameters

| Parameter | Default | Description | Source |
|-----------|---------|-------------|--------|
| `recency_decay` | 0.995 | Exponential decay per hour | Generative Agents |
| `importance_threshold` | 150 | Sum triggering reflection | Generative Agents |
| `similarity_threshold` | 0.90 | Cosine similarity for dedup | ACE |
| `reflection_interval` | 100 | Observations between reflections | Hybrid |
| `retrieval_limit` | 10 | Default top-k results | - |
| `alpha_recency` | 0.33 | Recency weight | Generative Agents |
| `alpha_importance` | 0.33 | Importance weight | Generative Agents |
| `alpha_relevance` | 0.33 | Relevance weight | Generative Agents |

---

## Known Limitations

### Cold Start Problem
New users have no memory, no importance calibration, no feedback counters.

**Mitigation:** Seed with high-quality examples, use conservative importance defaults (everything = 5).

### Recency Bias
Exponential decay (0.995^hours) means old memories decay to near-zero after ~2 weeks.

**Mitigation:** Reflection promotes important old memories to semantic storage (no decay).

### Embedding Drift
Model changes break similarity comparisons.

**Mitigation:** Store embedding model version, trigger re-embedding on model change.

### Context Window Limits
Can't retrieve full memory history in one query.

**Mitigation:** Multi-stage retrieval, paginated results.

---

## Failure Modes

| Mode | Trigger | Mitigation |
|------|---------|------------|
| Memory Explosion | High-volume sessions | TTL (30 days), cap at 10k per user |
| Retrieval Drift | Stale importance scores | Periodic recalibration via reflection |
| Reflection Loops | Reflector retrieves own outputs | Tag `is_synthetic=true`, filter |
| Dedup False Positives | Low threshold | Set ≥ 0.90, require same type |
| Importance Inflation | Everything scored 8-10 | Normalize per-user distribution |

---

## Open Questions

1. **Pgvector performance** at 100k+ observations
2. **Optimal decay factor** (0.995 vs 0.999)
3. **Reflection depth** (single-pass vs iterative refinement)
4. **Cross-user memory sharing** model
5. **Memory portability** format (JSON vs binary)

---

## References

1. Sumers, T.R., Yao, S., Narasimhan, K., Griffiths, T.L. (2024). Cognitive Architectures for Language Agents. TMLR.
2. Zhang, Q., et al. (2025). Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. arXiv:2510.04618.
3. Park, J.S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST '23.
