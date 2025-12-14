# Living Memory Architecture: Research Synthesis

> Consolidated findings from ACE, CoALA, and Generative Agents for building a persistent, evolving memory system for Claude Code.

**Date**: 2025-12-11
**Source Papers**: 3 (77 pages total)
**Implementation Target**: Claude Code CLI with MCP-based memory server

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Paper Analysis](#paper-analysis)
3. [Converging Principles](#converging-principles)
4. [Core Architecture](#core-architecture)
5. [Key Algorithms](#key-algorithms)
6. [Trade-off Analysis](#trade-off-analysis)
7. [Implementation Guidance](#implementation-guidance)
8. [Failure Modes & Mitigations](#failure-modes--mitigations)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Confidence Ratings](#confidence-ratings)
11. [Open Questions](#open-questions)

---

## Executive Summary

This document synthesizes research from three foundational papers on agent memory systems to provide actionable implementation guidance for a "living memory" architecture in Claude Code—a system that accumulates, refines, and organizes knowledge through agent interactions.

### Key Findings

| Finding | Confidence | Source |
|---------|------------|--------|
| Memory requires multi-layer hierarchy (episodic → semantic → procedural) | HIGH | All 3 papers |
| Retrieval needs three factors: recency × importance × relevance | HIGH | Generative Agents |
| Periodic reflection transforms observations into insights | HIGH | All 3 papers |
| Feedback loops (helpful/harmful counters) enable learning | HIGH | ACE |
| Incremental delta updates beat monolithic rewrites | MEDIUM-HIGH | ACE |

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIVING MEMORY SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  EPISODIC   │  │  SEMANTIC   │  │ PROCEDURAL  │             │
│  │  (events)   │──│  (insights) │──│ (workflows) │             │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘             │
│         │                │                                      │
│         ▼                ▼                                      │
│  ┌─────────────────────────────────────────────────┐           │
│  │           THREE-FACTOR RETRIEVAL                 │           │
│  │   score = α×recency + β×importance + γ×relevance │           │
│  └─────────────────────────────────────────────────┘           │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  GENERATOR  │──│  REFLECTOR  │──│   CURATOR   │             │
│  │  (observe)  │  │ (synthesize)│  │ (feedback)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Paper Analysis

### Paper 1: ACE - Agentic Context Engineering (2025)

**Authors**: Zhang et al. (Stanford, SambaNova, UC Berkeley)
**Type**: System/Empirical
**Fitness**: 22/25 (HIGH)
**Confidence**: 18/20 (HIGH)

**Core Contribution**: Framework treating contexts as "evolving playbooks" that accumulate, refine, and organize strategies through a modular process.

**Key Concepts**:
- **Generator-Reflector-Curator Pipeline**: Three-stage processing for memory evolution
- **Incremental Delta Updates**: Avoids "context collapse" from monolithic rewrites
- **Bullet Metadata**: Each memory has ID + helpful/harmful counters + content
- **Grow-and-Refine**: Balances context expansion with redundancy control
- **Dual Adaptation**: Offline (system prompts) + Online (test-time memory)

**Results**:
- +10.6% on agent benchmarks
- +8.6% on domain-specific benchmarks
- 86.9% lower adaptation latency than existing methods
- Works without labeled supervision using execution feedback

**Problems Solved**:
- **Brevity Bias**: Prior methods compress away useful detail
- **Context Collapse**: Iterative rewriting erodes knowledge over time

---

### Paper 2: CoALA - Cognitive Architectures for Language Agents (2024)

**Authors**: Sumers et al. (Princeton)
**Type**: Theoretical/Survey
**Fitness**: 16/25 (MEDIUM)
**Confidence**: 15/20 (MEDIUM)

**Core Contribution**: Conceptual framework organizing language agents via memory, action, and decision-making—drawing on cognitive science and symbolic AI.

**Key Concepts**:

**Memory Taxonomy**:
| Type | Description | Persistence | Example |
|------|-------------|-------------|---------|
| Working | Active context for current task | Session | Current goal, retrieved memories |
| Episodic | Specific experiences with temporal context | Long-term | "Fixed auth bug on Tuesday" |
| Semantic | Generalized knowledge and patterns | Long-term | "Always validate JWT tokens" |
| Procedural | Executable actions and workflows | Long-term | "How to add API endpoint" |

**Action Space**:
- **Internal**: Retrieval, Reasoning, Learning
- **External**: Grounding (tools, environment interaction)

**Decision Cycle**:
```
PROPOSE (candidates) → EVALUATE (score) → SELECT (choose) → EXECUTE → OBSERVE
```

**Surveyed Agents**: SayCan, ReAct, Voyager, Generative Agents, Tree of Thoughts

---

### Paper 3: Generative Agents (2023)

**Authors**: Park et al. (Stanford)
**Type**: System/Empirical
**Fitness**: 21/25 (HIGH)
**Confidence**: 17/20 (HIGH)

**Core Contribution**: Architecture for believable agents with persistent memory, demonstrated with 25 agents in a sandbox simulation.

**Key Concepts**:

**Memory Stream**: Comprehensive natural language record of all experiences
- Append-only storage
- Each entry: content + embedding + timestamp + importance + access_count

**Three-Factor Retrieval**:
```python
score = α_recency × recency + α_importance × importance + α_relevance × relevance

# Where:
recency = 0.995 ^ hours_since_access  # Half-life ~14 days
importance = LLM_scored(1-10)          # "How poignant is this?"
relevance = cosine_similarity(query, memory)
```

**Reflection Mechanism**:
1. Identify questions from recent memories: "What are 3 salient high-level questions?"
2. Retrieve relevant memories for each question
3. Extract insights with citations: "insight (because of 1, 5, 3)"
4. Store as semantic memory, linking to source episodic memories

**Reflection Trees**: Recursive reflection where insights spawn further reflection

**Planning Hierarchy**:
```
Day Plan (broad goals)
    └── Hour Plan (activity blocks)
            └── Minute Plan (5-15 min actions)
```

**Evaluation Results**:
- Agents spread information (4% → 32% knew about election)
- Network density increased 0.167 → 0.74 (relationship formation)
- 5/12 invited agents attended Valentine's Day party (coordination)

---

## Converging Principles

These principles are supported by multiple papers with high confidence:

### Principle 1: Multi-Layered Memory

**All three papers** distinguish between raw observations, synthesized knowledge, and actionable procedures.

| Layer | Source | Purpose |
|-------|--------|---------|
| Episodic | Generative Agents, CoALA | Store raw experiences |
| Semantic | Generative Agents, CoALA, ACE | Store synthesized insights |
| Procedural | CoALA, ACE (sections) | Store executable patterns |

**Implementation**: Implement at minimum three memory types with different storage and retrieval characteristics.

---

### Principle 2: Multi-Factor Retrieval

**Pure semantic similarity is insufficient**; temporal and importance factors are essential.

| Factor | Source | Purpose |
|--------|--------|---------|
| Recency | Generative Agents | Favor recent, contextually relevant memories |
| Importance | Generative Agents, ACE | Prioritize significant observations |
| Relevance | All three | Match query semantics |

**Implementation**: Use three-factor scoring with configurable weights for different scenarios.

---

### Principle 3: Reflection Creates Higher-Order Knowledge

**Periodic synthesis** of observations into insights is what makes memory "living" rather than just logging.

| Paper | Mechanism |
|-------|-----------|
| Generative Agents | Question generation → insight extraction with citations |
| ACE | Reflector component with iterative refinement |
| CoALA | Learning procedures that write to semantic memory |

**Implementation**: Implement triggered reflection that generates insights from accumulated observations.

---

### Principle 4: Feedback Enables Learning

**Memory quality improves** when outcomes inform future retrieval/storage.

| Paper | Mechanism |
|-------|-----------|
| ACE | Explicit helpful/harmful counters |
| CoALA | Learning procedures |
| Generative Agents | Implicit through retrieval success |

**Implementation**: Track `helpful_count` and `harmful_count` per memory; adjust effective importance.

---

### Principle 5: Incremental Over Monolithic

**Memory systems should accumulate** rather than replace.

| Paper | Mechanism |
|-------|-----------|
| Generative Agents | Append-only memory stream |
| ACE | Delta-based updates with deterministic merge |
| CoALA | Procedural updates (add, not replace) |

**Implementation**: Use append-only storage with delta merging; avoid full rewrites.

---

## Core Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIVING MEMORY SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────── Memory Layers ──────────────────────────┐      │
│   │                                                                  │      │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │      │
│   │  │  EPISODIC   │───▶│  SEMANTIC   │    │ PROCEDURAL  │         │      │
│   │  │  (events)   │    │  (insights) │    │ (workflows) │         │      │
│   │  │             │    │             │    │             │         │      │
│   │  │ • Raw obs   │    │ • Patterns  │    │ • Recipes   │         │      │
│   │  │ • Full ctx  │    │ • Lessons   │    │ • Commands  │         │      │
│   │  │ • Temporal  │    │ • Citations │    │ • Triggers  │         │      │
│   │  └─────────────┘    └─────────────┘    └─────────────┘         │      │
│   │                                                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌──────────────────────── Processing Pipeline ────────────────────┐      │
│   │                                                                  │      │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │      │
│   │  │  GENERATOR  │───▶│  REFLECTOR  │───▶│   CURATOR   │         │      │
│   │  │             │    │             │    │             │         │      │
│   │  │ • Observe   │    │ • Question  │    │ • Feedback  │         │      │
│   │  │ • Embed     │    │ • Extract   │    │ • Adjust    │         │      │
│   │  │ • Dedupe    │    │ • Cite      │    │ • Prune     │         │      │
│   │  │ • Store     │    │ • Refine    │    │             │         │      │
│   │  └─────────────┘    └─────────────┘    └─────────────┘         │      │
│   │                                                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌──────────────────────── Retrieval System ───────────────────────┐      │
│   │                                                                  │      │
│   │  score = α × recency + β × importance + γ × relevance           │      │
│   │                                                                  │      │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │      │
│   │  │   RECENCY   │    │ IMPORTANCE  │    │  RELEVANCE  │         │      │
│   │  │             │    │             │    │             │         │      │
│   │  │ 0.995^hours │    │ base_score  │    │ cosine_sim  │         │      │
│   │  │ half-life   │    │ + feedback  │    │ (query,mem) │         │      │
│   │  │ ~14 days    │    │ adjustment  │    │             │         │      │
│   │  └─────────────┘    └─────────────┘    └─────────────┘         │      │
│   │                                                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Model

```typescript
interface Memory {
  // Identity
  id: UUID;
  user_id: UUID;

  // Content
  content: string;              // Natural language (1-10000 chars)
  embedding: float[];           // Vector (1536 dims)

  // Classification
  memory_type: 'episodic' | 'semantic' | 'procedural';
  observation_type?: 'error' | 'instruction' | 'decision' | 'code_change' |
                     'insight' | 'test_result' | 'tool_output' | 'general';
  section?: 'strategies' | 'snippets' | 'pitfalls' | 'context' | 'preferences';

  // Scoring
  importance_score: float;      // Base importance [1.0, 10.0]
  helpful_count: int;           // Positive feedback signals
  harmful_count: int;           // Negative feedback signals

  // Provenance
  is_synthetic: boolean;        // True for reflections
  citations?: UUID[];           // Source memory IDs (for semantic)

  // Timestamps
  created_at: timestamp;
  last_accessed_at: timestamp;  // Updated on retrieval

  // Extensibility
  metadata?: Record<string, any>;
}

interface ReflectionState {
  user_id: UUID;
  accumulated_importance: float;  // Sum since last reflection
  observations_since_last: int;   // Count since last reflection
  last_reflection_at: timestamp;
}
```

---

## Key Algorithms

### Algorithm 1: Three-Factor Retrieval

**Source**: Generative Agents
**Complexity**: O(n log k + E) where n = memories, k = results, E = embedding time

```python
def retrieve_memories(
    query: str,
    memories: List[Memory],
    k: int = 10,
    alpha_recency: float = 0.33,
    alpha_importance: float = 0.33,
    alpha_relevance: float = 0.33
) -> List[Memory]:
    """
    Three-factor memory retrieval with normalization.
    """
    # 1. Embed query
    query_embedding = embed(query)

    # 2. Score all memories
    scored = []
    for memory in memories:
        hours = (now() - memory.last_accessed_at).hours
        recency = 0.995 ** hours

        importance = (
            memory.importance_score +
            (memory.helpful_count - memory.harmful_count) * 0.5
        ) / 10.0

        relevance = cosine_similarity(query_embedding, memory.embedding)

        scored.append({
            'memory': memory,
            'recency': recency,
            'importance': importance,
            'relevance': relevance
        })

    # 3. Min-max normalize each factor
    for factor in ['recency', 'importance', 'relevance']:
        values = [s[factor] for s in scored]
        min_v, max_v = min(values), max(values)
        for s in scored:
            s[f'{factor}_norm'] = (s[factor] - min_v) / (max_v - min_v + 1e-10)

    # 4. Combine scores
    for s in scored:
        s['final'] = (
            alpha_recency * s['recency_norm'] +
            alpha_importance * s['importance_norm'] +
            alpha_relevance * s['relevance_norm']
        )

    # 5. Return top-k, update access times
    scored.sort(key=lambda x: x['final'], reverse=True)
    results = []
    for s in scored[:k]:
        s['memory'].last_accessed_at = now()
        results.append(s['memory'])

    return results
```

**Weight Guidelines**:

| Scenario | Recency (α) | Importance (β) | Relevance (γ) |
|----------|-------------|----------------|---------------|
| Debugging | 0.5 | 0.2 | 0.3 |
| Architecture | 0.1 | 0.4 | 0.5 |
| Quick lookup | 0.2 | 0.3 | 0.5 |
| Error patterns | 0.3 | 0.4 | 0.3 |

---

### Algorithm 2: Reflection Trigger

**Source**: Generative Agents
**Complexity**: O(1)

```python
# Thresholds
IMPORTANCE_THRESHOLD = 150
OBSERVATION_THRESHOLD = 100
TIME_THRESHOLD_HOURS = 24

def should_reflect(state: ReflectionState) -> tuple[bool, str]:
    """Check if reflection should be triggered."""

    if state.accumulated_importance >= IMPORTANCE_THRESHOLD:
        return (True, f"importance >= {IMPORTANCE_THRESHOLD}")

    if state.observations_since_last >= OBSERVATION_THRESHOLD:
        return (True, f"observations >= {OBSERVATION_THRESHOLD}")

    hours = (now() - state.last_reflection_at).hours
    if hours >= TIME_THRESHOLD_HOURS:
        return (True, f"hours >= {TIME_THRESHOLD_HOURS}")

    return (False, "no threshold met")
```

---

### Algorithm 3: Insight Extraction with Citations

**Source**: Generative Agents + ACE
**Complexity**: O(L) where L = LLM inference time

```python
def extract_insights(
    question: str,
    memories: List[Memory],
    max_insights: int = 5
) -> List[Insight]:
    """Generate insights with citations to source memories."""

    # Format memories with indices
    statements = []
    memory_map = {}
    for i, mem in enumerate(memories[:50]):
        idx = i + 1
        statements.append(f"{idx}. {mem.content}")
        memory_map[idx] = mem.id

    prompt = f"""Based on these statements, answer the question with up to {max_insights}
insights. Format each as:
- Insight text (because of 1, 3, 7)

Question: {question}

Statements:
{chr(10).join(statements)}

Insights:"""

    response = llm.complete(prompt, max_tokens=500)

    # Parse insights and citations
    insights = []
    for line in response.split('\n'):
        match = re.match(r'-\s*(.+?)\s*\(because of\s+([\d,\s]+)\)', line)
        if match:
            text = match.group(1).strip()
            cited_ids = [memory_map[int(x.strip())]
                        for x in match.group(2).split(',')
                        if int(x.strip()) in memory_map]
            insights.append(Insight(text=text, citations=cited_ids))

    return insights[:max_insights]
```

---

### Algorithm 4: Delta Ingestion with Deduplication

**Source**: ACE
**Complexity**: O(log n + E) with vector index

```python
def ingest_observation(
    content: str,
    observation_type: str,
    similarity_threshold: float = 0.90
) -> Delta:
    """Ingest with deduplication."""

    # 1. Generate embedding
    embedding = embed(content)

    # 2. Find nearest neighbors
    candidates = db.query("""
        SELECT * FROM memories
        WHERE memory_type = 'episodic'
        ORDER BY embedding <=> $1
        LIMIT 5
    """, embedding)

    # 3. Check for duplicate
    for candidate in candidates:
        similarity = cosine_similarity(embedding, candidate.embedding)
        if similarity >= similarity_threshold:
            # Found duplicate: increment helpful counter
            candidate.helpful_count += 1
            return Delta(updated=[candidate])

    # 4. Create new memory
    importance = compute_importance(content, observation_type)
    new_memory = Memory(
        content=content,
        embedding=embedding,
        memory_type='episodic',
        observation_type=observation_type,
        importance_score=importance,
        helpful_count=1,
        harmful_count=0
    )

    return Delta(created=[new_memory])
```

---

### Algorithm 5: Importance Scoring (Heuristic)

**Source**: ACE
**Complexity**: O(m) where m = content length

```python
IMPORTANCE_HEURISTICS = {
    "instruction": 10,
    "error": 9,
    "decision": 8,
    "code_change": 7,
    "insight": 7,
    "test_result": 6,
    "general": 5,
    "tool_output": 3,
}

def compute_importance(content: str, observation_type: str) -> float:
    """Heuristic importance scoring."""

    base = IMPORTANCE_HEURISTICS.get(observation_type, 5)

    # Boost for critical markers
    if any(m in content.upper() for m in ["CRITICAL", "BREAKING", "SECURITY"]):
        base = min(base + 2, 10)

    # Boost for action markers
    if any(m in content.upper() for m in ["TODO", "FIXME", "HACK"]):
        base = min(base + 1, 10)

    return base
```

---

## Trade-off Analysis

### Trade-off 1: Heuristic vs LLM-Based Importance

| Aspect | Heuristic | LLM-Based |
|--------|-----------|-----------|
| Latency | <1ms | 500ms-2s |
| Cost | Free | ~$0.001-0.01/call |
| Accuracy | 70-80% | 90-95% |
| Consistency | Deterministic | Variable |

**Recommendation**: Hybrid cascade
1. Use heuristic for all observations (immediate)
2. Use LLM only for ambiguous cases (score 5-7)
3. Async LLM re-scoring during idle periods

---

### Trade-off 2: Synchronous vs Asynchronous Reflection

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| UX | Blocking delay | Non-blocking |
| Consistency | Immediate | Eventual |
| Complexity | Simple | Background processing |

**Recommendation**: Async default with sync fallback
- Default: Queue reflections, process during idle
- Provide `/reflect` command for explicit sync

---

### Trade-off 3: Delta Updates vs Monolithic Rewriting

| Aspect | Delta | Monolithic |
|--------|-------|------------|
| Merge complexity | High | None |
| Data loss risk | Low | High (concurrent) |
| Performance | O(Δ) | O(n) |
| Auditability | Full history | Snapshot only |

**Recommendation**: Delta with periodic compaction
- Append-only delta log for all changes
- Last-write-wins for conflicts
- Weekly compaction to snapshots

---

### Trade-off 4: Three-Factor vs Pure Similarity

| Aspect | Three-Factor | Pure Similarity |
|--------|--------------|-----------------|
| Relevance quality | Good | Excellent |
| Temporal awareness | Yes | No |
| Importance weighting | Yes | No |
| Complexity | Medium | Low |

**Recommendation**: Three-factor default
- Use three-factor for general retrieval
- Fall back to pure similarity for explicit "find similar"
- Expose weight tuning for power users

---

## Implementation Guidance

### When-to-Use-What Matrix

| Scenario | Retrieval Weights | Reflection | Importance | Updates |
|----------|-------------------|------------|------------|---------|
| Debugging | α=2.0, β=1.0, γ=1.0 | After resolution | Heuristic | Sync |
| Architecture | α=1.0, β=2.0, γ=1.0 | importance>=150 | LLM-based | Async |
| Quick question | α=1.0, β=1.0, γ=2.0 | Skip | Skip | Append |
| Multi-day project | α=1.0, β=1.0, γ=1.0 | observations>=100 | Hybrid | Delta |
| Code review | α=1.0, β=1.5, γ=1.5 | After completion | LLM-based | Sync |

### Memory Type Selection

| Information | Store In | Priority | Retention |
|-------------|----------|----------|-----------|
| Error + fix | Episodic | High recency | 90 days |
| Architecture decision | Semantic | High importance | Permanent |
| User preference | Procedural | High relevance | Permanent |
| File locations | Episodic | High recency | 30 days |
| Code patterns | Semantic | Balanced | 180 days |
| Command sequences | Procedural | High relevance | Permanent |

---

## Failure Modes & Mitigations

| Failure Mode | Trigger | Impact | Mitigation |
|--------------|---------|--------|------------|
| **Retrieval irrelevance** | Poor weights/embeddings | Wrong context | Monitor quality; expose tuning; keyword fallback |
| **Reflection explosion** | Low thresholds | Cost overrun | Rate limit; increase thresholds; async |
| **Memory bloat** | No forgetting | Slow retrieval; noise | TTL-based expiration; importance pruning |
| **Stale insights** | Codebase changes | Outdated recommendations | Track file versions; invalidate on change |
| **Feedback collapse** | Always positive | Score inflation | Normalize; detect suspicious patterns |
| **Cold start** | New user | No useful retrieval | Bootstrap defaults; graceful degradation |
| **Embedding drift** | Model upgrade | Broken similarity | Version embeddings; batch migration |

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Basic persistence with retrieval

- [ ] Memory record schema with embedding storage
- [ ] Append-only storage (SQLite or PostgreSQL)
- [ ] Embedding generation integration
- [ ] Pure similarity retrieval (baseline)

**Deliverable**: Store and retrieve memories by similarity

---

### Phase 2: Three-Factor Retrieval (Weeks 3-4)

**Goal**: Production-quality retrieval

- [ ] Recency scoring (0.995^hours decay)
- [ ] Importance scoring (heuristic)
- [ ] Three-factor combination with normalization
- [ ] Configurable weight tuning

**Deliverable**: Retrieval balances recency, importance, relevance

---

### Phase 3: Reflection System (Weeks 5-7)

**Goal**: Automatic insight synthesis

- [ ] Reflection state tracking
- [ ] Trigger evaluation (importance/count/time)
- [ ] Question generation via LLM
- [ ] Insight extraction with citations
- [ ] Async processing queue

**Deliverable**: System generates insights from observations

---

### Phase 4: Feedback Integration (Weeks 8-9)

**Goal**: Learning from outcomes

- [ ] Explicit feedback collection (helpful/harmful)
- [ ] Implicit feedback signals (used/not used)
- [ ] Importance adjustment formula
- [ ] Citation propagation

**Deliverable**: Memory quality improves with use

---

### Phase 5: Advanced Features (Weeks 10-12)

**Goal**: Production hardening

- [ ] Delta synchronization with conflict resolution
- [ ] Forgetting mechanism (TTL + importance threshold)
- [ ] LLM-based importance (optional async)
- [ ] Working memory persistence
- [ ] Compaction and archival

**Deliverable**: Production-ready system

---

## Confidence Ratings

### HIGH Confidence (Implement as described)

| Recommendation | Source | Rationale |
|----------------|--------|-----------|
| Append-only memory stream | Generative Agents | Proven; simple; auditable |
| Three-factor retrieval formula | Generative Agents | Validated in studies |
| Recency decay 0.995^hours | Generative Agents | Specific; reasonable half-life |
| Reflection trigger importance>=150 | Generative Agents | Empirically tuned |
| Citations in reflections | Generative Agents | Ensures grounding |
| Helpful/harmful counters | ACE | Enables learning |

### MEDIUM Confidence (Implement with monitoring)

| Recommendation | Source | Monitor For |
|----------------|--------|-------------|
| Heuristic importance default | Derived | Poor correlation |
| Async reflection | Derived | Consistency issues |
| Delta + last-write-wins | ACE | Edge case data loss |
| Three-level taxonomy | CoALA | Unclear boundaries |
| Iterative refinement (3 rounds) | ACE | Diminishing returns |

### LOW Confidence (Prototype first)

| Recommendation | Source | Evaluate |
|----------------|--------|----------|
| Recursive reflection | Generative Agents | Depth value |
| LLM importance scoring | Generative Agents | Cost/accuracy trade-off |
| Procedural memory execution | CoALA | Necessary structure |
| Forgetting decay rate | Derived | Optimal parameters |

---

## Open Questions

These questions are not answered by the research and need experimentation:

1. **Optimal reflection depth**: Does multi-level reflection (reflection on reflections) provide value proportional to cost?

2. **Forgetting algorithm**: What combination of TTL, access frequency, and importance threshold produces best retention?

3. **Cross-session working memory**: What state actually needs to persist between sessions vs. can be reconstructed?

4. **Weight auto-tuning**: Can feedback signals automatically optimize retrieval weights?

5. **Procedural memory execution**: How should stored workflows be triggered and executed?

6. **Multi-user learning**: Can insights from one user inform another (with privacy)?

---

## References

1. **ACE: Agentic Context Engineering** - Zhang, Q., et al. (2025)
   - Generator-Reflector-Curator pipeline
   - Delta updates and deduplication
   - Helpful/harmful feedback counters

2. **CoALA: Cognitive Architectures for Language Agents** - Sumers, T.R., et al. (2024)
   - Memory taxonomy (Working/Episodic/Semantic/Procedural)
   - Action space (Internal/External)
   - Decision cycle (Propose-Evaluate-Select-Execute)

3. **Generative Agents: Interactive Simulacra of Human Behavior** - Park, J.S., et al. (2023)
   - Memory stream architecture
   - Three-factor retrieval formula
   - Reflection mechanism with citations
   - Planning hierarchy

---

*Document generated via multi-phase research extraction pipeline. See source papers for complete details.*
