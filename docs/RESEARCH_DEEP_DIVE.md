# Engram Research Deep Dive

> Comprehensive analysis and synthesis of memory architectures for AI agents, grounding the design of engram - a portable memory graph for Claude Code interactions.

**Papers Analyzed:**
1. **CoALA**: Cognitive Architectures for Language Agents (Sumers, Yao, Narasimhan, Griffiths - Princeton, 2024)
2. **ACE**: Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models (Zhang et al. - Stanford, SambaNova, UC Berkeley, 2025)
3. **Generative Agents**: Interactive Simulacra of Human Behavior (Park et al. - Stanford, Google, 2023)

**Analysis Date:** 2025-12-09

---

## Table of Contents

1. [Paper Summaries](#paper-summaries)
2. [Memory Architecture Patterns](#memory-architecture-patterns)
3. [Retrieval Mechanisms](#retrieval-mechanisms)
4. [Learning and Ingestion](#learning-and-ingestion)
5. [Reflection and Synthesis](#reflection-and-synthesis)
6. [Cross-Paper Synthesis](#cross-paper-synthesis)
7. [Engram Design Implications](#engram-design-implications)
8. [Appendix: Key Prompts and Algorithms](#appendix-key-prompts-and-algorithms)

---

## Paper Summaries

### CoALA: Cognitive Architectures for Language Agents

**Type:** Theoretical framework with retrospective case studies

**Core Thesis:** Language agents can be understood through the lens of cognitive architectures from AI history (Soar, ACT-R), with LLMs serving as "probabilistic production systems" that map conditions (prompts) to actions (outputs).

**Key Insight:** The paper doesn't propose a new system but provides a **unifying vocabulary** for understanding existing agents. This is invaluable for engram because it gives us principled categories for memory organization.

**The CoALA Framework:**

```
                    ┌─────────────────────────────────────┐
                    │         LONG-TERM MEMORY            │
                    ├──────────┬──────────┬───────────────┤
                    │ Episodic │ Semantic │  Procedural   │
                    │ (events) │ (facts)  │ (skills/code) │
                    └────┬─────┴────┬─────┴───────┬───────┘
                         │          │             │
                    ┌────▼──────────▼─────────────▼────┐
                    │        WORKING MEMORY            │
                    │   (current context, active goals) │
                    └────────────────┬─────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
        ┌─────▼─────┐         ┌──────▼──────┐        ┌──────▼──────┐
        │ REASONING │         │  RETRIEVAL  │        │  LEARNING   │
        │ (internal)│         │  (internal) │        │  (internal) │
        └─────┬─────┘         └──────┬──────┘        └──────┬──────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  GROUNDING  │
                              │  (external) │
                              └─────────────┘
```

**Memory Types Defined:**

| Type | Contents | Persistence | Update Pattern |
|------|----------|-------------|----------------|
| **Working** | Current percepts, active goals, retrieved info, reasoning outputs | Session | Overwritten each cycle |
| **Episodic** | Past experiences, interaction traces, event sequences | Long-term | Append-only |
| **Semantic** | World knowledge, facts, inferences, reflections | Long-term | Append + occasional update |
| **Procedural** | Skills, code, prompts, retrieval procedures | Long-term | Versioned updates |

**Action Types:**

1. **Internal Actions** (within the agent):
   - **Reasoning**: Read/write working memory, generate plans, self-evaluate
   - **Retrieval**: Query long-term memory, bring relevant info to working memory
   - **Learning**: Write to long-term memory (episodic, semantic, or procedural)

2. **External Actions** (with environment):
   - **Grounding**: Tool use, API calls, dialogue, physical actions

**Decision Cycle:**
```
while goal_not_achieved:
    # Planning
    candidates = propose_actions(working_memory)
    values = evaluate_actions(candidates)
    action = select_action(candidates, values)

    # Execution
    result = execute(action)
    working_memory.update(result)

    # Optional Learning
    if should_learn(result):
        long_term_memory.update(result)
```

**Case Studies (How Real Agents Map to CoALA):**

| Agent | Memory Used | Key Innovation |
|-------|-------------|----------------|
| **ReAct** | Procedural only | Minimal: interleave reasoning + action |
| **Reflexion** | Episodic + Semantic | Learn from failure: store "lessons" |
| **Voyager** | Procedural (skill library) | Hierarchical skills, lifelong learning |
| **Generative Agents** | Episodic + Semantic | Reflection extracts semantic from episodic |
| **Tree of Thoughts** | Working only | Deliberate planning with backtracking |

---

### ACE: Agentic Context Engineering

**Type:** System paper with empirical validation

**Core Thesis:** Existing context optimization methods suffer from two failure modes:
1. **Brevity Bias**: Optimizers collapse toward short, generic prompts
2. **Context Collapse**: Iterative rewriting erases accumulated knowledge

ACE solves these by treating context as an **evolving playbook** of itemized bullets that grow incrementally.

**Key Insight:** The three-role separation (Generator/Reflector/Curator) prevents any single component from being overloaded, and delta updates preserve information across iterations.

**The ACE Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                          │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  GENERATOR  │───▶│  REFLECTOR  │───▶│   CURATOR   │     │
│  │             │    │             │    │             │     │
│  │ Executes    │    │ Analyzes    │    │ Synthesizes │     │
│  │ tasks with  │    │ trajectories│    │ insights    │     │
│  │ playbook    │    │ for errors  │    │ into deltas │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │             │
│         │                  │                  ▼             │
│         │                  │          ┌─────────────┐       │
│         │                  │          │   MERGE     │       │
│         │                  │          │ (non-LLM)   │       │
│         │                  │          └──────┬──────┘       │
│         │                  │                 │              │
│         ▼                  │                 ▼              │
│  ┌──────────────────────────────────────────────────┐      │
│  │                 CONTEXT PLAYBOOK                  │      │
│  │  ┌────────────────────────────────────────────┐  │      │
│  │  │ [ctx-00001] Strategy bullet...             │  │      │
│  │  │ [ctx-00002] Code snippet...                │  │      │
│  │  │ [ctx-00003] Pitfall warning...             │  │      │
│  │  └────────────────────────────────────────────┘  │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**The Three Roles:**

**1. Generator**
- Executes tasks using current playbook as context
- Produces execution trajectories (reasoning traces, tool calls, outputs)
- Tags which playbook bullets were helpful vs harmful during execution
- Stateless: doesn't modify playbook directly

**2. Reflector**
- Receives trajectory + ground truth (if available)
- Performs multi-round analysis to extract insights
- Outputs structured JSON with:
  - `reasoning`: Chain of thought analysis
  - `error_identification`: What went wrong
  - `root_cause_analysis`: Why it went wrong
  - `correct_approach`: What should have been done
  - `key_insight`: Generalizable principle
  - `bullet_tags`: Which existing bullets helped/hurt

**3. Curator**
- Synthesizes Reflector insights into delta updates
- Decides: create new bullet OR update existing bullet's counters
- Uses embedding similarity to detect semantic duplicates
- Non-LLM merge logic: deterministic, fast, reproducible

**Bullet Structure:**
```
[ctx-00263]
When processing time-sensitive transactions involving specific relationships:
always resolve identities from the correct source app (phone contacts), use
proper datetime range comparisons instead of string matching, and verify all
filtering criteria (relationship + time) are met before processing items.

Metadata:
  - helpful_count: 5
  - harmful_count: 0
  - embedding: [0.023, -0.156, ...]
  - section: strategies_and_hard_rules
```

**Delta vs Monolithic Update:**

| Approach | What Happens | Result |
|----------|--------------|--------|
| **Monolithic** | LLM rewrites entire context each iteration | Context collapse: 50 bullets → 5 generic rules |
| **Delta (ACE)** | Generate small change set, merge deterministically | Growth: 50 bullets → 51 bullets (preserves detail) |

**Empirical Results:**
- 10.6% improvement on AppWorld agent benchmark
- 8.6% improvement on financial reasoning (FiNER-ORD)
- 86.9% reduction in adaptation latency vs Dynamic Cheatsheet
- 75-84% reduction in token costs

**Key Innovation for Engram:** The playbook structure with metadata (ID, counters, section) is directly applicable to memory storage. The grow-and-refine deduplication via embeddings prevents memory bloat while preserving distinct insights.

---

### Generative Agents: Interactive Simulacra of Human Behavior

**Type:** System paper with empirical evaluation

**Core Thesis:** Believable human behavior requires long-term memory coherence. The paper demonstrates 25 agents living in a simulated town (Smallville) for 2 days, exhibiting emergent social behaviors like party planning, information diffusion, and relationship formation.

**Key Insight:** The memory stream + reflection + retrieval architecture produces coherent long-term behavior that pure LLM prompting cannot achieve. The three-factor retrieval scoring (recency + importance + relevance) is the key innovation.

**The Generative Agent Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATIVE AGENT                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    MEMORY STREAM                            │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │ [obs] Klaus woke up at 8:00am                        │  │ │
│  │  │ [obs] Klaus saw Maria in the café                    │  │ │
│  │  │ [obs] Klaus heard about the election from Sam        │  │ │
│  │  │ [ref] Klaus is passionate about research (1,2,5)     │  │ │
│  │  │ [plan] Today: work on paper, lunch with Maria        │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────┬──────────────────────────────────┘ │
│                            │                                     │
│       ┌────────────────────┼────────────────────┐               │
│       │                    │                    │               │
│  ┌────▼────┐         ┌─────▼─────┐        ┌─────▼─────┐        │
│  │RETRIEVAL│         │REFLECTION │        │ PLANNING  │        │
│  │         │         │           │        │           │        │
│  │ recency │         │ questions │        │ day plan  │        │
│  │+import. │         │ → insights│        │ → actions │        │
│  │+relev.  │         │ w/ cites  │        │ recursive │        │
│  └────┬────┘         └─────┬─────┘        └─────┬─────┘        │
│       │                    │                    │               │
│       └────────────────────┼────────────────────┘               │
│                            │                                     │
│                     ┌──────▼──────┐                             │
│                     │   ACTION    │                             │
│                     │             │                             │
│                     │ move, talk, │                             │
│                     │ use object  │                             │
│                     └─────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Memory Stream:**

An append-only list of memory objects, each containing:
```python
{
    "description": "Klaus heard about the mayoral election from Sam",
    "creation_timestamp": "2023-02-13 09:23:00",
    "last_access_timestamp": "2023-02-13 14:15:00",
    "importance_score": 8,  # 1-10, LLM-generated
    "embedding": [0.023, -0.156, ...],
    "memory_type": "observation",  # or "reflection" or "plan"
    "cited_memories": []  # For reflections: pointers to evidence
}
```

**Three-Factor Retrieval Scoring:**

```
score = α_recency × recency + α_importance × importance + α_relevance × relevance
```

Where (default α values = 1.0 for all):

1. **Recency** (temporal relevance):
   ```
   recency = 0.995 ^ hours_since_last_access
   ```
   - Exponential decay with ~14-day half-life
   - Uses `last_access_timestamp`, not creation time
   - Accessing a memory "refreshes" it

2. **Importance** (intrinsic significance):
   ```
   Prompt: "On a scale of 1-10, where 1 is mundane (brushing teeth)
           and 10 is extremely poignant (breakup, college acceptance),
           rate the likely poignancy of: {memory_description}"
   ```
   - Generated once at memory creation
   - Examples: "cleaning room" → 2, "asking crush on date" → 8

3. **Relevance** (contextual match):
   ```
   relevance = cosine_similarity(query_embedding, memory_embedding)
   ```
   - Standard semantic similarity
   - Query is the current situation or question

**Normalization:**
All three scores are min-max normalized to [0,1] before combining.

**Reflection Mechanism:**

Triggered when accumulated importance of recent observations exceeds threshold (default: 150).

**Step 1: Question Generation**
```
Given only the information above, what are 3 most salient high-level
questions we can answer about the subjects in the statements?

Statements:
1. Klaus woke up at 8:00am
2. Klaus saw Maria in the café
3. Klaus mentioned he's working on a research paper
...
```

**Step 2: Insight Extraction**
For each question, retrieve relevant memories and prompt:
```
Statements about Klaus:
1. Klaus mentioned he's been working on a research paper
2. Klaus spent 4 hours in the library yesterday
3. Klaus told Maria about his findings
...

What 5 high-level insights can you infer from the above statements?
(example format: insight (because of 1, 5, 3))
```

**Step 3: Storage**
Parse insights, create reflection memories with citations:
```python
{
    "description": "Klaus is deeply committed to his research",
    "memory_type": "reflection",
    "importance_score": 8,
    "cited_memories": ["mem_001", "mem_005", "mem_003"]
}
```

**Reflection Trees:**
Reflections can cite other reflections, building hierarchical abstraction:
```
Level 0: Raw observations
Level 1: First-order reflections (synthesize observations)
Level 2: Second-order reflections (synthesize reflections)
...
```

**Planning System:**

Hierarchical decomposition:
```
Day Plan (5-8 broad chunks)
  └─> Hour Plan (hour-long activities)
      └─> Action Plan (5-15 minute actions)
```

Plans are stored as memories and can be invalidated when reactions occur.

**Emergent Behaviors Observed:**
- Information diffusion (election news spread through town)
- Relationship formation (agents formed friendships based on interactions)
- Coordination (agents organized a Valentine's Day party)

**Failure Modes Documented:**
1. **Incomplete Retrieval**: Agent remembers what to do but not why
2. **Hallucinated Embellishments**: Agent adds details not in memory
3. **Overly Formal Dialogue**: Instruction-tuned models are too polite
4. **Location Inference Errors**: Over-generalization to new locations

---

## Memory Architecture Patterns

### Pattern 1: The Memory Taxonomy (CoALA)

CoALA's four-part memory taxonomy provides the conceptual foundation:

| Memory Type | Engram Mapping | Storage Strategy | Update Pattern |
|-------------|----------------|------------------|----------------|
| **Working** | Current session context | In-memory (Redis/dict) | Cleared on session end |
| **Episodic** | Session transcripts, tool calls | Postgres + pgvector | Append-only |
| **Semantic** | Extracted insights, patterns | Postgres + pgvector | Append via reflection |
| **Procedural** | Code snippets, tool patterns | Postgres | Versioned |

**Why This Taxonomy Matters:**

1. **Different retention policies**: Working memory is ephemeral, others are persistent
2. **Different retrieval patterns**: Episodic uses recency heavily, semantic uses relevance
3. **Different update semantics**: Episodic appends, semantic synthesizes, procedural versions
4. **Different storage optimizations**: Procedural may not need embeddings

### Pattern 2: The Evolving Playbook (ACE)

ACE's playbook structure provides the storage model:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENGRAM PLAYBOOK                          │
├─────────────────────────────────────────────────────────────┤
│ Section: strategies_and_patterns                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [eng-00001] When debugging async code, always check...  │ │
│ │ helpful: 12  harmful: 1  importance: 8                  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [eng-00002] For React components, prefer composition... │ │
│ │ helpful: 8   harmful: 0  importance: 7                  │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Section: code_snippets                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [eng-00003] ```python                                   │ │
│ │ async def retry_with_backoff(fn, max_retries=3):...     │ │
│ │ helpful: 5   harmful: 0  importance: 6                  │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Section: pitfalls_and_warnings                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [eng-00004] NEVER use eval() with user input...         │ │
│ │ helpful: 3   harmful: 0  importance: 10                 │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Sections for Engram:**
- `strategies_and_patterns`: High-level approaches
- `code_snippets`: Reusable code templates
- `pitfalls_and_warnings`: Things to avoid
- `project_context`: Project-specific knowledge
- `user_preferences`: Learned user patterns
- `tool_usage_patterns`: Effective tool combinations

### Pattern 3: The Memory Stream (Generative Agents)

The append-only memory stream with rich metadata:

```
Timeline ─────────────────────────────────────────────────────▶

[09:00] User asked to implement auth         importance: 8
        │
[09:05] Retrieved docs for NextAuth          importance: 3
        │
[09:15] Wrote auth middleware                importance: 7
        │                                         │
[09:20] Tests failed - missing env var       importance: 9
        │                                         │
[09:25] Fixed .env configuration             importance: 6
        │                                         │
[09:30] All tests passing                    importance: 5
        │
        ▼
[REFLECTION triggered: importance sum > 150]
        │
        ▼
[09:31] INSIGHT: Auth implementation requires careful    importance: 8
        env configuration; always verify env vars        cited: [09:20, 09:25]
        before running tests
```

---

## Retrieval Mechanisms

### The Three-Factor Formula

From Generative Agents, validated empirically:

```python
def compute_retrieval_score(memory, query, current_time):
    # Factor 1: Recency (temporal decay)
    hours_elapsed = (current_time - memory.last_access).total_seconds() / 3600
    recency = DECAY_FACTOR ** hours_elapsed  # 0.995 default

    # Factor 2: Importance (intrinsic significance)
    # Either pre-computed or feedback-adjusted
    importance = memory.importance_score / 10.0

    # Factor 3: Relevance (semantic similarity)
    relevance = cosine_similarity(query.embedding, memory.embedding)

    # Normalize each to [0, 1] across all memories
    # Then combine with equal weights
    return (normalize(recency) + normalize(importance) + normalize(relevance)) / 3
```

### Why Three Factors?

**Recency alone fails:** Old but important memories disappear
**Importance alone fails:** Irrelevant memories dominate
**Relevance alone fails:** Recent context is lost

**Combined scoring provides:**
- Temporal coherence (recency)
- Significance awareness (importance)
- Contextual appropriateness (relevance)

### Decay Function Analysis

The exponential decay `0.995^hours`:

| Time Elapsed | Recency Score | Interpretation |
|--------------|---------------|----------------|
| 1 hour | 0.995 | Almost full weight |
| 24 hours | 0.887 | ~89% weight |
| 1 week | 0.714 | ~71% weight |
| 2 weeks | 0.510 | ~51% weight (half-life) |
| 1 month | 0.260 | ~26% weight |
| 2 months | 0.068 | ~7% weight |

**Implications for Engram:**
- Memories older than 2 months are effectively "forgotten" unless:
  - They have high importance (compensates for low recency)
  - They're promoted to semantic memory (no decay)
  - They're explicitly accessed (refreshes timestamp)

### Retrieval SQL Pattern

```sql
WITH scored_memories AS (
    SELECT
        id,
        content,
        memory_type,
        -- Recency: exponential decay
        POWER(0.995, EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 3600) AS recency_raw,
        -- Importance: feedback-adjusted
        (importance_score + (helpful_count - harmful_count) * 0.5) / 10.0 AS importance_raw,
        -- Relevance: cosine similarity (pgvector)
        1 - (embedding <=> $1) AS relevance_raw
    FROM observations
    WHERE user_id = $2
      AND is_synthetic = FALSE
),
normalized AS (
    SELECT
        *,
        (recency_raw - MIN(recency_raw) OVER()) / NULLIF(MAX(recency_raw) OVER() - MIN(recency_raw) OVER(), 0) AS recency_norm,
        (importance_raw - MIN(importance_raw) OVER()) / NULLIF(MAX(importance_raw) OVER() - MIN(importance_raw) OVER(), 0) AS importance_norm,
        (relevance_raw - MIN(relevance_raw) OVER()) / NULLIF(MAX(relevance_raw) OVER() - MIN(relevance_raw) OVER(), 0) AS relevance_norm
    FROM scored_memories
)
SELECT
    id,
    content,
    memory_type,
    (COALESCE(recency_norm, 0) + COALESCE(importance_norm, 0) + COALESCE(relevance_norm, 0)) / 3 AS score
FROM normalized
ORDER BY score DESC
LIMIT $3;
```

---

## Learning and Ingestion

### The Delta Update Pattern (ACE)

**Problem:** Monolithic updates cause context collapse
**Solution:** Generate small deltas, merge deterministically

```python
class Delta:
    new_memories: List[Memory]      # Create these
    updated_memories: List[Memory]  # Modify counters on these

def ingest_observation(observation: str, context: List[Memory]) -> Delta:
    """
    Curator logic: decide whether to create new or update existing
    """
    delta = Delta(new_memories=[], updated_memories=[])

    # Generate embedding for new observation
    obs_embedding = embed(observation)

    # Check for semantic duplicates
    for memory in context:
        similarity = cosine_similarity(obs_embedding, memory.embedding)
        if similarity >= SIMILARITY_THRESHOLD:  # 0.90
            # Update existing memory's counters
            memory.helpful_count += 1
            delta.updated_memories.append(memory)
            return delta

    # No duplicate found: create new memory
    new_memory = Memory(
        id=generate_id(),
        content=observation,
        embedding=obs_embedding,
        importance_score=compute_importance(observation),
        memory_type="episodic",
        helpful_count=1,
        harmful_count=0
    )
    delta.new_memories.append(new_memory)
    return delta

def merge_delta(context: List[Memory], delta: Delta) -> List[Memory]:
    """
    Non-LLM merge: deterministic, fast, reproducible
    """
    context_map = {m.id: m for m in context}

    # Apply updates
    for updated in delta.updated_memories:
        context_map[updated.id] = updated

    # Add new
    for new in delta.new_memories:
        context_map[new.id] = new

    return list(context_map.values())
```

### Importance Scoring Strategies

**Option 1: LLM-Scored (Generative Agents)**
```python
def compute_importance_llm(observation: str) -> float:
    prompt = f"""On a scale of 1-10, where 1 is mundane (e.g., brushing teeth)
    and 10 is extremely significant (e.g., major bug fix, architecture decision),
    rate the importance of this observation for a coding assistant:

    Observation: {observation}

    Rating:"""

    response = llm.complete(prompt)
    return float(response.strip())
```

**Option 2: Rule-Based (Recommended for MVP)**
```python
IMPORTANCE_HEURISTICS = {
    "error": 9,           # Errors are highly important
    "user_instruction": 10, # User requests are top priority
    "code_change": 7,     # Code modifications matter
    "test_result": 6,     # Test outcomes inform quality
    "tool_output": 3,     # Tool outputs are context
    "observation": 5,     # Default for general observations
}

def compute_importance_heuristic(observation: str, obs_type: str) -> float:
    base = IMPORTANCE_HEURISTICS.get(obs_type, 5)

    # Adjustments based on content
    if "CRITICAL" in observation or "BREAKING" in observation:
        base = min(base + 2, 10)
    if "TODO" in observation or "FIXME" in observation:
        base = min(base + 1, 10)

    return base
```

**Option 3: Feedback-Driven (ACE pattern)**
```python
def compute_importance_feedback(memory: Memory) -> float:
    """Adjust importance based on actual usage"""
    base = memory.importance_score
    feedback_adjustment = (memory.helpful_count - memory.harmful_count) * 0.5
    return max(0, min(10, base + feedback_adjustment))
```

### Deduplication via Embeddings

```python
SIMILARITY_THRESHOLD = 0.90  # High to avoid false positives

def deduplicate_memories(memories: List[Memory]) -> List[Memory]:
    """
    Merge semantically similar memories
    Run periodically (every hour) or on-demand
    """
    if not memories:
        return memories

    deduplicated = []
    merged_ids = set()

    for i, mem_i in enumerate(memories):
        if mem_i.id in merged_ids:
            continue

        similar_group = [mem_i]

        for j, mem_j in enumerate(memories[i+1:], start=i+1):
            if mem_j.id in merged_ids:
                continue

            similarity = cosine_similarity(mem_i.embedding, mem_j.embedding)
            if similarity >= SIMILARITY_THRESHOLD:
                similar_group.append(mem_j)
                merged_ids.add(mem_j.id)

        # Merge the group
        merged = merge_memory_group(similar_group)
        deduplicated.append(merged)

    return deduplicated

def merge_memory_group(memories: List[Memory]) -> Memory:
    """Combine similar memories into one"""
    # Keep the most recent content
    memories.sort(key=lambda m: m.timestamp, reverse=True)
    primary = memories[0]

    # Sum counters
    total_helpful = sum(m.helpful_count for m in memories)
    total_harmful = sum(m.harmful_count for m in memories)

    # Average importance
    avg_importance = sum(m.importance_score for m in memories) / len(memories)

    return Memory(
        id=primary.id,
        content=primary.content,
        embedding=primary.embedding,
        importance_score=avg_importance,
        helpful_count=total_helpful,
        harmful_count=total_harmful,
        memory_type=primary.memory_type,
        timestamp=primary.timestamp
    )
```

---

## Reflection and Synthesis

### Triggering Reflection

**Generative Agents approach:** Threshold-based on accumulated importance

```python
REFLECTION_THRESHOLD = 150

class ReflectionTrigger:
    def __init__(self):
        self.accumulated_importance = 0
        self.last_reflection_time = datetime.now()

    def record_observation(self, importance: float):
        self.accumulated_importance += importance

    def should_reflect(self) -> bool:
        return self.accumulated_importance >= REFLECTION_THRESHOLD

    def mark_reflected(self):
        self.accumulated_importance = 0
        self.last_reflection_time = datetime.now()
```

**Hybrid approach for Engram:**
```python
def should_reflect(
    accumulated_importance: float,
    observations_since_last: int,
    hours_since_last: float
) -> bool:
    # Threshold-based (Generative Agents)
    if accumulated_importance >= 150:
        return True

    # Count-based (predictable)
    if observations_since_last >= 100:
        return True

    # Time-based (minimum frequency)
    if hours_since_last >= 24:
        return True

    return False
```

### The Reflection Algorithm

**Step 1: Question Generation**
```python
def generate_reflection_questions(recent_memories: List[Memory], n: int = 3) -> List[str]:
    # Format memories as statements
    statements = "\n".join(f"{i+1}. {m.content}" for i, m in enumerate(recent_memories[:100]))

    prompt = f"""Given only the information below, what are {n} most salient
high-level questions we can answer about patterns, learnings, or insights?

Statements:
{statements}

Questions:"""

    response = llm.complete(prompt)
    return parse_questions(response)
```

**Step 2: Insight Extraction with Citations**
```python
def extract_insights(question: str, relevant_memories: List[Memory], n: int = 5) -> List[Insight]:
    # Format memories with indices for citation
    statements = "\n".join(f"{i+1}. {m.content}" for i, m in enumerate(relevant_memories))

    prompt = f"""Based on these statements, answer the question with {n} high-level insights.
Format each insight as: insight text (because of 1, 5, 3)

Question: {question}

Statements:
{statements}

Insights:"""

    response = llm.complete(prompt)
    return parse_insights_with_citations(response, relevant_memories)
```

**Step 3: Store Reflections**
```python
def store_reflections(insights: List[Insight], memory_store: MemoryStore):
    for insight in insights:
        reflection = Memory(
            id=generate_id(),
            content=insight.text,
            embedding=embed(insight.text),
            importance_score=8,  # Reflections are inherently important
            memory_type="semantic",  # Promoted from episodic
            is_synthetic=True,  # Prevent reflection loops
            citations=[m.id for m in insight.cited_memories]
        )
        memory_store.add(reflection)
```

### ACE's Iterative Refinement

For critical insights, ACE uses multiple refinement rounds:

```python
def refine_insight(insight: str, trajectory: str, max_rounds: int = 5) -> str:
    current = insight

    for round_num in range(max_rounds):
        # Check if sufficiently detailed
        if is_insight_sufficient(current):
            break

        prompt = f"""Previous insight:
{current}

Original trajectory:
{trajectory}

Please refine this insight to be more specific and actionable.
Focus on:
1. Concrete root causes
2. Specific corrective actions
3. Generalizable principles

Refined insight:"""

        current = llm.complete(prompt)

    return current

def is_insight_sufficient(insight: str) -> bool:
    # Heuristic: check length and specificity
    return (
        len(insight) >= 100 and
        any(keyword in insight.lower() for keyword in ["because", "when", "always", "never", "should"])
    )
```

---

## Cross-Paper Synthesis

### Convergent Findings

These principles appear across all three papers:

| Principle | CoALA | ACE | Generative Agents |
|-----------|-------|-----|-------------------|
| **Memory is modular** | 4 types | Sectioned playbook | Stream + reflections |
| **Append, don't overwrite** | Learn module appends | Delta updates | Append-only stream |
| **Separate retrieval from storage** | Retrieval as action | Generator reads playbook | Retrieval module |
| **Synthesis requires separate process** | Learn module | Reflector + Curator | Reflection mechanism |
| **Metadata enriches memories** | Timestamps, types | IDs, counters, sections | Timestamps, importance |

### Complementary Contributions

Each paper adds unique value:

| Paper | Unique Contribution | Engram Application |
|-------|--------------------|--------------------|
| **CoALA** | Memory taxonomy, decision cycle | Structural organization |
| **ACE** | Delta updates, feedback counters | Ingestion pipeline |
| **Generative Agents** | Three-factor scoring, reflection | Retrieval and synthesis |

### Resolved Tensions

The papers surface trade-offs that engram must navigate:

**1. Simplicity vs Richness**
- CoALA: Rich taxonomy (4 types)
- Generative Agents: Simple stream (1 type with subtypes)
- **Resolution:** Start simple (stream), add types as needed

**2. LLM vs Rule-Based**
- Generative Agents: LLM scores importance
- ACE: Feedback-driven adjustment
- **Resolution:** Rule-based initial + feedback adjustment

**3. Eager vs Lazy Processing**
- ACE: Eager deduplication
- Generative Agents: Lazy reflection (threshold-triggered)
- **Resolution:** Lazy for expensive ops (reflection), eager for cheap (dedup on write)

---

## Engram Design Implications

### Architecture Decision Record

**ADR-001: Memory Type Taxonomy**
- **Decision:** Use CoALA's 4-type taxonomy (working/episodic/semantic/procedural)
- **Rationale:** Clean separation of concerns, different retention policies
- **Implementation:** `memory_type` enum column in Postgres

**ADR-002: Storage Backend**
- **Decision:** Postgres + pgvector (not dedicated vector DB)
- **Rationale:** Simpler ops, good enough for expected scale, rich query capabilities
- **Trade-off:** May need to revisit at >1M memories

**ADR-003: Retrieval Scoring**
- **Decision:** Three-factor (recency + importance + relevance) with equal weights
- **Rationale:** Empirically validated in Generative Agents
- **Implementation:** SQL query with pgvector cosine distance

**ADR-004: Ingestion Pattern**
- **Decision:** Delta updates (ACE pattern) with deduplication
- **Rationale:** Prevents context collapse, preserves detail
- **Implementation:** Embedding similarity check on write

**ADR-005: Importance Scoring**
- **Decision:** Rule-based heuristics + feedback adjustment
- **Rationale:** Avoids LLM cost, enables self-improvement
- **Implementation:** `observation_type` → base score, feedback counters adjust

**ADR-006: Reflection Triggering**
- **Decision:** Hybrid (threshold + count + time)
- **Rationale:** Balances responsiveness with predictability
- **Implementation:** Background job checks all three conditions

**ADR-007: Deduplication Threshold**
- **Decision:** 0.90 cosine similarity
- **Rationale:** Conservative to minimize false positives
- **Implementation:** Embedding comparison on write + periodic batch

### Data Model

```sql
-- Core memory table
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id UUID,

    -- Content
    content TEXT NOT NULL,
    embedding VECTOR(1536),

    -- Classification
    memory_type VARCHAR(20) NOT NULL CHECK (memory_type IN ('working', 'episodic', 'semantic', 'procedural')),
    observation_type VARCHAR(50),  -- 'error', 'instruction', 'code_change', etc.
    section VARCHAR(50),  -- ACE-style sectioning

    -- Scoring
    importance_score FLOAT NOT NULL CHECK (importance_score BETWEEN 0 AND 10),
    helpful_count INT DEFAULT 0,
    harmful_count INT DEFAULT 0,

    -- Provenance
    is_synthetic BOOLEAN DEFAULT FALSE,
    citations JSONB,  -- Array of memory IDs

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Extensibility
    metadata JSONB
);

-- Indexes
CREATE INDEX idx_memories_user ON memories(user_id);
CREATE INDEX idx_memories_session ON memories(session_id);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_accessed ON memories(last_accessed_at DESC);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Reflection tracking
CREATE TABLE reflection_state (
    user_id UUID PRIMARY KEY,
    accumulated_importance FLOAT DEFAULT 0,
    observations_since_last INT DEFAULT 0,
    last_reflection_at TIMESTAMPTZ DEFAULT NOW()
);
```

### FastMCP API Design

```python
from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("engram")

class SearchResult(BaseModel):
    id: str
    content: str
    memory_type: str
    score: float
    importance: float
    citations: list[str] | None

class IngestResult(BaseModel):
    id: str
    action: str  # "created" or "updated"
    importance: float

@mcp.tool()
async def memory_search(
    query: str,
    limit: int = 10,
    memory_types: list[str] | None = None,
    session_only: bool = False,
    min_importance: float = 0
) -> list[SearchResult]:
    """
    Search memories using three-factor scoring (recency + importance + relevance).

    Args:
        query: Natural language search query
        limit: Maximum results to return (default 10)
        memory_types: Filter by type ('episodic', 'semantic', 'procedural')
        session_only: Only search current session's memories
        min_importance: Minimum importance threshold

    Returns:
        List of matching memories with scores
    """
    pass

@mcp.tool()
async def memory_ingest(
    observation: str,
    observation_type: str = "general",
    importance: float | None = None,
    section: str | None = None
) -> IngestResult:
    """
    Ingest a new observation into the memory stream.

    Args:
        observation: The content to remember
        observation_type: Category ('error', 'instruction', 'code_change', etc.)
        importance: Override auto-computed importance (1-10)
        section: ACE-style section ('strategies', 'snippets', 'pitfalls')

    Returns:
        Result indicating if memory was created or merged with existing
    """
    pass

@mcp.tool()
async def memory_feedback(
    memory_id: str,
    helpful: bool
) -> dict:
    """
    Provide feedback on a memory's usefulness.

    Args:
        memory_id: ID of the memory to rate
        helpful: True if memory was helpful, False if harmful

    Returns:
        Updated memory stats
    """
    pass

@mcp.tool()
async def memory_reflect(
    focus: str | None = None,
    max_insights: int = 5,
    force: bool = False
) -> list[dict]:
    """
    Trigger reflection to synthesize insights from recent memories.

    Args:
        focus: Optional topic to focus reflection on
        max_insights: Maximum insights to generate
        force: Run even if threshold not met

    Returns:
        List of generated insights with citations
    """
    pass

@mcp.tool()
async def memory_forget(
    memory_id: str | None = None,
    older_than_days: int | None = None,
    memory_type: str | None = None
) -> dict:
    """
    Remove memories (use carefully).

    Args:
        memory_id: Specific memory to delete
        older_than_days: Delete memories older than N days
        memory_type: Only delete this type

    Returns:
        Count of deleted memories
    """
    pass
```

---

## Appendix: Key Prompts and Algorithms

### A. Importance Scoring Prompt (Generative Agents)

```
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth,
making bed) and 10 is extremely poignant (e.g., a break up, college
acceptance), rate the likely poignancy of the following piece of memory.

Memory: {memory_description}
Rating: <fill in>
```

### B. Reflection Question Generation Prompt

```
Given only the information above, what are 3 most salient high-level
questions we can answer about the subjects in the statements?

Statements:
1. {statement_1}
2. {statement_2}
...
n. {statement_n}

Questions:
```

### C. Insight Extraction Prompt

```
Statements about {subject}:
1. {statement_1}
2. {statement_2}
...

What 5 high-level insights can you infer from the above statements?
(example format: insight (because of 1, 5, 3))
```

### D. ACE Reflector Prompt

```
You are an expert analyst. Your job is to diagnose why the agent's
reasoning went wrong by analyzing the gap between predicted answer
and ground truth.

Agent Trajectory:
{trajectory}

Agent's Predicted Answer:
{predicted}

Ground Truth Answer:
{ground_truth}

Current Playbook Bullets Used:
{bullets}

Provide analysis in JSON format:
{
  "reasoning": "[Detailed analysis]",
  "error_identification": "[What went wrong]",
  "root_cause_analysis": "[Why it went wrong]",
  "correct_approach": "[What should have been done]",
  "key_insight": "[Generalizable principle]",
  "bullet_tags": [{"id": "ctx-001", "tag": "helpful"}]
}
```

### E. Three-Factor Retrieval Algorithm

```python
def retrieve(query: str, memories: List[Memory], k: int = 10) -> List[Memory]:
    """Complete retrieval algorithm"""
    current_time = datetime.now()
    query_embedding = embed(query)

    # Score all memories
    scored = []
    for m in memories:
        # Update last access for retrieved memories
        hours = (current_time - m.last_accessed_at).total_seconds() / 3600

        recency = 0.995 ** hours
        importance = (m.importance_score + (m.helpful_count - m.harmful_count) * 0.5) / 10
        relevance = cosine_similarity(query_embedding, m.embedding)

        scored.append((m, recency, importance, relevance))

    # Normalize each factor
    recencies = [s[1] for s in scored]
    importances = [s[2] for s in scored]
    relevances = [s[3] for s in scored]

    def normalize(values):
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return [0.5] * len(values)
        return [(v - min_v) / (max_v - min_v) for v in values]

    norm_rec = normalize(recencies)
    norm_imp = normalize(importances)
    norm_rel = normalize(relevances)

    # Combine with equal weights
    final_scores = []
    for i, (m, _, _, _) in enumerate(scored):
        score = (norm_rec[i] + norm_imp[i] + norm_rel[i]) / 3
        final_scores.append((m, score))

    # Sort and return top-k
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # Update last_accessed_at for returned memories
    results = []
    for m, score in final_scores[:k]:
        m.last_accessed_at = current_time
        results.append(m)

    return results
```

### F. Delta Ingestion Algorithm

```python
SIMILARITY_THRESHOLD = 0.90

def ingest(observation: str, obs_type: str, context: List[Memory]) -> Delta:
    """ACE-style delta ingestion"""
    embedding = embed(observation)

    # Check for duplicates
    for memory in context:
        if memory.memory_type != "episodic":
            continue
        similarity = cosine_similarity(embedding, memory.embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            # Update existing
            memory.helpful_count += 1
            return Delta(updated=[memory])

    # Create new
    importance = IMPORTANCE_HEURISTICS.get(obs_type, 5)
    new_memory = Memory(
        id=str(uuid4()),
        content=observation,
        embedding=embedding,
        memory_type="episodic",
        observation_type=obs_type,
        importance_score=importance,
        helpful_count=1,
        harmful_count=0
    )
    return Delta(created=[new_memory])
```

---

## References

1. Sumers, T.R., Yao, S., Narasimhan, K., Griffiths, T.L. (2024). **Cognitive Architectures for Language Agents**. Transactions on Machine Learning Research.

2. Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H. (2025). **Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models**. arXiv:2510.04618.

3. Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S. (2023). **Generative Agents: Interactive Simulacra of Human Behavior**. UIST '23.

---

*This document synthesizes research to guide engram implementation. It should be updated as the system evolves and new research emerges.*
