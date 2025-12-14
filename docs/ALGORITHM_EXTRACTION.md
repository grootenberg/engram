# Algorithm Extraction: Living Memory Architecture for Claude Code

> Comprehensive extraction of algorithms, pseudocode, complexity analysis, and implementation constraints from ACE, CoALA, and Generative Agents papers for implementing a persistent, evolving context system for Claude Code.

**Target System**: Engram - Living memory graph for Claude Code CLI
**Analysis Date**: 2025-12-11
**Source Papers**:
1. ACE: Agentic Context Engineering (Zhang et al., 2025) 
2. CoALA: Cognitive Architectures for Language Agents (Sumers et al., 2024)
3. Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)

---

## Executive Summary

This document extracts 12 core algorithms from three research papers, providing:
- Complete pseudocode for each algorithm
- Time/space/token complexity analysis
- Data structures and dependencies
- Implementation constraints and pitfalls
- Claude Code-specific adaptations

**Quick Reference: Algorithm Categories**

| Category | Algorithms | Primary Source |
|----------|-----------|----------------|
| **Memory Retrieval** | Three-Factor Scoring, Recency Decay, Importance Scoring | Generative Agents |
| **Memory Ingestion** | Delta Update with Dedup, Feedback Tracking | ACE |
| **Reflection & Synthesis** | Trigger Check, Question Generation, Insight Extraction, Iterative Refinement | Generative Agents + ACE |
| **Decision & Planning** | Propose-Evaluate-Select, Recursive Decomposition, Deviation Detection | CoALA + Generative Agents |

---

## Table of Contents

1. [Core Retrieval Algorithms](#core-retrieval-algorithms)
2. [Ingestion Algorithms](#ingestion-algorithms)
3. [Reflection Algorithms](#reflection-algorithms)
4. [Decision & Planning Algorithms](#decision--planning-algorithms)
5. [Data Structures](#data-structures)
6. [Complexity Summary](#complexity-summary)
7. [Dependencies](#dependencies)
8. [Claude Code Adaptations](#claude-code-adaptations)

---

## Core Retrieval Algorithms

### Algorithm 1: Three-Factor Memory Retrieval (Generative Agents)

**Purpose**: Retrieve most relevant memories by combining recency, importance, and semantic relevance

**Complexity**: O(n log k + E) where n=memories, k=results, E=embedding time (~50-200ms)

**Formula**:
```
score = α_recency × recency_norm + α_importance × importance_norm + α_relevance × relevance_norm

where:
  recency = 0.995^hours_since_access
  importance = (base_score + (helpful - harmful) * 0.5) / 10
  relevance = cosine_similarity(query_embedding, memory_embedding)
  *_norm = min-max normalization across all candidates
```

**Pseudocode**:
```python
def retrieve_memories(query: str, memories: List[Memory], k: int = 10) -> List[Memory]:
    # 1. Embed query (1 API call, ~50-200ms)
    query_emb = embed(query)  
    
    # 2. Score all memories O(n)
    for mem in memories:
        hours = (now - mem.last_accessed_at).hours
        recency_raw = 0.995 ** hours
        importance_raw = (mem.importance + (mem.helpful - mem.harmful) * 0.5) / 10
        relevance_raw = cosine_sim(query_emb, mem.embedding)
    
    # 3. Normalize to [0,1] O(n)
    recency_norm = normalize(all_recency_raw)
    importance_norm = normalize(all_importance_raw)
    relevance_norm = normalize(all_relevance_raw)
    
    # 4. Combine with weights (default 0.33 each) O(n)
    final_score = 0.33*recency_norm + 0.33*importance_norm + 0.33*relevance_norm
    
    # 5. Sort and return top-k O(n log k)
    results = heapq.nlargest(k, memories, key=final_score)
    
    # 6. Update access time for returned memories O(k)
    for mem in results:
        mem.last_accessed_at = now
    
    return results
```

**Key Implementation Details**:
- **Recency refresh**: Accessing a memory updates its `last_accessed_at`, keeping it fresh
- **Normalization required**: Without normalization, one factor dominates (typically relevance)
- **Zero-division handling**: If all values equal for a factor, set all to 0.5
- **Optimization**: Use pgvector IVFFlat index to pre-filter to top-100 by relevance, then full scoring

**Complexity Breakdown**:

| Operation | Time | Space | I/O | Tokens |
|-----------|------|-------|-----|--------|
| Embed query | O(E) | O(1536) | 1 API | ~200 |
| Score memories | O(n·1536) | O(n) | 0 | 0 |
| Normalize | O(n) | O(n) | 0 | 0 |
| Sort top-k | O(n log k) | O(k) | 0 | 0 |
| Update access | O(k) | O(1) | k writes | 0 |
| **Total** | **O(n log k + E)** | **O(n)** | **1 API + k DB** | **~200** |

**Pitfalls**:
- Embedding model change breaks all similarity scores (must re-embed all memories)
- Very old memories (>60 days) decay to ~5% even if important (mitigate: promote to semantic via reflection)
- Concurrent access updates may cause race conditions (use transactions)
- All equal scores causes division by zero in normalization (handle gracefully)

---

### Algorithm 2: Exponential Recency Decay (Generative Agents)

**Purpose**: Calculate temporal relevance of memories

**Complexity**: O(1)

**Formula**:
```
recency(t) = decay_factor^hours_elapsed
where decay_factor = 0.995 (default)

Half-life: ln(0.5) / ln(0.995) ≈ 138.6 hours ≈ 5.8 days
```

**Decay Schedule**:

| Time | Recency | Meaning |
|------|---------|---------|
| 0 hours | 1.000 | Just accessed |
| 1 hour | 0.995 | 99.5% relevance |
| 24 hours | 0.887 | ~89% relevance |
| 7 days | 0.714 | ~71% relevance |
| 14 days | 0.510 | 51% relevance (half-life) |
| 30 days | 0.260 | 26% relevance |
| 60 days | 0.068 | ~7% relevance |
| 90 days | 0.018 | ~2% relevance (effectively forgotten) |

**Pseudocode**:
```python
def compute_recency(last_accessed: datetime, now: datetime, decay: float = 0.995) -> float:
    hours = (now - last_accessed).total_seconds() / 3600
    return decay ** hours
```

**Implementation Notes**:
- For hours > 2000, return 0.0 directly (optimization, avoids underflow)
- Never returns exactly 0 (asymptotic decay), but <0.001 for old memories
- Accessing a memory resets its decay (last_accessed updated)
- Use integer hours for discrete bucketing if needed

**Alternative Decay Curves**:

| Decay Factor | Half-Life | Use Case |
|--------------|-----------|----------|
| 0.999 | 693 hours (29 days) | Long-term memory (coding projects) |
| 0.995 | 139 hours (6 days) | Medium-term (default) |
| 0.99 | 69 hours (3 days) | Short-term (rapid iteration) |
| 0.95 | 13.5 hours (0.5 days) | Very short-term (single session) |

---

### Algorithm 3: Importance Scoring (Hybrid: Heuristic + Feedback)

**Purpose**: Assign intrinsic significance score to observations

**Complexity**: O(m) where m = observation length (for keyword scan)

**Two-Stage Approach**:

**Stage 1: Heuristic Base Score**
```python
IMPORTANCE_HEURISTICS = {
    "instruction": 10,      # User directives highest
    "error": 9,             # Errors critical
    "decision": 8,          # Architectural decisions
    "code_change": 7,       # Code modifications
    "insight": 7,           # Synthesized insights
    "test_result": 6,       # Test outcomes
    "general": 5,           # Baseline
    "tool_output": 3,       # Context only
}

def base_importance(obs_type: str, content: str) -> float:
    score = IMPORTANCE_HEURISTICS.get(obs_type, 5)
    
    # Content boosts
    if any(kw in content.upper() for kw in ["CRITICAL", "BREAKING", "SECURITY"]):
        score = min(score + 2, 10)
    if any(kw in content.upper() for kw in ["TODO", "FIXME", "HACK"]):
        score = min(score + 1, 10)
    
    return score
```

**Stage 2: Feedback Adjustment (ACE Pattern)**
```python
def effective_importance(memory: Memory) -> float:
    base = memory.importance_score
    feedback = (memory.helpful_count - memory.harmful_count) * 0.5
    return max(0, min(10, base + feedback))
```

**LLM Alternative** (more expensive):
```python
def llm_importance(observation: str) -> float:
    """
    Complexity: O(L) where L = LLM inference (~500-2000ms)
    Token usage: ~100-200 tokens
    """
    prompt = f"""Rate 1-10 (1=mundane, 10=critical) for a coding assistant:
    
    {observation}
    
    Rating:"""
    
    response = llm.complete(prompt)  # ~1s, $0.015 per 1M tokens
    return float(response.strip())
```

**Recommendation**: Use heuristic for all initial scores (fast, free), reserve LLM for uncertain cases.

---

## Ingestion Algorithms

### Algorithm 4: Delta Ingestion with Semantic Deduplication (ACE)

**Purpose**: Append new observations while preventing semantic duplicates

**Complexity**: 
- Unoptimized: O(n·d + E) where n=memories, d=embedding dim (1536)
- With vector index: O(log n·d + E)

**Core Idea**: Generate embedding for new observation, check if similar memory exists. If yes, increment its `helpful_count`. If no, create new memory.

**Pseudocode**:
```python
SIMILARITY_THRESHOLD = 0.90  # High to avoid false positives

def ingest_observation(obs: str, obs_type: str) -> Delta:
    """
    Returns Delta with either:
    - created: [new_memory] (if unique)
    - updated: [existing_memory] (if duplicate found)
    """
    # 1. Generate embedding (1 API call)
    obs_emb = embed(obs)  # ~50-200ms
    
    # 2. Find most similar episodic memory
    # Option A: Linear scan O(n·d)
    similar = find_most_similar(obs_emb, memories, threshold=0.90)
    
    # Option B: Vector index O(log n·d) - PREFERRED
    candidates = db.query("""
        SELECT * FROM memories 
        WHERE memory_type = 'episodic'
        ORDER BY embedding <=> $1 
        LIMIT 5
    """, obs_emb)
    
    for cand in candidates:
        sim = cosine_sim(obs_emb, cand.embedding)
        if sim >= SIMILARITY_THRESHOLD:
            similar = cand
            break
    
    # 3. Update or create
    if similar:
        similar.helpful_count += 1
        return Delta(updated=[similar])
    else:
        importance = compute_importance(obs, obs_type)
        new_mem = Memory(
            content=obs,
            embedding=obs_emb,
            memory_type="episodic",
            observation_type=obs_type,
            importance_score=importance,
            helpful_count=1,
            harmful_count=0
        )
        return Delta(created=[new_mem])

def merge_delta(memories: List[Memory], delta: Delta) -> List[Memory]:
    """Deterministic non-LLM merge O(n)"""
    mem_map = {m.id: m for m in memories}
    
    for updated in delta.updated:
        mem_map[updated.id] = updated
    
    for created in delta.created:
        mem_map[created.id] = created
    
    return list(mem_map.values())
```

**Threshold Selection**:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.95+ | Very conservative, duplicates likely | High-precision needs |
| 0.90 | Recommended default | Balanced |
| 0.85 | Moderate merging | Storage-constrained |
| <0.80 | Aggressive merging, false positives | Not recommended |

**Pitfalls**:
- Threshold too low: Merges distinct observations (information loss)
- Threshold too high: Storage bloat from near-duplicates
- Embedding model change: All similarities invalid (must re-embed)
- Race condition: Two concurrent ingestions create duplicates (use DB transactions)

**Optimization**: Pgvector IVFFlat Index
```sql
CREATE INDEX idx_memories_embedding 
ON memories USING ivfflat(embedding vector_cosine_ops) 
WITH (lists = 100);

-- Reduces similarity search from O(n) to O(log n)
-- lists = sqrt(n) where n = expected row count
```

---

### Algorithm 5: Feedback Counter Update (ACE)

**Purpose**: Track helpful/harmful signals to adjust importance dynamically

**Complexity**: O(1)

**Pseudocode**:
```python
def update_feedback(memory_id: str, helpful: bool):
    mem = db.get(memory_id)
    
    if helpful:
        mem.helpful_count += 1
    else:
        mem.harmful_count += 1
    
    db.update(mem)
    
    # Effective importance computed at retrieval time:
    # effective = base + (helpful - harmful) * 0.5
    # Clamped to [0, 10]
```

**Feedback Impact Examples**:

| Base | Helpful | Harmful | Effective |
|------|---------|---------|-----------|
| 5 | 0 | 0 | 5.0 |
| 5 | 10 | 0 | 10.0 (capped) |
| 5 | 0 | 10 | 0.0 (floored) |
| 7 | 5 | 2 | 8.5 |
| 8 | 3 | 5 | 7.0 |

**Pitfalls**:
- Inflation: Over time, popular memories drift to importance=10 (consider decay)
- Spam: Same agent repeatedly marks memory helpful (rate limit)
- No context: Old feedback weighted same as recent (consider time-weighting)

**Improvement: Time-Weighted Feedback**
```python
def effective_importance_timeweighted(mem: Memory, now: datetime) -> float:
    base = mem.importance_score
    
    # Decay feedback influence over time
    feedback_age_hours = (now - mem.last_feedback_at).hours
    feedback_weight = 0.99 ** feedback_age_hours  # Slower decay than recency
    
    feedback_delta = (mem.helpful - mem.harmful) * 0.5 * feedback_weight
    
    return max(0, min(10, base + feedback_delta))
```

---

## Reflection Algorithms

### Algorithm 6: Multi-Condition Reflection Trigger (Generative Agents)

**Purpose**: Decide when to synthesize episodic memories into semantic insights

**Complexity**: O(1)

**Triggers** (ANY of these):
```python
IMPORTANCE_THRESHOLD = 150   # Sum of importance scores
OBSERVATION_THRESHOLD = 100  # Count of observations
TIME_THRESHOLD_HOURS = 24    # Hours since last reflection

def should_reflect(state: ReflectionState, force: bool = False) -> tuple[bool, str]:
    if force:
        return (True, "force_triggered")
    
    if state.accumulated_importance >= 150:
        return (True, "importance_threshold")
    
    if state.observations_since_last >= 100:
        return (True, "observation_threshold")
    
    hours = (now - state.last_reflection_at).hours
    if hours >= 24:
        return (True, "time_threshold")
    
    return (False, "no_threshold_met")
```

**State Management**:
```python
class ReflectionState:
    user_id: str
    accumulated_importance: float = 0
    observations_since_last: int = 0
    last_reflection_at: datetime = now()
    
    def record_observation(self, importance: float):
        self.accumulated_importance += importance
        self.observations_since_last += 1
    
    def reset_after_reflection(self):
        self.accumulated_importance = 0
        self.observations_since_last = 0
        self.last_reflection_at = now()
```

**Thresholds Rationale**:
- **Importance=150**: ~15-30 significant observations (errors, decisions) or ~30-50 moderate ones
- **Observations=100**: Ensures reflection at least every 100 actions, even if low importance
- **Time=24h**: Prevents staleness, ensures daily synthesis for active users

---

### Algorithm 7: Reflection Question Generation (Generative Agents)

**Purpose**: Generate high-level questions from recent episodic memories

**Complexity**: O(L + n·m) where L=LLM time (~1-3s), n=memories, m=avg length

**Token Usage**: ~500-1500 input, ~100-300 output

**Pseudocode**:
```python
def generate_questions(recent_memories: List[Memory], n: int = 3) -> List[str]:
    # 1. Format last 100 memories as numbered statements
    statements = "\n".join(
        f"{i+1}. {mem.content}" 
        for i, mem in enumerate(recent_memories[:100])
    )
    
    # 2. Prompt LLM for salient questions
    prompt = f"""Given these observations from coding work, what are {n} most 
salient high-level questions about patterns, learnings, or insights?

Observations:
{statements}

Questions:"""
    
    response = llm.complete(prompt, max_tokens=300, temperature=0.7)
    
    # 3. Parse numbered list
    questions = parse_numbered_list(response)
    return questions[:n]
```

**Example**:
```
Observations:
1. Fixed authentication bug in login endpoint
2. Added password validation tests
3. Refactored session management
4. Discovered race condition in token refresh
5. Implemented token expiry cleanup job
...

Generated Questions:
1. What patterns emerged in authentication implementation and testing?
2. What lessons were learned about session and token management?
3. What best practices should be applied to future auth work?
```

**Quality Heuristics**:
- Good questions are answerable from provided observations
- Good questions generalize beyond specific observations
- Good questions guide actionable insights

---

### Algorithm 8: Insight Extraction with Citations (Generative Agents)

**Purpose**: Generate semantic insights from episodic memories with evidence

**Complexity**: O(L + n·m + p) where p=parsing time

**Token Usage**: ~600-2000 input, ~200-500 output

**Pseudocode**:
```python
def extract_insights(question: str, relevant_memories: List[Memory], max: int = 5) -> List[Insight]:
    # 1. Format memories with citation indices
    statements = "\n".join(
        f"{i+1}. {mem.content}"
        for i, mem in enumerate(relevant_memories[:50])
    )
    memory_map = {i+1: mem.id for i, mem in enumerate(relevant_memories[:50])}
    
    # 2. Prompt for insights with citations
    prompt = f"""Answer this question with up to {max} insights about coding 
patterns, learnings, or best practices.

Format each as: 
- Insight text (because of 1, 3, 7)

Question: {question}

Observations:
{statements}

Insights:"""
    
    response = llm.complete(prompt, max_tokens=500, temperature=0.7)
    
    # 3. Parse insights and citations
    insights = []
    for line in response.split('\n'):
        match = re.match(r'-\s*(.+?)\s*\(because of\s+([\d,\s]+)\)', line)
        if match:
            text = match.group(1).strip()
            indices = [int(x.strip()) for x in match.group(2).split(',')]
            cited_ids = [memory_map[i] for i in indices if i in memory_map]
            
            insights.append(Insight(
                text=text,
                cited_memory_ids=cited_ids
            ))
    
    return insights[:max]
```

**Storage as Semantic Memory**:
```python
for insight in insights:
    semantic_mem = Memory(
        content=insight.text,
        memory_type="semantic",
        importance_score=8,  # Reflections inherently important
        is_synthetic=True,   # Prevents reflection loops
        citations=insight.cited_memory_ids
    )
    db.insert(semantic_mem)
```

**Citation Benefits**:
- Enables tracing insights back to source observations
- Prevents hallucination (insights must reference evidence)
- Supports iterative refinement (check if citations support claim)

---

### Algorithm 9: Iterative Insight Refinement (ACE)

**Purpose**: Improve insight quality through multi-round prompting

**Complexity**: O(R·L) where R=rounds (1-3), L=LLM time per round

**Token Usage**: ~300-500 per round × R rounds

**Pseudocode**:
```python
def refine_insight(initial: str, max_rounds: int = 3) -> str:
    current = initial
    
    for round in range(max_rounds):
        if is_sufficient(current):
            break
        
        prompt = f"""Refine this insight to be more specific and actionable.
        
Previous: {current}

Focus on:
1. Concrete root causes or patterns
2. Specific corrective actions
3. Generalizable principles

Refined:"""
        
        current = llm.complete(prompt, max_tokens=500, temperature=0.5)
    
    return current

def is_sufficient(insight: str) -> bool:
    """Heuristic quality check"""
    if len(insight) < 100:
        return False
    
    actionable_keywords = [
        "because", "when", "always", "never", "should", 
        "avoid", "ensure", "verify", "use", "instead"
    ]
    
    count = sum(1 for kw in actionable_keywords if kw in insight.lower())
    return count >= 2
```

**Example Refinement**:
```
Round 0 (initial): "Authentication should be handled carefully"
Sufficient? No (too vague, only 45 chars)

Round 1: "When implementing authentication, always validate tokens 
on both client and server to prevent security vulnerabilities"
Sufficient? No (missing actionable keywords, 131 chars but generic)

Round 2: "Authentication implementation should always validate JWT 
tokens on both client and server. Specifically: (1) verify signature 
with secret key, (2) check expiration timestamp, (3) validate issuer 
and audience claims. Never trust client-side validation alone because 
it can be bypassed."
Sufficient? Yes (270 chars, contains "always", "never", "should", "because")
```

**When to Use**:
- Critical insights (importance ≥ 8)
- Insights that will guide future work
- User-requested reflections

**When to Skip**:
- Low-importance observations
- Time-sensitive contexts (multi-round too slow)
- Cost-constrained scenarios (3× LLM calls)

---

## Decision & Planning Algorithms

### Algorithm 10: Propose-Evaluate-Select Decision Cycle (CoALA)

**Purpose**: Choose next action from candidates using value estimates

**Complexity**: O(C·L) where C=candidates (typically 5), L=LLM time

**Pseudocode**:
```python
def propose_evaluate_select(working_memory: WorkingMemory) -> Action:
    # 1. Generate candidate actions
    candidates = propose_actions(working_memory, limit=5)
    # O(L) - single LLM call, ~500-1000 input tokens
    
    # 2. Evaluate each candidate
    values = [evaluate_action(c, working_memory) for c in candidates]
    # O(C) if heuristic, O(C·L) if LLM-based
    
    # 3. Select highest value
    best = candidates[argmax(values)]
    
    return best

def propose_actions(wm: WorkingMemory, limit: int) -> List[Action]:
    prompt = f"""Given context and goals, propose {limit} possible next actions.

Context: {wm.current_context}
Goals: {', '.join(wm.active_goals)}
Recent: {', '.join(a.type for a in wm.recent_actions[-5:])}

Types: retrieval (search memory), reasoning (analyze), 
       learning (store insight), grounding (use tool)

Actions:"""
    
    response = llm.complete(prompt, max_tokens=400)
    return parse_actions(response)[:limit]

def evaluate_action(action: Action, wm: WorkingMemory) -> float:
    """Heuristic evaluation O(1) - fast alternative to LLM"""
    if action.type == "retrieval":
        # High value if working memory lacks info
        coverage = estimate_coverage(wm.current_context)
        return 1.0 - coverage
    
    elif action.type == "grounding":
        # High value if goals unmet
        progress = estimate_goal_progress(wm.active_goals)
        return 1.0 - progress
    
    elif action.type == "learning":
        # High value after significant activity
        activity = len(wm.recent_actions)
        return min(1.0, activity / 100.0)
    
    else:  # "reasoning"
        return 0.5  # Always moderately valuable
```

**Claude Code Mapping**:
- **Retrieval**: Search memory for similar past work
- **Reasoning**: Analyze current task, plan approach
- **Learning**: Store pattern or insight for future
- **Grounding**: Execute tool (Write, Edit, Bash, etc.)

**Example Decision**:
```
Context: "User asked to fix authentication bug"
Goals: ["Identify root cause", "Fix bug", "Verify fix"]

Proposed Actions:
1. Retrieval: "Search memory for past auth bugs"
2. Grounding: "Read auth.py to understand code"
3. Reasoning: "Analyze error message for clues"
4. Grounding: "Run tests to reproduce bug"
5. Retrieval: "Search for similar error patterns"

Evaluation:
1. value=0.8 (high - working memory lacks auth context)
2. value=0.6 (medium - need code understanding)
3. value=0.7 (medium-high - error analysis valuable)
4. value=0.9 (high - reproduction essential for debugging)
5. value=0.6 (medium - similar to action 1)

Selected: Action 4 (Run tests to reproduce bug) - highest value
```

---

### Algorithm 11: Recursive Plan Decomposition (Generative Agents)

**Purpose**: Break down high-level goals into executable chunks

**Complexity**: O(D·N·L) where D=depth (2-3), N=nodes per level (3-8), L=LLM time

**Pseudocode**:
```python
def recursive_decompose(goal: str, duration_min: int, chunk_size: int = 15, 
                        depth: int = 0, max_depth: int = 3) -> Plan:
    """
    Hierarchical decomposition:
    - Day plan (480 min) → Hour chunks (60 min) → Action chunks (15 min)
    """
    # Base case: atomic action
    if duration_min <= chunk_size or depth >= max_depth:
        return Plan(goal=goal, duration=duration_min, subplans=[])
    
    # Recursive case: decompose
    num_subgoals = min(8, duration_min // chunk_size)
    
    prompt = f"""Break "{goal}" into {num_subgoals} sequential subgoals.
    
Total time: {duration_min} minutes

Format:
1. Subgoal (duration minutes)
...

Subgoals:"""
    
    response = llm.complete(prompt, max_tokens=400)
    subgoals = parse_subgoals(response)  # [(text, duration), ...]
    
    # Recursively decompose each subgoal
    subplans = [
        recursive_decompose(text, dur, chunk_size, depth+1, max_depth)
        for text, dur in subgoals
    ]
    
    return Plan(goal=goal, duration=duration_min, subplans=subplans)
```

**Example Decomposition**:
```
Goal: "Implement user authentication" (240 min)
├─ "Design database schema" (60 min)
│  ├─ "Create users table" (20 min) [ATOMIC]
│  ├─ "Create sessions table" (20 min) [ATOMIC]
│  └─ "Write migration" (20 min) [ATOMIC]
├─ "Implement login endpoint" (120 min)
│  ├─ "Password validation" (30 min)
│  │  ├─ "Hash function" (15 min) [ATOMIC]
│  │  └─ "Verify function" (15 min) [ATOMIC]
│  ├─ "POST /login route" (30 min)
│  │  ├─ "Route handler" (15 min) [ATOMIC]
│  │  └─ "Validation middleware" (15 min) [ATOMIC]
│  └─ "Integration tests" (60 min)
│     ├─ "Happy path tests" (30 min) [ATOMIC]
│     └─ "Error case tests" (30 min) [ATOMIC]
└─ "JWT token generation" (60 min)
   ├─ "Token create" (20 min) [ATOMIC]
   ├─ "Token verify" (20 min) [ATOMIC]
   └─ "Refresh logic" (20 min) [ATOMIC]
```

**Chunk Size Guidelines**:

| Chunk Size | Use Case | Pros | Cons |
|------------|----------|------|------|
| 5 min | Micro-tasks | Very granular | High overhead |
| 15 min | Default | Balanced | - |
| 30 min | Complex tasks | Fewer nodes | Less flexible |
| 60 min | High-level only | Minimal LLM calls | Too coarse |

---

### Algorithm 12: Plan Deviation Detection (Generative Agents)

**Purpose**: Decide when to abandon current plan and replan

**Complexity**: O(1) heuristic or O(L) LLM

**Pseudocode**:
```python
def should_deviate(current_plan: Plan, new_obs: str, context: str) -> tuple[bool, str]:
    # Fast heuristic rules (O(1))
    
    # Rule 1: Errors always trigger replan
    if any(kw in new_obs.upper() for kw in ["ERROR", "FAILED", "EXCEPTION"]):
        return (True, "error_detected")
    
    # Rule 2: User interruption
    if "user:" in new_obs.lower() or "instruction:" in new_obs.lower():
        return (True, "user_instruction")
    
    # Rule 3: Goal satisfied early
    if is_goal_satisfied(current_plan.goal, context):
        return (True, "goal_satisfied")
    
    # Rule 4: Blocked by dependency
    if "blocked" in new_obs.lower() or "missing" in new_obs.lower():
        return (True, "blocked")
    
    # Slow LLM fallback for uncertain cases
    prompt = f"""Should agent abandon current plan and replan?

Current plan: {current_plan.goal}
Context: {context}
New observation: {new_obs}

Answer YES or NO with reason.
Decision:"""
    
    response = llm.complete(prompt, max_tokens=50, temperature=0.3)
    
    if "YES" in response.upper():
        reason = response.replace("YES", "").strip()
        return (True, reason)
    
    return (False, "continue_plan")
```

**Example Scenarios**:

| Observation | Deviation Decision | Reason |
|-------------|-------------------|--------|
| "Tests passed" | No | Plan proceeding normally |
| "Error: Module not found" | Yes | Error requires attention |
| "User: Actually, do this instead" | Yes | User override |
| "Already implemented in branch X" | Yes | Goal satisfied |
| "Warning: Deprecated API" | No | Warning, not blocker |
| "Build failed: syntax error" | Yes | Error blocks progress |

**Debouncing**: Don't replan more than once per 5 minutes to avoid thrashing.

---

## Data Structures

### Memory Node

```python
@dataclass
class Memory:
    # Identity
    id: str  # UUID
    user_id: str
    
    # Content
    content: str  # 1-10000 chars
    embedding: List[float]  # 1536 dims (text-embedding-3-small)
    
    # Classification
    memory_type: str  # "episodic" | "semantic" | "procedural"
    observation_type: Optional[str]  # "error", "instruction", "code_change", etc.
    section: Optional[str]  # "strategies", "snippets", "pitfalls", etc.
    
    # Scoring
    importance_score: float  # [1.0, 10.0]
    helpful_count: int  # >= 0
    harmful_count: int  # >= 0
    
    # Provenance
    is_synthetic: bool  # True for reflections
    citations: Optional[List[str]]  # Memory IDs (for reflections)
    
    # Timestamps
    created_at: datetime
    last_accessed_at: datetime  # Updated on retrieval
    
    # Extensibility
    metadata: Optional[dict]
```

**Storage**: PostgreSQL + pgvector
```sql
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    content TEXT CHECK (length(content) BETWEEN 1 AND 10000),
    embedding VECTOR(1536),
    memory_type VARCHAR(20) CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    observation_type VARCHAR(50),
    section VARCHAR(50),
    importance_score FLOAT CHECK (importance_score BETWEEN 1 AND 10),
    helpful_count INT DEFAULT 0,
    harmful_count INT DEFAULT 0,
    is_synthetic BOOLEAN DEFAULT FALSE,
    citations JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Indexes
CREATE INDEX ON memories(user_id, created_at DESC);
CREATE INDEX ON memories(memory_type);
CREATE INDEX ON memories(last_accessed_at DESC);
CREATE INDEX ON memories USING ivfflat(embedding vector_cosine_ops) WITH (lists=100);
```

---

### Reflection State

```python
@dataclass
class ReflectionState:
    user_id: str
    accumulated_importance: float = 0
    observations_since_last: int = 0
    last_reflection_at: datetime = field(default_factory=datetime.now)
```

**Storage**: PostgreSQL
```sql
CREATE TABLE reflection_state (
    user_id UUID PRIMARY KEY,
    accumulated_importance FLOAT DEFAULT 0,
    observations_since_last INT DEFAULT 0,
    last_reflection_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### Delta Change Set

```python
@dataclass
class Delta:
    created: List[Memory] = field(default_factory=list)
    updated: List[Memory] = field(default_factory=list)
```

---

## Complexity Summary

### Algorithm Complexity Table

| Algorithm | Time | Space | Token | LLM |
|-----------|------|-------|-------|-----|
| **Three-Factor Retrieval** | O(n log k + E) | O(n) | ~200 | 1 |
| **Recency Decay** | O(1) | O(1) | 0 | 0 |
| **Importance (Heuristic)** | O(m) | O(1) | 0 | 0 |
| **Importance (LLM)** | O(L) | O(1) | ~150 | 1 |
| **Delta Ingestion** | O(n·d + E) | O(d) | ~200 | 1 |
| **Delta Ingestion (Indexed)** | O(log n·d + E) | O(d) | ~200 | 1 |
| **Feedback Update** | O(1) | O(1) | 0 | 0 |
| **Reflection Trigger** | O(1) | O(1) | 0 | 0 |
| **Question Generation** | O(L + n·m) | O(n·m) | ~1000 | 1 |
| **Insight Extraction** | O(L + n·m) | O(n·m) | ~1500 | 1 |
| **Iterative Refinement** | O(R·L) | O(m) | ~1200 | 1-3 |
| **Propose-Evaluate-Select** | O(C·L) | O(C) | ~1000 | 1-6 |
| **Recursive Planning** | O(D·N·L) | O(D·N) | ~3000 | O(N) |
| **Deviation Detection** | O(1) or O(L) | O(1) | ~100 | 0-1 |

**Legend**:
- n = memories (100s-100k)
- k = result limit (5-20)
- m = avg content length (~100 chars)
- d = embedding dimension (1536)
- E = embedding time (50-200ms)
- L = LLM time (500-3000ms)
- R = refinement rounds (1-3)
- C = candidates (3-8)
- D = depth (2-4)
- N = nodes/level (3-8)

---

## Dependencies

### Required Services

**OpenAI API** (Embeddings)
- Model: `text-embedding-3-small`
- Dimensions: 1536
- Cost: $0.020 / 1M tokens
- Latency: 50-200ms
- Rate limit: 3000 req/min

**Anthropic API** (Reflection LLM)
- Model: `claude-opus-4-5` or `claude-sonnet-4-5`
- Cost: $15/1M input, $75/1M output (Opus)
- Latency: 500-3000ms
- Rate limit: 50 req/min

**PostgreSQL + pgvector**
- Version: PostgreSQL 15+, pgvector 0.5+
- Extensions: `vector`, `uuid-ossp`
- Indexes: IVFFlat for approximate NN search

### Python Libraries

```
fastmcp>=0.2.0          # MCP server
anthropic>=0.40.0       # Claude API
openai>=1.0.0           # OpenAI API
psycopg2>=2.9.0         # PostgreSQL
pgvector>=0.3.0         # Vector extension
sqlalchemy>=2.0.0       # ORM
pydantic>=2.0.0         # Validation
```

---

## Claude Code Adaptations

### 1. Auto-Capture via PostToolUse Hooks

**Problem**: Manual `memory_observe` calls are tedious.

**Solution**: Plugin hooks auto-capture significant events.

```json
{
  "hooks": [
    {
      "matcher": {"tool_name": "Write"},
      "type": "PostToolUse",
      "agent": "memory-manager",
      "prompt": "If file change is significant (architectural decision, bug fix, pattern), call memory_observe with:\n- observation: concise summary\n- observation_type: 'code_change' or 'decision'\n- importance: 5-9\n\nContext: {tool_params.file_path}, {tool_result.summary}"
    },
    {
      "matcher": {"tool_name": "Bash"},
      "type": "PostToolUse",
      "agent": "memory-manager",
      "prompt": "If command shows error or significant result, call memory_observe.\n\nContext: {tool_params.command}, exit={tool_result.exit_code}"
    }
  ]
}
```

### 2. Context-Aware Query Generation

**Problem**: User doesn't know what to search for.

**Solution**: Auto-generate retrieval queries.

```python
def smart_memory_search(current_task: str, recent_errors: List[str]) -> dict:
    # Extract keywords from task + errors
    keywords = extract_keywords(current_task)
    if recent_errors:
        keywords.extend(extract_keywords(recent_errors[0]))
    
    query = " ".join(keywords)
    return memory_retrieve(query, limit=10)

# Example:
# Task: "Fix authentication bug"
# Error: "JWT token expired"
# Query: "authentication bug JWT token expired"
```

### 3. Proactive Reflection Notifications

**Problem**: User doesn't know when to reflect.

**Solution**: Notify when thresholds met.

```python
async def check_reflection_status():
    state = get_reflection_state(user_id)
    should, reason = state.should_reflect()
    
    if should:
        return {
            "notification": f"Reflection recommended: {reason}",
            "prompt": "Would you like me to synthesize recent learnings? (~3-5 insights)",
            "state": {
                "importance": state.accumulated_importance,
                "observations": state.observations_since_last,
                "hours": hours_since_last
            }
        }
```

### 4. Domain-Specific Importance Scoring

**Problem**: Generic heuristics don't fit coding context.

**Solution**: Code-specific importance rules.

```python
CODE_IMPORTANCE = {
    "production_incident": 10,
    "security_issue": 10,
    "architectural_decision": 9,
    "error": 9,
    "test_failure": 8,
    "api_design": 8,
    "bug_fix": 7,
    "feature_implementation": 7,
    "refactoring": 6,
    "configuration_change": 7,
    "code_review_feedback": 8,
    "dependency_change": 6,
    "test_success": 5,
    "general": 5,
    "file_read": 2,
}

# Content boosts:
# +3: "BREAKING", "CRITICAL", "SECURITY", "CVE"
# +2: "test" + "failed"
# +1: "prod", "production", "main branch"
```

---

## References

1. **ACE: Agentic Context Engineering** - Zhang et al. (2025)
   - Delta ingestion with deduplication
   - Feedback-driven importance
   - Iterative refinement

2. **CoALA: Cognitive Architectures for Language Agents** - Sumers et al. (2024)
   - Memory taxonomy (episodic/semantic/procedural)
   - Propose-evaluate-select decision cycle
   - Working memory structure

3. **Generative Agents: Interactive Simulacra of Human Behavior** - Park et al. (2023)
   - Three-factor retrieval scoring
   - Exponential recency decay
   - Reflection triggering
   - Question generation
   - Insight extraction with citations
   - Recursive planning
   - Deviation detection

---

**Document Status**: Implementation-Ready
**Version**: 1.0
**Date**: 2025-12-11

This document provides complete implementable specifications for all core algorithms needed to build a living memory system for Claude Code. For system architecture, see `/Users/chris/Documents/code/birdseye/docs/ARCHITECTURE.md`. For full research analysis, see `/Users/chris/Documents/code/birdseye/docs/RESEARCH_DEEP_DIVE.md`.
