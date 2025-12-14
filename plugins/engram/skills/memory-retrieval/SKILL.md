---
description: Query formulation and context assembly patterns for memory retrieval. Use when you need to search the memory graph effectively.
---

# Memory Retrieval Skill

This skill provides guidance on effective memory retrieval strategies.

## Query Formulation

### Semantic Query Construction

Write queries that capture the conceptual meaning:

```
Good: "user preferences for error handling in TypeScript"
Bad: "error handling" (too generic)
Bad: "that thing we discussed yesterday" (temporal, not semantic)
```

### Query Decomposition

For complex retrievals, break into sub-queries:

1. **Topic queries**: What subject matter?
2. **Type queries**: Decisions? Instructions? Patterns?
3. **Context queries**: What project? What problem domain?

## Weight Tuning

### Three-Factor Scoring

Adjust weights based on retrieval intent:

| Intent          | Recency | Importance | Relevance |
| --------------- | ------- | ---------- | --------- |
| Recent context  | 0.5     | 0.2        | 0.3       |
| Best practices  | 0.1     | 0.4        | 0.5       |
| Specific recall | 0.2     | 0.3        | 0.5       |
| Error patterns  | 0.3     | 0.4        | 0.3       |

### Memory Type Filters

- `episodic`: Raw observations, recent context
- `semantic`: Synthesized insights, best practices
- `procedural`: Workflows, how-to knowledge

## Context Assembly

### Result Synthesis

When presenting retrieved memories:

1. **Relevance filter**: Show top 3-5 most relevant
2. **Score threshold**: Ignore combined_score < 0.4
3. **Grouping**: Cluster by topic or type
4. **Citation**: Reference memory IDs for feedback

### Insufficient Results

If retrieval returns < 3 results with score > 0.4:

1. Broaden the query semantically
2. Reduce relevance_weight, boost others
3. Remove type filters
4. Consider the query may be novel (no prior memories)

## Anti-Patterns

❌ **Over-retrieval**: Don't retrieve for every interaction
❌ **Generic queries**: "help" or "code" match everything poorly
❌ **Ignoring scores**: Low-scoring results may mislead
❌ **No feedback**: Retrieved results should get feedback signals
