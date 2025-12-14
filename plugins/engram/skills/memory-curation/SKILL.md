---
description: Memory quality management through feedback, deduplication awareness, and importance calibration. Use when curating memory quality.
---

# Memory Curation Skill

This skill guides memory quality management and maintenance.

## Feedback Loop

### When to Provide Feedback

Provide `feedback` signals when:

- A retrieved memory directly helped solve a problem
- A memory misled or wasted time
- An insight proved incorrect in practice
- A memory keeps surfacing but isn't useful

### Feedback Signals

**Helpful (+1 effective importance)**

- Memory saved time or prevented mistakes
- Memory provided accurate context
- Insight was actionable and correct

**Harmful (-1 effective importance)**

- Memory was outdated or incorrect
- Memory was misleading in context
- Memory added noise without value

### Effective Importance

```
effective_importance = base_importance + (helpful - harmful) * 0.5
```

Clamped to [1, 10] range. Heavy negative feedback can suppress memories without deletion.

## Deduplication Awareness

### Built-in Deduplication

Observations with >90% cosine similarity to recent memories (7 days) are rejected.

### Manual Deduplication

For older duplicates:

1. Retrieve with broad query
2. Identify redundant memories
3. Provide `harmful` feedback to lower-quality duplicates
4. Keep the best-formulated version

### Avoiding Duplicates

When ingesting observations:

- Check if similar memory exists via retrieval
- Reformulate to add new information
- Use updates rather than repetitions

## Importance Calibration

### Heuristic Defaults

| Observation Type | Default Importance |
| ---------------- | ------------------ |
| instruction      | 10                 |
| error            | 9                  |
| decision         | 8                  |
| code_change      | 7                  |
| insight          | 7                  |
| test_result      | 6                  |
| general          | 5                  |
| tool_output      | 3                  |

### When to Override

Override importance when:

- A "general" observation is actually critical
- An "error" is minor/transient
- Context makes default inappropriate

### Calibration via Feedback

If memories are consistently:

- Under-utilized: Check importance scores, consider boosting
- Over-surfacing: Provide harmful feedback or lower initial importance
- Missing from retrieval: Verify they exist, check query formulation

## Maintenance Operations

### Health Monitoring

Use `stats(scope="detailed")` to monitor:

- Distribution by memory type
- Feedback ratios (helpful vs harmful)
- Top importance memories

### Signs of Unhealthy Memory

- High harmful feedback ratio
- Mostly low-importance memories
- Unbalanced type distribution (all episodic, no semantic)
- Stale last_accessed_at timestamps

### Remediation

For unhealthy memory:

1. Run focused reflection to synthesize
2. Review and provide feedback on low-quality memories
3. Adjust ingestion criteria (be more selective)
4. Consider archival of old episodic memories (future feature)

## Best Practices

1. **Feedback often** - Every retrieval should consider feedback
2. **Quality over quantity** - Fewer high-quality memories beat many low-quality
3. **Regular reflection** - Keep episodic → semantic flow healthy
4. **Monitor stats** - Weekly health checks prevent degradation
5. **Be specific** - Detailed observations dedupe better than vague ones
