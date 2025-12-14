---
description: Procedures for synthesizing episodic memories into semantic insights. Use when managing reflection cycles.
---

# Memory Reflection Skill

This skill guides the episodic → semantic synthesis process.

## Reflection Triggers

### Automatic Triggers

Reflection fires when ANY threshold is met:

- **Importance sum** ≥ 150 accumulated
- **Observation count** ≥ 100 since last reflection
- **Time elapsed** ≥ 24 hours since last reflection

### Manual Triggers

Use `force=True` when:

- User requests memory consolidation
- Session is ending and valuable context exists
- A major milestone or decision was reached

## Reflection Flow

1. Enqueue reflection via `reflect` (non-blocking). Poll completion via `reflect_status` or check `stats(scope="reflection")`.
2. Generate up to 3 high-level reflection questions from recent episodic memories (optionally biased by `focus`).
3. For each question, retrieve the most relevant episodic memories by semantic similarity.
4. Extract semantic insights **and** procedural workflows (short title + 3-7 steps) with citations to source memory IDs.
5. Respect `max_insights` across both insights and procedures.

## Focus Strategies

### Focused Reflection

Provide a focus when:

- Working on a specific domain (e.g., "authentication patterns")
- Addressing a recurring issue (e.g., "test failures")
- Consolidating project knowledge (e.g., "project architecture")

### Unfocused Reflection

Omit focus when:

- General consolidation is needed
- No clear theme in recent observations
- Exploring emergent patterns

## Quality Indicators

### Good Insights

- Self-contained: Understandable without context
- Actionable: Can be applied in future situations
- Specific: Names concrete patterns or decisions
- Cited: References source observations

### Bad Insights

- Vague: "Testing is important"
- Context-dependent: "The bug in the auth module"
- Uncited: No link to source observations

## Post-Reflection

### Review Generated Insights

After reflection completes:

1. Scan generated insights for quality
2. Consider providing feedback on helpful/harmful
3. Note if certain topics need more observation

### When Reflection Fails

If status is "skipped":

- Check reason (insufficient memories, triggers not met)
- Use `force=True` if appropriate
- Ensure observations are being ingested

## Reflection Cadence

### Recommended Schedule

- **High activity**: Reflect every 50-100 observations
- **Normal activity**: Reflect when triggers fire naturally
- **Session end**: Consider forced reflection on valuable sessions

### Over-Reflection

Signs of too frequent reflection:

- Redundant insights across cycles
- Insights losing specificity
- Episodic memory pool depleted too quickly

### Under-Reflection

Signs of too infrequent reflection:

- Large episodic backlog (>500 memories)
- Stale insights (no recent semantic memories)
- Repeated similar observations
