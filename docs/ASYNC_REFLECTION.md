# Async Reflection & Maintenance Pattern

This document sketches how to make reflection (and other heavy maintenance) non-blocking in Claude Code while keeping the MCP interface predictable.

## Goals
- Keep user turns snappy; never block on LLM or long DB work.
- Preserve clear contracts for MCP tools (no surprise latency).
- Provide a simple poll/notify path so agents know when background work completes.
- Enforce backpressure (one active reflection per user; queue or drop duplicates).

## Tool Contracts (Proposed)

### `reflect` (enqueue)
- **Request:** `reflect(user_id="...", focus=None, max_insights=5, force=False)`
- **Response (always fast):**
  ```json
  {
    "status": "queued",
    "job_id": "uuid",
    "queued_at": "2025-02-14T22:17:00Z",
    "eta_seconds": 30
  }
  ```
- **Behavior:** Validates triggers (unless `force=True`), enqueues a background job, and returns immediately. If a job is already running for the user, return `{"status": "already_running", "job_id": ...}`.

### `reflect_status` (poll)
- **Request:** `reflect_status(job_id="...", user_id="...")`
- **Response:**
  - Pending:
    ```json
    {"status": "running", "job_id": "...", "queued_at": "...", "started_at": "...", "progress": 0.3}
    ```
  - Completed:
    ```json
    {
      "status": "completed",
      "job_id": "...",
      "completed_at": "...",
      "insights_created": 4,
      "procedures_created": 1,
      "insights": [...],
      "procedures": [...],
      "memories_analyzed": 47
    }
    ```
  - Failed:
    ```json
    {"status": "failed", "job_id": "...", "reason": "error message"}
    ```
  - Not found: `{"status": "not_found"}`

### `stats(scope="reflection")` (lightweight check)
- Add fields:
  ```json
  {
    "pending_jobs": 1,
    "running_jobs": 1,
    "last_completed_job": {"job_id": "...", "completed_at": "...", "insights_created": 3}
  }
  ```

### Optional: `cancel_reflection(job_id, user_id)`
- Cancel queued jobs; no-op for completed/failed.

## Background Worker Responsibilities
- Dequeue jobs (FIFO per user; enforce one running per user).
- Run reflection (question generation + per-question synthesis).
- Persist results as semantic/procedural memories.
- Emit a completion record for `reflect_status` and `stats`.
- Handle retries with a small cap; record failures with a reason.

## Agent/UX Pattern in Claude Code
1. User or trigger requests reflection → call `reflect` (async enqueue).
2. Agent tells user: “Reflecting in the background; I’ll share results when ready.”
3. Poll `reflect_status` after a delay or on the next turn; when `completed`, present results (optionally re-retrieve semantic/procedural memories to include scores).
4. Surface failures succinctly; allow re-run.
5. Avoid flooding: if `already_running`, don’t enqueue another; maybe offer to cancel/retry later.

## Sequencing With Other Work
- `retrieve`, `observe`, `feedback`, `stats` remain synchronous.
- Heavy maintenance (compaction, re-embedding) should also be queued, with similar `job_id`/`status` patterns.

## Minimal Server Changes
1. Add a job table/model (`id`, `user_id`, `type`, `status`, `payload`, `result`, timestamps).
2. Change `reflect` tool to enqueue and return `job_id`.
3. Add `reflect_status` tool (read-only).
4. Add a background worker (async task loop or separate process) to process reflection jobs.
5. Extend `stats(scope="reflection")` to surface job counts and last completion.
6. Optional: emit a small notification memory on completion/failure so the agent can pick it up without polling.

## Backpressure & Limits
- One active reflection per user; queue additional requests or return `already_running`.
- Rate-limit forced reflections to prevent cost spikes.
- Cap retries to avoid runaway failures.
