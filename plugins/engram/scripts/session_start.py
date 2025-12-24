#!/usr/bin/env python3
"""SessionStart hook for engram plugin.

Checks server health and injects retrieval guidance at session start.
Output becomes additionalContext for Claude.

Exit codes:
- 0: Success, stdout becomes context
- 1: Error (server down), message shown to user
"""

import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engram_client import check_health, stats, EngramUnavailable


def main():
    try:
        # Check server health
        check_health()

        # Get quick stats to include in guidance
        try:
            stat_result = stats(scope="summary")
            memory_count = stat_result.get("total_memories", 0)
            by_type = stat_result.get("by_type", {})
            episodic_count = by_type.get("episodic", 0)
            semantic_count = by_type.get("semantic", 0)

            stats_line = f"Current state: {memory_count} memories ({episodic_count} episodic, {semantic_count} semantic)"
        except Exception:
            stats_line = ""

        # Output guidance that becomes additionalContext
        guidance = """[engram] Memory system active.

To find relevant context from past sessions, use the `retrieve` tool with a semantic query.
Available memory types: errors, decisions, patterns, instructions, code changes.

Examples:
- retrieve(query="authentication implementation patterns")
- retrieve(query="errors with database connections", memory_types=["episodic"])
- retrieve(query="user preferences for this project", relevance_weight=0.6)"""

        if stats_line:
            guidance += f"\n\n{stats_line}"

        print(guidance)
        sys.exit(0)

    except EngramUnavailable as e:
        # Fail loudly if server is down
        print(f"[engram] ERROR: Memory server unavailable - {e}", file=sys.stderr)
        print("[engram] ERROR: Start the server with: cd mcp && ./start_dev_server.sh", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
