#!/usr/bin/env python3
"""Stop hook for engram plugin.

Runs when the session ends. Checks reflection triggers and reminds user if needed.
Can optionally analyze transcript for uncaptured insights.

Input (stdin): JSON with session_id, transcript_path, cwd
Output (stdout): Reminder about reflection if triggers are met

Exit codes:
- 0: Success
"""

import json
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    try:
        # Read input
        input_data = json.load(sys.stdin)
        transcript_path = input_data.get("transcript_path", "")

        # Import after path setup
        from engram_client import stats, EngramUnavailable

        try:
            # Check reflection triggers
            stat_result = stats(scope="reflection")

            # Extract trigger state
            triggers = stat_result.get("reflection_triggers", {})
            importance_sum = triggers.get("importance_sum", 0)
            obs_count = triggers.get("observation_count", 0)
            hours_since = triggers.get("hours_since_reflection", 0)

            # Check if any triggers are met
            triggers_met = []
            if importance_sum >= 150:
                triggers_met.append(f"importance sum ({importance_sum:.0f} >= 150)")
            if obs_count >= 100:
                triggers_met.append(f"observation count ({obs_count} >= 100)")
            if hours_since >= 24:
                triggers_met.append(f"time since reflection ({hours_since:.1f}h >= 24h)")

            if triggers_met:
                print(f"\n[engram] Reflection triggers met: {', '.join(triggers_met)}")
                print("[engram] Consider running reflection to synthesize recent observations into insights.")
                print("[engram] Use: reflect(force=True) or wait for automatic reflection.")

            # Show session summary
            total = stat_result.get("total_memories", 0)
            recent = stat_result.get("recent_observations", 0)
            if recent > 0:
                print(f"\n[engram] Session captured {recent} new observations. Total memories: {total}")

        except EngramUnavailable:
            # Server down at session end - just skip
            pass

        sys.exit(0)

    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as e:
        # Log but don't fail
        print(f"[engram] Session end hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
