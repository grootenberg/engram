#!/usr/bin/env python3
"""PostToolUse hook for Bash tool.

Analyzes command output and stores observations synchronously.

Input (stdin): JSON with tool_name, tool_input, tool_response, cwd
Output: None (runs silently)

Exit codes:
- 0: Always
"""

import json
import os
import sys


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    try:
        # Add scripts directory to path for imports
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)

        from analyzer import analyze_bash
        from engram_client import observe, EngramUnavailable

        tool_input = input_data.get("tool_input", {})
        tool_response = input_data.get("tool_response", {})
        cwd = input_data.get("cwd", "")

        # Analyze the command output
        decision = analyze_bash(tool_input, tool_response)

        if decision.capture:
            # Include cwd and command in metadata
            metadata = {"cwd": cwd}
            command = tool_input.get("command", "")
            if command:
                metadata["command"] = command[:500]  # Truncate long commands

            # Store the observation
            observe(
                content=decision.observation,
                observation_type=decision.observation_type,
                importance=decision.importance,
                section=decision.section,
                metadata=metadata
            )

    except EngramUnavailable:
        # Server down - silently skip
        pass
    except Exception:
        # Any error - silently skip
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
