#!/usr/bin/env python3
"""PostToolUse hook for Write/Edit tools.

Analyzes file changes and stores observations synchronously.

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

        from analyzer import analyze_write
        from engram_client import observe, EngramUnavailable

        tool_input = input_data.get("tool_input", {})
        tool_response = input_data.get("tool_response", {})
        cwd = input_data.get("cwd", "")

        # Analyze the file change
        decision = analyze_write(tool_input, tool_response)

        if decision.capture:
            # Include cwd in metadata for context
            metadata = {"cwd": cwd}
            file_path = tool_input.get("file_path", "")
            if file_path:
                metadata["file_path"] = file_path

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
