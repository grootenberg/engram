#!/usr/bin/env python3
"""UserPromptSubmit hook for engram plugin.

Detects when memory retrieval would be valuable and injects a contextual reminder.
Analyzes the user's prompt for memory-relevant triggers.

Input (stdin): JSON with prompt, session_id, cwd, etc.
Output (stdout): Context to inject, or nothing if no triggers detected.

Exit codes:
- 0: Success (with or without output)
"""

from __future__ import annotations

import json
import re
import sys


# Explicit memory trigger keywords
EXPLICIT_TRIGGERS = [
    r"\bremember\b",
    r"\brecall\b",
    r"\blast time\b",
    r"\bbefore\b",
    r"\bpreviously\b",
    r"\bearlier\b",
    r"\bwe discussed\b",
    r"\byou mentioned\b",
    r"\bwe tried\b",
    r"\bwhat did we\b",
    r"\bhow did we\b",
    r"\bwhat was\b",
]

# Implicit triggers - topics where past context is often valuable
IMPLICIT_TRIGGERS = [
    r"\berror\b",
    r"\bbug\b",
    r"\bpattern\b",
    r"\bdecision\b",
    r"\barchitecture\b",
    r"\bapproach\b",
    r"\bstrategy\b",
    r"\bconvention\b",
    r"\bpreference\b",
    r"\bsimilar\b",
    r"\blike we did\b",
    r"\bsame as\b",
]

# Project-specific terms that might benefit from memory lookup
# These are detected from cwd and file references
PROJECT_INDICATORS = [
    r"engram",
    r"memory",
    r"mcp",
]


def extract_topic(prompt: str, cwd: str) -> str | None:
    """Extract the most relevant topic from the prompt for retrieval suggestion."""
    # Check for explicit memory triggers first
    for pattern in EXPLICIT_TRIGGERS:
        if re.search(pattern, prompt, re.IGNORECASE):
            # Try to extract what they're asking about
            # Look for nouns/phrases after the trigger
            return None  # Let Claude figure out the query

    # Check for implicit triggers
    for pattern in IMPLICIT_TRIGGERS:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            # Return the trigger word as a hint
            return match.group(0).lower()

    return None


def should_suggest_retrieval(prompt: str, cwd: str) -> tuple[bool, str | None]:
    """Determine if we should suggest memory retrieval.

    Returns:
        Tuple of (should_suggest, topic_hint)
    """
    prompt_lower = prompt.lower()

    # Check explicit triggers - always suggest
    for pattern in EXPLICIT_TRIGGERS:
        if re.search(pattern, prompt_lower):
            return True, None

    # Check implicit triggers
    for pattern in IMPLICIT_TRIGGERS:
        if re.search(pattern, prompt_lower):
            topic = extract_topic(prompt, cwd)
            return True, topic

    # Check for project-specific indicators in cwd
    cwd_lower = cwd.lower()
    for indicator in PROJECT_INDICATORS:
        if indicator in cwd_lower and re.search(r"\b(how|what|why|where)\b", prompt_lower):
            return True, indicator

    return False, None


def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        prompt = input_data.get("prompt", "")
        cwd = input_data.get("cwd", "")

        if not prompt:
            sys.exit(0)

        should_suggest, topic = should_suggest_retrieval(prompt, cwd)

        if should_suggest:
            if topic:
                context = f"[engram] Consider retrieving relevant memories about '{topic}' using the retrieve tool."
            else:
                context = "[engram] This query might benefit from past context. Consider using the retrieve tool to find relevant memories."

            print(context)

        # Exit 0 whether or not we output anything
        sys.exit(0)

    except json.JSONDecodeError:
        # No input or invalid JSON - silent exit
        sys.exit(0)
    except Exception as e:
        # Log errors but don't block the user
        print(f"[engram] Hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
