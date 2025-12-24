#!/usr/bin/env python3
"""Dual-mode analyzer for engram observation capture.

Supports two modes:
- Heuristics (default): Fast pattern-based analysis
- LLM (ENGRAM_USE_LLM=true): Local ollama for smarter decisions

Used by PostToolUse hooks to decide what's worth capturing.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

# Configuration
USE_LLM = os.environ.get("ENGRAM_USE_LLM", "").lower() in ("true", "1", "yes")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "10"))


@dataclass
class CaptureDecision:
    """Result of analysis - whether to capture and how."""
    capture: bool
    observation: str = ""
    observation_type: str = "general"
    importance: int = 5
    section: str | None = None


# File extensions by importance tier
HIGH_IMPORTANCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".rb"}
MEDIUM_IMPORTANCE_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".sql", ".sh"}
LOW_IMPORTANCE_EXTENSIONS = {".md", ".txt", ".css", ".html"}

# Patterns that indicate significant content
PITFALL_PATTERNS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bHACK\b",
    r"\bBUG\b",
    r"\bXXX\b",
    r"\bWARNING\b",
]

# Test file patterns
TEST_FILE_PATTERNS = [
    r"test_.*\.py$",
    r".*_test\.py$",
    r".*\.test\.(ts|js|tsx|jsx)$",
    r".*\.spec\.(ts|js|tsx|jsx)$",
    r"tests?/.*\.(py|ts|js)$",
]

# Error indicators in bash output
ERROR_PATTERNS = [
    r"\berror\b",
    r"\bError\b",
    r"\bERROR\b",
    r"\bfailed\b",
    r"\bFailed\b",
    r"\bFAILED\b",
    r"\bfailure\b",
    r"\bException\b",
    r"\bTraceback\b",
]

# Test command patterns
TEST_COMMAND_PATTERNS = [
    r"\bpytest\b",
    r"\bpython.*-m\s+pytest\b",
    r"\bjest\b",
    r"\bmocha\b",
    r"\bvitest\b",
    r"\bnpm\s+test\b",
    r"\byarn\s+test\b",
    r"\bgo\s+test\b",
    r"\bcargo\s+test\b",
]

# Build/deploy patterns
BUILD_PATTERNS = [
    r"\bbuild\b",
    r"\bdeploy\b",
    r"\bnpm\s+run\s+build\b",
    r"\byarn\s+build\b",
    r"\bmake\b",
    r"\bcargo\s+build\b",
]


def _get_file_extension(file_path: str) -> str:
    """Extract file extension from path."""
    return os.path.splitext(file_path)[1].lower()


def _is_test_file(file_path: str) -> bool:
    """Check if file is a test file."""
    return any(re.search(p, file_path) for p in TEST_FILE_PATTERNS)


def _has_pitfall_markers(content: str) -> list[str]:
    """Find pitfall markers in content."""
    found = []
    for pattern in PITFALL_PATTERNS:
        if re.search(pattern, content):
            found.append(pattern.replace(r"\b", ""))
    return found


def _has_errors(text: str) -> bool:
    """Check if text contains error indicators."""
    return any(re.search(p, text, re.IGNORECASE) for p in ERROR_PATTERNS)


def _is_test_command(command: str) -> bool:
    """Check if command is a test command."""
    return any(re.search(p, command) for p in TEST_COMMAND_PATTERNS)


def _is_build_command(command: str) -> bool:
    """Check if command is a build/deploy command."""
    return any(re.search(p, command) for p in BUILD_PATTERNS)


def _analyze_with_llm(tool_name: str, tool_input: dict, tool_response: dict) -> CaptureDecision:
    """Use local LLM to analyze tool output.

    Falls back to heuristics if LLM is unavailable.
    """
    prompt = f"""Analyze this tool output for memory capture. Output JSON only.
Tool: {tool_name}
Input: {json.dumps(tool_input, indent=2)[:2000]}
Output: {json.dumps(tool_response, indent=2)[:2000]}

Decide if this is worth capturing as a memory. Consider:
- Is this a significant code change, error, decision, or insight?
- Would this be valuable to recall in future sessions?
- Is it routine/trivial (don't capture) or noteworthy (capture)?

Respond with ONLY valid JSON, no other text:
{{"capture": true/false, "observation": "clear description", "type": "error|code_change|decision|insight|test_result|general", "importance": 1-10, "section": "strategies|snippets|pitfalls|context|null"}}

If not worth capturing, respond: {{"capture": false}}"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
            llm_response = json.loads(result.get("response", "{}"))

            if not llm_response.get("capture", False):
                return CaptureDecision(capture=False)

            return CaptureDecision(
                capture=True,
                observation=llm_response.get("observation", ""),
                observation_type=llm_response.get("type", "general"),
                importance=llm_response.get("importance", 5),
                section=llm_response.get("section")
            )

    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        # Fall back to heuristics if LLM fails
        return _analyze_write_heuristic(tool_input, tool_response) if tool_name in ("Write", "Edit") \
            else _analyze_bash_heuristic(tool_input, tool_response)


def _analyze_write_heuristic(tool_input: dict, tool_response: dict) -> CaptureDecision:
    """Heuristic analysis for Write/Edit tool output."""
    file_path = tool_input.get("file_path", "")
    content = tool_input.get("content", "") or tool_input.get("new_string", "")

    # Skip if no meaningful content
    if not file_path or not content:
        return CaptureDecision(capture=False)

    ext = _get_file_extension(file_path)
    is_test = _is_test_file(file_path)
    pitfalls = _has_pitfall_markers(content)

    # Determine importance based on file type
    if ext in HIGH_IMPORTANCE_EXTENSIONS:
        base_importance = 7
    elif ext in MEDIUM_IMPORTANCE_EXTENSIONS:
        base_importance = 6
    else:
        base_importance = 4

    # Boost for test files
    if is_test:
        base_importance = max(base_importance, 6)
        obs_type = "test_result"
    else:
        obs_type = "code_change"

    # Check for pitfalls
    section = None
    if pitfalls:
        section = "pitfalls"
        base_importance = max(base_importance, 7)

    # Large changes are more significant
    line_count = content.count("\n")
    if line_count > 50:
        base_importance = min(base_importance + 1, 9)

    # Skip trivial changes
    if line_count < 5 and not pitfalls and ext in LOW_IMPORTANCE_EXTENSIONS:
        return CaptureDecision(capture=False)

    # Build observation description
    file_name = os.path.basename(file_path)
    if is_test:
        observation = f"Test file modified: {file_name} ({line_count} lines)"
    elif pitfalls:
        observation = f"Code with {', '.join(pitfalls)} markers in {file_name}"
    else:
        observation = f"Modified {file_name} ({line_count} lines changed)"

    return CaptureDecision(
        capture=True,
        observation=observation,
        observation_type=obs_type,
        importance=base_importance,
        section=section
    )


def _analyze_bash_heuristic(tool_input: dict, tool_response: dict) -> CaptureDecision:
    """Heuristic analysis for Bash tool output."""
    command = tool_input.get("command", "")
    output = str(tool_response.get("output", "") or tool_response.get("stdout", ""))
    exit_code = tool_response.get("exit_code", tool_response.get("exitCode", 0))

    # Error detection - high priority
    if exit_code != 0:
        error_snippet = output[:500] if output else "No output"
        return CaptureDecision(
            capture=True,
            observation=f"Command failed (exit {exit_code}): {command[:100]}\n{error_snippet}",
            observation_type="error",
            importance=9,
            section="pitfalls"
        )

    # Check for error patterns in output even with exit 0
    if _has_errors(output):
        return CaptureDecision(
            capture=True,
            observation=f"Errors in output of: {command[:100]}",
            observation_type="error",
            importance=8,
            section="pitfalls"
        )

    # Test results
    if _is_test_command(command):
        # Determine if tests passed or had issues
        if "passed" in output.lower() or "ok" in output.lower():
            importance = 6
            observation = f"Tests passed: {command[:100]}"
        else:
            importance = 7
            observation = f"Test run: {command[:100]}"

        return CaptureDecision(
            capture=True,
            observation=observation,
            observation_type="test_result",
            importance=importance
        )

    # Build/deploy events
    if _is_build_command(command):
        return CaptureDecision(
            capture=True,
            observation=f"Build/deploy command: {command[:100]}",
            observation_type="tool_output",
            importance=6,
            section="context"
        )

    # Skip routine commands
    routine_commands = ["ls", "cd", "pwd", "cat", "echo", "which", "type", "git status", "git diff"]
    if any(command.strip().startswith(rc) for rc in routine_commands):
        return CaptureDecision(capture=False)

    # Default: don't capture unless significant
    return CaptureDecision(capture=False)


def analyze_write(tool_input: dict, tool_response: dict) -> CaptureDecision:
    """Analyze Write/Edit tool output for capture decision.

    Args:
        tool_input: The tool input (file_path, content/new_string)
        tool_response: The tool response (success, etc.)

    Returns:
        CaptureDecision with capture decision and metadata
    """
    if USE_LLM:
        return _analyze_with_llm("Write", tool_input, tool_response)
    return _analyze_write_heuristic(tool_input, tool_response)


def analyze_bash(tool_input: dict, tool_response: dict) -> CaptureDecision:
    """Analyze Bash tool output for capture decision.

    Args:
        tool_input: The tool input (command)
        tool_response: The tool response (output, exit_code)

    Returns:
        CaptureDecision with capture decision and metadata
    """
    if USE_LLM:
        return _analyze_with_llm("Bash", tool_input, tool_response)
    return _analyze_bash_heuristic(tool_input, tool_response)


if __name__ == "__main__":
    # Test with sample data
    print(f"LLM mode: {USE_LLM}")

    # Test write analysis
    write_result = analyze_write(
        {"file_path": "/test/example.py", "content": "def foo():\n    # TODO: fix this\n    pass\n"},
        {"success": True}
    )
    print(f"Write analysis: {write_result}")

    # Test bash analysis
    bash_result = analyze_bash(
        {"command": "pytest tests/"},
        {"output": "5 passed", "exit_code": 0}
    )
    print(f"Bash analysis: {bash_result}")
