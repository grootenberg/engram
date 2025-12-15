#!/bin/bash
# Stdio server for engram MCP (for use with reloaderoo or direct testing)

cd "$(dirname "$0")"

# Set debug mode for development
export ENGRAM_DEBUG=true

exec uv run python server_stdio.py
