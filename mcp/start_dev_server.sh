#!/bin/bash
# Development server with hot reload for engram MCP server

cd "$(dirname "$0")"

# Set debug mode for development
export ENGRAM_DEBUG=true

# Suppress noisy MCP streamable_http errors (known issue with stateless mode)
export LOG_LEVEL=info

exec uv run uvicorn server:app \
    --reload \
    --reload-dir app \
    --host 0.0.0.0 \
    --port 8787 \
    --log-level info
