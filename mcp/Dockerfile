FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY app/ ./app/
COPY server.py ./
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Expose the MCP server port
EXPOSE 8787

# Run migrations and start server
CMD ["sh", "-c", "uv run alembic upgrade head && uv run python server.py"]
