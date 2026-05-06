FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Make the venv available without needing to activate it
ENV PATH="/app/.venv/bin:$PATH"

RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('jinaai/jina-embeddings-v2-small-en'); SparseTextEmbedding('Qdrant/bm25')"

EXPOSE 7860

CMD ["uv", "run", "python", "-m", "streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]