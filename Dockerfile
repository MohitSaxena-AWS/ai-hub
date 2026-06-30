# Single-stage image for the AI Hub backend.
#
# Dependencies have a single source of truth: pyproject.toml. The image installs
# the project (and, by default, the optional `embeddings` extra) from it, so
# there is no separate requirements file to keep in sync.
#
# A single build arg, SIMILARITY_BACKEND, selects the duplicate-detection
# backend AND is baked into the image as the runtime default, so the packaged
# image is always self-consistent — the ML stack is installed iff the image will
# actually run the embedding backend. This removes any chance of build/runtime
# drift (a lean image silently falling back to lexical at runtime).
#
# By default (`embedding`) the image installs the local semantic backend and
# bakes the model weights in, so paraphrase detection works out of the box and
# no text (nor model download) leaves the container at runtime. For a smaller,
# fully offline image without the ML stack, build with
# `--build-arg SIMILARITY_BACKEND=lexical`.
FROM python:3.12-slim

# "embedding" (default, installs torch + model) or "lexical" (lean, no ML).
ARG SIMILARITY_BACKEND=embedding
# Embedding model to bake in; only used when SIMILARITY_BACKEND=embedding.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Absolute path so the prompt config resolves regardless of working dir.
    PROMPT_CONFIG_PATH=/app/config/prompt.yaml \
    # Bake the selected backend as the runtime default so the image is
    # self-consistent: it defaults to exactly the backend it was built with,
    # regardless of any compose-level default.
    SIMILARITY_BACKEND=${SIMILARITY_BACKEND} \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    # Keep the model cache under /app so the non-root runtime user can read the
    # weights baked in at build time (default cache lives under root's home).
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

WORKDIR /app

# Install dependencies straight from pyproject.toml (the single source of truth).
# Copying the package before install means a code change invalidates the dep
# layer; acceptable here in exchange for not maintaining a second manifest.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir . \
    && if [ "$SIMILARITY_BACKEND" = "embedding" ]; then \
         pip install --no-cache-dir ".[embeddings]" \
         # Pre-download the model so runtime is fully offline.
         && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"; \
       fi

COPY config ./config

# Run as an unprivileged user (defence in depth for the banking context).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
