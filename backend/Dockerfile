# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into a target prefix so we can
# copy only the installed packages into the final image.
COPY requirements.txt .
RUN python - <<'PY'
from pathlib import Path

lines = Path("requirements.txt").read_text().splitlines()
filtered = [
    line for line in lines
    if line.strip() and not line.lstrip().startswith(("torch", "torchvision", "torchaudio", "triton"))
]
Path("requirements-docker.txt").write_text("\n".join(filtered) + "\n")
Path("constraints-docker.txt").write_text("torch==2.5.1\n")
PY
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1
RUN pip install --no-cache-dir --prefix=/install \
    -c constraints-docker.txt \
    -r requirements-docker.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create directories expected at runtime
RUN mkdir -p data indexes

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health-check so Docker / compose knows when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)"

# Default command — override in docker-compose for other services
CMD ["uvicorn", "06_api_deployment:app", "--host", "0.0.0.0", "--port", "8000"]
