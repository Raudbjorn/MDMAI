# Multi-stage build for TTRPG Assistant MCP Server
# Stage 1: Builder
FROM python:3.9-slim as builder

# Build arguments
ARG GPU_SUPPORT=none
ARG PYTHON_VERSION=3.9

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-cpu.txt ./
COPY setup.py pyproject.toml ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch based on GPU support
RUN if [ "$GPU_SUPPORT" = "cuda" ]; then \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$GPU_SUPPORT" = "rocm" ]; then \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7; \
    else \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install Python dependencies
COPY . /build/
RUN pip install -e .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 2: Runtime
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r ttrpg && useradd -r -g ttrpg ttrpg

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=ttrpg:ttrpg . /app/

# Create necessary directories
RUN mkdir -p /data /config /logs && \
    chown -R ttrpg:ttrpg /data /config /logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_DB_PATH=/data/chromadb \
    CACHE_DIR=/data/cache \
    LOG_FILE=/logs/ttrpg-assistant.log \
    CONFIG_DIR=/config

# Volume mounts
VOLUME ["/data", "/config", "/logs"]

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Switch to non-root user
USER ttrpg

# Entry point
ENTRYPOINT ["python", "-m", "src.main"]

# Default command (can be overridden)
CMD []