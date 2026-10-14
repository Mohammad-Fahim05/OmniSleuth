# ==============================================================================
# OmniSleuth — Production Multi-Agent Data Investigation System Dockerfile
# Multi-stage optimized, non-root hardened container for Local / Cloud Run / ECS
# ==============================================================================

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

WORKDIR /app

# Install minimal OS dependencies for building native packages & health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer Optimization: Copy dependency specifications first for caching
COPY requirements.txt pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create dedicated non-root user (UID/GID 10001) for security compliance
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -m -s /bin/bash appuser

# Copy application source code and data
COPY omnisleuth/ omnisleuth/
COPY data/ data/
COPY .env.example .env

# Create persistent storage directories and assign ownership to non-root appuser
RUN mkdir -p /app/persisted_data /app/data /home/appuser/.streamlit && \
    chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck validating Streamlit internal health status
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Launch Streamlit with dynamic Cloud Run $PORT and 0.0.0.0 address binding
ENTRYPOINT ["sh", "-c", "exec streamlit run omnisleuth/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
