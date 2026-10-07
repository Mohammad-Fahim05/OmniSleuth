# OmniSleuth Multi-Agent Data Investigation System Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY omnisleuth/ omnisleuth/
COPY data/ data/
COPY .env.example .env

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "omnisleuth/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
