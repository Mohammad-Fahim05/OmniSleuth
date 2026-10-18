# OmniSleuth — Setup and Operations Guide

This guide details local environment setup, application execution, testing workflows, dataset management, persistence operations, containerization status, and troubleshooting.

---

## 1. Environment & Dependency Setup

### Requirements
- **Operating System:** Linux, macOS, or Windows 10/11 (with PowerShell or Bash).
- **Python Version:** Python `>=3.10` (Python 3.11, 3.12, or 3.13 supported).

### Virtual Environment Creation
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS (Bash)
python3 -m venv .venv
source .venv/bin/activate
```

### Dependency Installation
```powershell
# Install core package in editable mode with development & UI dependencies
pip install --upgrade pip
pip install -e ".[dev,ui]"
```

---

## 2. Running OmniSleuth Locally

### Launching the Streamlit Interactive Workspace
```powershell
# Run from repository root
streamlit run omnisleuth/ui/app.py --server.port 8501 --server.address 0.0.0.0
```
Open your browser to: `http://localhost:8501`

### Offline-First Default Mode
By default, OmniSleuth requires **no external API keys or cloud credentials**. If no LLM keys are detected, the system safely operates in deterministic offline mode:
- All descriptive and trend queries are parsed and executed using deterministic Python/pandas math.
- Diagnostic "why" questions execute rule-based hypothesis generation against the allowlisted analytical tools.
- All numbers and claims are validated against the local evidence store.

### Optional OpenRouter Configuration
To enable LLM-assisted semantic query planning using OpenRouter:
1. Copy the example environment template:
   ```powershell
   cp .env.example .env
   ```
2. Edit `.env` and configure your API key (never commit `.env` to Git):
   ```env
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   OPENROUTER_MODEL=openai/gpt-4o-mini
   LLM_PROVIDER=openrouter
   ```
3. Restart Streamlit. The sidebar will indicate: `Configured (openrouter · openai/gpt-4o-mini)`.

---

## 3. Running Automated Tests

OmniSleuth has a comprehensive 509-test test suite covering core analytics, upload validation, security hardening, evidence grounding, and the Streamlit UI.

### Focused UI and Regression Suite (Fast)
```powershell
pytest tests/test_ui.py tests/test_ui_flow.py -v
```

### Security and Upload Validation Suite
```powershell
pytest tests/test_security_audit.py tests/test_upload_validation.py tests/test_persistence.py tests/test_guardrails.py -v
```

### Full Test Suite (All 509 Tests)
```powershell
pytest -q -p no:cacheprovider
```

---

## 4. Dataset Operations

### Loading the Benchmark Sample Dataset
1. Open the sidebar in the Streamlit UI.
2. Select **Sample Sales Dataset**.
3. Click **Load Sample Sales Dataset**.
4. The system deterministically synthesizes `sample_sales_2024.csv` (1,000 retail sales transactions with planted revenue anomalies), computes SHA-256 fingerprinting, runs data profiling, and rehydrates the session.

### Uploading a Custom Dataset
1. In the sidebar, select **Upload Custom File**.
2. Click **Browse files** or drag and drop a `.csv`, `.parquet`, or `.xlsx` file (up to 50 MB).
3. The upload pipeline automatically:
   - Validates MIME and binary magic bytes to reject disguised executables or archives.
   - Tests OpenXML structure for Excel files to defeat decompression bombs.
   - Verifies tabular shape (max 1,000 columns to prevent DoS).
   - Atomically streams the file into `data/uploads/` with a non-colliding UUID name.
   - Stores schema metadata in the SQLite repository under your active session ID.

---

## 5. Storage, Persistence & Backups

### Storage File Locations
- **SQLite Database:** `data/omnisleuth.db` (Configurable via `SQLITE_DB_PATH`)
- **Uploaded Datasets:** `data/uploads/` (Configurable via `UPLOADS_DIR`)
- **Application Logs:** `logs/omnisleuth.log` (Configurable via `LOG_FILE`)

### Online SQLite Backup (Without Stopping Application)
SQLite supports safe online snapshots using the `.backup` command:
```powershell
# Create timestamped database backup
sqlite3 data/omnisleuth.db ".backup 'data/backup_omnisleuth.db'"
```

### Backing Up Uploaded Files
```powershell
# Archive upload directory
tar -czvf uploads_backup.tar.gz data/uploads/
```

### Database Restore Procedure
1. Stop the Streamlit application.
2. Restore database file:
   ```powershell
   cp data/backup_omnisleuth.db data/omnisleuth.db
   ```
3. Restart the Streamlit application. `DatabaseManager` will automatically verify table structure and apply any pending additive migrations.

---

## 6. Container Deployment Status & Instructions

> [!NOTE]
> **Host Environment Status (T-007 Audit):**
> The Docker CLI is installed on this host (`Docker version 28.5.1`). However, the Docker Desktop daemon service (`com.docker.service`) was in a `Stopped` state and requires Windows Administrator elevation to start. Therefore, static container configurations (`Dockerfile`, `.dockerignore`, `docker-compose.yml`) have been verified, but actual runtime container execution awaits administrator daemon startup.

### Docker Build Command (Once Daemon is Active)
```bash
docker build -t omnisleuth:latest .
```

### Docker Run Command (Single-Instance with Persistent Mounts)
```bash
docker run -d \
  --name omnisleuth-app \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e APP_ENV=production \
  -e LOG_LEVEL=INFO \
  omnisleuth:latest
```

### Verifying Container Health
```bash
# Check container status
docker ps -f name=omnisleuth-app

# Probe Streamlit internal health endpoint
curl -I http://localhost:8501/_stcore/health
```

---

## 7. Troubleshooting Common Issues

### 1. Port Conflict (`Port 8501 is not available`)
- **Cause:** Another process or previous Streamlit instance is bound to port 8501.
- **Fix:** Specify an alternative port:
  ```powershell
  streamlit run omnisleuth/ui/app.py --server.port 8504
  ```
  Or identify and terminate the process holding the port:
  ```powershell
  Get-NetTCPConnection -LocalPort 8501 | Select-Object OwningProcess
  Stop-Process -Id <PID> -Force
  ```

### 2. SQLite Database Locked (`sqlite3.OperationalError: database is locked`)
- **Cause:** A long-running write transaction or concurrent processes accessing SQLite without WAL mode.
- **Mitigation:** OmniSleuth's `DatabaseManager` automatically configures `PRAGMA busy_timeout = 30000;` (30 seconds) on every connection. Ensure your local filesystem does not have network-drive file locking delays.

### 3. OpenRouter API Key Error / Fallback Notice
- **Symptom:** UI displays notice: `LLM planner selected but no provider is configured. Deterministic planner is used instead.`
- **Cause:** `OPENROUTER_API_KEY` is missing or invalid in `.env`.
- **Mitigation:** The system intentionally continues operating safely using rule-based planning. To use LLM planning, verify your key in `.env` and verify provider is set to `openrouter`.

### 4. Upload Rejected (`Invalid CSV file: File appears to be a Windows executable`)
- **Cause:** The uploaded file's binary magic bytes matched an executable or archive signature rather than plain text CSV.
- **Mitigation:** Ensure custom files are valid UTF-8/ASCII CSV or valid Parquet/OpenXML Excel files.
