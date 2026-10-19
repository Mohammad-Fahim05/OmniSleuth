"""Unit and smoke tests for Phase 20 - Docker, Cloud Deployment & CI/CD Pipeline."""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_exists_and_security_configuration():
    """Verify Dockerfile exists and adheres to container security and runtime specifications."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root."

    content = dockerfile_path.read_text(encoding="utf-8")

    # 1. Slim Python base image
    assert re.search(r"FROM\s+python:3\.\d+-slim", content), "Must use official python slim base image."

    # 2. Non-root user setup & enforcement
    assert "appuser" in content, "Must create dedicated non-root appuser."
    assert "10001" in content, "Must use UID/GID 10001 for non-root user."
    assert re.search(r"USER\s+appuser", content), "Must switch to non-root user before entrypoint."

    # 3. Dynamic $PORT support for Cloud Run / ECS
    assert "PORT" in content, "Must define or support PORT environment variable."
    assert "8501" in content, "Must default or expose port 8501."

    # 4. Streamlit binding to 0.0.0.0
    assert "0.0.0.0" in content, "Streamlit must bind to 0.0.0.0."

    # 5. Healthcheck endpoint
    assert "HEALTHCHECK" in content, "Dockerfile must define a container HEALTHCHECK."
    assert "_stcore/health" in content, "Must check Streamlit internal /_stcore/health endpoint."

    # 6. Optimized layer caching (requirements copied before source code)
    req_pos = content.find("requirements.txt")
    src_pos = content.find("omnisleuth/")
    assert req_pos != -1 and src_pos != -1, "Both requirements and source code must be copied."
    assert req_pos < src_pos, "Requirements must be copied and installed prior to copying full source."

    # 7. Entrypoint supports CLI argument passthrough
    assert 'exec "$@"' in content or 'exec \\"$@\\"' in content, "Dockerfile ENTRYPOINT must support CLI argument passthrough."



def test_dockerignore_exclusions():
    """Verify .dockerignore exists and excludes build caches, virtualenvs, secrets, and logs."""
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist at repository root."

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    mandatory_patterns = [
        ".git",
        ".venv",
        "__pycache__",
        ".env",
        ".coverage",
        ".pytest_cache",
    ]
    for pattern in mandatory_patterns:
        matched = any(pattern in line for line in lines)
        assert matched, f".dockerignore must exclude '{pattern}'."


def test_docker_compose_configuration():
    """Verify docker-compose.yml exists and configures the omnisleuth service with volumes."""
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at repository root."

    content = compose_path.read_text(encoding="utf-8")

    assert "omnisleuth" in content, "Compose must define omnisleuth service."
    assert "8501" in content, "Compose must expose/map port 8501."
    assert "volumes:" in content, "Compose must configure persistent volumes."
    assert "./data" in content or "data:" in content, "Compose must mount data directory."
    assert "persisted_data" in content, "Compose must mount persistent database volume."


def test_github_actions_ci_workflow():
    """Verify GitHub Actions CI workflow configures testing, matrix, and container build."""
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), ".github/workflows/ci.yml must exist."

    content = ci_path.read_text(encoding="utf-8")

    # Workflow triggers
    assert "push:" in content and "pull_request:" in content, "CI must trigger on push and pull_request."

    # Pytest execution and offline testing
    assert "pytest" in content, "CI must execute pytest suite."

    # Python version matrix
    assert "matrix:" in content, "CI must test across Python version matrix."
    assert "3.10" in content or "3.11" in content, "Matrix must test supported Python 3.10+ versions."

    # Container build verification
    assert "docker" in content.lower(), "CI must include Docker container build / validation job."


def test_packaging_and_environment_files():
    """Verify pyproject.toml, requirements.txt, .env.example, and UI entrypoints exist."""
    assert (REPO_ROOT / "pyproject.toml").exists(), "pyproject.toml must exist."
    assert (REPO_ROOT / "requirements.txt").exists(), "requirements.txt must exist."
    assert (REPO_ROOT / ".env.example").exists(), ".env.example must exist."
    assert (REPO_ROOT / "omnisleuth" / "ui" / "app.py").exists(), "Streamlit app.py entrypoint must exist."
    assert (REPO_ROOT / "omnisleuth" / "cli.py").exists(), "CLI entrypoint must exist."


def test_docker_security_audit_claims():
    """Phase E audit: verify non-root user, persistent volume paths, and filesystem permissions."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    # 1. Non-root user compliance
    assert "groupadd -g 10001 appuser" in dockerfile
    assert "useradd -u 10001" in dockerfile
    assert "USER appuser" in dockerfile

    # 2. Dedicated directory ownership for non-root user
    assert "/app/data/uploads" in dockerfile
    assert "/app/logs" in dockerfile
    assert "/home/appuser/.streamlit" in dockerfile
    assert "chown -R appuser:appuser" in dockerfile

    # 3. Persistent SQLite & upload storage volume in compose
    assert "./data:/app/data" in compose
    assert "SQLITE_DB_PATH=/app/data/omnisleuth.db" in compose
    assert "UPLOADS_DIR=/app/data/uploads" in compose

    # 4. Verified healthcheck and port mapping
    assert "8501:8501" in compose
    assert "/_stcore/health" in compose

    # 5. Root filesystem in compose is standard writable (not read_only)
    # Streamlit, SQLite WAL sidecars, and logging require writable target paths
    assert "read_only: true" not in compose
