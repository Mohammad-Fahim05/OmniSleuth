"""Unit tests for Phase 19 — LLM Gateway, timeout, retries, fallback, tracing, and caching."""

import time
from typing import Optional
from unittest.mock import MagicMock, patch
import pytest

from omnisleuth.core.config import get_settings, reset_settings
from omnisleuth.llm import (
    BaseProvider,
    DeterministicFallbackProvider,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    OpenAIProvider,
    SemanticCache,
    UsageMetrics,
    calculate_usage_and_cost,
    generate_request_id,
)
from omnisleuth.llm.retry import execute_with_retry, is_transient_error


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gateway & Model Abstraction Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_request_id_generation():
    """Verify request ID format."""
    req_id = generate_request_id()
    assert req_id.startswith("llm_req_")
    assert len(req_id) > 16


def test_deterministic_fallback_provider():
    """Verify DeterministicFallbackProvider returns structured offline response."""
    provider = DeterministicFallbackProvider()
    req = LLMRequest(prompt="Analyze sales trend", investigation_id="inv_123")
    res = provider.generate(req, request_id="llm_req_test")

    assert res.request_id == "llm_req_test"
    assert res.provider == "fallback"
    assert res.model == "deterministic"
    assert res.fallback_used is True
    assert res.usage.estimated_cost_usd == 0.0
    assert res.usage.is_actual is False
    assert "[OFFLINE DETERMINISTIC SYNTHESIS]" in res.text


def test_gateway_offline_fallback():
    """Verify LLMGateway uses fallback provider when no API credentials exist."""
    gateway = LLMGateway()
    req = LLMRequest(prompt="Why did revenue drop in March?", dataset_version="ver_123")
    res = gateway.generate(req)

    assert res.fallback_used is True
    assert res.provider == "fallback"
    assert res.model == "deterministic"
    assert res.latency_ms >= 0.0
    assert len(gateway.list_traces()) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Timeout & Retry Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_transient_vs_permanent_error_classification():
    """Test transient error detection for retry logic."""
    assert is_transient_error(Exception("Connection timed out")) is True
    assert is_transient_error(Exception("Rate limit exceeded 429")) is True
    assert is_transient_error(Exception("503 Service Unavailable")) is True

    # Permanent errors should not be retried
    assert is_transient_error(Exception("401 Unauthorized invalid_api_key")) is False
    assert is_transient_error(Exception("400 Bad Request invalid model")) is False


def test_execute_with_retry_transient_success():
    """Test retry mechanism succeeds after a transient failure."""
    call_count = 0

    def mock_flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("503 Service Unavailable")
        return "success"

    res, retries = execute_with_retry(mock_flaky_func, max_retries=3, base_delay=0.01)
    assert res == "success"
    assert retries == 1
    assert call_count == 2


def test_execute_with_retry_permanent_failure_immediate_abort():
    """Test retry mechanism immediately aborts on permanent error without retrying."""
    call_count = 0

    def mock_perm_fail():
        nonlocal call_count
        call_count += 1
        raise Exception("401 Unauthorized invalid_api_key")

    with pytest.raises(Exception, match="401 Unauthorized"):
        execute_with_retry(mock_perm_fail, max_retries=3, base_delay=0.01)

    assert call_count == 1  # Aborted immediately, 0 retries


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom Provider & Mock Tests
# ─────────────────────────────────────────────────────────────────────────────

class MockSuccessProvider(BaseProvider):
    """Mock provider returning a successful response."""

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        return LLMResponse(
            request_id=request_id,
            text="Mocked LLM analytical output.",
            provider="mock",
            model="mock-gpt",
            latency_ms=15.0,
            fallback_used=False,
            usage=UsageMetrics(input_tokens=100, output_tokens=50, total_tokens=150, estimated_cost_usd=0.000045, is_actual=True),
        )


class MockTransientFailProvider(BaseProvider):
    """Mock provider raising transient 429 rate limit error."""

    def generate(self, request: LLMRequest, request_id: str) -> LLMResponse:
        raise Exception("429 Rate limit exceeded")


def test_gateway_with_mock_success_provider():
    """Test LLMGateway with a custom successful provider."""
    gateway = LLMGateway(provider=MockSuccessProvider())
    req = LLMRequest(prompt="Summarize trend")
    res = gateway.generate(req)

    assert res.fallback_used is False
    assert res.text == "Mocked LLM analytical output."
    assert res.provider == "mock"
    assert res.usage.input_tokens == 100
    assert res.usage.output_tokens == 50
    assert res.usage.estimated_cost_usd == 0.000045


def test_gateway_fallback_on_transient_failure_exhaustion():
    """Test gateway falling back to deterministic provider when primary provider retries are exhausted."""
    gateway = LLMGateway(provider=MockTransientFailProvider())

    # Lower delays for fast test execution
    with patch.object(gateway.settings, "llm_max_retries", 2), \
         patch.object(gateway.settings, "llm_retry_base_delay", 0.01):
        req = LLMRequest(prompt="Complex question")
        res = gateway.generate(req)

        assert res.fallback_used is True
        assert res.provider == "fallback"
        assert res.model == "deterministic"
        assert res.retry_count == 2
        assert "Primary provider failed" in res.error


# ─────────────────────────────────────────────────────────────────────────────
# 4. Token & Cost Logging Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_usage_and_cost_calculation():
    """Test token and cost calculation rules."""
    # Actual tokens with configured pricing
    usage = calculate_usage_and_cost(
        input_tokens=1000,
        output_tokens=500,
        provider="openai",
        input_price_per_1k=0.0015,
        output_price_per_1k=0.0020,
    )
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 500
    assert usage.total_tokens == 1500
    assert usage.estimated_cost_usd == 0.0025  # (1*0.0015) + (0.5*0.002)
    assert usage.is_actual is True

    # Offline / Fallback
    fallback_usage = calculate_usage_and_cost(provider="fallback")
    assert fallback_usage.estimated_cost_usd == 0.0
    assert fallback_usage.is_actual is False

    # Missing token metadata
    no_tokens_usage = calculate_usage_and_cost(input_tokens=None, output_tokens=None, provider="custom")
    assert no_tokens_usage.input_tokens is None
    assert no_tokens_usage.estimated_cost_usd is None
    assert no_tokens_usage.is_actual is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Request Tracing & Correlation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tracing_and_investigation_correlation():
    """Test logging and retrieving traces correlated with investigation IDs."""
    gateway = LLMGateway(provider=MockSuccessProvider())

    req1 = LLMRequest(prompt="Query 1", investigation_id="inv_alpha")
    req2 = LLMRequest(prompt="Query 2", investigation_id="inv_alpha")
    req3 = LLMRequest(prompt="Query 3", investigation_id="inv_beta")

    gateway.generate(req1)
    gateway.generate(req2)
    gateway.generate(req3)

    traces_alpha = gateway.get_traces_for_investigation("inv_alpha")
    assert len(traces_alpha) == 2
    assert all(t.investigation_id == "inv_alpha" for t in traces_alpha)

    traces_beta = gateway.get_traces_for_investigation("inv_beta")
    assert len(traces_beta) == 1
    assert traces_beta[0].investigation_id == "inv_beta"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Semantic Cache Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_semantic_cache_hit_and_dataset_isolation():
    """Test cache hit, cache miss, and dataset-version key isolation."""
    cache = SemanticCache(enabled=True)
    gateway = LLMGateway(provider=MockSuccessProvider(), cache=cache)

    req_v1 = LLMRequest(prompt="What is total sales?", dataset_version="dataset_v1")
    req_v2 = LLMRequest(prompt="What is total sales?", dataset_version="dataset_v2")

    # Initial call -> cache miss -> executes provider -> puts in cache
    res1 = gateway.generate(req_v1)
    assert res1.cached is False

    # Second call with same dataset_version -> cache hit
    res1_hit = gateway.generate(req_v1)
    assert res1_hit.cached is True

    # Call with DIFFERENT dataset_version -> cache miss (dataset isolation!)
    res2 = gateway.generate(req_v2)
    assert res2.cached is False


def test_disabled_cache():
    """Test that disabled cache never returns cached hits."""
    cache = SemanticCache(enabled=False)
    gateway = LLMGateway(provider=MockSuccessProvider(), cache=cache)

    req = LLMRequest(prompt="What is total sales?", dataset_version="v1")
    res1 = gateway.generate(req)
    res2 = gateway.generate(req)

    assert res1.cached is False
    assert res2.cached is False
