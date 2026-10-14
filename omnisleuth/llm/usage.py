"""Token usage and pricing cost calculation utility."""

from typing import Optional
from omnisleuth.core.config import get_settings
from omnisleuth.llm.models import UsageMetrics


def calculate_usage_and_cost(
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    provider: str = "fallback",
    input_price_per_1k: Optional[float] = None,
    output_price_per_1k: Optional[float] = None,
) -> UsageMetrics:
    """Calculate token metrics and cost.

    If provider is fallback or offline, return cost=0.0 and is_actual=False.
    If actual token counts are provided by an API, calculate exact cost.
    If no token counts are provided, keep tokens and cost as None without inventing numbers.
    """
    settings = get_settings()

    # Offline / Fallback mode
    if provider == "fallback":
        return UsageMetrics(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            is_actual=False,
        )

    # If provider returns token counts
    if input_tokens is not None or output_tokens is not None:
        in_tok = input_tokens or 0
        out_tok = output_tokens or 0
        tot_tok = total_tokens if total_tokens is not None else (in_tok + out_tok)

        in_price = input_price_per_1k if input_price_per_1k is not None else settings.llm_input_price_per_1k
        out_price = output_price_per_1k if output_price_per_1k is not None else settings.llm_output_price_per_1k

        cost: Optional[float] = None
        if in_price is not None and out_price is not None:
            cost = round((in_tok / 1000.0 * in_price) + (out_tok / 1000.0 * out_price), 6)

        return UsageMetrics(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            estimated_cost_usd=cost,
            is_actual=True,
        )

    # Provider supplied no token metadata
    return UsageMetrics(
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        estimated_cost_usd=None,
        is_actual=False,
    )
