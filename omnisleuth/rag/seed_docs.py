"""Initial seed domain documents for the contextual RAG knowledge base."""

from typing import List, Tuple
from omnisleuth.rag.models import DocumentType

SEED_DOCUMENTS: List[Tuple[str, str, DocumentType, str]] = [
    (
        "doc_data_dict",
        "OmniSleuth Sales Data Dictionary",
        DocumentType.DATA_DICTIONARY,
        """# OmniSleuth Sales Data Dictionary
## Field Definitions
- `order_id`: Unique identifier for every customer transaction.
- `order_date`: Timestamp when the purchase was executed (YYYY-MM-DD).
- `customer_id`: Unique identifier for the purchasing client or organization.
- `region`: Geographic sales territory ('North', 'South', 'East', 'West').
- `category`: Product grouping category ('Electronics', 'Furniture', 'Clothing').
- `revenue`: Gross financial transaction value in USD before applicable refunds.
- `units_sold`: Count of discrete product units included in the transaction.
- `is_returned`: Boolean flag indicating whether the order was refunded or reversed.
""",
    ),
    (
        "doc_kpi_defs",
        "Executive KPI Definitions & Methodology",
        DocumentType.KPI_DEFINITION,
        """# Executive KPI Definitions & Business Metrics
## Core Financial KPIs
- **Revenue**: Sum total of all recognized gross transaction values across settled orders.
- **Order Volume**: Total count of unique completed transactions (`COUNT(DISTINCT order_id)`).
- **Average Order Value (AOV)**: Mean expenditure per transaction calculated as `Revenue / Order Volume`.
- **Segment Contribution**: The proportion of overall period-over-period metric change attributable to an individual dimension: `(Delta_segment / Delta_total) * 100%`.
- **Return Rate**: Percentage of processed orders flagged with `is_returned = True`.
""",
    ),
    (
        "doc_q1_report",
        "Q1 2024 Quarterly Business Review Report",
        DocumentType.QUARTERLY_REPORT,
        """# Q1 2024 Quarterly Business Review (QBR)
## Executive Operational Summary
In March 2024, our primary tier-1 microchip and electronics component supplier in the North region experienced a severe 3-week facility shutdown due to localized grid outages.
Consequently, our North regional distribution center suffered significant inventory stockouts in high-margin Electronics products from March 1st through March 22nd.
While customer demand remained healthy, fulfilled order volume and available stock in Electronics dropped sharply during March.
Operations resumed standard capacity in April 2024 after secondary supply lines were activated.
""",
    ),
    (
        "doc_biz_rules",
        "OmniSleuth Analytical Guardrails & Business Rules",
        DocumentType.BUSINESS_RULE,
        """# Business Rules & Analytical Policy
## Investigation Standards
1. **Rule of Determinism**: All figures, totals, and percentages must be calculated directly from data using approved tools—never guessed.
2. **Attribution vs Causality**: Segment declines describe mathematical contributors. External events in quarterly reports provide contextual association, not absolute causal proof.
3. **Anomaly Flagging**: Extreme outlier transactions exceeding 3 standard deviations must be flagged for manual auditing before final accounting sign-off.
""",
    ),
]
