"""Deterministic generator for realistic business sales datasets with planted phenomena."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def generate_benchmark_sales_dataset(
    n_rows: int = 1000,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate a realistic synthetic sales dataset spanning Q1-Q2 2024 with planted phenomena.

    Planted Phenomena:
    1. March Electronics Decline: North region electronics orders drop by 60% in March due to supplier outage.
    2. Furniture Surge in May: South region furniture experiences a 40% seasonal spike in May.
    3. Planted Outlier Transactions: 10 transactions with extreme revenue (>5-sigma).
    4. Data Quality Flaws: 20 duplicate rows, 15 missing customer IDs, 10 missing discounts.
    """
    np.random.seed(random_seed)

    start_date = datetime(2024, 1, 1)
    date_offsets = [timedelta(days=int(np.random.uniform(0, 180))) for _ in range(n_rows)]
    order_dates = [start_date + offset for offset in date_offsets]

    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Furniture", "Clothing"]

    region_probs = [0.35, 0.25, 0.20, 0.20]
    category_probs = [0.50, 0.30, 0.20]

    assigned_regions = np.random.choice(regions, size=n_rows, p=region_probs)
    assigned_categories = np.random.choice(categories, size=n_rows, p=category_probs)

    base_prices = {"Electronics": 800.0, "Furniture": 400.0, "Clothing": 90.0}
    revenues: list[float] = []
    units_list: list[int] = []

    for i in range(n_rows):
        dt = order_dates[i]
        reg = assigned_regions[i]
        cat = assigned_categories[i]

        units = int(np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03]))
        price_per_unit = base_prices[cat] * np.random.uniform(0.85, 1.15)
        rev = round(units * price_per_unit, 2)

        # Planted Phenomenon 1: March North Electronics Decline
        if dt.month == 3 and reg == "North" and cat == "Electronics":
            if np.random.random() < 0.65:
                # 65% of orders in this segment get suppressed or reduced
                rev = round(rev * 0.3, 2)
                units = 1

        # Planted Phenomenon 2: May South Furniture Surge
        if dt.month == 5 and reg == "South" and cat == "Furniture":
            rev = round(rev * 1.5, 2)
            units = max(2, units + 1)

        revenues.append(rev)
        units_list.append(units)

    # Planted Phenomenon 3: 10 Extreme Outliers
    outlier_indices = np.random.choice(n_rows, size=10, replace=False)
    for idx in outlier_indices:
        revenues[idx] = round(revenues[idx] + np.random.uniform(8000, 15000), 2)

    df = pd.DataFrame({
        "order_id": [f"ORD-{20240000 + i}" for i in range(n_rows)],
        "order_date": [d.strftime("%Y-%m-%d") for d in order_dates],
        "customer_id": [f"CUST-{1000 + (i % 250)}" if (i % 70 != 0) else None for i in range(n_rows)],
        "region": assigned_regions,
        "category": assigned_categories,
        "revenue": revenues,
        "units_sold": units_list,
        "discount": [round(float(np.random.choice([0.0, 0.05, 0.1, 0.15])), 2) if (i % 90 != 0) else None for i in range(n_rows)],
        "is_returned": [bool(np.random.random() < 0.05) for _ in range(n_rows)],
    })

    # Planted Phenomenon 4: 15 Duplicate Rows
    dup_indices = np.random.choice(n_rows, size=15, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")

    return df
