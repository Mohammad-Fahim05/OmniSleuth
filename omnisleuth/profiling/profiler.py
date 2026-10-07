"""Deterministic profiling and non-destructive data quality assessment engine."""

from typing import List, Optional
import numpy as np
import pandas as pd
from omnisleuth.core.logging import get_logger
from omnisleuth.ingestion.models import ColumnType, DatasetMetadata
from omnisleuth.profiling.models import (
    CategoricalSummary,
    CategoryFrequency,
    DatetimeSummary,
    NumericSummary,
    QualityReport,
    QualityWarning,
    WarningSeverity,
)

logger = get_logger("profiling")


class DatasetProfiler:
    """Profiles DataFrames and generates data quality reports."""

    def __init__(
        self,
        high_missing_warning_threshold: float = 20.0,
        high_missing_critical_threshold: float = 50.0,
        high_cardinality_threshold: int = 100,
        top_categories_count: int = 10,
    ):
        self.high_missing_warning_threshold = high_missing_warning_threshold
        self.high_missing_critical_threshold = high_missing_critical_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.top_categories_count = top_categories_count

    def profile_numeric(self, series: pd.Series, col_name: str) -> Optional[NumericSummary]:
        """Compute statistical properties for a numeric series."""
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) == 0:
            return None

        q25 = float(np.percentile(clean, 25))
        q75 = float(np.percentile(clean, 75))
        iqr = float(q75 - q25)

        return NumericSummary(
            column_name=col_name,
            count=int(len(clean)),
            missing_count=int(series.isna().sum()),
            mean=round(float(clean.mean()), 4),
            std=round(float(clean.std(ddof=1) if len(clean) > 1 else 0.0), 4),
            median=round(float(clean.median()), 4),
            min=round(float(clean.min()), 4),
            max=round(float(clean.max()), 4),
            q25=round(q25, 4),
            q75=round(q75, 4),
            iqr=round(iqr, 4),
            zero_count=int((clean == 0).sum()),
            negative_count=int((clean < 0).sum()),
        )

    def profile_categorical(self, series: pd.Series, col_name: str) -> CategoricalSummary:
        """Compute cardinality and top frequency metrics for a categorical series."""
        clean = series.dropna().astype(str)
        cardinality = int(clean.nunique())
        missing_count = int(series.isna().sum())
        total_non_null = len(clean)

        top_freqs: List[CategoryFrequency] = []
        if total_non_null > 0:
            val_counts = clean.value_counts().head(self.top_categories_count)
            for val, count in val_counts.items():
                pct = round((count / total_non_null) * 100.0, 2)
                top_freqs.append(CategoryFrequency(value=str(val), count=int(count), percentage=pct))

        return CategoricalSummary(
            column_name=col_name,
            cardinality=cardinality,
            missing_count=missing_count,
            top_frequencies=top_freqs,
        )

    def profile_datetime(self, series: pd.Series, col_name: str) -> Optional[DatetimeSummary]:
        """Compute date ranges and temporal span for a datetime series."""
        clean_dates = pd.to_datetime(series, errors="coerce").dropna()
        if len(clean_dates) == 0:
            return None

        min_date = clean_dates.min()
        max_date = clean_dates.max()
        span_days = round((max_date - min_date).total_seconds() / 86400.0, 2)

        return DatetimeSummary(
            column_name=col_name,
            min_date=str(min_date),
            max_date=str(max_date),
            timespan_days=span_days,
            missing_count=int(series.isna().sum()),
        )

    def check_quality_warnings(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        numeric_summaries: dict[str, NumericSummary],
        categorical_summaries: dict[str, CategoricalSummary],
    ) -> List[QualityWarning]:
        """Check rule-based data quality issues."""
        warnings: List[QualityWarning] = []
        total_rows = len(df)

        if total_rows == 0:
            warnings.append(
                QualityWarning(
                    severity=WarningSeverity.CRITICAL,
                    rule="empty_dataset",
                    message="Dataset contains 0 rows.",
                )
            )
            return warnings

        # 1. Duplicate row check
        dup_count = metadata.duplicate_rows
        if dup_count > 0:
            dup_pct = (dup_count / total_rows) * 100.0
            sev = WarningSeverity.WARNING if dup_pct < 10.0 else WarningSeverity.CRITICAL
            warnings.append(
                QualityWarning(
                    severity=sev,
                    rule="duplicate_rows",
                    message=f"Dataset has {dup_count} duplicate rows ({dup_pct:.1f}% of total).",
                )
            )

        # 2. Missing value checks per column
        for col_schema in metadata.columns:
            missing_pct = col_schema.missing_percentage
            if missing_pct >= self.high_missing_critical_threshold:
                warnings.append(
                    QualityWarning(
                        column_name=col_schema.name,
                        severity=WarningSeverity.CRITICAL,
                        rule="high_missing_values",
                        message=f"Column '{col_schema.name}' has {missing_pct:.1f}% missing values (exceeds critical threshold {self.high_missing_critical_threshold}%).",
                    )
                )
            elif missing_pct >= self.high_missing_warning_threshold:
                warnings.append(
                    QualityWarning(
                        column_name=col_schema.name,
                        severity=WarningSeverity.WARNING,
                        rule="high_missing_values",
                        message=f"Column '{col_schema.name}' has {missing_pct:.1f}% missing values.",
                    )
                )

        # 3. High cardinality on categorical columns
        for col_name, cat_sum in categorical_summaries.items():
            if cat_sum.cardinality > self.high_cardinality_threshold:
                warnings.append(
                    QualityWarning(
                        column_name=col_name,
                        severity=WarningSeverity.INFO,
                        rule="high_cardinality",
                        message=f"Categorical column '{col_name}' has high cardinality ({cat_sum.cardinality} distinct values).",
                    )
                )

        # 4. Zero variance / constant columns
        for col_name, num_sum in numeric_summaries.items():
            if num_sum.std == 0.0 and num_sum.count > 1:
                warnings.append(
                    QualityWarning(
                        column_name=col_name,
                        severity=WarningSeverity.WARNING,
                        rule="zero_variance",
                        message=f"Numeric column '{col_name}' is constant with zero variance (all values = {num_sum.mean}).",
                    )
                )

        return warnings

    def generate_report(self, df: pd.DataFrame, metadata: DatasetMetadata) -> QualityReport:
        """Produce a complete, reproducible QualityReport for a dataset."""
        logger.info(f"Generating quality report for dataset '{metadata.name}' ({metadata.dataset_id})")

        numeric_summaries: dict[str, NumericSummary] = {}
        categorical_summaries: dict[str, CategoricalSummary] = {}
        datetime_summaries: dict[str, DatetimeSummary] = {}

        for col_schema in metadata.columns:
            col_name = col_schema.name
            if col_schema.data_type == ColumnType.NUMERIC:
                summary = self.profile_numeric(df[col_name], col_name)
                if summary:
                    numeric_summaries[col_name] = summary
            elif col_schema.data_type in [ColumnType.CATEGORICAL, ColumnType.IDENTIFIER]:
                categorical_summaries[col_name] = self.profile_categorical(df[col_name], col_name)
            elif col_schema.data_type == ColumnType.DATETIME:
                dt_summary = self.profile_datetime(df[col_name], col_name)
                if dt_summary:
                    datetime_summaries[col_name] = dt_summary

        # Calculate overall dataset completeness
        total_cells = df.size
        missing_cells = int(df.isna().sum().sum())
        completeness = round(((total_cells - missing_cells) / total_cells * 100.0), 2) if total_cells > 0 else 0.0

        dup_pct = round((metadata.duplicate_rows / len(df) * 100.0), 2) if len(df) > 0 else 0.0

        warnings = self.check_quality_warnings(df, metadata, numeric_summaries, categorical_summaries)

        report = QualityReport(
            dataset_id=metadata.dataset_id,
            version_hash=metadata.version_hash,
            row_count=metadata.row_count,
            column_count=metadata.column_count,
            duplicate_rows=metadata.duplicate_rows,
            duplicate_percentage=dup_pct,
            completeness_score=completeness,
            warnings=warnings,
            numeric_summaries=numeric_summaries,
            categorical_summaries=categorical_summaries,
            datetime_summaries=datetime_summaries,
        )

        logger.info(
            f"Quality report created for '{metadata.name}': Completeness={report.completeness_score}%, "
            f"Warnings={len(report.warnings)}"
        )
        return report
