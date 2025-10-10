"""Great Expectations data quality utilities for the ML portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:  # pragma: no cover - optional dependency
    import great_expectations as ge

    GE_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when dependency missing
    ge = None  # type: ignore
    GE_AVAILABLE = False


@dataclass
class DataQualityReport:
    """Lightweight summary of a Great Expectations validation run."""

    success: bool
    statistics: Dict[str, Any]
    expectation_results: List[Dict[str, Any]]
    skipped: bool = False
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "statistics": self.statistics,
            "expectation_results": self.expectation_results,
            "skipped": self.skipped,
            "reason": self.reason,
        }


def _execute_expectations(df: pd.DataFrame) -> Any:
    """Create and run Great Expectations validations for the Walmart raw dataset."""
    if ge is None:
        raise RuntimeError("Great Expectations is not installed. Install it to run data-quality checks.")

    validator = ge.from_pandas(df)

    validator.expect_column_values_to_not_be_null("Store")
    validator.expect_column_values_to_be_between("Store", min_value=1, max_value=45)

    validator.expect_column_values_to_not_be_null("Date")
    validator.expect_column_values_to_match_strftime_format("Date", "%d-%m-%Y")

    validator.expect_column_values_to_be_between("Weekly_Sales", min_value=0)

    validator.expect_column_values_to_be_in_set("Holiday_Flag", value_set=[0, 1])

    validator.expect_column_values_to_be_between("Temperature", min_value=-50, max_value=140)
    validator.expect_column_values_to_be_between("Fuel_Price", min_value=1.5, max_value=5.0)

    validator.expect_column_values_to_be_between("CPI", min_value=150, max_value=300)
    validator.expect_column_values_to_be_between("Unemployment", min_value=0, max_value=25)

    return validator.validate(result_format="SUMMARY")


def validate_walmart_raw_dataset(data_path: Path, raise_on_failure: bool = True) -> Dict[str, Any]:
    """Run Great Expectations checks on the Walmart raw dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found for validation: {data_path}")

    if ge is None:
        report = DataQualityReport(
            success=False,
            statistics={},
            expectation_results=[],
            skipped=True,
            reason="Great Expectations is not installed.",
        )
        return report.to_dict()

    df = pd.read_csv(data_path, dayfirst=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        if pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = df["Date"].dt.strftime("%d-%m-%Y")
    result = _execute_expectations(df)

    expectation_results = [
        {
            "expectation": res.expectation_config.expectation_type,
            "success": res.success,
            "observed_value": res.result.get("observed_value"),
            "details": res.result.get("details"),
        }
        for res in result.results
    ]

    report = DataQualityReport(
        success=result.success,
        statistics=result.statistics,
        expectation_results=expectation_results,
    )

    if raise_on_failure and not report.success:
        raise ValueError("Great Expectations validation failed. See expectation_results for details.")

    return report.to_dict()
