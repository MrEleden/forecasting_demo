"""Tests for Great Expectations data quality utilities."""

from pathlib import Path

import pytest

# Skip all tests in this module on Windows due to Great Expectations lark parser crash
try:
    from ml_portfolio.data.quality import validate_walmart_raw_dataset

    GREAT_EXPECTATIONS_AVAILABLE = True
except Exception:
    GREAT_EXPECTATIONS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GREAT_EXPECTATIONS_AVAILABLE, reason="Great Expectations causes access violation on Windows (lark parser crash)"
)


@pytest.mark.parametrize(
    "csv_content,expected_success",
    [
        (
            "Store,Date,Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment\n"
            "1,05-02-2010,1643690.9,0,42.31,2.572,211.0963582,8.106\n",
            True,
        ),
        (
            "Store,Date,Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment\n"
            "99,05-02-2010,-5,2,200,8.0,400,50\n",
            False,
        ),
    ],
)
def test_validate_walmart_raw_dataset(tmp_path: Path, csv_content: str, expected_success: bool) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    report = validate_walmart_raw_dataset(csv_path, raise_on_failure=False)

    assert report["success"] is expected_success
    assert "statistics" in report


def test_validate_walmart_raw_dataset_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        validate_walmart_raw_dataset(missing_path)
