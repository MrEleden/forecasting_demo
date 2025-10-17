#!/usr/bin/env python3
"""
Download Transportation Services Index (TSI) data from U.S. Bureau of Transportation Statistics.

This script downloads official TSI data from the BTS Data Portal.
TSI is a seasonally-adjusted economic indicator tracking freight and passenger traffic.

Usage:
    python download_tsi_data.py

Data Source:
    - Bureau of Transportation Statistics (BTS) Data Portal
    - URL: https://data.bts.gov/Research-and-Statistics/
      Transportation-Services-Index-and-Seasonally-Adjus/bw6n-ddqk/about_data
    - API Endpoint: https://data.bts.gov/resource/bw6n-ddqk.csv
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


def download_tsi_data():
    """Download TSI data from BTS Data Portal via API."""

    # Set up paths
    project_root = Path(__file__).parent.parent
    data_raw = project_root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {data_raw}")

    # BTS Data Portal API endpoint for TSI data
    # This is the official Socrata API endpoint for the dataset
    api_url = "https://data.bts.gov/resource/bw6n-ddqk.csv"

    try:
        print("Downloading Transportation Services Index from BTS Data Portal...")

        # Download the complete dataset
        # Use simpler API call without complex parameters first
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()

        # Save raw CSV data
        csv_file = data_raw / "tsi_official.csv"
        csv_file.write_text(response.text)

        # Load and validate the data
        df = pd.read_csv(csv_file)

        size_kb = csv_file.stat().st_size / 1024
        print(f"Downloaded: tsi_official.csv ({size_kb:.1f} KB)")
        print(f"{len(df)} records with {len(df.columns)} columns")

        # Show column information
        print(f"Columns: {list(df.columns)}")

        # Show date range if available
        date_columns = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                print(f"Date range: {df[date_col].min().strftime('%Y-%m')} to {df[date_col].max().strftime('%Y-%m')}")
            except Exception as e:
                print(f"Could not parse dates in {date_col}: {e}")

        # Show sample data
        print("Sample data (first 3 rows):")
        print(df.head(3).to_string())

        return csv_file

    except requests.exceptions.RequestException as e:
        print(f"Failed to download from BTS API: {e}")
        print("\nAPI download failed. Creating sample data for development...")
        return create_sample_tsi_data(data_raw)

    except Exception as e:
        print(f"Error processing TSI data: {e}")
        print("\nProcessing failed. Creating sample data for development...")
        return create_sample_tsi_data(data_raw)


def create_sample_tsi_data(data_raw):
    """Create sample TSI data for development purposes."""

    # Generate sample TSI data

    import numpy as np

    # Create monthly time series from 2010 to 2024
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 12, 31)

    # Generate monthly dates
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    # Create realistic TSI patterns
    np.random.seed(42)  # For reproducibility

    # Base trend with seasonality and economic cycles
    n_months = len(dates)
    trend = 100 + np.linspace(0, 50, n_months)  # Long-term growth
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_months) / 12)  # Seasonal pattern
    economic_cycle = 10 * np.sin(2 * np.pi * np.arange(n_months) / 60)  # 5-year cycle
    noise = np.random.normal(0, 2, n_months)

    # Overall TSI
    tsi_overall = trend + seasonal + economic_cycle + noise

    # Freight (slightly more volatile)
    freight_multiplier = 1.2
    freight_noise = np.random.normal(0, 3, n_months)
    tsi_freight = (trend + seasonal + economic_cycle) * freight_multiplier + freight_noise

    # Passenger (more seasonal variation)
    passenger_seasonal = 8 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    tsi_passenger = trend + passenger_seasonal + economic_cycle * 0.8 + noise * 0.8

    # Create comprehensive dataset
    tsi_data = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "tsi_overall": np.round(tsi_overall, 2),
            "tsi_freight": np.round(tsi_freight, 2),
            "tsi_passenger": np.round(tsi_passenger, 2),
            "seasonally_adjusted": True,
        }
    )

    # Add economic recession periods (2008-2009, 2020 COVID)
    recession_2020 = (tsi_data["date"] >= "2020-03-01") & (tsi_data["date"] <= "2020-12-01")
    tsi_data.loc[recession_2020, "tsi_overall"] *= 0.85
    tsi_data.loc[recession_2020, "tsi_freight"] *= 0.80
    tsi_data.loc[recession_2020, "tsi_passenger"] *= 0.60

    # Save sample data
    sample_file = data_raw / "tsi_sample.csv"
    tsi_data.to_csv(sample_file, index=False)

    size_kb = sample_file.stat().st_size / 1024
    print(f"Created sample TSI data: tsi_sample.csv ({size_kb:.1f} KB)")
    print(f"Data range: {dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}")
    print(f"{len(tsi_data)} monthly observations")

    return sample_file


def validate_tsi_data():
    """Validate TSI data files."""
    project_root = Path(__file__).parent.parent
    data_raw = project_root / "data" / "raw"

    print("\nValidating TSI data files...")

    csv_files = list(data_raw.glob("*.csv"))
    if not csv_files:
        print("No CSV files found")
        return False

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"{csv_file.name}: {len(df)} rows, {len(df.columns)} columns ({size_mb:.2f} MB)")

            # Show first few column names
            if len(df.columns) > 5:
                cols_preview = f"{', '.join(df.columns[:5])}, ..."
            else:
                cols_preview = ", ".join(df.columns)
            print(f"  Columns: {cols_preview}")

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")

    return True


if __name__ == "__main__":
    print("Transportation Services Index (TSI) Data Downloader")
    print("=" * 55)
    print("Source: BTS Data Portal")
    print(
        "URL: https://data.bts.gov/Research-and-Statistics/"
        "Transportation-Services-Index-and-Seasonally-Adjus/bw6n-ddqk/about_data"
    )

    # Download data
    try:
        data_file = download_tsi_data()

        # Validate data
        if validate_tsi_data():
            print("\nTSI data setup complete!")
            print("Next steps:")
            print("   1. Explore data: notebooks/01_eda.ipynb")
            print("   2. Check seasonal adjustments and economic patterns")
            print("   3. Train time series models (ARIMA, Prophet)")
            print("   4. Analyze economic cycle correlations")
        else:
            print("\nData validation failed. Check downloaded files.")

    except Exception as e:
        print(f"\nDownload process failed: {e}")
        print("\nManual download alternative:")
        print(
            "   1. Visit: https://data.bts.gov/Research-and-Statistics/"
            "Transportation-Services-Index-and-Seasonally-Adjus/bw6n-ddqk/about_data"
        )
        print("   2. Click 'Export' → 'CSV'")
        print("   3. Save as: projects/transportation_tsi/data/raw/tsi_official.csv")
