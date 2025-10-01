#!/usr/bin/env python3
"""
Download Walmart sales forecasting dataset from Kaggle.

This script downloads the Walmart dataset from yasserh/walmart-dataset
on Kaggle. Requires Kaggle API credentials.

Dataset URL: https://www.kaggle.com/datasets/yasserh/walmart-dataset

Usage:
    python download_data.py

Requirements:
    - Kaggle API installed: pip install kaggle
    - Kaggle API credentials configured: ~/.kaggle/kaggle.json
"""

import os
import zipfile
from pathlib import Path
import subprocess
import sys


def setup_kaggle_api():
    """Check if Kaggle API is installed and configured."""
    try:
        import kaggle

        print("Kaggle API is installed")
        return True
    except ImportError:
        print("Kaggle API not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle

        return True


def download_walmart_data():
    """Download Walmart sales forecasting dataset from Kaggle."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_raw = project_root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {data_raw}")

    try:
        # Download the Walmart dataset from yasserh
        dataset_name = "yasserh/walmart-dataset"

        print(f"Downloading {dataset_name} dataset...")

        # Use kaggle CLI to download dataset (not competition)
        cmd = ["kaggle", "datasets", "download", dataset_name, "--path", str(data_raw)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("Download completed successfully")

            # Extract zip files
            zip_files = list(data_raw.glob("*.zip"))
            for zip_file in zip_files:
                print(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(data_raw)

                # Remove zip file after extraction
                zip_file.unlink()
                print(f"Extracted and removed {zip_file.name}")

            # List downloaded files
            csv_files = list(data_raw.glob("*.csv"))
            print(f"\nDownloaded files:")
            for csv_file in csv_files:
                size_mb = csv_file.stat().st_size / (1024 * 1024)
                print(f"  - {csv_file.name} ({size_mb:.1f} MB)")

        else:
            print(f"Download failed: {result.stderr}")
            print("Make sure you have:")
            print("   1. Kaggle API installed: pip install kaggle")
            print("   2. API credentials configured: ~/.kaggle/kaggle.json")
            print("   3. Accepted the competition rules on Kaggle website")

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nAlternative download methods:")
        print("   1. Manual download from: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data")
        print("   2. Place CSV files in:", data_raw)


def validate_data():
    """Validate that all required files are present."""
    project_root = Path(__file__).parent.parent
    data_raw = project_root / "data" / "raw"

    required_files = ["train.csv", "test.csv", "features.csv", "stores.csv"]
    missing_files = []

    print("\nValidating downloaded files...")
    for file_name in required_files:
        file_path = data_raw / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {file_name} ({size_mb:.1f} MB)")
        else:
            missing_files.append(file_name)
            print(f"  {file_name} - MISSING")

    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("Expected Walmart dataset structure:")
        print("  - train.csv: Historical sales data")
        print("  - test.csv: Test period for predictions")
        print("  - features.csv: Store features (temperature, fuel price, etc.)")
        print("  - stores.csv: Store metadata (type, size)")
        return False
    else:
        print("\nAll required files present!")
        return True


if __name__ == "__main__":
    print("Walmart Sales Dataset Downloader")
    print("=" * 50)
    print("Source: yasserh/walmart-dataset (Kaggle)")
    print("URL: https://www.kaggle.com/datasets/yasserh/walmart-dataset")
    print()

    # Setup Kaggle API
    if setup_kaggle_api():
        # Download data
        download_walmart_data()

        # Validate download
        validate_data()

        print("\nData download complete!")
        print("Next steps:")
        print("   1. Run EDA notebook: notebooks/01_eda.ipynb")
        print("   2. Preprocess data: scripts/build_matrix.py")
        print("   3. Train models: scripts/train_torch.py")
