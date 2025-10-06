#!/usr/bin/env python3
"""
Master data downloader for ML Portfolio Forecasting Projects.

This script coordinates data acquisition for all forecasting projects:
1. Walmart Retail Sales (yasserh/walmart-dataset - Kaggle)
2. Ola Bike Ride-Sharing (Generated)
3. Inventory Forecasting (Generated)
4. U.S. Transportation Services Index (BTS)

Usage:
    python download_all_data.py
    python download_all_data.py --dataset walmart
    python download_all_data.py --dataset all
"""

import argparse
import subprocess
import sys
from pathlib import Path


def download_dataset(dataset_name: str) -> None:
    """Download or generate data for a specific dataset by calling project scripts."""

    if dataset_name == "walmart":
        print("Downloading Walmart dataset...")
        print("Source: yasserh/walmart-dataset (Kaggle)")
        print("URL: https://www.kaggle.com/datasets/yasserh/walmart-dataset")
        script_path = Path("projects/retail_sales_walmart/scripts/download_data.py")
        if script_path.exists():
            result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                print("Walmart data ready!")
            else:
                print(f"Walmart download failed: {result.stderr}")
        else:
            print(f"Script not found: {script_path}")

    elif dataset_name == "ola":
        print("Generating Ola ride-sharing dataset...")
        script_path = Path("projects/rideshare_demand_ola/scripts/generate_data.py")
        if script_path.exists():
            result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                print("Ola data ready!")
            else:
                print(f"Ola generation failed: {result.stderr}")
        else:
            print(f"Script not found: {script_path}")

    elif dataset_name == "inventory":
        print("Generating inventory dataset...")
        script_path = Path("projects/inventory_forecasting/scripts/generate_data.py")
        if script_path.exists():
            result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                print("Inventory data ready!")
            else:
                print(f"Inventory generation failed: {result.stderr}")
        else:
            print(f"Script not found: {script_path}")

    elif dataset_name == "tsi":
        print("Downloading TSI dataset...")
        script_path = Path("projects/transportation_tsi/scripts/download_data.py")
        if script_path.exists():
            result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                print("TSI data ready!")
            else:
                print(f"TSI download failed: {result.stderr}")
        else:
            print(f"Script not found: {script_path}")

    else:
        print(f"Unknown dataset: {dataset_name}")


def main():
    """Main entry point for data download orchestration."""
    parser = argparse.ArgumentParser(description="Download forecasting datasets")
    parser.add_argument(
        "--dataset", choices=["walmart", "ola", "inventory", "tsi", "all"], default="all", help="Dataset to download"
    )

    args = parser.parse_args()

    print("ML Portfolio Forecasting - Data Downloader")
    print("=" * 50)

    if args.dataset == "all":
        datasets = ["walmart", "ola", "inventory", "tsi"]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        print(f"\n{'='*20} {dataset.upper()} {'='*20}")
        download_dataset(dataset)

    print("\nData download process complete!")
    print("\nNext steps:")
    print("   1. Explore data in notebooks/")
    print("   2. Check data quality and preprocessing")
    print("   3. Run baseline models")
    print("   4. Start with EDA: notebooks/01_eda.ipynb")


if __name__ == "__main__":
    main()
