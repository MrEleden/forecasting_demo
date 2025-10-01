#!/usr/bin/env python3
"""
Generate realistic inventory demand forecasting data.

This script creates synthetic but realistic inventory demand data
for multiple product categories with realistic patterns including:
- Seasonal trends (holiday, back-to-school, etc.)
- Product category effects
- Price elasticity
- Promotional impacts
- Cross-category influences

Usage:
    python generate_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_inventory_data():
    """Generate realistic inventory demand data."""

    # Set up paths
    project_root = Path(__file__).parent.parent
    data_raw = project_root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    print("Creating sample inventory forecasting data...")

    # Generate weekly data for 3 years
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 29)
    weeks = pd.date_range(start=start_date, end=end_date, freq="W")

    # Define product categories and SKUs
    categories = {
        "Electronics": ["SKU_001", "SKU_002", "SKU_003", "SKU_004"],
        "Clothing": ["SKU_005", "SKU_006", "SKU_007", "SKU_008"],
        "Home & Garden": ["SKU_009", "SKU_010", "SKU_011", "SKU_012"],
        "Sports & Outdoors": ["SKU_013", "SKU_014", "SKU_015", "SKU_016"],
        "Books & Media": ["SKU_017", "SKU_018", "SKU_019", "SKU_020"],
    }

    # Price ranges by category
    price_ranges = {
        "Electronics": (200, 800),
        "Clothing": (50, 200),
        "Home & Garden": (100, 400),
        "Sports & Outdoors": (75, 300),
        "Books & Media": (10, 50),
    }

    # Base demand by category
    base_demand = {"Electronics": 15, "Clothing": 25, "Home & Garden": 20, "Sports & Outdoors": 18, "Books & Media": 30}

    data_rows = []

    # Set random seed for reproducibility
    np.random.seed(42)

    for category, skus in categories.items():
        for sku in skus:
            # Set consistent price for each SKU
            min_price, max_price = price_ranges[category]
            sku_price = np.random.uniform(min_price, max_price)

            for week in weeks:
                week_of_year = week.isocalendar()[1]
                month = week.month

                # Seasonal patterns
                seasonal_factor = 1.0

                # Electronics: Q4 holiday boost
                if category == "Electronics" and month in [11, 12]:
                    seasonal_factor = 1.5

                # Clothing: Spring/Fall fashion seasons
                elif category == "Clothing" and month in [3, 4, 9, 10]:
                    seasonal_factor = 1.3

                # Home & Garden: Spring/Summer peak
                elif category == "Home & Garden" and month in [4, 5, 6, 7]:
                    seasonal_factor = 1.4

                # Sports: Summer activities
                elif category == "Sports & Outdoors" and month in [5, 6, 7, 8]:
                    seasonal_factor = 1.3

                # Books: Back-to-school
                elif category == "Books & Media" and month in [8, 9]:
                    seasonal_factor = 1.2

                # Price elasticity (higher price = lower demand)
                price_factor = max(0.5, 1.5 - (sku_price / max_price))

                # Promotion effect (random promotions)
                has_promotion = np.random.choice([0, 1], p=[0.8, 0.2])
                promotion_factor = 1.3 if has_promotion else 1.0

                # Calculate demand
                base_sku_demand = base_demand[category]
                demand = int(
                    base_sku_demand * seasonal_factor * price_factor * promotion_factor + np.random.normal(0, 3)
                )
                demand = max(0, demand)  # No negative demand

                data_rows.append(
                    {
                        "date": week,
                        "sku_id": sku,
                        "category": category,
                        "demand": demand,
                        "price": round(sku_price, 2),
                        "promotion": has_promotion,
                        "week_of_year": week_of_year,
                    }
                )

    # Create DataFrame
    inventory_data = pd.DataFrame(data_rows)

    # Save data
    output_file = data_raw / "inventory_demand.csv"
    inventory_data.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Created: {output_file.name} ({size_mb:.1f} MB)")
    print(f"{len(inventory_data)} records, {len(categories)} categories")
    print(f"Date range: {inventory_data['date'].min()} to {inventory_data['date'].max()}")

    # Show category breakdown
    print(f"\nCategory breakdown:")
    for category in categories.keys():
        cat_data = inventory_data[inventory_data["category"] == category]
        avg_demand = cat_data["demand"].mean()
        print(f"  {category}: {len(cat_data)} records, avg demand: {avg_demand:.1f}")

    return output_file


if __name__ == "__main__":
    print("Inventory Demand Data Generator")
    print("=" * 40)
    generate_inventory_data()
    print("\nData generation complete!")
    print("Next steps:")
    print("   1. Explore data: notebooks/01_eda.ipynb")
    print("   2. Analyze seasonal patterns")
    print("   3. Study price elasticity")
    print("   4. Train hierarchical forecasting models")
