#!/usr/bin/env python3
"""
Generate realistic Ola ride-sharing demand data.

This script creates synthetic but realistic ride-sharing demand data
for major Indian cities with realistic patterns including:
- Daily peaks (morning/evening rush hours)
- Weekly patterns (weekday vs weekend)
- Seasonal effects (monsoon, festivals)
- Weather impact (rain, temperature)
- City-specific demand patterns

Usage:
    python generate_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_ola_data():
    """Generate realistic Ola ride-sharing demand data."""

    # Set up paths
    project_root = Path(__file__).parent.parent
    data_raw = project_root / "data" / "raw"
    data_raw.mkdir(parents=True, exist_ok=True)

    print("Creating sample Ola bike ride-sharing data...")

    # Generate hourly data for 2 years
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    hours = pd.date_range(start=start_date, end=end_date, freq="h")

    # Define pickup zones (major Indian cities)
    pickup_zones = [
        "Mumbai_Central",
        "Mumbai_Andheri",
        "Mumbai_Bandra",
        "Delhi_CP",
        "Delhi_Gurgaon",
        "Delhi_Noida",
        "Bangalore_Koramangala",
        "Bangalore_Whitefield",
        "Bangalore_MG_Road",
        "Chennai_T_Nagar",
        "Pune_Koregaon_Park",
        "Hyderabad_Hitech_City",
    ]

    data_rows = []

    # Set random seed for reproducibility
    np.random.seed(42)

    for zone in pickup_zones:
        for hour in hours[::4]:  # Sample every 4 hours to reduce size
            # Base demand with patterns
            hour_of_day = hour.hour
            day_of_week = hour.weekday()

            # Peak hours: 8-10 AM, 6-8 PM
            peak_morning = 1.5 if 8 <= hour_of_day <= 10 else 1.0
            peak_evening = 1.8 if 18 <= hour_of_day <= 20 else 1.0
            peak_factor = max(peak_morning, peak_evening)

            # Weekend patterns
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0

            # Base demand by zone type
            if "Mumbai" in zone:
                base_demand = 25
            elif "Delhi" in zone:
                base_demand = 22
            elif "Bangalore" in zone:
                base_demand = 20
            else:
                base_demand = 15

            # Weather effects (simplified)
            weather_factor = np.random.uniform(0.8, 1.2)

            # Calculate ride requests
            ride_requests = int(base_demand * peak_factor * weekend_factor * weather_factor + np.random.poisson(3))

            # Weather data (simplified)
            temp = np.random.normal(28, 8)  # Indian climate
            humidity = np.random.uniform(60, 90)
            is_raining = np.random.choice([0, 1], p=[0.85, 0.15])

            data_rows.append(
                {
                    "datetime": hour,
                    "pickup_zone": zone,
                    "ride_requests": max(0, ride_requests),
                    "temperature": round(temp, 1),
                    "humidity": round(humidity, 1),
                    "is_raining": is_raining,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "is_weekend": day_of_week >= 5,
                    "month": hour.month,
                }
            )

    # Create DataFrame and save
    ola_data = pd.DataFrame(data_rows)

    # Add some special events (festivals, holidays)
    # Diwali effect (increased rides)
    diwali_dates = ["2022-10-24", "2023-11-12"]
    for diwali in diwali_dates:
        mask = ola_data["datetime"].dt.date == pd.to_datetime(diwali).date()
        ola_data.loc[mask, "ride_requests"] = (ola_data.loc[mask, "ride_requests"] * 1.4).astype(int)

    # Monsoon effect (decreased rides)
    monsoon_months = [6, 7, 8, 9]  # June to September
    monsoon_mask = ola_data["datetime"].dt.month.isin(monsoon_months)
    rain_mask = ola_data["is_raining"] == 1
    ola_data.loc[monsoon_mask & rain_mask, "ride_requests"] = (
        ola_data.loc[monsoon_mask & rain_mask, "ride_requests"] * 0.6
    ).astype(int)

    # Save data
    output_file = data_raw / "ola_ride_requests.csv"
    ola_data.to_csv(output_file, index=False)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Created: {output_file.name} ({size_mb:.1f} MB)")
    print(f"{len(ola_data)} records, {len(pickup_zones)} zones")
    print(f"Date range: {ola_data['datetime'].min()} to {ola_data['datetime'].max()}")

    return output_file


if __name__ == "__main__":
    print("Ola Ride-sharing Data Generator")
    print("=" * 40)
    generate_ola_data()
    print("\nData generation complete!")
    print("Next steps:")
    print("   1. Explore data: notebooks/01_eda.ipynb")
    print("   2. Analyze peak hour patterns")
    print("   3. Study weather impact on demand")
    print("   4. Train time series models")
