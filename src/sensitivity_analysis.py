import os
import sys
from pathlib import Path

import runtime_support
import numpy as np
import pandas as pd

# Bind project environment path rules cleanly
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import config


def _demand_pressure(df):
    if "predicted_utilization_rate" in df.columns:
        return pd.to_numeric(df["predicted_utilization_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    station_peak = df.groupby(["context_tag", "station_id"])["predicted_volume_kwh"].transform(
        lambda values: values.quantile(0.90)
    )
    global_peak = df["predicted_volume_kwh"].quantile(0.90)
    if pd.isna(global_peak) or global_peak <= 0:
        global_peak = 1.0

    denominator = station_peak.replace(0, np.nan).fillna(global_peak)
    return (df["predicted_volume_kwh"] / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def _sigmoid_congestion(pressure):
    """
    Calculates operational congestion probability.
    UPDATED: Set steepness denominator to 0.05 to mirror the 0.4/0.6 policy model curves
    and keep sensitivity scoring mathematically locked to runtime execution.
    """
    return 1.0 / (1.0 + np.exp(-((pressure - config.SURGE_THRESHOLD_UPPER) / 0.05)))


def _consumer_acceptability_label(surge_cap):
    if surge_cap <= 1.25:
        return "Excellent"
    if surge_cap <= 1.30:
        return "High"
    if surge_cap <= 1.34:
        return "Optimal/Safe"
    if surge_cap <= 1.40:
        return "Marginal"
    return "Unacceptable/High-Risk"


def generate_policy_sensitivity_matrix():
    """
    Multi-objective policy sensitivity analysis.

    Evaluates combinations of:
    - Surge pricing multipliers
    - Off-peak discount multipliers

    Outputs:
    Revenue Growth
    Congestion Mitigation
    Consumer Acceptability
    """

    if not config.FORECAST_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Forecast cache missing at {config.FORECAST_CACHE_PATH}. Run demand_agent.py first."
        )

    forecast_df = pd.read_csv(
        config.FORECAST_CACHE_PATH,
        dtype={"station_id": str}
    )

    required_cols = [
        "timestamp",
        "station_id",
        "context_tag",
        "predicted_volume_kwh"
    ]

    missing_cols = [
        column
        for column in required_cols
        if column not in forecast_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Forecast cache missing required columns: {missing_cols}"
        )

    forecast_df["predicted_volume_kwh"] = pd.to_numeric(
        forecast_df["predicted_volume_kwh"],
        errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    pressure = _demand_pressure(forecast_df)

    base_tariff = config.BASE_TARIFF_INR_PER_KWH
    elasticity = config.DEMAND_ELASTICITY

    baseline_revenue = (
        forecast_df["predicted_volume_kwh"] * base_tariff
    )

    baseline_total = float(baseline_revenue.sum())

    # ----------------------------------------------------------
    # Search Space Matrix Mapping
    # Expanded factors to include operational bounds (1.3333x surge, 0.80x discount)
    # ----------------------------------------------------------
    surge_caps = [1.20, 1.25, 1.30, 1.3333, 1.35, 1.40]
    discount_caps = [0.70, 0.75, 0.80, 0.85, 0.90]

    rows = []

    for surge_cap in surge_caps:
        for discount_cap in discount_caps:

            tariffs = np.select(
                [
                    pressure >= config.SURGE_THRESHOLD_UPPER,
                    pressure <= config.DISCOUNT_THRESHOLD_LOWER,
                ],
                [
                    base_tariff * surge_cap,
                    base_tariff * discount_cap,
                ],
                default=base_tariff,
            )

            price_change_pct = (
                tariffs - base_tariff
            ) / base_tariff

            response_factor = (
                1.0 + elasticity * price_change_pct
            ).clip(0.75, 1.25)

            expected_volume = (
                forecast_df["predicted_volume_kwh"]
                * response_factor
            )

            projected_revenue = (
                expected_volume * tariffs
            )

            pressure_after = (
                pressure * response_factor
            ).clip(0.0, 1.0)

            congestion_before = _sigmoid_congestion(
                pressure
            )

            congestion_after = _sigmoid_congestion(
                pressure_after
            )

            congestion_mitigation = (
                (
                    congestion_before.mean()
                    - congestion_after.mean()
                )
                / congestion_before.mean()
                * 100
                if congestion_before.mean() > 0
                else 0.0
            )

            projected_total = float(
                projected_revenue.sum()
            )

            revenue_growth = (
                (
                    projected_total
                    - baseline_total
                )
                / baseline_total
                * 100
                if baseline_total > 0
                else 0.0
            )

            rows.append(
                {
                    "Surge Cap Factor": surge_cap,
                    "Discount Factor": discount_cap,
                    "Peak Tariff (INR/kWh)": round(
                        base_tariff * surge_cap,
                        2
                    ),
                    "OffPeak Tariff (INR/kWh)": round(
                        base_tariff * discount_cap,
                        2
                    ),
                    "Projected Revenue Growth (%)": round(
                        revenue_growth,
                        2
                    ),
                    "Congestion Mitigation Rate (%)": round(
                        congestion_mitigation,
                        2
                    ),
                    "Consumer Acceptability Score":
                        _consumer_acceptability_label(
                            surge_cap
                        ),
                }
            )

    df = pd.DataFrame(rows)

    # Sort by best congestion reduction first to match your 0.6 priority split,
    # then resolve ties using revenue growth metrics.
    df = df.sort_values(
        [
            "Congestion Mitigation Rate (%)",
            "Projected Revenue Growth (%)"
        ],
        ascending=[False, False]
    )

    config.OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        config.OUTPUTS_DIR
        / "policy_sensitivity_matrix.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Sensitivity analysis matrix exported to {output_path}"
    )

    print("\nTop Policy Candidates (Ordered by Congestion Mitigation Priority):")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        generate_policy_sensitivity_matrix()
        print("Sensitivity matrix analysis completed successfully.")
    except Exception as exc:
        print(f"[ERROR] Sensitivity analysis failed: {exc}")
        sys.exit(1)