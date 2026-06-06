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
import utils


# Establish parameters mapping clean constants from config.py
BASE_TARIFF = config.BASE_TARIFF_INR_PER_KWH
SURGE_THRESHOLD = config.SURGE_THRESHOLD_UPPER
DISCOUNT_THRESHOLD = config.DISCOUNT_THRESHOLD_LOWER


def _float_from_env(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


SURGE_MULTIPLIER = _float_from_env(
    "SURGE_MULTIPLIER_OVERRIDE", config.DEFAULT_SURGE_MULTIPLIER
)
DISCOUNT_MULTIPLIER = _float_from_env(
    "DISCOUNT_MULTIPLIER_OVERRIDE", config.DEFAULT_DISCOUNT_MULTIPLIER
)
DEMAND_ELASTICITY = _float_from_env("DEMAND_ELASTICITY_OVERRIDE", config.DEMAND_ELASTICITY)


def add_demand_pressure(df):
    """
    Builds a 0-1 pressure score from forecasted demand signals only.
    It prefers the Demand Agent's predicted utilization rate and falls back to
    station-wise predicted demand relative to each station's high-demand level.
    """
    df = df.copy()

    if "predicted_utilization_rate" in df.columns:
        df["demand_pressure"] = pd.to_numeric(
            df["predicted_utilization_rate"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        df["demand_pressure_source"] = "predicted_utilization_rate"
        return df

    station_peak = df.groupby(["context_tag", "station_id"])["predicted_volume_kwh"].transform(
        lambda values: values.quantile(0.90)
    )

    global_peak = df["predicted_volume_kwh"].quantile(0.90)
    if pd.isna(global_peak) or global_peak <= 0:
        global_peak = 1.0

    denominator = station_peak.replace(0, np.nan).fillna(global_peak)
    df["demand_pressure"] = (df["predicted_volume_kwh"] / denominator).clip(0.0, 1.0)
    df["predicted_utilization_rate"] = df["demand_pressure"]
    df["demand_pressure_source"] = "station_forecast_percentile"

    return df


def calculate_dynamic_tariff(row):
    pressure = row["demand_pressure"]

    if pressure >= SURGE_THRESHOLD:
        return BASE_TARIFF * SURGE_MULTIPLIER

    if pressure <= DISCOUNT_THRESHOLD:
        return BASE_TARIFF * DISCOUNT_MULTIPLIER

    return BASE_TARIFF


def classify_pricing_action(row):
    pressure = row["demand_pressure"]

    if pressure >= SURGE_THRESHOLD:
        return "surge"

    if pressure <= DISCOUNT_THRESHOLD:
        return "discount"

    return "baseline"


def _sigmoid_congestion(pressure):
    """
    Calculates operational congestion probability.
    Using a denominator of 0.05 (sharper slope) aligns with the 0.4/0.6 policy weight
    by mapping high-utilization states into definitive congestion signals quickly.
    """
    return 1.0 / (1.0 + np.exp(-((pressure - SURGE_THRESHOLD) / 0.05)))


def classify_pricing_confidence(row):
    pressure = row["demand_pressure"]
    if row["pricing_action"] == "surge":
        margin = pressure - SURGE_THRESHOLD
    elif row["pricing_action"] == "discount":
        margin = DISCOUNT_THRESHOLD - pressure
    else:
        margin = min(SURGE_THRESHOLD - pressure, pressure - DISCOUNT_THRESHOLD)

    if margin >= 0.20:
        return "high"
    if margin >= 0.08:
        return "medium"
    return "watch"


def explain_tariff_decision(row):
    action = row["pricing_action"]
    pressure = row["demand_pressure"]
    if action == "surge":
        return (
            f"Predicted utilization {pressure:.2f} crosses {SURGE_THRESHOLD:.2f}; "
            "surge protects capacity and peak revenue."
        )
    if action == "discount":
        return (
            f"Predicted utilization {pressure:.2f} is below {DISCOUNT_THRESHOLD:.2f}; "
            "discount encourages off-peak charging."
        )
    return (
        f"Predicted utilization {pressure:.2f} stays within the balanced band; "
        "baseline tariff avoids unnecessary intervention."
    )


def execute_pricing_optimization():
    utils.print_agent_header("Dynamic Tariff Optimization Matrix Sequence")

    print(f"Loading cached demand predictions from: {config.FORECAST_CACHE_PATH}")
    if not config.FORECAST_CACHE_PATH.exists():
        raise FileNotFoundError("Missing demand forecast cache. Run demand_agent.py first.")

    df = pd.read_csv(config.FORECAST_CACHE_PATH, dtype={"station_id": str})
    required_cols = ["timestamp", "station_id", "context_tag", "predicted_volume_kwh"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Forecast cache missing required columns: {missing_cols}")

    df["predicted_volume_kwh"] = pd.to_numeric(
        df["predicted_volume_kwh"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    df = add_demand_pressure(df)
    if "congestion_probability" not in df.columns:
        df["congestion_probability"] = _sigmoid_congestion(df["demand_pressure"])
    else:
        df["congestion_probability"] = pd.to_numeric(
            df["congestion_probability"], errors="coerce"
        ).fillna(_sigmoid_congestion(df["demand_pressure"]))

    print("Calculating dynamic tariff tiers...")
    df["pricing_action"] = df.apply(classify_pricing_action, axis=1)
    df["dynamic_tariff_inr"] = df.apply(calculate_dynamic_tariff, axis=1).round(2)

    price_change_pct = (df["dynamic_tariff_inr"] - BASE_TARIFF) / BASE_TARIFF
    df["demand_response_factor"] = (1.0 + DEMAND_ELASTICITY * price_change_pct).clip(0.75, 1.25)

    df["expected_volume_after_pricing_kwh"] = (
        df["predicted_volume_kwh"] * df["demand_response_factor"]
    )
    df["expected_utilization_after_pricing"] = (
        df["demand_pressure"] * df["demand_response_factor"]
    ).clip(0.0, 1.0)
    df["congestion_probability_after_pricing"] = _sigmoid_congestion(
        df["expected_utilization_after_pricing"]
    )

    df["baseline_revenue_inr"] = df["predicted_volume_kwh"] * BASE_TARIFF
    df["projected_revenue_inr"] = (
        df["expected_volume_after_pricing_kwh"] * df["dynamic_tariff_inr"]
    )
    df["pricing_confidence"] = df.apply(classify_pricing_confidence, axis=1)
    df["tariff_reason"] = df.apply(explain_tariff_decision, axis=1)

    pricing_schedule_cols = [
        "timestamp",
        "station_id",
        "context_tag",
        "predicted_volume_kwh",
        "predicted_utilization_rate",
        "congestion_probability",
        "demand_pressure",
        "demand_pressure_source",
        "pricing_action",
        "pricing_confidence",
        "dynamic_tariff_inr",
        "expected_volume_after_pricing_kwh",
        "expected_utilization_after_pricing",
        "congestion_probability_after_pricing",
        "baseline_revenue_inr",
        "projected_revenue_inr",
        "tariff_reason",
    ]

    pricing_schedule_df = df[pricing_schedule_cols].copy()
    utils.save_dataframe(config.DYNAMIC_TARIFF_PATH, pricing_schedule_df)

    action_summary = (
        df.groupby("pricing_action", as_index=False)
        .agg(
            intervals=("pricing_action", "size"),
            mean_pressure=("demand_pressure", "mean"),
            mean_tariff_inr=("dynamic_tariff_inr", "mean"),
            predicted_demand_kwh=("predicted_volume_kwh", "sum"),
            expected_demand_kwh=("expected_volume_after_pricing_kwh", "sum"),
            baseline_revenue_inr=("baseline_revenue_inr", "sum"),
            projected_revenue_inr=("projected_revenue_inr", "sum"),
        )
        .sort_values("pricing_action")
    )
    action_summary["revenue_lift_inr"] = (
        action_summary["projected_revenue_inr"] - action_summary["baseline_revenue_inr"]
    )
    utils.save_dataframe(config.OUTPUTS_DIR / "pricing_action_summary.csv", action_summary)

    total_predicted_demand = float(df["predicted_volume_kwh"].sum())
    total_expected_demand = float(df["expected_volume_after_pricing_kwh"].sum())
    total_projected_revenue = float(df["projected_revenue_inr"].sum())
    total_baseline_revenue = float(df["baseline_revenue_inr"].sum())

    net_revenue_lift_inr = total_projected_revenue - total_baseline_revenue
    percentage_gain = (
        net_revenue_lift_inr / total_baseline_revenue * 100
        if total_baseline_revenue > 0
        else 0.0
    )
    discount_mask = df["pricing_action"] == "discount"
    surge_mask = df["pricing_action"] == "surge"
    discount_uplift_kwh = float(
        (
            df.loc[discount_mask, "expected_volume_after_pricing_kwh"]
            - df.loc[discount_mask, "predicted_volume_kwh"]
        ).sum()
    )
    discount_baseline_kwh = float(df.loc[discount_mask, "predicted_volume_kwh"].sum())
    off_peak_uplift_percentage = (
        discount_uplift_kwh / discount_baseline_kwh * 100
        if discount_baseline_kwh > 0
        else 0.0
    )
    peak_demand_reduction_kwh = float(
        (
            df.loc[surge_mask, "predicted_volume_kwh"]
            - df.loc[surge_mask, "expected_volume_after_pricing_kwh"]
        ).sum()
    )
    surge_baseline_kwh = float(df.loc[surge_mask, "predicted_volume_kwh"].sum())
    average_wait_reduction_proxy_pct = (
        peak_demand_reduction_kwh / surge_baseline_kwh * 100
        if surge_baseline_kwh > 0
        else 0.0
    )
    customer_response_rate_proxy_pct = (
        float(
            (
                df["expected_volume_after_pricing_kwh"] - df["predicted_volume_kwh"]
            ).abs().sum()
        )
        / total_predicted_demand
        * 100
        if total_predicted_demand > 0
        else 0.0
    )
    pricing_efficiency_inr_per_kwh = (
        total_projected_revenue / total_expected_demand
        if total_expected_demand > 0
        else 0.0
    )
    congestion_risk_share_pct = float(
        (
            df["congestion_probability"]
            >= config.CONGESTION_PROBABILITY_THRESHOLD
        ).mean()
        * 100
    )
    congestion_risk_after_pricing_pct = float(
        (
            df["congestion_probability_after_pricing"]
            >= config.CONGESTION_PROBABILITY_THRESHOLD
        ).mean()
        * 100
    )

    business_kpis = {
        "total_predicted_demand_kwh": total_predicted_demand,
        "expected_demand_after_pricing_kwh": total_expected_demand,
        "baseline_revenue_flat_inr": total_baseline_revenue,
        "optimized_revenue_dynamic_inr": total_projected_revenue,
        "net_revenue_lift_inr": net_revenue_lift_inr,
        "revenue_growth_percentage": percentage_gain,
        "average_dynamic_tariff_per_kwh": float(df["dynamic_tariff_inr"].mean()),
        "surge_share_percentage": float((df["pricing_action"] == "surge").mean() * 100),
        "discount_share_percentage": float((df["pricing_action"] == "discount").mean() * 100),
        "baseline_share_percentage": float((df["pricing_action"] == "baseline").mean() * 100),
        "off_peak_uplift_percentage": off_peak_uplift_percentage,
        "average_wait_reduction_proxy_pct": average_wait_reduction_proxy_pct,
        "customer_response_rate_proxy_pct": customer_response_rate_proxy_pct,
        "pricing_efficiency_inr_per_kwh": pricing_efficiency_inr_per_kwh,
        "congestion_risk_share_pct": congestion_risk_share_pct,
        "congestion_risk_after_pricing_pct": congestion_risk_after_pricing_pct,
        "surge_multiplier": SURGE_MULTIPLIER,
        "discount_multiplier": DISCOUNT_MULTIPLIER,
        "demand_elasticity": DEMAND_ELASTICITY,
    }

    utils.save_model_metrics(config.BUSINESS_OUTCOMES_PATH, business_kpis)

    utils.print_kv_table(
        "Financial Performance Optimization Summary",
        [
            ("Total predicted demand", f"{total_predicted_demand:,.2f} kWh"),
            ("Expected demand after pricing", f"{total_expected_demand:,.2f} kWh"),
            ("Baseline flat revenue", f"INR {total_baseline_revenue:,.2f}"),
            ("Optimized dynamic revenue", f"INR {total_projected_revenue:,.2f}"),
            ("Revenue lift", f"INR {net_revenue_lift_inr:,.2f}"),
            ("Net revenue growth", f"{percentage_gain:.2f}%"),
            ("Off-peak uplift proxy", f"{off_peak_uplift_percentage:.2f}%"),
            ("Peak wait reduction proxy", f"{average_wait_reduction_proxy_pct:.2f}%"),
            ("Pricing efficiency", f"INR {pricing_efficiency_inr_per_kwh:.2f}/kWh"),
        ],
    )
    utils.print_reasoning_block(
        "Tariff Pricing Agent Reasoning Trace",
        objective="Convert predicted utilization into dynamic tariffs that balance revenue, congestion, and off-peak utilization.",
        evidence=[
            f"Surge rule: utilization >= {SURGE_THRESHOLD:.2f} at {SURGE_MULTIPLIER:.2f}x baseline",
            f"Discount rule: utilization <= {DISCOUNT_THRESHOLD:.2f} at {DISCOUNT_MULTIPLIER:.2f}x baseline",
            f"Action mix: surge={business_kpis['surge_share_percentage']:.2f}%, discount={business_kpis['discount_share_percentage']:.2f}%, baseline={business_kpis['baseline_share_percentage']:.2f}%",
            f"Congestion risk changes from {congestion_risk_share_pct:.2f}% to {congestion_risk_after_pricing_pct:.2f}% after pricing response",
        ],
        decision=(
            "Publish interval-level tariff schedule with explicit action, confidence, "
            "and tariff reason for each station-hour."
        ),
        assumptions=[
            f"Demand response follows a constant price elasticity of demand framework (PED = {DEMAND_ELASTICITY}).",
            "Congestion mitigations assume that price-penalized users respond deterministically by shifting or shedding load.",
        ],
        next_actions=[
            "Monitoring agent will audit whether revenue lift and congestion proxies are improving.",
        ],
    )


if __name__ == "__main__":
    try:
        execute_pricing_optimization()
        print("Pricing optimization run finished successfully.")
    except Exception as exc:
        print(f"[ERROR] Pricing optimization failed: {exc}")
        sys.exit(1)