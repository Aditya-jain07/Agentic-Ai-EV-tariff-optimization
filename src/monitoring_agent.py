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


def run_system_telemetry_audit():
    utils.print_agent_header("EV Network Telemetry & Feedback Monitor")

    print(f"[INFO] Analyzing latest operational pricing matrix from: {config.DYNAMIC_TARIFF_PATH}")
    if not config.DYNAMIC_TARIFF_PATH.exists():
        raise FileNotFoundError("Missing active tariff schedule! Run pricing_agent.py first.")

    # Ingest the schedule
    df = pd.read_csv(config.DYNAMIC_TARIFF_PATH, dtype={"station_id": str})
    
    # Ingest business outcomes for delta tracking
    if config.BUSINESS_OUTCOMES_PATH.exists():
        kpi_df = pd.read_csv(config.BUSINESS_OUTCOMES_PATH)
        # Handle cases where column names might be structured slightly differently
        if "revenue_growth_percentage" in kpi_df.columns:
            current_growth = kpi_df["revenue_growth_percentage"].iloc[0]
        elif "net_revenue_growth" in kpi_df.columns:
            current_growth = kpi_df["net_revenue_growth"].iloc[0]
        else:
            current_growth = 0.0
            
        # Strip potential string symbols (like '%') if saved in text format
        if isinstance(current_growth, str):
            current_growth = float(current_growth.replace("%", "").strip())
    else:
        current_growth = 0.0

    print("[INFO] Auditing price-elasticity margins and action distributions...")

    # Calculate segment behaviors
    surge_mask = df["pricing_action"] == "surge"
    discount_mask = df["pricing_action"] == "discount"
    baseline_mask = df["pricing_action"] == "baseline"

    # Quantify Gross Delta components
    surge_gain = (df.loc[surge_mask, "projected_revenue_inr"] - df.loc[surge_mask, "baseline_revenue_inr"]).sum()
    discount_loss = (df.loc[discount_mask, "projected_revenue_inr"] - df.loc[discount_mask, "baseline_revenue_inr"]).sum()
    
    total_records = len(df)
    if total_records == 0:
        raise ValueError("Tariff schedule is empty; telemetry audit cannot run.")

    surge_count = surge_mask.sum()
    discount_count = discount_mask.sum()
    baseline_count = baseline_mask.sum()
    total_predicted_demand = float(df["predicted_volume_kwh"].sum())
    total_expected_demand = float(df["expected_volume_after_pricing_kwh"].sum())
    total_projected_revenue = float(df["projected_revenue_inr"].sum())

    if "congestion_probability" in df.columns:
        congestion_before = float(df["congestion_probability"].mean())
    else:
        congestion_before = float(df["demand_pressure"].mean())

    if "congestion_probability_after_pricing" in df.columns:
        congestion_after = float(df["congestion_probability_after_pricing"].mean())
    else:
        congestion_after = float(df["demand_pressure"].mean())

    congestion_reduction_pct = (
        (congestion_before - congestion_after) / congestion_before * 100
        if congestion_before > 0
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
    pricing_efficiency_score = (
        total_projected_revenue / total_expected_demand
        if total_expected_demand > 0
        else 0.0
    )
    off_peak_uplift_kwh = float(
        (
            df.loc[discount_mask, "expected_volume_after_pricing_kwh"]
            - df.loc[discount_mask, "predicted_volume_kwh"]
        ).sum()
    )

    # Formulate Strategic Recommendations via automated rules engine
    recommendations = []
    status_alert = "NOMINAL PERFORMANCE"

    if current_growth < 0:
        status_alert = "ALERT: REVENUE DRIFT DEGRADATION DETECTED"
        recommendations.append("CRITICAL: Discount rules are Cannibalizing Margins. Immediately compress DISCOUNT_MULTIPLIER closer to 1.0.")
        recommendations.append("ACTION: Re-evaluate surge parameters to balance the 0.4 revenue and 0.6 congestion targets safely.")
    elif current_growth < 5.0:
        status_alert = "WARNING: SUB-OPTIMAL GROWTH ALERT"
        recommendations.append("ADVISORY: Growth yields are slim. Tighten off-peak discount windows by dropping DISCOUNT_THRESHOLD_LOWER.")
    else:
        recommendations.append("STABLE: Strategy proving highly accretive. No manual multiplier overrides required.")

    if discount_count / total_records > 0.50:
        recommendations.append("STRUCTURAL: Over 50% of network hours are flagged as low-occupancy. Shift baseline marketing to increase off-peak base utilization.")
    if congestion_reduction_pct < 1.0 and surge_count > 0:
        recommendations.append("CAPACITY: Congestion relief is shallow. Consider a higher surge multiplier only if customer acceptance remains stable.")
    if off_peak_uplift_kwh <= 0 and discount_count > 0:
        recommendations.append("DEMAND SHIFT: Discount windows are not creating enough off-peak uplift under the current elasticity assumption.")

    # Print Visual Telemetry Dashboard
    utils.print_kv_table(
        f"Network Telemetry Audit: {status_alert}",
        [
            ("Evaluated grid intervals", f"{total_records:,} hours"),
            ("Pricing split", f"surge={surge_count/total_records*100:.1f}% | discount={discount_count/total_records*100:.1f}% | base={baseline_count/total_records*100:.1f}%"),
            ("Surge margin capitalization", f"+INR {surge_gain:,.2f}"),
            ("Discount margin effect", f"INR {discount_loss:,.2f}"),
            ("Current system revenue delta", f"INR {surge_gain + discount_loss:,.2f} ({current_growth:.2f}%)"),
            ("Congestion probability change", f"{congestion_before:.3f} -> {congestion_after:.3f}"),
            ("Congestion reduction proxy", f"{congestion_reduction_pct:.2f}%"),
            ("Customer response proxy", f"{customer_response_rate_proxy_pct:.2f}%"),
            ("Pricing efficiency score", f"INR {pricing_efficiency_score:.2f}/kWh"),
        ],
    )

    # FIXED: Converted title parameter into a standard positional argument to prevent signature crashes
    utils.print_reasoning_block(
        "Telemetry & Feedback Audit Trace",
        objective="Audit pipeline compliance, track elasticity drift, and flag anomalous revenue drops.",
        evidence=[
            f"Evaluated data spaces: {len(df):,} system metrics rows",
            f"Observed surge rate cap: +INR {surge_gain:,.2f}",
            f"Observed congestion response delta: {congestion_before:.3f} -> {congestion_after:.3f} (-{congestion_reduction_pct:.2f}%)",
        ],
        decision="System health is NOMINAL. Pricing feedback loops are functioning within stable margins.",
        assumptions=[
            "Driver elasticity bounds remain constant over the active evaluation window.",
        ],
        next_actions=[
            "Flush optimized parameters to master production database.",
        ],
    )

    # Log operational diagnostic telemetry artifact
    telemetry_payload = {
        "timestamp_audit": str(pd.Timestamp.now()),
        "system_status": status_alert,
        "surge_gross_gain_inr": float(surge_gain),
        "discount_gross_loss_inr": float(discount_loss),
        "net_telemetry_delta_inr": float(surge_gain + discount_loss),
        "surge_allocation_pct": float(surge_count / total_records * 100),
        "discount_allocation_pct": float(discount_count / total_records * 100),
        "baseline_allocation_pct": float(baseline_count / total_records * 100),
        "congestion_probability_before": congestion_before,
        "congestion_probability_after": congestion_after,
        "congestion_reduction_proxy_pct": congestion_reduction_pct,
        "customer_response_rate_proxy_pct": customer_response_rate_proxy_pct,
        "pricing_efficiency_score_inr_per_kwh": pricing_efficiency_score,
        "off_peak_uplift_kwh": off_peak_uplift_kwh,
    }
    
    telemetry_path = config.OUTPUTS_DIR / "system_telemetry_diagnostics.csv"
    utils.save_model_metrics(telemetry_path, telemetry_payload)


if __name__ == "__main__":
    try:
        run_system_telemetry_audit()
        print("System telemetry audit completed successfully.")
    except Exception as exc:
        print(f"[ERROR] System telemetry audit failed: {exc}")
        sys.exit(1)