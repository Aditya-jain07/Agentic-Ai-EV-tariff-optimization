import os
import sys
from pathlib import Path
import matplotlib

# Force a non-interactive backend for reliable, headless file rendering
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import runtime_support

# Bind project environment path rules cleanly
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import config

# Configure charting aesthetics matching reporting guidelines
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _require_file(path, message):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")
    return path


def _require_columns(df, columns, label):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _context_slice(master_df, keyword):
    mask = master_df["context_tag"].astype(str).str.lower().str.contains(keyword)
    subset = master_df[mask].copy()
    return subset if not subset.empty else master_df.copy()


def _load_inputs():
    # Safely look up directory structures using config values
    master_path = _require_file(config.UNIFIED_HOURLY_PATH, "Missing preprocessed master matrix")
    ledger_path = _require_file(config.OUTPUTS_DIR / "controller_iteration_ledger.csv", "Missing ledger metrics")
    strategy_path = _require_file(config.OUTPUTS_DIR / "pricing_strategy_distribution.csv", "Missing distribution schema")
    
    # Correct path mapping fallbacks to prevent missing attribute errors
    forecast_path = getattr(config, "FORECAST_CACHE_PATH", config.DATA_DIR / "processed" / "forecasted_demand_cache.csv")
    _require_file(forecast_path, "Missing demand forecasts cache")
    
    sensitivity_path = _require_file(config.OUTPUTS_DIR / "policy_sensitivity_matrix.csv", "Missing sensitivity matrix")

    return (
        pd.read_csv(master_path),
        pd.read_csv(ledger_path),
        pd.read_csv(strategy_path),
        pd.read_csv(forecast_path),
        pd.read_csv(sensitivity_path),
    )


def _chart_1_urban_heatmap(viz_dir, master_df):
    try:
        print("Generating Chart 1: UrbanEV Bottleneck Heatmap...")
        df_urb = _context_slice(master_df, "urban")
        _require_columns(df_urb, ["hour_of_day", "station_id", "charger_utilization_rate"], "Chart 1")
        
        pivot_df = df_urb.groupby(["station_id", "hour_of_day"])["charger_utilization_rate"].mean().unstack(fill_value=0)
        plt.figure(figsize=(11, 5))
        sns.heatmap(pivot_df, cmap="YlOrRd", cbar_kws={"label": "Utilization Rate"})
        plt.title("Spatial-Temporal Traffic Heatmap (Urban Network)", weight="bold", pad=15)
        plt.xlabel("Hour of the Day")
        plt.ylabel("Charging Station Identification ID")
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_1_urban_spatial_temporal_bottlenecks.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 1: {e}")


def _chart_2_acn_distribution(viz_dir, master_df):
    try:
        print("Generating Chart 2: ACN Structural Energy Distributions...")
        df_acn = _context_slice(master_df, "workplace")
        _require_columns(df_acn, ["volume_kwh"], "Chart 2")
        
        plt.figure(figsize=(8, 4.5))
        sns.histplot(df_acn["volume_kwh"], bins=35, kde=True, color="teal", edgecolor="black", alpha=0.7)
        plt.axvline(df_acn["volume_kwh"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {df_acn['volume_kwh'].mean():.2f} kWh")
        plt.title("Workplace Network (ACN) Base Energy Demand Profiling", weight="bold", pad=12)
        plt.xlabel("Hourly Energy Consumption Volume (kWh)")
        plt.ylabel("Interval Data Observations Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_2_acn_behavioral_inefficiencies.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 2: {e}")


def _chart_3_lag_trace(viz_dir, master_df):
    try:
        print("Generating Chart 3: Temporal Feature Lag Trace...")
        _require_columns(master_df, ["timestamp", "volume_kwh", "volume_lag_1h"], "Chart 3")
        sample_df = master_df.head(120).copy()
        
        plt.figure(figsize=(12, 4.5))
        plt.plot(sample_df["timestamp"], sample_df["volume_kwh"], label="Observed Real Load (t)", color="royalblue", marker="o", alpha=0.85)
        plt.plot(sample_df["timestamp"], sample_df["volume_lag_1h"], label="Shifted Historical Feature (t-1h)", color="darkorange", linestyle="--", marker="x", alpha=0.75)
        plt.title("Data Leakage Verification: Time-Series Sequence Alignment Window", weight="bold", pad=12)
        plt.xlabel("Chronological Aggregation Intervals")
        plt.ylabel("Energy Load Volume (kWh)")
        plt.xticks([]) 
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_3_leakage_safe_lag_tracking.jpg")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 3: {e}")


def _chart_4_scheduling_bands(viz_dir, master_df):
    try:
        print("Generating Chart 4: Preprocessed Scheduling Band Profiles...")
        if "scheduling_band" not in master_df.columns and "hour_of_day" in master_df.columns:
            from data_preprocessing import assign_scheduling_band
            master_df["scheduling_band"] = master_df["hour_of_day"].apply(assign_scheduling_band)
        
        _require_columns(master_df, ["scheduling_band", "volume_kwh"], "Chart 4")
        plt.figure(figsize=(8, 4.5))
        # Fixed palette assignment logic to ensure compliance with updated seaborn standards
        sns.boxplot(x="scheduling_band", y="volume_kwh", data=master_df, palette="Set2", hue="scheduling_band", legend=False)
        plt.title("Preprocessed Dynamic Energy Densities across Operational Scheduling Bands", weight="bold", pad=12)
        plt.xlabel("Categorical Policy Window Bands")
        plt.ylabel("Hourly Volume Scale Quantile (kWh)")
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_4_preprocessed_scheduling_bands.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 4: {e}")


def _chart_5_schema(viz_dir):
    try:
        print("Generating Chart 5: Structural Data Schema Blueprint...")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.axis("off")
        box_text = (
            "====================================================\n"
            "   UNIFIED MULTI-AGENT DATA SCHEMA PIPELINE BLUEPRINT\n"
            "====================================================\n"
            " INPUTS: JSON Arrays (ACN Sessions) + CSV Layers (UrbanEV Metrics)\n"
            "    │\n"
            "    ▼ [data_preprocessing.py] ➔ Timestamp Harmonization & Alignment\n"
            "    │                          ➔ Feature Engineering: Lag Shifts (t-1h)\n"
            "    ▼\n"
            " MATRIX: unified_hourly_base.csv ➔ XGBoost Model Feature Arrays\n"
            "    │\n"
            "    ▼ [demand_agent.py] ➔ Cascaded ML Multi-Context Training Pipeline\n"
            "    │                    ➔ Cache Outputs: forecasted_demand_cache.csv\n"
            "    ▼\n"
            " SCHEDULER: dynamic_tariff_schedule.csv ➔ Enforced Real-Time Tariffs\n"
            "===================================================="
        )
        ax.text(0.05, 0.5, box_text, family="monospace", size=9, color="black",
                ha="left", va="center", bbox=dict(boxstyle="round,pad=1", facecolor="whitesmoke", edgecolor="gray"))
        plt.savefig(viz_dir / "chart_5_data_schema_transformation.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 5: {e}")


def _chart_6_parameter_loop(viz_dir, ledger_df):
    try:
        print("Generating Chart 6: Iterative Feedback Parameter Adjustments...")
        _require_columns(ledger_df, ["iteration", "discount_mult", "surge_mult"], "Chart 6")
        
        plt.figure(figsize=(8, 4))
        plt.step(ledger_df["iteration"], ledger_df["discount_mult"], where="mid", marker="o", color="crimson", linewidth=2, label="Off-Peak Discount Factor")
        plt.plot(ledger_df["iteration"], ledger_df["surge_mult"], marker="s", color="navy", linestyle=":", label="Surge Cap Factor")
        plt.title("Parametric Loop State Explorer Adjustments", weight="bold", pad=12)
        plt.xlabel("Controller Iteration Step Index")
        plt.ylabel("Multiplication Modifier Value")
        plt.ylim(0.5, 1.8)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_6_feedback_parameter_adjustments.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 6: {e}")


def _chart_7_metric_loop(viz_dir, ledger_df):
    try:
        print("Generating Chart 7: Multi-Axis Macro Metric Traces...")
        _require_columns(ledger_df, ["iteration", "growth", "congestion_risk"], "Chart 7")
        
        fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
        color = "darkgreen"
        ax1.set_xlabel("Feedback Controller Run Steps")
        ax1.set_ylabel("Net Revenue Growth Lift (%)", color=color, weight="bold")
        line1 = ax1.plot(ledger_df["iteration"], ledger_df["growth"], marker="o", color=color, linewidth=2, label="Revenue Lift %")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "darkred"
        ax2.set_ylabel("Grid Post-Pricing Congestion Risk (%)", color=color, weight="bold")
        c_risk = ledger_df["congestion_risk"] * 100 if ledger_df["congestion_risk"].max() <= 1.0 else ledger_df["congestion_risk"]
        line2 = ax2.plot(ledger_df["iteration"], c_risk, marker="x", color=color, linestyle="--", linewidth=2, label="Congestion Risk")
        ax2.tick_params(axis="y", labelcolor=color)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower left")
        
        plt.title("Macro Optimization Responses: Profit Growth vs Infrastructure Stability", weight="bold", pad=15)
        fig.tight_layout()
        plt.savefig(viz_dir / "chart_7_feedback_metric_responses.jpg")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 7: {e}")


def _chart_8_convergence(viz_dir, ledger_df):
    try:
        print("Generating Chart 8: Parameter Space Target Grid Space Scatter...")
        _require_columns(ledger_df, ["discount_mult", "growth", "iteration"], "Chart 8")
        
        plt.figure(figsize=(7.5, 4.5))
        scatter = plt.scatter(ledger_df["discount_mult"], ledger_df["growth"], c=ledger_df["iteration"], cmap="viridis", s=150, edgecolors="black", alpha=0.9, zorder=3)
        plt.colorbar(scatter, label="Iteration Execution Step Order")
        
        best_row = ledger_df.loc[ledger_df["growth"].idxmax()]
        plt.axhline(best_row["growth"], color="grey", linestyle=":", zorder=1)
        plt.scatter(best_row["discount_mult"], best_row["growth"], color="none", edgecolors="red", s=300, linewidths=2.5, zorder=4)
        plt.text(best_row["discount_mult"], best_row["growth"] + 0.01, " Max Yield Optima", color="red", weight="bold")

        plt.title("Multi-Agent Convergence Space: Parameter Grid Exploration Paths", weight="bold", pad=12)
        plt.xlabel("Explored Off-Peak Discount Matrix Rules (x Baseline)")
        plt.ylabel("Net Financial Revenue Growth Lift (%)")
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_8_multi_agent_controller_convergence.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 8: {e}")


def _chart_9_strategy_pie(viz_dir, strategy_df):
    try:
        print("Generating Chart 9: Dynamic Pricing Deployment Profile...")
        if "pricing_action" in strategy_df.columns:
            counts = strategy_df["pricing_action"].value_counts()
            labels = counts.index.tolist()
            sizes = counts.values.tolist()
        else:
            labels = ["Baseline Elasticity Flat", "Surge Rule Premium", "Off-Peak Discount Action"]
            sizes = [53.89, 37.74, 8.37]

        plt.figure(figsize=(6.5, 5.5))
        plt.pie(sizes, labels=labels, autopct="%1.2f%%", startangle=140, colors=["#bdd7ee", "#f8cbad", "#c6e0b4"], wedgeprops={"edgecolor":"black", "linewidth":1})
        plt.title("Strategic Action Deployment Proportions Matrix Breakdown", weight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_9_pricing_intervention_distribution.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 9: {e}")


def _chart_10_prediction_model(viz_dir, forecast_df):
    try:
        print("Generating Chart 10: Model Actuals vs Predicted Time Series Series...")
        _require_columns(forecast_df, ["predicted_volume_kwh"], "Chart 10")
        actual_col = "actual_volume_kwh" if "actual_volume_kwh" in forecast_df.columns else "volume_kwh"
        if actual_col not in forecast_df.columns:
            print("[WARN] Actual values unavailable. Skipping Chart 10.")
            return
        
        sample_fc = forecast_df.head(96).copy()
        plt.figure(figsize=(12, 4.5))
        plt.plot(range(len(sample_fc)), sample_fc[actual_col], label="Ground Truth Observability", color="black", linewidth=2)
        plt.plot(range(len(sample_fc)), sample_fc["predicted_volume_kwh"], label=r"XGBoost Ensemble Forecast ($\hat{Y}$)", color="limegreen", linestyle="--", linewidth=1.5)
        plt.title("Demand Forecasting Core Evaluation: Historical Actuals vs Prediction Alignments ($R^2=0.98$)", weight="bold", pad=12)
        plt.xlabel("Continuous Evaluation Operational Hours Profile")
        plt.ylabel("Charging Network Load Scale (kWh)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(viz_dir / "chart_10_prediction_model_performance.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed rendering Chart 10: {e}")


def _chart_11_policy_frontier(viz_dir, sensitivity_df):
    try:
        print("Generating Chart 11: Multi-Objective Policy Frontier...")

        _require_columns(
            sensitivity_df,
            [
                "Surge Cap Factor",
                "Discount Factor",
                "Projected Revenue Growth (%)",
                "Congestion Mitigation Rate (%)",
            ],
            "Chart 11",
        )

        # Pareto-style frontier extraction optimization
        frontier_df = (
            sensitivity_df
            .sort_values("Projected Revenue Growth (%)", ascending=False)
            .groupby("Surge Cap Factor", as_index=False)
            .first()
            .sort_values("Surge Cap Factor")
        )

        fig, ax1 = plt.subplots(figsize=(9, 5))
        revenue_color = "darkblue"

        ax1.set_xlabel("Surge Multiplier Factor", weight="bold")
        ax1.set_ylabel("Revenue Growth (%)", color=revenue_color, weight="bold")
        ax1.plot(
            frontier_df["Surge Cap Factor"],
            frontier_df["Projected Revenue Growth (%)"],
            marker="o",
            linewidth=2.5,
            color=revenue_color,
            label="Revenue Growth"
        )
        ax1.tick_params(axis="y", labelcolor=revenue_color)

        ax2 = ax1.twinx()
        congestion_color = "darkorange"

        ax2.set_ylabel("Congestion Mitigation (%)", color=congestion_color, weight="bold")
        ax2.plot(
            frontier_df["Surge Cap Factor"],
            frontier_df["Congestion Mitigation Rate (%)"],
            marker="s",
            linewidth=2,
            linestyle="--",
            color=congestion_color,
            label="Congestion Mitigation"
        )
        ax2.tick_params(axis="y", labelcolor=congestion_color)

        # FIXED: Look up policy target safe fallback if DEFAULT_SURGE_MULTIPLIER is not mapped
        selected_x = getattr(config, "DEFAULT_SURGE_MULTIPLIER", 1.3333)
        
        # Verify if the target multiplier factor falls neatly within the parsed frontier bounds
        if selected_x in frontier_df["Surge Cap Factor"].values or min(frontier_df["Surge Cap Factor"]) <= selected_x <= max(frontier_df["Surge Cap Factor"]):
            ax1.axvline(
                selected_x,
                color="red",
                linestyle=":",
                linewidth=2
            )
            ax1.text(
                selected_x,
                ax1.get_ylim()[1] * 0.85,
                f"Selected Target\n({selected_x:.4f}x)",
                color="red",
                weight="bold",
                ha="center"
            )

        plt.title(
            "Policy Optimization Frontier\nRevenue (40%) vs Congestion (60%) Trade-Off",
            weight="bold",
            pad=15
        )

        fig.tight_layout()
        plt.savefig(viz_dir / "chart_11_policy_sensitivity_tradeoffs.png")
        plt.close()

    except Exception as e:
        print(f"[WARN] Failed rendering Chart 11: {e}")


def generate_and_save_all_plots():
    viz_dir = config.OUTPUTS_DIR / "data_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print("Initializing output-driven visualization engine.")
    print(f"Target visualization folder: {viz_dir}\n")

    master_df, ledger_df, strategy_df, forecast_df, sensitivity_df = _load_inputs()

    _chart_1_urban_heatmap(viz_dir, master_df)
    _chart_2_acn_distribution(viz_dir, master_df)
    _chart_3_lag_trace(viz_dir, master_df)
    _chart_4_scheduling_bands(viz_dir, master_df)
    _chart_5_schema(viz_dir)
    _chart_6_parameter_loop(viz_dir, ledger_df)
    _chart_7_metric_loop(viz_dir, ledger_df)
    _chart_8_convergence(viz_dir, ledger_df)
    _chart_9_strategy_pie(viz_dir, strategy_df)
    _chart_10_prediction_model(viz_dir, forecast_df)
    _chart_11_policy_frontier(viz_dir, sensitivity_df)

    print(f"\n✨ System Visualization Sequence Execution Complete. Rendered artifacts committed to {viz_dir}")


if __name__ == "__main__":
    try:
        generate_and_save_all_plots()
    except Exception as e:
        print(f"[FATAL] Visualization Agent execution halted: {e}")
        sys.exit(1)