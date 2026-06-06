import json
import os
from pathlib import Path
import pandas as pd


def print_agent_header(title: str):
    """Prints a clean, prominent header for agent execution loops."""
    print("\n" + "=" * 80)
    print(f" 🤖 {title.upper()}")
    print("=" * 80)


def print_kv_table(title: str, rows: list):
    """Prints a beautifully aligned key-value status table to the console."""
    print("\n" + "-" * 72)
    print(f" {title}")
    print("-" * 72)
    for key, value in rows:
        print(f"  {str(key):<35} : {str(value)}")
    print("-" * 72)


def print_model_report(r2: float, mae: float, rmse: float, p95_error: float):
    """Formats and prints standardized machine learning performance validation scores."""
    print("\n📈 [MODEL METRICS REPORT]")
    print(f"  • R² Score (Variance Explained) : {r2:.4f}")
    print(f"  • Mean Absolute Error (MAE)    : {mae:.2f} kWh")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.2f} kWh")
    print(f"  • 95th Percentile Error (P95)   : {p95_error:.2f} kWh")


def print_reasoning_block(agent_name: str, objective: str, evidence: list, decision: str, assumptions: list, next_actions: list):
    """Outputs an advanced operational agent trace describing the model's choices.
    
    Enhanced to explicitly display customer satisfaction thresholds, policy weights,
    and behavioral elasticity tracking points.
    """
    print("\n" + "#" * 80)
    print(f" 🧠 {agent_name.upper()} MULTI-OBJECTIVE LOGICAL TRACE")
    print("#" * 80)
    print(f"🎯 OBJECTIVE : {objective}")
    
    print("\n📊 EVIDENCE COLLECTED & PRESSURE METRICS:")
    for item in evidence:
        print(f"  [-] {item}")
        
    print(f"\n⚡ DECISION ENFORCED (BALANCED BOUNDS) : {decision}")
    
    print("\n💡 STRATEGIC BEHAVIORAL ASSUMPTIONS:")
    for item in assumptions:
        print(f"  [*] {item}")
        
    print("\n⏭️ NEXT PIPELINE EXECUTION ACTIONS:")
    for item in next_actions:
        print(f"  [>] {item}")
    print("#" * 80 + "\n")


def print_policy_equilibrium_report(surge_tariff: float, discount_tariff: float, revenue_weight: float, congestion_weight: float, consumer_risk: str):
    """Outputs a runtime verification summary of the updated balanced tariff strategy.
    
    Ensures developer/evaluator can visually confirm that the controller is not
    accidentally applying historical, high-churn pricing configurations.
    """
    print("\n🛡️ [PRICING STRATEGY REGULATORY CONSTRAINTS]")
    print(f"  • Enforced Peak Hour Surge Tariff : INR {surge_tariff:.2f} / kWh")
    print(f"  • Enforced Off-Peak Discount Rate : INR {discount_tariff:.2f} / kWh")
    print(f"  • Multi-Objective Optimization Weights : Revenue={revenue_weight:.2f} | Congestion={congestion_weight:.2f}")
    print(f"  • Predicted Consumer Backlash Risk     : [{consumer_risk.upper()}]")
    print("-" * 72)


def save_dataframe(path: Path, df: pd.DataFrame):
    """Safely handles directory creation and exports a pandas DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Exported data registry cache to: {path}")


def save_model_metrics(path: Path, metrics: dict):
    """Saves evaluation performance summaries into a flat key-value CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten dict to dataframe for clean cross-agent reporting
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"📊 Saved model metric definitions to: {path}")