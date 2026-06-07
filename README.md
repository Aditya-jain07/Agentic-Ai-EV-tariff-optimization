# Optimizing EV Charging Networks via Multi-Agent Closed-Loop Tariff Control

> **Author:** Aditya Jain · Roll No: 23118005  
> **Domain:** Predictive Analytics · Behavioral Economics · Multi-Agent Control  
> **Stack:** Python · XGBoost · Multi-Agent Systems · Dynamic Pricing

---

## Project resources

https://drive.google.com/drive/folders/1dACwD8poTVCXj09iNiZpHe7cl3C5hvvy?usp=drive_link

---

## Abstract

EV adoption is outpacing grid capacity. Charging networks face two simultaneous failures: **localized peak congestion** that threatens transformers, and **chronic off-peak idle capacity**. Physical grid upgrades are capital-intensive and slow to deploy.

This project implements a **closed-loop, multi-agent tariff control system** that uses data-driven demand forecasting and price-elasticity economics to shift driver behaviour — turning pricing itself into the control lever that hardware used to be. The result: more throughput, a flatter grid, and higher revenue — **at the same physical assets**.

---

## Core Research Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROLLER (ORCHESTRATOR)                    │
│        Sweeps tariff multipliers · Harvests telemetry ·        │
│               Locks optimal policy each cycle                   │
└───────────┬─────────────────────────────────────┬──────────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────┐                 ┌───────────────────────┐
│  01 DEMAND AGENT  │                 │   02 PRICING AGENT    │
│                   │                 │                       │
│  Context-isolated │  ─── signal ──► │  Converts forecast    │
│  XGBoost forecasts│                 │  pressure into        │
│  volume, util &   │                 │  Surge / Baseline /   │
│  congestion prob  │                 │  Discount tariffs     │
│  per station-hour │                 │  + elasticity demand  │
│                   │                 │  response             │
└───────────────────┘                 └──────────┬────────────┘
            ▲                                    │
            │                                    ▼
            │                      ┌─────────────────────────┐
            │                      │   03 MONITORING AGENT   │
            │                      │                         │
            └────── re-tunes ──────│  Audits revenue lift &  │
                  next cycle       │  congestion deltas,     │
                                   │  flags drift, emits     │
                                   │  status alerts &        │
                                   │  recommendations        │
                                   └─────────────────────────┘

        ↻  CLOSED-LOOP FEEDBACK: measured driver response
           re-tunes the next cycle's multipliers automatically
```

---

## Key Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Closed-loop multi-agent control** | Forecasting, pricing, monitoring & orchestration as cooperating, auditable agents |
| 2 | **Explicit behavioral model** | Price-elasticity demand response with realistic ±25% bounds — not assumed |
| 3 | **Constrained multi-objective policy** | Revenue-positivity + congestion-priority (0.40/0.60) selection rule |
| 4 | **Explainable by construction** | Reasoning traces, pricing confidence & per-decision tariff reasons |

---

## Project Structure

```
project/
│
├── data/
│   ├── processed/                          # Cleaned, feature-engineered datasets
│   └── raw/
│       ├── ACN Data_ 25 April 2018 to 16 Dec 2018/
│       │   └── acndata_sessions.json.xlsx  # Workplace ACN charging session logs
│       └── UrbanEV_SZ_districts/           # Shenzhen urban EV district data
│           ├── adj.csv                     # District adjacency matrix
│           ├── distance.csv                # Inter-district distances
│           ├── duration.csv                # Charging session durations
│           ├── information.csv             # Station metadata
│           ├── occupancy.csv               # Occupancy rates per station
│           ├── price.csv                   # Historical pricing data
│           ├── stations.csv                # Station inventory & attributes
│           ├── time.csv                    # Temporal index
│           └── volume.csv                  # Charging volume per station-hour
│
├── outputs/
│   ├── data_visualization/
│   │   ├── chart_1_urban_spatial_temporal_bottlenecks.png
│   │   ├── chart_2_acn_behavioral_inefficiencies.png
│   │   ├── chart_3_leakage_safe_lag_tracking.jpg
│   │   ├── chart_4_preprocessed_scheduling_bands.png
│   │   ├── chart_5_data_schema_transformation.png
│   │   ├── chart_6_feedback_parameter_adjustments.png
│   │   ├── chart_7_feedback_metric_responses.jpg
│   │   ├── chart_8_multi_agent_controller_convergence.png
│   │   ├── chart_9_pricing_intervention_distribution.png
│   │   ├── chart_10_prediction_model_performance.png
│   │   └── chart_11_policy_sensitivity_tradeoffs.png
│   ├── business_outcomes.csv               # Aggregated financial & operational KPIs
│   ├── controller_iteration_ledger.csv     # Per-iteration convergence trace
│   ├── demand_forecast_top_errors.csv      # Worst-case forecast samples
│   ├── dynamic_tariff_schedule.csv         # Final tariff schedule per station-hour
│   ├── model_accuracy_by_context.csv       # R², MAE, RMSE split by context
│   ├── model_accuracy_scores.csv           # Global model accuracy metrics
│   ├── policy_sensitivity_matrix.csv       # Surge-cap frontier analysis
│   ├── pricing_action_summary.csv          # Tier deployment counts & lift
│   ├── pricing_strategy_distribution.csv   # Action mix proportions
│   ├── system_telemetry_diagnostics.csv    # Per-cycle monitoring agent output
│   └── validation_split_summary.csv        # Train/test split statistics
│
├── src/
│   ├── config.py                           # Global hyperparameters & constants
│   ├── data_preprocessing.py               # Schema unification & feature engineering
│   ├── demand_agent.py                     # Context-isolated XGBoost forecasting
│   ├── monitoring_agent.py                 # Drift detection, alerts & recommendations
│   ├── pricing_agent.py                    # Tiered tariff engine + elasticity model
│   ├── runtime_support.py                  # Logging, telemetry & session utilities
│   ├── sensitivity_analysis.py             # Surge-cap frontier & policy sweep
│   ├── utils.py                            # Shared helpers & data I/O
│   └── visualization_agent.py             # Chart generation for all 11 outputs
│
└── main.py                                 # Pipeline entry point — runs end-to-end
```

---

## Datasets

### 1. UrbanEV — Shenzhen Districts (`data/raw/UrbanEV_SZ_districts/`)

Real-world public EV charging data from Shenzhen's urban districts. High daily demand variance makes this context ideal for learning peak-congestion patterns.

| File | Description |
|------|-------------|
| `stations.csv` | Station IDs, district mapping, capacity |
| `volume.csv` | Charging volume (kWh) per station per hour |
| `occupancy.csv` | Occupancy rate per station-hour |
| `price.csv` | Historical tariff per station |
| `duration.csv` | Session length distributions |
| `distance.csv` | Distance matrix between districts |
| `adj.csv` | Binary adjacency matrix for spatial graph |
| `information.csv` | Station-level metadata |
| `time.csv` | Temporal index (timestamps) |

### 2. ACN Data — Workplace Charging (`data/raw/ACN Data_…/acndata_sessions.json.xlsx`)

Caltech ACN workplace charging sessions (25 Apr 2018 – 16 Dec 2018). Flat demand profile (~9 kWh mean) — evaluated on MAE rather than R² due to near-zero variance.

---

## Understanding the outputs

###  Most Important Outputs

If you only review a few files, start with these:

| File                              | Why It Matters                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------- |
| `business_outcomes.csv`           | Final business KPIs: revenue growth, congestion reduction, demand shifting, and pricing efficiency |
| `controller_iteration_ledger.csv` | Complete optimization trace showing how the controller converged to the final policy               |
| `model_accuracy_scores.csv`       | Overall forecasting performance (R², MAE, RMSE, P95 Error)                                         |
| `policy_sensitivity_matrix.csv`   | Revenue–congestion trade-off frontier used to validate the chosen pricing policy                   |
| `dynamic_tariff_schedule.csv`     | Final station-hour tariff assignments generated by the pricing engine                              |

---

### Visualization Outputs

| Chart                                            | Purpose                                                                     |
| ------------------------------------------------ | --------------------------------------------------------------------------- |
| `chart_1_urban_spatial_temporal_bottlenecks.png` | Identifies localized congestion hotspots across districts and time periods  |
| `chart_2_acn_behavioral_inefficiencies.png`      | Reveals under-utilized workplace charging demand and off-peak opportunities |
| `chart_3_leakage_safe_lag_tracking.jpg`          | Verifies that lag features do not introduce target leakage                  |
| `chart_4_preprocessed_scheduling_bands.png`      | Shows demand segmentation into Peak, Shoulder, and Off-Peak operating bands |
| `chart_5_data_schema_transformation.png`         | Visualizes raw-to-feature-engineered data transformation pipeline           |
| `chart_6_feedback_parameter_adjustments.png`     | Displays controller adjustments applied during optimization                 |
| `chart_7_feedback_metric_responses.jpg`          | Shows how operational KPIs respond to policy changes                        |
| `chart_8_multi_agent_controller_convergence.png` | Visualizes controller convergence toward the final policy                   |
| `chart_9_pricing_intervention_distribution.png`  | Distribution of Discount, Baseline, and Surge pricing actions               |
| `chart_10_prediction_model_performance.png`      | Predicted vs actual demand and forecasting accuracy analysis                |
| `chart_11_policy_sensitivity_tradeoffs.png`      | Revenue, congestion, and acceptability trade-off frontier                   |

---

### Structured Output Files

| File                                | Description                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| `business_outcomes.csv`             | Final financial and operational KPI summary                           |
| `controller_iteration_ledger.csv`   | Per-iteration optimization history and convergence metrics            |
| `demand_forecast_top_errors.csv`    | Largest forecasting errors for model diagnostics                      |
| `dynamic_tariff_schedule.csv`       | Generated tariff schedule for every station-hour                      |
| `model_accuracy_by_context.csv`     | Forecast performance split by Urban Public and Workplace ACN contexts |
| `model_accuracy_scores.csv`         | Global forecasting accuracy metrics                                   |
| `policy_sensitivity_matrix.csv`     | Sensitivity analysis across surge-cap policies                        |
| `pricing_action_summary.csv`        | Revenue contribution of each pricing tier                             |
| `pricing_strategy_distribution.csv` | Percentage distribution of Discount, Baseline, and Surge actions      |
| `system_telemetry_diagnostics.csv`  | Monitoring agent status reports, alerts, and recommendations          |
| `validation_split_summary.csv`      | Train/test split statistics and dataset partition information         |

---

### Recommended Review Order

To understand the entire system in the shortest time:

1. `business_outcomes.csv`
2. `model_accuracy_scores.csv`
3. `controller_iteration_ledger.csv`
4. `policy_sensitivity_matrix.csv`
5. `dynamic_tariff_schedule.csv`
6. Visualization charts (`chart_1` → `chart_11`)

---

## Source Modules

### `config.py` — Central Configuration
Defines all tunable hyperparameters in one place: XGBoost settings, tariff multipliers, elasticity bounds, controller grid dimensions, and objective weights.

### `data_preprocessing.py` — Schema Unification & Feature Engineering
- Merges heterogeneous raw CSVs into a unified station-hour schema
- Engineers **leakage-safe lag features**: 1h / 2h / 24h / 168h lags + rolling means
- Applies chronological 80/20 train-test split **per context** (never trains on future)
- Infers utilization from predicted demand vs each station's own 90th-percentile peak

### `demand_agent.py` — XGBoost Demand Forecaster

Trains one model per network context (Urban Public vs Workplace ACN) because pooling would average away behavioural differences between the two.

```
Model Config:
  Estimators / Depth  : 450 / 5
  Learning Rate       : 0.045
  Subsample/Colsample : 0.90 / 0.90
  Regularization α/λ  : 0.05 / 1.50
```

**Forecast Accuracy:**
| Context | Samples | R² | MAE (kWh) | RMSE (kWh) |
|---|---|---|---|---|
| Urban Public | 35,568 | **0.9785** | 61.28 | 187.49 |
| Workplace ACN | 2,979 | 0.007 | **6.10** | 8.56 |
| Global | 38,547 | **0.9788** | 57.01 | 180.11 |

> **The Contextual Paradox:** Near-zero R² on ACN means "no variance to explain," not a failed model. MAE of 6.10 kWh against a mean actual of 9.20 kWh is strong performance.

### `pricing_agent.py` — Dynamic Tariff Engine

Converts the demand-pressure signal (predicted utilization) into one of three tariff tiers. Every decision is logged with a confidence level and a plain-language reason.

```
Demand Response Model:
  Qafter = Qpred × ( 1 + ε · Δprice% )
  ε = −0.25   |   response clipped to [0.75, 1.25]

Congestion Probability (Logistic):
  P(congestion) = σ( (u − 0.75) / 0.05 )
```

| Tier | Trigger | Multiplier | Tariff (INR) | Purpose |
|---|---|---|---|---|
| 🟢 Discount | util < 35% | 0.90x | ₹13.50 | Fill idle valleys |
| 🟡 Baseline | 35–75% | 1.00x | ₹15.00 | Normal operation |
| 🔴 Surge | util > 75% | 1.40x | ₹21.00 | Relieve congestion |

### `monitoring_agent.py` — Drift Detection & Alerting
- Emits **NOMINAL / WARNING / ALERT** status on every cycle
- Auto-generates recommendations (e.g., "discount cannibalizing margins → compress multiplier")
- Audits revenue lift and congestion deltas across iterations

### `sensitivity_analysis.py` — Policy Frontier Sweep
Sweeps a 5 × 5 grid of surge × discount multipliers to map the full revenue–congestion tradeoff curve, identifying the safe operating boundary before a policy is locked.

### `visualization_agent.py` — Chart Generation
Produces all 11 publication-ready charts covering spatial bottlenecks, ACN behavioral patterns, leakage-safe lag verification, schema transformation, controller convergence, pricing distribution, prediction performance, and the policy sensitivity frontier.

---

## Controller & Optimization

The controller sweeps a **5 × 5 grid** of `surge × discount` multipliers and applies a two-stage selection rule:

```
Selection Rule:
  Step 1 → Keep only policies where revenue growth ≥ 7%
  Step 2 → Among those, maximize congestion mitigation (weight 0.60)
  Result → Locks 1.40x Surge / 0.90x Discount
```

**Convergence Trace:**

| Iteration | Surge | Discount | Revenue Growth | Congestion ↓ |
|---|---|---|---|---|
| 1 | 1.25x | 0.70x | 7.20% | 30.7% |
| 10 | 1.30x | 0.90x | 10.53% | 36.3% |
| **25 (Lock)** | **1.40x** | **0.90x** | **13.73%** | **37.20%** |

---

## Business Impact

```
┌─────────────────┬────────────────────────────────────────┐
│ +13.73%         │ Net Revenue Growth                     │
│                 │ +INR 35.54M vs flat-tariff baseline    │
├─────────────────┼────────────────────────────────────────┤
│ −37.20%         │ Peak Congestion Reduction              │
│                 │ Risk score: 0.572 → 0.359              │
├─────────────────┼────────────────────────────────────────┤
│ 41,227 kWh      │ Off-Peak Demand Uplift                 │
│                 │ Successfully pulled into idle valleys  │
├─────────────────┼────────────────────────────────────────┤
│ INR 18.02       │ Pricing Efficiency                     │
│                 │ Revenue per delivered kWh              │
└─────────────────┴────────────────────────────────────────┘
```

**Revenue Decomposition by Tier:**

| Tier | Intervals | Net Lift (INR) | Role |
|---|---|---|---|
| 🔴 Surge | 17,140 | +37,460,864 | Revenue engine |
| 🟢 Discount | 6,082 | −1,917,065 | Congestion investment |
| **Net** | **38,547** | **+35,543,799** | **+13.73% over baseline** |

**Action Mix:** Surge 44.5% · Baseline 39.8% · Discount 15.8% of station-hours

---

## Satisfaction & Policy Sensitivity

Aggressive pricing can win short-term revenue and lose customers. Surge is bounded inside a measured acceptability envelope:

| Surge Cap | Rev. Growth | Congestion ↓ | Acceptability |
|---|---|---|---|
| 1.30x | 10.53% | 13.00% | High Churn Risk |
| 1.333x | 11.63% | 14.85% | Optimal / Safe |
| **1.40x ✓** | **13.73%** | **18.82%** | **Marginal Boundary** |
| 1.45x | 15.24% | — | ❌ Unacceptable |

**Demand Response Proxies:**
- Customer response rate: **5.81%** of demand shifted on price
- Peak wait reduction: **≈10%** less queueing at hot stations
- Off-peak uplift: **41,227 kWh** of valleys successfully filled

---

## Getting Started

### Prerequisites

```bash
python >= 3.9
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

### Run the Full Pipeline

```bash
# Clone and navigate
cd project/

# Run end-to-end: preprocessing → forecasting → pricing → optimization → outputs
python main.py
```

### Run Individual Components (not necessary if main.py is already executed)

```bash
# Data preprocessing only
python -c "from src.data_preprocessing import run_preprocessing; run_preprocessing()"

# Sensitivity analysis sweep
python -c "from src.sensitivity_analysis import run_sweep; run_sweep()"
```

All outputs (CSV tables + charts) are written to `outputs/`.

---

## 🔬 Research Gap Closed

| Approach | Real-time | Models Driver Behaviour | Self-corrects |
|---|---|---|---|
| Flat tariff | ❌ | ❌ | ❌ |
| Static TOU | Partly | ❌ | ❌ |
| Rule-based surge | ✅ | ❌ | ❌ |
| **This work** | ✅ | ✅ | ✅ |

---

## Future Work

| # | Direction | Description |
|---|---|---|
| 01 | **Live hardware telemetry** | Replace batch processing with real-time charger status feeds |
| 02 | **Contextual enrichment** | Add weather, holidays & wholesale energy prices as model features |
| 03 | **Reinforcement learning** | Safety-constrained RL to replace the static grid search |
| 04 | **Personalized nudges** | Carbon-aware, app-level delayed-charging offers per individual driver |

---

## Research Pipeline

```
Problem → Modeling → Incentives → Optimization → Impact → Scalability
   │           │            │             │            │          │
Grid        XGBoost     Price-       Multi-agent   +13.73%     Live
congestion  per-context elasticity   controller    revenue     telemetry
+ idle      forecasting ±25% bounds  5×5 sweep    −37.20%     + RL
capacity                             convergence  congestion
```
---

*A charging network re-imagined as self-regulating infrastructure: it forecasts, prices, measures response, and re-tunes itself — turning price into the control lever that hardware used to be.*
