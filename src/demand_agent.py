import os
import sys
from pathlib import Path

import runtime_support
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Bind project environment path rules cleanly
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import config
import utils

INPUT_DATA_PATH = config.UNIFIED_HOURLY_PATH


def load_and_prepare_data(file_path):
    print(f"Loading master training matrix from: {file_path}")
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("Missing master data file. Run data_preprocessing.py first.")

    df = pd.read_csv(file_path, dtype={"station_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "volume_kwh"]).reset_index(drop=True)

    df["station_id_raw"] = df["station_id"].astype(str)
    df["context_tag_raw"] = df["context_tag"].astype(str)

    required_lags = [
        "volume_lag_1h",
        "volume_lag_2h",
        "utilization_lag_1h",
        "occupancy_lag_1h",
        "queue_lag_1h",
    ]
    df = df.dropna(subset=required_lags).reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = pd.get_dummies(
        df,
        columns=["scheduling_band", "context_tag", "station_id"],
        drop_first=False,
        dtype=float,
    )
    return df


def _feature_columns(df):
    base_features = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "energy_cost_per_kwh",
        "volume_lag_1h",
        "volume_lag_2h",
        "volume_lag_24h",
        "volume_roll_mean_3h",
        "volume_roll_mean_24h",
        "utilization_lag_1h",
        "occupancy_lag_1h",
        "queue_lag_1h",
    ]
    base_features = [column for column in base_features if column in df.columns]

    raw_metadata_columns = {"station_id_raw", "context_tag_raw"}
    dummy_features = [
        column
        for column in df.columns
        if column not in raw_metadata_columns
        and (
            column.startswith("scheduling_band_")
            or column.startswith("context_tag_")
            or column.startswith("station_id_")
        )
    ]
    return base_features + dummy_features


def test_meta_block(test_df):
    columns = ["timestamp", "station_id_raw", "context_tag_raw", "volume_kwh"]
    optional_columns = [
        "occupancy_density",
        "charger_utilization_rate",
        "queue_length_proxy",
    ]
    columns.extend([column for column in optional_columns if column in test_df.columns])
    meta = test_df[columns].copy()
    meta = meta.rename(
        columns={
            "station_id_raw": "station_id",
            "context_tag_raw": "context_tag",
            "volume_kwh": "actual_volume_kwh",
        }
    )
    return meta


def execute_time_series_split(df):
    df = df.sort_values(["context_tag_raw", "timestamp"]).reset_index(drop=True)

    train_parts = []
    test_parts = []
    cutoff_rows = []

    # Safe fallback parameter extraction to protect against config omissions
    test_fraction = getattr(config, "TEST_FRACTION", 0.20)

    for context_tag, group in df.groupby("context_tag_raw", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 2:
            continue

        split_idx = int(len(group) * (1.0 - test_fraction))
        split_idx = min(max(split_idx, 1), len(group) - 1)

        train_part = group.iloc[:split_idx].copy()
        test_part = group.iloc[split_idx:].copy()
        train_parts.append(train_part)
        test_parts.append(test_part)
        cutoff_rows.append(
            {
                "context_tag": context_tag,
                "train_rows": int(len(train_part)),
                "test_rows": int(len(test_part)),
                "cutoff_timestamp": group.loc[split_idx, "timestamp"],
                "test_window_start": test_part["timestamp"].min(),
                "test_window_end": test_part["timestamp"].max(),
            }
        )

    if not train_parts or not test_parts:
        raise ValueError("Context-aware chronological split did not produce train/test data.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    cutoff_summary = pd.DataFrame(cutoff_rows)

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty train or test set.")

    feature_cols = _feature_columns(df)
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    fill_values = X_train.median(numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    y_train = train_df["volume_kwh"].astype(float)
    y_test = test_df["volume_kwh"].astype(float)
    train_context = train_df["context_tag_raw"].reset_index(drop=True)
    test_context = test_df["context_tag_raw"].reset_index(drop=True)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_meta_block(test_df),
        cutoff_summary,
        train_context,
        test_context,
    )


def _build_xgb_model():
    random_seed = getattr(config, "RANDOM_SEED", 42)
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        max_depth=5,
        learning_rate=0.045,
        min_child_weight=4,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=1.50,
        tree_method="hist",
        random_state=random_seed,  
        n_jobs=-1,
    )


def train_demand_forecaster(X_train, y_train, train_context=None):
    utils.print_agent_header("XGBoost Demand Agent Training Loop")

    if train_context is None:
        model = _build_xgb_model()
        model.fit(X_train, y_train)
        print("Training complete.")
        return model

    models = {}
    for context_tag in sorted(train_context.astype(str).unique()):
        mask = train_context.astype(str) == context_tag
        model = _build_xgb_model()
        model.fit(X_train.loc[mask], y_train.loc[mask])
        models[context_tag] = model
        print(f"Trained context model: {context_tag} ({int(mask.sum()):,} rows)")

    print("Context-specific training complete.")
    return models


def predict_demand(model_bundle, X_test, test_context=None):
    if not isinstance(model_bundle, dict):
        return model_bundle.predict(X_test)

    if test_context is None:
        raise ValueError("Context labels are required for context-specific model predictions.")

    predictions = np.zeros(len(X_test), dtype=float)
    default_model = next(iter(model_bundle.values()))
    for context_tag in sorted(test_context.astype(str).unique()):
        mask = test_context.astype(str) == context_tag
        model = model_bundle.get(context_tag, default_model)
        predictions[mask.to_numpy()] = model.predict(X_test.loc[mask])

    return predictions


def _context_metrics(meta_df):
    rows = []
    for context_tag, group in meta_df.groupby("context_tag"):
        actual = group["actual_volume_kwh"].to_numpy(dtype=float)
        predicted = group["predicted_volume_kwh"].to_numpy(dtype=float)
        payload = {
            "context_tag": context_tag,
            "samples": int(len(group)),
            "r2_score": float(r2_score(actual, predicted)) if len(group) > 1 else np.nan,
            "mean_absolute_error_kwh": float(mean_absolute_error(actual, predicted)),
            "root_mean_squared_error_kwh": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mean_actual_kwh": float(np.mean(actual)),
            "mean_predicted_kwh": float(np.mean(predicted)),
        }
        if "predicted_utilization_rate" in group.columns:
            payload["mean_predicted_utilization_rate"] = float(
                group["predicted_utilization_rate"].mean()
            )
        if "congestion_probability" in group.columns:
            payload["mean_congestion_probability"] = float(
                group["congestion_probability"].mean()
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def _add_forecast_operational_signals(meta_df):
    """Augments raw volume forecasts with advanced multi-objective behavioral metrics."""
    meta_df = meta_df.copy()
    grouped = meta_df.groupby(["context_tag", "station_id"])["predicted_volume_kwh"]
    station_peak = grouped.transform(lambda values: values.quantile(0.90))
    context_peak = meta_df.groupby("context_tag")["predicted_volume_kwh"].transform(
        lambda values: values.quantile(0.90)
    )
    global_peak = meta_df["predicted_volume_kwh"].quantile(0.90)
    if pd.isna(global_peak) or global_peak <= 0:
        global_peak = 1.0

    denominator = (
        station_peak.replace(0, np.nan)
        .fillna(context_peak.replace(0, np.nan))
        .fillna(global_peak)
    )
    meta_df["predicted_utilization_rate"] = (
        meta_df["predicted_volume_kwh"] / denominator
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    
    # RE-OPTIMIZED CONGESTION RESPONSE MATRIX:
    # Scaling steepness to 0.05 (from 0.08) creates an earlier and more definitive 
    # warning slope, triggering congestion protection features in the system sooner.
    meta_df["congestion_probability"] = 1.0 / (
        1.0
        + np.exp(
            -(
                meta_df["predicted_utilization_rate"] - config.CONGESTION_PROBABILITY_THRESHOLD
            )
            / 0.05
        )
    )
    
    meta_df["expected_queue_length_proxy"] = (
        (meta_df["predicted_utilization_rate"] - 0.85).clip(lower=0.0) * 10.0
    )
    meta_df["forecast_risk_band"] = np.select(
        [
            meta_df["predicted_utilization_rate"] >= config.SURGE_THRESHOLD_UPPER,
            meta_df["predicted_utilization_rate"] <= config.DISCOUNT_THRESHOLD_LOWER,
        ],
        ["High congestion risk", "Low utilization opportunity"],
        default="Balanced demand window",
    )
    return meta_df


def evaluate_predictions(model, X_test, y_test, meta_df, test_context=None):
    predictions = predict_demand(model, X_test, test_context)
    predictions = np.clip(predictions, 0.0, None)

    r2 = r2_score(y_test, predictions) if len(y_test) > 1 and np.var(y_test) > 0 else 1.0
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    absolute_error = np.abs(y_test.to_numpy(dtype=float) - predictions)
    p95_error = np.percentile(absolute_error, 95) if len(absolute_error) > 0 else 0.0

    utils.print_model_report(r2, mae, rmse, p95_error)

    metrics_payload = {
        "total_test_samples": int(len(y_test)),
        "r2_score": float(r2),
        "mean_absolute_error_kwh": float(mae),
        "root_mean_squared_error_kwh": float(rmse),
        "p95_absolute_error_kwh": float(p95_error),
        "mean_actual_kwh": float(y_test.mean()),
        "median_actual_kwh": float(y_test.median()) if len(y_test) > 0 else 0.0,
    }
    utils.save_model_metrics(config.MODEL_METRICS_PATH, metrics_payload)

    meta_df = meta_df.copy()
    meta_df["predicted_volume_kwh"] = predictions
    meta_df["absolute_error_kwh"] = absolute_error
    meta_df["absolute_percentage_error"] = np.where(
        meta_df["actual_volume_kwh"] > 0,
        meta_df["absolute_error_kwh"] / meta_df["actual_volume_kwh"],
        np.nan,
    )
    meta_df = _add_forecast_operational_signals(meta_df)

    context_metrics = _context_metrics(meta_df)
    utils.save_dataframe(config.CONTEXT_METRICS_PATH, context_metrics)

    top_errors = meta_df.sort_values("absolute_error_kwh", ascending=False).head(100)
    utils.save_dataframe(config.TOP_ERROR_PATH, top_errors)

    evaluated_contexts = ", ".join(context_metrics["context_tag"].astype(str).tolist())
    high_risk_share = float(
        (meta_df["predicted_utilization_rate"] >= config.SURGE_THRESHOLD_UPPER).mean() * 100
    )
    low_util_share = float(
        (meta_df["predicted_utilization_rate"] <= config.DISCOUNT_THRESHOLD_LOWER).mean() * 100
    )
    utils.print_reasoning_block(
        "Demand Agent Reasoning Trace",
        objective="Forecast hourly charging load and translate it into utilization and congestion signals.",
        evidence=[
            f"Model strategy: {'context-specific XGBoost ensemble' if isinstance(model, dict) else 'single global XGBoost model'}",
            f"Validation covers: {evaluated_contexts}",
            f"Overall R2={r2:.4f}, MAE={mae:.2f} kWh, RMSE={rmse:.2f} kWh",
            f"High congestion risk windows={high_risk_share:.2f}% of validation schedule",
            f"Low-utilization discount windows={low_util_share:.2f}% of validation schedule",
        ],
        decision="Publish forecast cache with predicted load, utilization rate, congestion probability, and risk band.",
        assumptions=[
            "Utilization is inferred by comparing each station forecast with its own high-demand forecast level.",
            "This avoids using future actual occupancy inside the pricing agent.",
        ],
        next_actions=[
            "Pricing agent will use predicted utilization, not actual test occupancy, as its tariff pressure signal.",
        ],
    )

    return meta_df


if __name__ == "__main__":
    try:
        master_df = load_and_prepare_data(INPUT_DATA_PATH)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            evaluation_tracker,
            cutoff_summary,
            train_context,
            test_context,
        ) = execute_time_series_split(master_df)

        utils.print_kv_table(
            "Chronological Validation Design",
            [
                ("Split strategy", "Context-aware 80/20 time split"),
                ("Train rows", X_train.shape[0]),
                ("Validation rows", X_test.shape[0]),
                ("Contexts evaluated", cutoff_summary["context_tag"].nunique()),
            ],
        )
        print(cutoff_summary.to_string(index=False))
        utils.save_dataframe(config.OUTPUTS_DIR / "validation_split_summary.csv", cutoff_summary)

        forecasting_model = train_demand_forecaster(X_train, y_train, train_context)
        predictions_metadata = evaluate_predictions(
            forecasting_model, X_test, y_test, evaluation_tracker, test_context
        )

        utils.save_dataframe(config.FORECAST_CACHE_PATH, predictions_metadata)

    except Exception as exc:
        import traceback
        print("[FATAL ERROR TRACEBACK]")
        traceback.print_exc()
        sys.exit(1)