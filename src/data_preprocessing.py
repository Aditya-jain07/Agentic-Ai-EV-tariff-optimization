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


def assign_scheduling_band(hour):
    if hour in [8, 9, 10, 17, 18, 19, 20]:
        return "Peak"
    if hour in [23, 0, 1, 2, 3, 4, 5, 6]:
        return "Off-Peak"
    return "Shoulder"


def get_band_tariff(hour):
    """Returns the baseline tariff based on the scheduling band."""
    band = assign_scheduling_band(hour)
    if band == "Peak":
        return config.BASE_TARIFF_INR_PER_KWH * config.DEFAULT_SURGE_MULTIPLIER      # Evaluates to 15 * 1.3333 = 20.00
    elif band == "Off-Peak":
        return config.BASE_TARIFF_INR_PER_KWH * config.DEFAULT_DISCOUNT_MULTIPLIER   # Evaluates to 15 * 0.80 = 12.00
    return config.BASE_TARIFF_INR_PER_KWH                                            # Baseline Shoulder = 15.00


def _parse_datetime(series):
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def _require_columns(df, required, file_label):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def process_acn_data(file_path):
    file_path = Path(file_path)
    print(f"Loading ACN sessions from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"ACN file not found at: {file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")
    _require_columns(df, ["connectionTime", "disconnectTime", "kWhDelivered"], "ACN workbook")

    if "stationID" not in df.columns:
        if "spaceID" in df.columns:
            df["stationID"] = df["spaceID"]
        else:
            df["stationID"] = "unknown"

    if "doneChargingTime" not in df.columns:
        df["doneChargingTime"] = df["disconnectTime"]

    df["connectionTime"] = _parse_datetime(df["connectionTime"])
    df["disconnectTime"] = _parse_datetime(df["disconnectTime"])
    df["doneChargingTime"] = _parse_datetime(df["doneChargingTime"])
    df["stationID"] = df["stationID"].fillna("unknown").astype(str).str.strip()
    df["kWhDelivered"] = pd.to_numeric(df["kWhDelivered"], errors="coerce")

    median_kwh = df["kWhDelivered"].median()
    if pd.isna(median_kwh):
        median_kwh = 0.0
    df["kWhDelivered"] = df["kWhDelivered"].fillna(median_kwh)

    df = df.dropna(subset=["connectionTime", "disconnectTime"])
    df = df[df["disconnectTime"] > df["connectionTime"]]
    df = df[df["kWhDelivered"] >= 0]

    df["doneChargingTime"] = df["doneChargingTime"].fillna(df["disconnectTime"])
    df["doneChargingTime"] = df[["doneChargingTime", "disconnectTime"]].min(axis=1)
    df["doneChargingTime"] = df[["doneChargingTime", "connectionTime"]].max(axis=1)

    df["total_session_hours"] = (
        df["disconnectTime"] - df["connectionTime"]
    ).dt.total_seconds() / 3600.0
    df["active_charging_hours"] = (
        df["doneChargingTime"] - df["connectionTime"]
    ).dt.total_seconds() / 3600.0
    df["active_charging_hours"] = df["active_charging_hours"].clip(
        lower=0.0, upper=df["total_session_hours"]
    )
    df["idling_hours_waste"] = (
        df["total_session_hours"] - df["active_charging_hours"]
    ).clip(lower=0.0)

    df["charger_utilization_rate"] = np.where(
        df["total_session_hours"] > 0,
        df["active_charging_hours"] / df["total_session_hours"],
        0.0,
    )
    
    # DYNAMIC TRACKING FIX: Calculate baseline tariffs mapped directly to session intervals
    session_hours = df["connectionTime"].dt.hour
    df["session_base_tariff"] = session_hours.apply(get_band_tariff)
    df["revenue_per_session"] = df["kWhDelivered"] * df["session_base_tariff"]
    
    df["energy_cost_per_kwh"] = (
        config.BASE_TARIFF_INR_PER_KWH * config.ACN_ENERGY_COST_FACTOR
    )
    df["idle_time_ratio"] = np.where(
        df["total_session_hours"] > 0,
        df["idling_hours_waste"] / df["total_session_hours"],
        0.0,
    )

    return df.sort_values("connectionTime").reset_index(drop=True)


def _read_urbanev_timeline(urban_dir):
    time_path = urban_dir / "time.csv"
    if not time_path.exists():
        raise FileNotFoundError(f"UrbanEV time.csv not found at: {time_path}")

    time_df = pd.read_csv(time_path)
    required = ["year", "month", "day", "hour", "minute"]
    _require_columns(time_df, required, "UrbanEV time.csv")

    return pd.to_datetime(
        pd.DataFrame(
            {
                "year": time_df["year"],
                "month": time_df["month"],
                "day": time_df["day"],
                "hour": time_df["hour"],
                "minute": time_df["minute"],
                "second": time_df["second"] if "second" in time_df.columns else 0,
            }
        ),
        errors="coerce",
    )


def _matrix_station_columns(matrix):
    return [column for column in matrix.columns if str(column).lower() != "timestamp"]


def _matrix_values(urban_dir, metric, station_cols, valid_rows):
    path = urban_dir / f"{metric}.csv"
    if not path.exists():
        raise FileNotFoundError(f"UrbanEV matrix not found: {path}")

    matrix = pd.read_csv(path)
    missing_cols = [column for column in station_cols if column not in matrix.columns]
    if missing_cols:
        raise ValueError(f"{metric}.csv is missing station columns: {missing_cols[:8]}")

    values = matrix.loc[valid_rows, station_cols].apply(pd.to_numeric, errors="coerce")
    return values.to_numpy(dtype=float).reshape(-1, order="F")


def _load_capacity_map(urban_dir, station_cols):
    info_name = "information.csv" if (urban_dir / "information.csv").exists() else "stations.csv"
    info_path = urban_dir / info_name
    if not info_path.exists():
        return {}

    info_df = pd.read_csv(info_path)
    station_col_set = {str(column).strip() for column in station_cols}

    if "grid" in info_df.columns and set(info_df["grid"].astype(str)).intersection(station_col_set):
        id_col = "grid"
    elif "num" in info_df.columns:
        id_col = "num"
    else:
        id_col = info_df.columns[0]

    if "count" not in info_df.columns:
        return {}

    station_ids = info_df[id_col].astype(str).str.strip()
    capacities = pd.to_numeric(info_df["count"], errors="coerce")
    return dict(zip(station_ids, capacities))


def process_urban_ev_data(urban_dir):
    urban_dir = Path(urban_dir)
    print(f"Loading UrbanEV matrices from: {urban_dir}")
    if not urban_dir.exists():
        raise FileNotFoundError(f"UrbanEV directory not found at: {urban_dir}")

    timeline = _read_urbanev_timeline(urban_dir)
    valid_rows = timeline.notna().to_numpy()
    timeline = timeline.loc[valid_rows].reset_index(drop=True)

    volume_matrix = pd.read_csv(urban_dir / "volume.csv")
    station_cols = [str(column) for column in _matrix_station_columns(volume_matrix)]
    n_times = len(timeline)

    urban_master = pd.DataFrame(
        {
            "station_id": np.repeat(station_cols, n_times),
            "timestamp": np.tile(timeline.to_numpy(), len(station_cols)),
        }
    )

    for metric in ["volume", "occupancy", "duration", "price"]:
        print(f"Processing UrbanEV layer: {metric}.csv")
        urban_master[metric] = _matrix_values(urban_dir, metric, station_cols, valid_rows)

    capacity_map = _load_capacity_map(urban_dir, station_cols)
    urban_master["station_id"] = urban_master["station_id"].astype(str).str.strip()
    urban_master["total_pile_capacity"] = (
        urban_master["station_id"].map(capacity_map).fillna(config.DEFAULT_PILE_CAPACITY)
    )
    urban_master["total_pile_capacity"] = pd.to_numeric(
        urban_master["total_pile_capacity"], errors="coerce"
    ).fillna(config.DEFAULT_PILE_CAPACITY)
    urban_master["total_pile_capacity"] = urban_master["total_pile_capacity"].clip(lower=1.0)

    for column in ["volume", "occupancy", "duration", "price"]:
        urban_master[column] = pd.to_numeric(urban_master[column], errors="coerce")

    urban_master["volume"] = (
        urban_master["volume"].fillna(0.0).clip(lower=0.0)
        * config.URBAN_VOLUME_TO_KWH_FACTOR
    )
    urban_master["occupancy"] = urban_master["occupancy"].fillna(0.0).clip(lower=0.0)

    median_price = urban_master["price"].median()
    if pd.isna(median_price):
        median_price = config.BASE_TARIFF_INR_PER_KWH * config.ACN_ENERGY_COST_FACTOR
    urban_master["price"] = urban_master["price"].fillna(median_price).clip(lower=0.0)

    urban_master["occupancy_density"] = (
        urban_master["occupancy"] / urban_master["total_pile_capacity"]
    ).clip(0.0, 1.0)
    urban_master["charger_utilization_rate"] = urban_master["occupancy_density"]
    
    # DYNAMIC TRACKING FIX: Base revenue on preprocessed bands rather than flat 15
    urban_hours = urban_master["timestamp"].dt.hour
    urban_master["session_base_tariff"] = urban_hours.apply(get_band_tariff)
    urban_master["revenue_per_session"] = urban_master["volume"] * urban_master["session_base_tariff"]
    
    urban_master["energy_cost_per_kwh"] = urban_master["price"]

    overload = urban_master["occupancy"] - urban_master["total_pile_capacity"] * 0.90
    urban_master["queue_length_proxy"] = overload.clip(lower=0.0)

    return urban_master.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def _add_lag_features(master):
    group_cols = ["context_tag", "station_id"]
    master = master.sort_values(group_cols + ["timestamp"]).reset_index(drop=True)

    master["volume_lag_1h"] = master.groupby(group_cols)["volume_kwh"].shift(1)
    master["volume_lag_2h"] = master.groupby(group_cols)["volume_kwh"].shift(2)
    master["volume_lag_24h"] = master.groupby(group_cols)["volume_kwh"].shift(24)

    master["utilization_lag_1h"] = master.groupby(group_cols)["charger_utilization_rate"].shift(1)
    master["occupancy_lag_1h"] = master.groupby(group_cols)["occupancy_density"].shift(1)
    master["queue_lag_1h"] = master.groupby(group_cols)["queue_length_proxy"].shift(1)

    shifted_volume = master.groupby(group_cols)["volume_kwh"].shift(1)
    master["volume_roll_mean_3h"] = (
        shifted_volume.groupby([master["context_tag"], master["station_id"]])
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    master["volume_roll_mean_24h"] = (
        shifted_volume.groupby([master["context_tag"], master["station_id"]])
        .rolling(24, min_periods=3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    lag_cols = [
        "volume_lag_1h", "volume_lag_2h", "volume_lag_24h",
        "utilization_lag_1h", "occupancy_lag_1h", "queue_lag_1h",
        "volume_roll_mean_3h", "volume_roll_mean_24h"
    ]
    master[lag_cols] = master.groupby(group_cols)[lag_cols].bfill().ffill().fillna(0.0)

    return master


def assemble_unified_hourly_master(acn_df, urban_df):
    print("Building unified hourly training matrix...")

    acn_work = acn_df.copy()
    acn_work["timestamp"] = acn_work["connectionTime"].dt.floor("h")
    acn_hourly = (
        acn_work.groupby(["stationID", "timestamp"], as_index=False)
        .agg(
            volume_kwh=("kWhDelivered", "sum"),
            charger_utilization_rate=("charger_utilization_rate", "mean"),
            revenue_generated=("revenue_per_session", "sum"),
            energy_cost_per_kwh=("energy_cost_per_kwh", "mean"),
        )
    )
    acn_hourly = acn_hourly.rename(columns={"stationID": "station_id"})
    acn_hourly["occupancy_density"] = acn_hourly["charger_utilization_rate"].clip(0.0, 1.0)
    acn_hourly["queue_length_proxy"] = np.where(
        acn_hourly["occupancy_density"] >= 0.85, 1.0, 0.0
    )
    acn_hourly["context_tag"] = "Workplace_ACN"

    urban_work = urban_df.copy()
    urban_work["timestamp"] = urban_work["timestamp"].dt.floor("h")
    urban_hourly = (
        urban_work.groupby(["station_id", "timestamp"], as_index=False)
        .agg(
            volume_kwh=("volume", "sum"),
            charger_utilization_rate=("charger_utilization_rate", "mean"),
            revenue_generated=("revenue_per_session", "sum"),
            energy_cost_per_kwh=("energy_cost_per_kwh", "mean"),
            occupancy_density=("occupancy_density", "mean"),
            queue_length_proxy=("queue_length_proxy", "mean"),
        )
    )
    urban_hourly["context_tag"] = "UrbanPublic_UrbanEV"

    for frame in [acn_hourly, urban_hourly]:
        frame["station_id"] = frame["station_id"].astype(str).str.strip()

    target_ordering = [
        "timestamp",
        "station_id",
        "volume_kwh",
        "charger_utilization_rate",
        "occupancy_density",
        "queue_length_proxy",
        "revenue_generated",
        "energy_cost_per_kwh",
        "context_tag",
    ]
    master = pd.concat(
        [acn_hourly[target_ordering], urban_hourly[target_ordering]],
        ignore_index=True,
    )

    master = master.sort_values(["context_tag", "station_id", "timestamp"]).reset_index(drop=True)
    master["hour_of_day"] = master["timestamp"].dt.hour
    master["day_of_week"] = master["timestamp"].dt.dayofweek
    master["is_weekend"] = (master["day_of_week"] >= 5).astype(int)
    master["scheduling_band"] = master["hour_of_day"].apply(assign_scheduling_band)

    print("Generating leakage-safe historical lag features...")
    master = _add_lag_features(master)
    return master.sort_values(["timestamp", "context_tag", "station_id"]).reset_index(drop=True)


if __name__ == "__main__":
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ACN Raw Reference Path: {config.ACN_RAW_PATH}")
    print(f"UrbanEV Raw Reference Folder: {config.URBAN_RAW_DIR}")

    try:
        acn_cleaned = process_acn_data(config.ACN_RAW_PATH)
        acn_cleaned.to_csv(config.CLEAN_ACN_PATH, index=False)
        print(f"Saved ACN sessions: {config.CLEAN_CLEAN_ACN_PATH if hasattr(config, 'CLEAN_CLEAN_ACN_PATH') else config.CLEAN_ACN_PATH}")

        urban_cleaned = process_urban_ev_data(config.URBAN_RAW_DIR)
        urban_cleaned.to_csv(config.CLEAN_URBAN_PATH, index=False)
        print(f"Saved UrbanEV sessions: {config.CLEAN_URBAN_PATH}")

        print("Building unified hourly master demand matrix...")
        master_matrix = assemble_unified_hourly_master(acn_cleaned, urban_cleaned)
        master_matrix.to_csv(config.UNIFIED_HOURLY_PATH, index=False)
        print(f"✨ Data Preprocessing Phase Complete. Master Matrix Exported: {config.UNIFIED_HOURLY_PATH}")

    except Exception as e:
        print(f"[FATAL] Data Preprocessing pipeline execution halted: {e}")
        sys.exit(1)