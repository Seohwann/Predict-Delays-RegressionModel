# -*- coding: utf-8 -*-

import os
import json
import pickle
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET = "avg_delay_minutes_next_30m"
ID_COL = "ID"
GROUP_COL = "scenario_id"
LAYOUT_KEY = "layout_id"


def ensure_dirs(models_dir: str = "./models", outputs_dir: str = "./outputs"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def detect_existing_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def load_data():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    layout = pd.read_csv("./data/layout_info.csv")
    submission = pd.read_csv("./data/sample_submission.csv")
    return train, test, layout, submission


def merge_layout(train: pd.DataFrame, test: pd.DataFrame, layout: pd.DataFrame):
    if LAYOUT_KEY in train.columns and LAYOUT_KEY in test.columns and LAYOUT_KEY in layout.columns:
        train = train.merge(layout, on=LAYOUT_KEY, how="left")
        test = test.merge(layout, on=LAYOUT_KEY, how="left")
    return train, test


def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_row_order"] = np.arange(len(df))

    if GROUP_COL in df.columns:
        df["slot_idx"] = df.groupby(GROUP_COL).cumcount()
        df["slot_from_end"] = 24 - df["slot_idx"]
        df["is_first_slot"] = (df["slot_idx"] == 0).astype(int)
        df["is_last_slot"] = (df["slot_idx"] == 24).astype(int)
        df["slot_progress"] = df["slot_idx"] / 24.0
        df["slot_progress_sq"] = df["slot_progress"] ** 2
        df["slot_mid_distance"] = np.abs(df["slot_idx"] - 12)
        df["slot_bin_early"] = (df["slot_idx"] <= 7).astype(int)
        df["slot_bin_mid"] = ((df["slot_idx"] >= 8) & (df["slot_idx"] <= 16)).astype(int)
        df["slot_bin_late"] = (df["slot_idx"] >= 17).astype(int)
    else:
        df["slot_idx"] = np.arange(len(df))
        df["slot_from_end"] = 24 - df["slot_idx"]
        df["is_first_slot"] = 0
        df["is_last_slot"] = 0
        df["slot_progress"] = 0.0
        df["slot_progress_sq"] = 0.0
        df["slot_mid_distance"] = 0
        df["slot_bin_early"] = 0
        df["slot_bin_mid"] = 0
        df["slot_bin_late"] = 0

    if "shift_hour" in df.columns:
        x = df["shift_hour"].fillna(0)
        df["shift_hour_sin"] = np.sin(2 * np.pi * x / 24.0)
        df["shift_hour_cos"] = np.cos(2 * np.pi * x / 24.0)

    if "day_of_week" in df.columns:
        x = df["day_of_week"].fillna(0)
        df["day_of_week_sin"] = np.sin(2 * np.pi * x / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * x / 7.0)

    return df


def add_robot_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    robot_cols = detect_existing_columns(df, ["robot_active", "robot_idle", "robot_charging"])
    if len(robot_cols) >= 2:
        df["total_robot_count"] = df[robot_cols].fillna(0).sum(axis=1)

    if {"robot_active", "total_robot_count"}.issubset(df.columns):
        df["active_robot_ratio"] = safe_div(df["robot_active"], df["total_robot_count"])
    if {"robot_idle", "total_robot_count"}.issubset(df.columns):
        df["idle_robot_ratio"] = safe_div(df["robot_idle"], df["total_robot_count"])
    if {"robot_charging", "total_robot_count"}.issubset(df.columns):
        df["charging_robot_ratio"] = safe_div(df["robot_charging"], df["total_robot_count"])

    if {"order_inflow_15m", "robot_active"}.issubset(df.columns):
        df["orders_per_active_robot"] = safe_div(df["order_inflow_15m"], df["robot_active"] + 1)

    if {"order_inflow_15m", "total_robot_count"}.issubset(df.columns):
        df["orders_per_total_robot"] = safe_div(df["order_inflow_15m"], df["total_robot_count"] + 1)

    if {"robot_utilization", "total_robot_count"}.issubset(df.columns):
        df["util_robot_interaction"] = df["robot_utilization"] * df["total_robot_count"]

    if {"total_robot_count", "charging_station_count"}.issubset(df.columns):
        df["robots_per_charging_station"] = safe_div(df["total_robot_count"], df["charging_station_count"] + 1)

    if {"total_robot_count", "packing_station_count"}.issubset(df.columns):
        df["robots_per_packing_station"] = safe_div(df["total_robot_count"], df["packing_station_count"] + 1)

    return df


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"order_inflow_15m", "staff_on_floor"}.issubset(df.columns):
        df["orders_per_staff"] = safe_div(df["order_inflow_15m"], df["staff_on_floor"] + 1)

    if {"order_inflow_15m", "pack_utilization"}.issubset(df.columns):
        df["pack_pressure_index"] = df["order_inflow_15m"] * df["pack_utilization"]

    if {"order_inflow_15m", "loading_dock_util"}.issubset(df.columns):
        df["dock_pressure_index"] = df["order_inflow_15m"] * df["loading_dock_util"]

    if {"order_inflow_15m", "congestion_score"}.issubset(df.columns):
        df["traffic_load_index"] = df["order_inflow_15m"] * df["congestion_score"]

    if {"low_battery_ratio", "robot_utilization"}.issubset(df.columns):
        df["battery_stress_index"] = df["low_battery_ratio"] * df["robot_utilization"]

    if {"order_inflow_15m", "packing_station_count"}.issubset(df.columns):
        df["orders_per_packing_station"] = safe_div(df["order_inflow_15m"], df["packing_station_count"] + 1)

    if {"order_inflow_15m", "charging_station_count"}.issubset(df.columns):
        df["orders_per_charging_station"] = safe_div(df["order_inflow_15m"], df["charging_station_count"] + 1)

    if {"order_inflow_15m", "intersection_count"}.issubset(df.columns):
        df["orders_per_intersection"] = safe_div(df["order_inflow_15m"], df["intersection_count"] + 1)

    if {"order_inflow_15m", "total_aisle_length_m"}.issubset(df.columns):
        df["orders_per_aisle_length"] = safe_div(df["order_inflow_15m"], df["total_aisle_length_m"] + 1)

    if {"congestion_score", "intersection_count"}.issubset(df.columns):
        df["congestion_per_intersection"] = safe_div(df["congestion_score"], df["intersection_count"] + 1)

    if {"congestion_score", "total_aisle_length_m"}.issubset(df.columns):
        df["congestion_per_aisle_length"] = safe_div(df["congestion_score"], df["total_aisle_length_m"] + 1)

    if {"pack_utilization", "loading_dock_util"}.issubset(df.columns):
        df["pack_dock_interaction"] = df["pack_utilization"] * df["loading_dock_util"]

    if {"congestion_score", "robot_utilization"}.issubset(df.columns):
        df["congestion_robot_interaction"] = df["congestion_score"] * df["robot_utilization"]

    if {"low_battery_ratio", "order_inflow_15m"}.issubset(df.columns):
        df["battery_order_interaction"] = df["low_battery_ratio"] * df["order_inflow_15m"]

    if {"backorder_ratio", "order_inflow_15m"}.issubset(df.columns):
        df["backorder_load_interaction"] = df["backorder_ratio"] * df["order_inflow_15m"]

    if {"order_inflow_15m", "congestion_score", "pack_utilization"}.issubset(df.columns):
        df["triple_pressure_index"] = (
            df["order_inflow_15m"] * df["congestion_score"] * df["pack_utilization"]
        )

    if {"order_inflow_15m", "low_battery_ratio", "robot_utilization"}.issubset(df.columns):
        df["battery_load_util_index"] = (
            df["order_inflow_15m"] * df["low_battery_ratio"] * df["robot_utilization"]
        )

    if {"congestion_score", "loading_dock_util", "pack_utilization"}.issubset(df.columns):
        df["process_bottleneck_index"] = (
            df["congestion_score"] * df["loading_dock_util"] * df["pack_utilization"]
        )

    return df


def add_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"warehouse_temp_avg", "humidity_pct"}.issubset(df.columns):
        df["env_risk_index"] = df["warehouse_temp_avg"] * df["humidity_pct"]

    if {"wms_response_time_ms", "network_latency_ms"}.issubset(df.columns):
        df["system_latency_sum"] = df["wms_response_time_ms"] + df["network_latency_ms"]
        df["system_latency_mul"] = df["wms_response_time_ms"] * df["network_latency_ms"]

    if "order_inflow_15m" in df.columns:
        df["log_order_inflow_15m"] = np.log1p(df["order_inflow_15m"].clip(lower=0))

    for c in ["wms_response_time_ms", "network_latency_ms", "outbound_truck_wait_min"]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if GROUP_COL not in df.columns:
        return df

    lag_candidates = [
        "order_inflow_15m",
        "robot_utilization",
        "battery_mean",
        "low_battery_ratio",
        "congestion_score",
        "pack_utilization",
        "loading_dock_util",
        "wms_response_time_ms",
        "network_latency_ms",
        "outbound_truck_wait_min",
        "backorder_ratio",
        "sort_accuracy_pct",
        "robot_active",
        "robot_idle",
        "robot_charging",
    ]
    lag_cols = detect_existing_columns(df, lag_candidates)

    for col in lag_cols:
        grp = df.groupby(GROUP_COL)[col]

        df[f"{col}_lag1"] = grp.shift(1)
        df[f"{col}_lag2"] = grp.shift(2)

        df[f"{col}_diff1"] = df[col] - grp.shift(1)
        df[f"{col}_diff2"] = df[col] - grp.shift(2)

        lag1 = grp.shift(1)
        lag2 = grp.shift(2)

        df[f"{col}_diff1_ratio"] = safe_div(df[col] - lag1, lag1.abs() + 1.0)
        df[f"{col}_diff2_ratio"] = safe_div(df[col] - lag2, lag2.abs() + 1.0)

        roll3 = grp.shift(1).rolling(3)
        roll5 = grp.shift(1).rolling(5)

        df[f"{col}_roll3_mean"] = roll3.mean().reset_index(level=0, drop=True)
        df[f"{col}_roll3_std"] = roll3.std().reset_index(level=0, drop=True)
        df[f"{col}_roll3_min"] = roll3.min().reset_index(level=0, drop=True)
        df[f"{col}_roll3_max"] = roll3.max().reset_index(level=0, drop=True)

        df[f"{col}_roll5_mean"] = roll5.mean().reset_index(level=0, drop=True)
        df[f"{col}_roll5_std"] = roll5.std().reset_index(level=0, drop=True)

        df[f"{col}_ratio_to_roll3_mean"] = safe_div(df[col], df[f"{col}_roll3_mean"] + 1.0)
        df[f"{col}_ratio_to_roll5_mean"] = safe_div(df[col], df[f"{col}_roll5_mean"] + 1.0)

    return df


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if GROUP_COL not in df.columns:
        return df

    cum_candidates = [
        "order_inflow_15m",
        "congestion_score",
        "pack_utilization",
        "loading_dock_util",
        "robot_utilization",
        "low_battery_ratio",
    ]
    cum_cols = detect_existing_columns(df, cum_candidates)

    for col in cum_cols:
        grp = df.groupby(GROUP_COL)[col]

        df[f"{col}_cum_sum"] = grp.cumsum()
        df[f"{col}_cum_mean"] = grp.expanding().mean().reset_index(level=0, drop=True)
        df[f"{col}_cum_max"] = grp.cummax()
        df[f"{col}_cum_min"] = grp.cummin()

    return df


def add_group_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if GROUP_COL not in df.columns:
        return df

    rel_candidates = [
        "order_inflow_15m",
        "congestion_score",
        "pack_utilization",
        "loading_dock_util",
        "robot_utilization",
        "low_battery_ratio",
    ]
    rel_cols = detect_existing_columns(df, rel_candidates)

    for col in rel_cols:
        grp = df.groupby(GROUP_COL)[col]
        grp_mean = grp.transform("mean")
        grp_std = grp.transform("std")
        grp_max = grp.transform("max")
        grp_min = grp.transform("min")

        df[f"{col}_grp_mean_ratio"] = safe_div(df[col], grp_mean + 1.0)
        df[f"{col}_grp_zscore"] = safe_div(df[col] - grp_mean, grp_std + 1.0)
        df[f"{col}_grp_max_ratio"] = safe_div(df[col], grp_max + 1.0)
        df[f"{col}_grp_min_diff"] = df[col] - grp_min

    return df


def add_slot_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    key_cols = [
        "congestion_score",
        "pack_utilization",
        "loading_dock_util",
        "robot_utilization",
        "low_battery_ratio",
        "order_inflow_15m",
    ]

    for col in key_cols:
        if col in df.columns:
            df[f"slot_x_{col}"] = df["slot_idx"] * df[col]
            df[f"slot_progress_x_{col}"] = df["slot_progress"] * df[col]
            df[f"late_slot_x_{col}"] = df["slot_bin_late"] * df[col]

    return df


def reduce_memory_and_fix_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            nunique = df[col].nunique(dropna=False)
            if nunique < max(50, len(df) * 0.05):
                df[col] = df[col].astype("category")

        elif str(df[col].dtype).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        elif str(df[col].dtype).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_basic_time_features(df)
    df = add_robot_features(df)
    df = add_pressure_features(df)
    df = add_environment_features(df)
    df = add_lag_features(df)
    df = add_cumulative_features(df)
    df = add_group_relative_features(df)
    df = add_slot_interaction_features(df)

    if "_row_order" in df.columns:
        df = df.sort_values("_row_order").reset_index(drop=True)

    df = reduce_memory_and_fix_types(df)
    return df


def build_datasets():
    train, test, layout, submission = load_data()
    train, test = merge_layout(train, test, layout)
    train = make_features(train)
    test = make_features(test)
    return train, test, submission


def get_feature_columns(train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    drop_cols = [c for c in [ID_COL, TARGET, "_row_order"] if c in train.columns]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    if GROUP_COL in feature_cols:
        feature_cols.remove(GROUP_COL)

    # 너무 잡음이 될 수 있는 std 계열은 기본적으로 제외
    drop_feature_keywords = [
        "_roll5_std",
        "_roll3_std",
    ]
    feature_cols = [
        c for c in feature_cols
        if not any(k in c for k in drop_feature_keywords)
    ]

    cat_candidates = [
        "layout_id",
        "layout_type",
        "day_of_week",
        "zone_type",
        "shift_type",
    ]
    cat_cols = [c for c in cat_candidates if c in feature_cols]

    return feature_cols, cat_cols


def fill_missing_for_models(train_x: pd.DataFrame, valid_x: pd.DataFrame, test_x: pd.DataFrame, cat_cols: List[str]):
    train_x = train_x.copy()
    valid_x = valid_x.copy()
    test_x = test_x.copy()

    for col in train_x.columns:
        if col in cat_cols:
            train_x[col] = train_x[col].astype(str).fillna("missing")
            valid_x[col] = valid_x[col].astype(str).fillna("missing")
            test_x[col] = test_x[col].astype(str).fillna("missing")

    return train_x, valid_x, test_x


def convert_cat_for_lgb(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def convert_for_xgb(train_x: pd.DataFrame, valid_x: pd.DataFrame, test_x: pd.DataFrame, cat_cols: List[str]):
    train_x = train_x.copy()
    valid_x = valid_x.copy()
    test_x = test_x.copy()

    for col in cat_cols:
        train_x[col] = train_x[col].astype("category")
        valid_x[col] = valid_x[col].astype("category")
        test_x[col] = test_x[col].astype("category")

        all_cats = pd.Index(train_x[col].cat.categories)\
            .union(valid_x[col].cat.categories)\
            .union(test_x[col].cat.categories)

        train_x[col] = train_x[col].cat.set_categories(all_cats).cat.codes
        valid_x[col] = valid_x[col].cat.set_categories(all_cats).cat.codes
        test_x[col] = test_x[col].cat.set_categories(all_cats).cat.codes

    return train_x, valid_x, test_x


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)