# -*- coding: utf-8 -*-

import os
import gc
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# =========================================================
# 1. 설정
# =========================================================
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "avg_delay_minutes_next_30m"

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
LAYOUT_PATH = "./data/layout_info.csv"
SUBMISSION_PATH = "./data/sample_submission.csv"
OUTPUT_PATH = "./submission_cat_lgb_ensemble.csv"


# =========================================================
# 2. 데이터 로드
# =========================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
layout = pd.read_csv(LAYOUT_PATH)
submission = pd.read_csv(SUBMISSION_PATH)

print("train shape:", train.shape)
print("test shape :", test.shape)
print("layout shape:", layout.shape)


# =========================================================
# 3. layout merge
# =========================================================
train = train.merge(layout, on="layout_id", how="left")
test = test.merge(layout, on="layout_id", how="left")

print("merged train shape:", train.shape)
print("merged test shape :", test.shape)


# =========================================================
# 4. 특징 생성 함수
# =========================================================
def add_time_order_features(df):
    """
    scenario_id 내부의 현재 row 순서를 시간순으로 가정.
    각 scenario는 25개 슬롯으로 구성되어 있으므로 slot_idx 생성.
    """
    df = df.copy()

    # 원래 순서 복원용
    df["_row_order"] = np.arange(len(df))

    # 현재 파일 내 scenario별 row 순서가 시간순이라고 가정
    df["slot_idx"] = df.groupby("scenario_id").cumcount()
    df["slot_from_end"] = 24 - df["slot_idx"]

    df["is_first_slot"] = (df["slot_idx"] == 0).astype(int)
    df["is_last_slot"] = (df["slot_idx"] == 24).astype(int)

    # 주기형 인코딩
    if "shift_hour" in df.columns:
        df["shift_hour_sin"] = np.sin(2 * np.pi * df["shift_hour"].fillna(0) / 24.0)
        df["shift_hour_cos"] = np.cos(2 * np.pi * df["shift_hour"].fillna(0) / 24.0)

    if "day_of_week" in df.columns:
        # 0~6 또는 1~7 어느 경우든 크게 문제 없이 동작
        day = df["day_of_week"].fillna(0)
        df["day_of_week_sin"] = np.sin(2 * np.pi * day / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * day / 7.0)

    return df


def add_ratio_interaction_features(df):
    df = df.copy()
    eps = 1e-6

    # 로봇 총량
    if {"robot_active", "robot_idle", "robot_charging"}.issubset(df.columns):
        df["total_robot_count"] = (
            df["robot_active"].fillna(0)
            + df["robot_idle"].fillna(0)
            + df["robot_charging"].fillna(0)
        )

        df["active_robot_ratio"] = df["robot_active"] / (df["total_robot_count"] + eps)
        df["idle_robot_ratio"] = df["robot_idle"] / (df["total_robot_count"] + eps)
        df["charging_robot_ratio"] = df["robot_charging"] / (df["total_robot_count"] + eps)

    # 부하 / 자원 비율
    if {"order_inflow_15m", "robot_active"}.issubset(df.columns):
        df["orders_per_active_robot"] = df["order_inflow_15m"] / (df["robot_active"] + 1)

    if {"order_inflow_15m", "staff_on_floor"}.issubset(df.columns):
        df["orders_per_staff"] = df["order_inflow_15m"] / (df["staff_on_floor"] + 1)

    if {"order_inflow_15m", "pack_utilization"}.issubset(df.columns):
        df["pack_pressure_index"] = df["order_inflow_15m"] * df["pack_utilization"]

    if {"order_inflow_15m", "loading_dock_util"}.issubset(df.columns):
        df["dock_pressure_index"] = df["order_inflow_15m"] * df["loading_dock_util"]

    if {"order_inflow_15m", "congestion_score"}.issubset(df.columns):
        df["traffic_load_index"] = df["order_inflow_15m"] * df["congestion_score"]

    if {"low_battery_ratio", "robot_utilization"}.issubset(df.columns):
        df["battery_stress_index"] = df["low_battery_ratio"] * df["robot_utilization"]

    if {"warehouse_temp_avg", "humidity_pct"}.issubset(df.columns):
        df["env_risk_index"] = df["warehouse_temp_avg"] * df["humidity_pct"]

    # layout_info와 결합한 비율
    if {"order_inflow_15m", "packing_station_count"}.issubset(df.columns):
        df["orders_per_packing_station"] = df["order_inflow_15m"] / (df["packing_station_count"] + 1)

    if {"total_robot_count", "packing_station_count"}.issubset(df.columns):
        df["robots_per_packing_station"] = df["total_robot_count"] / (df["packing_station_count"] + 1)

    if {"total_robot_count", "charging_station_count"}.issubset(df.columns):
        df["robots_per_charging_station"] = df["total_robot_count"] / (df["charging_station_count"] + 1)

    if {"order_inflow_15m", "total_aisle_length_m"}.issubset(df.columns):
        df["orders_per_aisle_length"] = df["order_inflow_15m"] / (df["total_aisle_length_m"] + 1)

    if {"congestion_score", "intersection_count"}.issubset(df.columns):
        df["congestion_per_intersection"] = df["congestion_score"] / (df["intersection_count"] + 1)

    return df


def add_lag_features(df):
    """
    scenario 내부 시간순 기준으로 과거 정보만 사용.
    현재 row보다 이전 시점 정보만 참조하도록 shift 사용.
    """
    df = df.copy()

    lag_cols = [
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
    ]

    existing_lag_cols = [c for c in lag_cols if c in df.columns]

    for col in existing_lag_cols:
        grp = df.groupby("scenario_id")[col]

        df[f"{col}_lag1"] = grp.shift(1)
        df[f"{col}_lag2"] = grp.shift(2)

        df[f"{col}_diff1"] = df[col] - grp.shift(1)
        df[f"{col}_diff2"] = df[col] - grp.shift(2)

        # 과거 3개 평균 (현재 포함 X)
        df[f"{col}_roll3_mean"] = grp.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
        df[f"{col}_roll3_std"] = grp.shift(1).rolling(3).std().reset_index(level=0, drop=True)

    return df


def make_features(df):
    df = add_time_order_features(df)
    df = add_ratio_interaction_features(df)
    df = add_lag_features(df)

    # 원래 순서 복구
    df = df.sort_values("_row_order").reset_index(drop=True)
    return df


train = make_features(train)
test = make_features(test)

print("feature engineered train shape:", train.shape)
print("feature engineered test shape :", test.shape)


# =========================================================
# 5. 학습용 컬럼 정의
# =========================================================
drop_cols = ["ID", TARGET, "_row_order"]
feature_cols = [c for c in train.columns if c not in drop_cols]

# scenario_id는 fold grouping에는 쓰되, feature에는 넣지 않는 것을 권장
# 이유: high-cardinality ID이고 일반화에 도움 되기보다 과적합 유도 가능
if "scenario_id" in feature_cols:
    feature_cols.remove("scenario_id")

# 범주형 후보
cat_cols = [c for c in ["layout_id", "layout_type", "day_of_week"] if c in feature_cols]

X = train[feature_cols].copy()
y = train[TARGET].copy()
X_test = test[feature_cols].copy()
groups = train["scenario_id"].copy()

print("n_features:", len(feature_cols))
print("categorical columns:", cat_cols)


# =========================================================
# 6. dtype 처리
# =========================================================
# CatBoost / LightGBM 둘 다 범주형 처리 가능하도록 object 유지 또는 category 변환
for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# LightGBM에는 category dtype 권장
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# CatBoost용은 string/object 유지
X_cat = X.copy()
X_test_cat = X_test.copy()


# =========================================================
# 7. CV + 모델 학습
# =========================================================
gkf = GroupKFold(n_splits=N_SPLITS)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_ens = np.zeros(len(X))

pred_lgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))
pred_ens = np.zeros(len(X_test))

lgb_scores = []
cat_scores = []
ens_scores = []

for fold, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    print(f"\n==================== Fold {fold} ====================")

    X_train_lgb = X_lgb.iloc[train_idx].copy()
    X_valid_lgb = X_lgb.iloc[valid_idx].copy()

    X_train_cat = X_cat.iloc[train_idx].copy()
    X_valid_cat = X_cat.iloc[valid_idx].copy()

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # -------------------------
    # LightGBM
    # -------------------------
    lgb_model = LGBMRegressor(
        objective="mae",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
    )

    lgb_model.fit(
        X_train_lgb,
        y_train,
        eval_set=[(X_valid_lgb, y_valid)],
        eval_metric="l1",
        callbacks=[
            early_stopping(stopping_rounds=300, verbose=False),
            log_evaluation(period=0),
        ],
    )

    valid_pred_lgb = lgb_model.predict(X_valid_lgb, num_iteration=lgb_model.best_iteration_)
    test_pred_lgb = lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration_)

    fold_mae_lgb = mean_absolute_error(y_valid, valid_pred_lgb)
    lgb_scores.append(fold_mae_lgb)
    print(f"Fold {fold} LGB MAE: {fold_mae_lgb:.6f}")

    oof_lgb[valid_idx] = valid_pred_lgb
    pred_lgb += test_pred_lgb / N_SPLITS

    # -------------------------
    # CatBoost
    # -------------------------
    cat_feature_indices = [X_train_cat.columns.get_loc(col) for col in cat_cols]

    cat_model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=RANDOM_STATE + fold,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        verbose=0,
    )

    cat_model.fit(
        X_train_cat,
        y_train,
        eval_set=(X_valid_cat, y_valid),
        cat_features=cat_feature_indices,
        use_best_model=True,
        early_stopping_rounds=300,
        verbose=False,
    )

    valid_pred_cat = cat_model.predict(X_valid_cat)
    test_pred_cat = cat_model.predict(X_test_cat)

    fold_mae_cat = mean_absolute_error(y_valid, valid_pred_cat)
    cat_scores.append(fold_mae_cat)
    print(f"Fold {fold} CAT MAE: {fold_mae_cat:.6f}")

    oof_cat[valid_idx] = valid_pred_cat
    pred_cat += test_pred_cat / N_SPLITS

    # -------------------------
    # Ensemble
    # -------------------------
    # 기본은 0.5 / 0.5
    # 보통 CatBoost가 missing / categorical에 강해서
    # 필요시 0.6*Cat + 0.4*LGB로 바꿔볼 수 있음
    valid_pred_ens = 0.5 * valid_pred_lgb + 0.5 * valid_pred_cat
    test_pred_ens = 0.5 * test_pred_lgb + 0.5 * test_pred_cat

    fold_mae_ens = mean_absolute_error(y_valid, valid_pred_ens)
    ens_scores.append(fold_mae_ens)
    print(f"Fold {fold} ENS MAE: {fold_mae_ens:.6f}")

    oof_ens[valid_idx] = valid_pred_ens
    pred_ens += test_pred_ens / N_SPLITS

    del lgb_model, cat_model
    del X_train_lgb, X_valid_lgb, X_train_cat, X_valid_cat
    del y_train, y_valid
    gc.collect()


# =========================================================
# 8. OOF 성능 출력
# =========================================================
print("\n==================== CV Result ====================")
print(f"LGB CV MAE: {np.mean(lgb_scores):.6f}")
print(f"CAT CV MAE: {np.mean(cat_scores):.6f}")
print(f"ENS CV MAE: {np.mean(ens_scores):.6f}")

print("\nOOF MAE")
print(f"LGB OOF MAE: {mean_absolute_error(y, oof_lgb):.6f}")
print(f"CAT OOF MAE: {mean_absolute_error(y, oof_cat):.6f}")
print(f"ENS OOF MAE: {mean_absolute_error(y, oof_ens):.6f}")


# =========================================================
# 9. 제출 파일 생성
# =========================================================
# 앙상블 결과 사용
submission[TARGET] = pred_ens
submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\nSaved submission to: {OUTPUT_PATH}")