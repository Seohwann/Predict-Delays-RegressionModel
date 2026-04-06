# -*- coding: utf-8 -*-

import argparse
import gc
import itertools
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from utils import (
    TARGET,
    GROUP_COL,
    ensure_dirs,
    build_datasets,
    get_feature_columns,
    fill_missing_for_models,
    convert_cat_for_lgb,
    convert_for_xgb,
    save_pickle,
    save_json,
)


RANDOM_STATE = 42
N_SPLITS = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="models 하위에 사용할 저장 폴더명 (예: exp01 -> ./models/exp01)",
    )
    return parser.parse_args()


def find_best_ensemble_weights(y_true, oof_lgb, oof_cat, oof_xgb, oof_et):
    best_score = 1e18
    best_weights = None

    grid = np.arange(0.0, 1.01, 0.05)

    for w_lgb, w_cat, w_xgb in itertools.product(grid, grid, grid):
        w_et = 1.0 - w_lgb - w_cat - w_xgb
        if w_et < 0 or w_et > 1:
            continue

        pred = w_lgb * oof_lgb + w_cat * oof_cat + w_xgb * oof_xgb + w_et * oof_et
        score = mean_absolute_error(y_true, pred)

        if score < best_score:
            best_score = score
            best_weights = (
                round(float(w_lgb), 3),
                round(float(w_cat), 3),
                round(float(w_xgb), 3),
                round(float(w_et), 3),
            )

    return best_weights, best_score


def main():
    args = parse_args()

    model_dir = os.path.join("./models", args.name) if args.name else "./models"
    output_dir = "./outputs"

    ensure_dirs(models_dir=model_dir, outputs_dir=output_dir)

    train, test, submission = build_datasets()

    feature_cols, cat_cols = get_feature_columns(train)

    X = train[feature_cols].copy()
    y = train[TARGET].copy()
    groups = train[GROUP_COL].copy()
    X_test = test[feature_cols].copy()

    print("train shape:", train.shape)
    print("test shape :", test.shape)
    print("n_features :", len(feature_cols))
    print("cat_cols   :", cat_cols)

    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_et  = np.zeros(len(X))

    pred_lgb = np.zeros(len(X_test))
    pred_cat = np.zeros(len(X_test))
    pred_xgb = np.zeros(len(X_test))
    pred_et  = np.zeros(len(X_test))

    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\n================ Fold {fold} ================")

        X_train = X.iloc[tr_idx].copy()
        X_valid = X.iloc[va_idx].copy()
        y_train = y.iloc[tr_idx].copy()
        y_valid = y.iloc[va_idx].copy()

        X_train, X_valid, X_test_tmp = fill_missing_for_models(X_train, X_valid, X_test.copy(), cat_cols)

        # ----------------------------------
        # LightGBM
        # ----------------------------------
        X_train_lgb = convert_cat_for_lgb(X_train, cat_cols)
        X_valid_lgb = convert_cat_for_lgb(X_valid, cat_cols)
        X_test_lgb = convert_cat_for_lgb(X_test_tmp, cat_cols)

        lgb_model = LGBMRegressor(
            objective="mae",
            n_estimators=6000,
            learning_rate=0.02,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )

        lgb_model.fit(
            X_train_lgb,
            y_train,
            eval_set=[(X_valid_lgb, y_valid)],
            eval_metric="l1",
            callbacks=[
                early_stopping(stopping_rounds=400, verbose=False),
                log_evaluation(period=0),
            ],
        )

        valid_pred_lgb = lgb_model.predict(X_valid_lgb, num_iteration=lgb_model.best_iteration_)
        test_pred_lgb = lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration_)

        lgb_mae = mean_absolute_error(y_valid, valid_pred_lgb)
        print(f"Fold {fold} LGB MAE: {lgb_mae:.6f}")

        oof_lgb[va_idx] = valid_pred_lgb
        pred_lgb += test_pred_lgb / N_SPLITS

        save_pickle(lgb_model, os.path.join(model_dir, f"lgb_fold{fold}.pkl"))

        # ----------------------------------
        # CatBoost
        # ----------------------------------
        cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

        cat_model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=7000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE + fold,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            verbose=0,
        )

        cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_feature_indices,
            use_best_model=True,
            early_stopping_rounds=400,
            verbose=False,
        )

        valid_pred_cat = cat_model.predict(X_valid)
        test_pred_cat = cat_model.predict(X_test_tmp)

        cat_mae = mean_absolute_error(y_valid, valid_pred_cat)
        print(f"Fold {fold} CAT MAE: {cat_mae:.6f}")

        oof_cat[va_idx] = valid_pred_cat
        pred_cat += test_pred_cat / N_SPLITS

        cat_model.save_model(os.path.join(model_dir, f"cat_fold{fold}.cbm"))

        # ----------------------------------
        # XGBoost
        # ----------------------------------
        X_train_xgb, X_valid_xgb, X_test_xgb = convert_for_xgb(
            X_train.copy(), X_valid.copy(), X_test_tmp.copy(), cat_cols
        )

        xgb_model = XGBRegressor(
            objective="reg:absoluteerror",
            eval_metric="mae",
            n_estimators=6000,
            early_stopping_rounds=400,
            learning_rate=0.02,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=2.0,
            gamma=0.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            tree_method="hist",
        )

        xgb_model.fit(
            X_train_xgb,
            y_train,
            eval_set=[(X_valid_xgb, y_valid)],
            verbose=False,
        )

        best_iter = getattr(xgb_model, "best_iteration", None)
        if best_iter is not None and best_iter >= 0:
            valid_pred_xgb = xgb_model.predict(X_valid_xgb, iteration_range=(0, best_iter + 1))
            test_pred_xgb = xgb_model.predict(X_test_xgb, iteration_range=(0, best_iter + 1))
        else:
            valid_pred_xgb = xgb_model.predict(X_valid_xgb)
            test_pred_xgb = xgb_model.predict(X_test_xgb)

        xgb_mae = mean_absolute_error(y_valid, valid_pred_xgb)
        print(f"Fold {fold} XGB MAE: {xgb_mae:.6f}")

        oof_xgb[va_idx] = valid_pred_xgb
        pred_xgb += test_pred_xgb / N_SPLITS

        save_pickle(xgb_model, os.path.join(model_dir, f"xgb_fold{fold}.pkl"))

        # ----------------------------------
        # ExtraTrees
        # sklearn은 NaN 미지원 → train 기준 median으로 수치형 결측 채움
        # ----------------------------------
        num_cols = [c for c in X_train.columns if c not in cat_cols]

        imputer = SimpleImputer(strategy="median")
        X_train_et = X_train.copy()
        X_valid_et = X_valid.copy()
        X_test_et  = X_test_tmp.copy()

        X_train_et[num_cols] = imputer.fit_transform(X_train_et[num_cols])
        X_valid_et[num_cols] = imputer.transform(X_valid_et[num_cols])
        X_test_et[num_cols]  = imputer.transform(X_test_et[num_cols])

        # 범주형 컬럼은 label encoding (category → codes)
        for col in cat_cols:
            all_cats = pd.Categorical(
                pd.concat([X_train_et[col], X_valid_et[col], X_test_et[col]])
            ).categories
            X_train_et[col] = pd.Categorical(X_train_et[col], categories=all_cats).codes
            X_valid_et[col] = pd.Categorical(X_valid_et[col], categories=all_cats).codes
            X_test_et[col]  = pd.Categorical(X_test_et[col],  categories=all_cats).codes

        et_model = ExtraTreesRegressor(
            n_estimators=300,
            criterion="squared_error",
            max_features=0.5,
            min_samples_leaf=10,
            max_depth=None,
            bootstrap=False,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )

        et_model.fit(X_train_et, y_train)

        valid_pred_et = et_model.predict(X_valid_et)
        test_pred_et  = et_model.predict(X_test_et)

        et_mae = mean_absolute_error(y_valid, valid_pred_et)
        print(f"Fold {fold} ET  MAE: {et_mae:.6f}")

        oof_et[va_idx] = valid_pred_et
        pred_et += test_pred_et / N_SPLITS

        save_pickle(et_model, os.path.join(model_dir, f"et_fold{fold}.pkl"))

        fold_scores.append({
            "fold": fold,
            "lgb_mae": float(lgb_mae),
            "cat_mae": float(cat_mae),
            "xgb_mae": float(xgb_mae),
            "et_mae":  float(et_mae),
        })

        del lgb_model, cat_model, xgb_model, et_model
        del X_train, X_valid, y_train, y_valid
        del X_train_lgb, X_valid_lgb, X_test_lgb
        del X_train_xgb, X_valid_xgb, X_test_xgb
        del X_train_et, X_valid_et, X_test_et
        del X_test_tmp
        gc.collect()

    # ----------------------------------
    # OOF 기준 앙상블 weight 탐색
    # ----------------------------------
    best_weights, best_score = find_best_ensemble_weights(y, oof_lgb, oof_cat, oof_xgb, oof_et)
    w_lgb, w_cat, w_xgb, w_et = best_weights

    print("\n========== Ensemble Weight Search ==========")
    print(f"best weights -> LGB: {w_lgb}, CAT: {w_cat}, XGB: {w_xgb}, ET: {w_et}")
    print(f"best OOF MAE -> {best_score:.6f}")

    oof_ens  = w_lgb * oof_lgb  + w_cat * oof_cat  + w_xgb * oof_xgb  + w_et * oof_et
    pred_ens = w_lgb * pred_lgb + w_cat * pred_cat + w_xgb * pred_xgb + w_et * pred_et

    # ----------------------------------
    # OOF 저장
    # ----------------------------------
    oof_df = train[["ID", GROUP_COL]].copy()
    oof_df["target"] = y.values
    oof_df["oof_lgb"] = oof_lgb
    oof_df["oof_cat"] = oof_cat
    oof_df["oof_xgb"] = oof_xgb
    oof_df["oof_et"]  = oof_et
    oof_df["oof_ens"] = oof_ens
    oof_df.to_csv(os.path.join(output_dir, "oof.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------------
    # submission 저장
    # ----------------------------------
    submission[TARGET] = pred_ens
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------------
    # meta 저장
    # ----------------------------------
    meta = {
        "target": TARGET,
        "group_col": GROUP_COL,
        "n_splits": N_SPLITS,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "ensemble_weights": {
            "lgb": w_lgb,
            "cat": w_cat,
            "xgb": w_xgb,
            "et":  w_et,
        },
        "cv_scores": fold_scores,
        "oof_mae_lgb": float(mean_absolute_error(y, oof_lgb)),
        "oof_mae_cat": float(mean_absolute_error(y, oof_cat)),
        "oof_mae_xgb": float(mean_absolute_error(y, oof_xgb)),
        "oof_mae_et":  float(mean_absolute_error(y, oof_et)),
        "oof_mae_ens": float(mean_absolute_error(y, oof_ens)),
    }
    save_json(meta, os.path.join(model_dir, "meta.json"))

    print("\n================ Result ================")
    print(f"OOF LGB MAE : {meta['oof_mae_lgb']:.6f}")
    print(f"OOF CAT MAE : {meta['oof_mae_cat']:.6f}")
    print(f"OOF XGB MAE : {meta['oof_mae_xgb']:.6f}")
    print(f"OOF ET  MAE : {meta['oof_mae_et']:.6f}")
    print(f"OOF ENS MAE : {meta['oof_mae_ens']:.6f}")

    print("\nSaved:")
    print(f"- {os.path.join(output_dir, 'oof.csv')}")
    print(f"- {os.path.join(output_dir, 'submission.csv')}")
    print(f"- {os.path.join(model_dir, 'meta.json')}")
    print(f"- {os.path.join(model_dir, 'lgb_fold*.pkl')}")
    print(f"- {os.path.join(model_dir, 'cat_fold*.cbm')}")
    print(f"- {os.path.join(model_dir, 'xgb_fold*.pkl')}")
    print(f"- {os.path.join(model_dir, 'et_fold*.pkl')}")


if __name__ == "__main__":
    main()