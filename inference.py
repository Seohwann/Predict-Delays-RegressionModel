# -*- coding: utf-8 -*-

import numpy as np
from catboost import CatBoostRegressor

from utils import (
    TARGET,
    ensure_dirs,
    build_datasets,
    load_pickle,
    load_json,
    fill_missing_for_models,
    convert_cat_for_lgb,
    convert_for_xgb,
)


def main():
    ensure_dirs()

    train, test, submission = build_datasets()

    meta = load_json("./models/meta.json")
    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    w_lgb = meta["ensemble_weights"]["lgb"]
    w_cat = meta["ensemble_weights"]["cat"]
    w_xgb = meta["ensemble_weights"]["xgb"]
    n_splits = meta["n_splits"]

    X_train_dummy = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    X_train_dummy, X_valid_dummy, X_test = fill_missing_for_models(
        X_train_dummy.iloc[:10].copy(),
        X_train_dummy.iloc[:10].copy(),
        X_test,
        cat_cols,
    )

    X_test_lgb = convert_cat_for_lgb(X_test.copy(), cat_cols)
    _, _, X_test_xgb = convert_for_xgb(
        X_train_dummy.copy(),
        X_valid_dummy.copy(),
        X_test.copy(),
        cat_cols,
    )

    pred_lgb = np.zeros(len(X_test))
    pred_cat = np.zeros(len(X_test))
    pred_xgb = np.zeros(len(X_test))

    for fold in range(1, n_splits + 1):
        lgb_model = load_pickle(f"./models/lgb_fold{fold}.pkl")
        pred_lgb += lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration_) / n_splits

        cat_model = CatBoostRegressor()
        cat_model.load_model(f"./models/cat_fold{fold}.cbm")
        pred_cat += cat_model.predict(X_test) / n_splits

        xgb_model = load_pickle(f"./models/xgb_fold{fold}.pkl")
        pred_xgb += xgb_model.predict(X_test_xgb) / n_splits

        print(f"loaded and predicted fold {fold}")

    pred_ens = w_lgb * pred_lgb + w_cat * pred_cat + w_xgb * pred_xgb

    submission[TARGET] = pred_ens
    submission.to_csv("./outputs/submission_inference.csv", index=False, encoding="utf-8-sig")

    print("\nSaved: ./outputs/submission_inference.csv")


if __name__ == "__main__":
    main()