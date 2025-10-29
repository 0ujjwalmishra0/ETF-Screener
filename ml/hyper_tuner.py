import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import numpy as np
import json
import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def tune_lightgbm(X, y, n_trials=25, save_path="models/best_lgb_params.json"):
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        }

        tscv = TimeSeriesSplit(n_splits=4)
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.log_evaluation(period=0)]  # disable logging
            )

            preds = model.predict(X_val)
            aucs.append(roc_auc_score(y_val, preds))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    print(f"\n🏆 Best params: {best_params}")
    print(f"📊 Best AUC: {best_score:.4f}")

    # Save best params
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"💾 Saved best parameters to {save_path}")

    return best_params, best_score

