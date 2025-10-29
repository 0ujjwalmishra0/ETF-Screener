# train_model.py
import os
import argparse
import pandas as pd
import lightgbm as lgb
import json
from datetime import datetime
from strategies.smart_hybrid_strategy import SmartHybridStrategy
from data_fetch import fetch_etf_data
from ml.hyper_tuner import tune_lightgbm
from ml.walkforward_eval import walk_forward_eval
from config.ticker_config import (
    TICKERS,
    MODEL_DATA_FETCH_PERIOD,
    DEFAULT_INTERVAL,
    MODEL_CONFIG,
    MODEL_PATH,
)



os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ----------------------------------------------------------
def fetch_all_data(tickers, period="2y", interval="1d"):
    """Fetches and combines data for multiple ETFs."""
    dfs = []
    for ticker in tickers:
        try:
            print(f"📥 Fetching data for {ticker} ...")
            df = fetch_etf_data(ticker, period=period, interval=interval)
            df["Ticker"] = ticker
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to fetch {ticker}: {e}")
    if not dfs:
        raise ValueError("No valid ETF data fetched.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Combined dataset size: {len(combined):,} rows from {len(dfs)} tickers.")
    return combined


def train_combined_model(tune=False, walkforward=False):
    """Trains SmartHybridStrategy on multiple ETFs and saves model."""
    df = fetch_all_data(TICKERS, period=MODEL_DATA_FETCH_PERIOD, interval=DEFAULT_INTERVAL)
    strategy = SmartHybridStrategy()

    # Prepare training data
    feats = strategy._prepare_features(df)
    X = strategy._features_for_model(feats)
    y = (feats["FwdRet"] > 0.005).astype(int).fillna(0)

    best_params_path = "models/best_lgb_params.json"
    best_params = None
    auc_score = None

    if os.path.exists(best_params_path):
        print(f"📂 Loading existing tuned parameters from {best_params_path}")
        with open(best_params_path, "r") as f:
            best_params = json.load(f)

    if tune:
        print("\n🔍 Running Optuna hyperparameter tuning ...")
        best_params, auc_score = tune_lightgbm(
            strategy._features_for_model(strategy._prepare_features(df)),
            (strategy._prepare_features(df)["FwdRet"] > 0.005).astype(int),
        )
        print(f"🎯 Best tuned AUC: {auc_score:.4f}")

    if walkforward:
        print("\n📈 Running walk-forward evaluation ...")
        model = lgb.LGBMClassifier(**best_params)
        aucs = walk_forward_eval(model, X, y)
        print(f"📊 Mean walk-forward AUC: {sum(aucs)/len(aucs):.4f}")

    # Final training
    print("\n🏋️ Training final model ...")
    results = strategy.train_model(
        df,
        save_path=MODEL_PATH,
        use_lightgbm=True if "learning_rate" in best_params else False,
    )

    print("\n🎯 Training complete:")
    print(results)

    # Log training details
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tickers_trained": ", ".join(TICKERS),
        "auc": results.get("auc") or auc_score,
        "rows_used": len(df),
        "tuned": tune,
        "walkforward": walkforward,
    }
    log_df = pd.DataFrame([log_entry])
    log_path = "models/training_log.csv"
    log_df.to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))
    print(f"🧾 Logged training details → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartHybrid ETF model.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward validation before final training")
    args = parser.parse_args()

    train_combined_model(tune=args.tune, walkforward=args.walkforward)
