"""
auto_retrain.py

Auto-train SmartHybridStrategy ML model weekly and persist to disk.
Usage:
    python auto_retrain.py --data historical_etf.csv --model models/breakout_model.pkl
"""

import os
import argparse
import json
from datetime import datetime, timedelta

import pandas as pd
from strategies.smart_hybrid_strategy import SmartHybridStrategy

TRAIN_LOG = "models/retrain_log.json"

def load_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def save_log(info, path=TRAIN_LOG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, default=str, indent=2)

def read_log(path=TRAIN_LOG):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def auto_retrain(historical_csv, model_path="models/breakout_model.pkl", force=False, use_lightgbm=True):
    df = load_csv(historical_csv)
    s = SmartHybridStrategy(cfg={"model_path": model_path, "use_ml": True})
    log = read_log()
    last = log.get("last_train_date")
    last_date = pd.to_datetime(last) if last else None
    now = pd.Timestamp.now()

    # If last trained within 7 days and not forced, skip
    if not force and last_date is not None:
        if (now - last_date) < pd.Timedelta(days=7):
            print("Model was trained recently on", last_date)
            return log

    print("Training model on data:", historical_csv)
    try:
        res = s.train_model(df, save_path=model_path, use_lightgbm=use_lightgbm)
        info = {
            "last_train_date": now.isoformat(),
            "model_path": model_path,
            "train_result": res
        }
        save_log(info)
        print("Model trained and saved:", model_path, "res:", res)
        print(f"✅ Weekly retraining done — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return info
    except Exception as e:
        print("Training failed:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="historical csv path")
    p.add_argument("--model", default="models/breakout_model.pkl", help="model path to save")
    p.add_argument("--force", action="store_true", help="force retrain even if recent")
    p.add_argument("--no-lgb", action="store_true", help="do not use LightGBM even if installed")
    args = p.parse_args()

    cfg = {"model_path": args.model, "use_ml": not args.no_lgb}
    info = auto_retrain(args.data, model_path=args.model, force=args.force, use_lightgbm=not args.no_lgb)
    print(info)
