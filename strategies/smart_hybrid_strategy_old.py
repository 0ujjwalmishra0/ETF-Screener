# # strategies/smart_hybrid_strategy.py
# import os
# import math
# import numpy as np
# import pandas as pd
# import joblib
# from .base_strategy import BaseStrategy
#
#
# # Optional ML imports
# try:
#     import lightgbm as lgb
#
#     LGB_INSTALLED = True
# except Exception:
#     LGB_INSTALLED = False
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, accuracy_score
# from  lightgbm import early_stopping, log_evaluation
#
# # small safe divide
# def safe_div(a, b, default=0.0):
#     try:
#         return a / b if b not in (0, None, np.nan) else default
#     except Exception:
#         return default
#
# class SmartHybridStrategy(BaseStrategy):
#     """
#     Smart hybrid strategy that combines:
#       1) Technical Intelligence Layer (rich features),
#       2) Adaptive Scoring Engine (short-term correlation weights),
#       3) Pattern Recognition (squeeze, breakout, divergence),
#       4) Predictive Layer (ML model producing breakout/up probability).
#     """
#
#     def __init__(self, cfg=None, model_path="models/breakout_model.pkl"):
#         defaults = {
#             "forward_horizon": 5,
#             "corr_window": 90,
#             "min_history": 90,
#             "squeeze_bb_width": 0.06,
#             "volume_z_thresh": 1.2,
#             "atr_period": 14,
#             "bb_period": 20,
#             "rsi_period": 14,
#             "obv_window": 10,
#             "model_path": model_path,
#             "use_ml": True,
#         }
#         self.cfg = {**defaults, **(cfg or {})}
#         self.model = None
#         self.model_path = self.cfg["model_path"]
#         # Try to load model if available
#         if self.cfg["use_ml"]:
#             if os.path.exists(self.model_path):
#                 try:
#                     self.model = joblib.load(self.model_path)
#                 except Exception:
#                     self.model = None
#
#     # -------------------------
#     # Feature engineering
#     # -------------------------
#     def _prepare_features(self, df_in):
#         df = df_in.copy().reset_index(drop=True)
#         if df is None or len(df) < 50:
#             return pd.DataFrame()
#
#         df = df.copy()
#         # --- Base cleanup ---
#         df = df.sort_index()
#         df = df.ffill().bfill()
#
#         # basic moving averages
#         df["50DMA"] = df["Close"].rolling(50, min_periods=1).mean()
#         df["200DMA"] = df["Close"].rolling(200, min_periods=1).mean()
#         df["20DMA"] = df["Close"].rolling(20, min_periods=1).mean()
#
#         # RSI
#         delta = df["Close"].diff().fillna(0)
#         up = delta.clip(lower=0)
#         down = -delta.clip(upper=0)
#         avg_up = up.rolling(self.cfg["rsi_period"], min_periods=1).mean()
#         avg_down = down.rolling(self.cfg["rsi_period"], min_periods=1).mean()
#         rs = safe_div(avg_up, avg_down, np.nan)
#         df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
#
#         # MACD + signal + hist
#         exp12 = df["Close"].ewm(span=12, adjust=False).mean()
#         exp26 = df["Close"].ewm(span=26, adjust=False).mean()
#         df["MACD"] = exp12 - exp26
#         df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
#         df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
#
#         # ATR
#         tr1 = df["High"] - df["Low"]
#         tr2 = (df["High"] - df["Close"].shift()).abs()
#         tr3 = (df["Low"] - df["Close"].shift()).abs()
#         tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#         df["ATR"] = tr.rolling(self.cfg["atr_period"], min_periods=1).mean()
#         df["ATR_Pct"] = safe_div(df["ATR"], df["Close"], 0) * 100
#
#         # Bollinger bands and width / %B
#         ma = df["Close"].rolling(self.cfg["bb_period"], min_periods=1).mean()
#         std = df["Close"].rolling(self.cfg["bb_period"], min_periods=1).std().fillna(0)
#         df["BB_Upper"] = ma + 2 * std
#         df["BB_Lower"] = ma - 2 * std
#         df["BB_Width"] = safe_div(df["BB_Upper"] - df["BB_Lower"], ma, 0)
#         df["PctB"] = safe_div(df["Close"] - df["BB_Lower"], df["BB_Upper"] - df["BB_Lower"], 0)
#
#         # OBV and slope
#         obv = (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).fillna(0).cumsum()
#         df["OBV"] = obv
#         df["OBV_Slope"] = obv.diff().rolling(self.cfg["obv_window"], min_periods=1).mean()
#
#         # Volume z-score (30-day)
#         vol_mean = df["Volume"].rolling(30, min_periods=1).mean()
#         vol_std = df["Volume"].rolling(30, min_periods=1).std().replace(0, np.nan).fillna(1)
#         df["Vol_Z"] = safe_div(df["Volume"] - vol_mean, vol_std, 0)
#
#         # price z
#         recent_mean = df["Close"].rolling(50, min_periods=1).mean()
#         recent_std = df["Close"].rolling(50, min_periods=1).std().replace(0, np.nan).fillna(1)
#         df["Price_Z"] = safe_div(df["Close"] - recent_mean, recent_std, 0)
#
#         # MA slope
#         df["MA50_Slope"] = df["50DMA"].diff().rolling(5, min_periods=1).mean()
#
#         # rolling highs/lows
#         df["Rolling_Max_20"] = df["High"].rolling(20, min_periods=1).max()
#         df["Rolling_Min_20"] = df["Low"].rolling(20, min_periods=1).min()
#
#         # forward return (for adaptive weights)
#         fh = self.cfg["forward_horizon"]
#         df["FwdRet"] = df["Close"].shift(-fh) / df["Close"] - 1.0
#
#         # small extras
#         df["MA50_above_200"] = (df["50DMA"] > df["200DMA"]).astype(int)
#         # --- Drop early NaNs ---
#         df = df.dropna().copy()
#         df = df.replace([np.inf, -np.inf], 0).fillna(0)
#
#         return df
#
#     # -------------------------
#     # Pattern detectors
#     # -------------------------
#     def _is_squeeze(self, row):
#         return row.get("BB_Width", 1.0) < self.cfg["squeeze_bb_width"]
#
#     def _is_breakout(self, row):
#         # breakout: price above recent high and volume z high
#         return (row["Close"] > row.get("Rolling_Max_20", row["Close"]) * 1.002) and (row.get("Vol_Z", 0) >= self.cfg["volume_z_thresh"])
#
#     def _rsi_divergence_score(self, df, idx):
#         # small heuristic for divergence: compare recent peaks of price vs RSI
#         if idx < 6:
#             return 0
#         window = df.iloc[max(0, idx-8): idx+1]
#         ph = window["Close"]
#         rh = window["RSI"]
#         # price makes higher high but RSI lower high -> bearish divergence
#         if ph.iloc[-1] > ph.max() and rh.iloc[-1] < rh.max():
#             return -1
#         # price makes lower low but RSI higher low -> bullish divergence
#         if ph.iloc[-1] < ph.min() and rh.iloc[-1] > rh.min():
#             return 1
#         return 0
#
#     # -------------------------
#     # Adaptive weights via short-term correlation
#     # -------------------------
#     def _adaptive_weights(self, feats):
#         window = self.cfg["corr_window"]
#         tail = feats.tail(window).dropna(subset=["FwdRet"])
#         features = {
#             "RSI": lambda x: (35 - x)/10.0,
#             "MACD_Hist": lambda x: x,
#             "MA50_Slope": lambda x: x,
#             "OBV_Slope": lambda x: x,
#             "Vol_Z": lambda x: x,
#             "BB_Width": lambda x: -x,
#             "PctB": lambda x: x,
#             "Price_Z": lambda x: -x,
#             "ATR_Pct": lambda x: -x,
#         }
#         if tail.empty or len(tail) < max(12, int(window/4)):
#             # fallback base weights
#             base = {"RSI":1.0, "MACD_Hist":1.0, "MA50_Slope":1.0, "OBV_Slope":0.8, "Vol_Z":0.8,
#                     "BB_Width":0.8, "PctB":0.6, "Price_Z":0.8, "ATR_Pct":0.6}
#             s = sum(abs(v) for v in base.values()) or 1.0
#             return {k: v/s for k,v in base.items()}
#
#         corrs = {}
#         for k, fn in features.items():
#             try:
#                 ser = tail[k].map(fn)
#                 corr = ser.corr(tail["FwdRet"])
#                 corrs[k] = 0.0 if np.isnan(corr) else float(corr)
#             except Exception:
#                 corrs[k] = 0.0
#
#         # scale & clip
#         raw = {k: np.sign(v) * min(abs(v), 0.6) for k, v in corrs.items()}
#         total = sum(abs(v) for v in raw.values()) or 1.0
#         weights = {k: raw[k]/total for k in raw}
#         return weights
#
#     # -------------------------
#     # ML helper: build feature matrix from feats df
#     # -------------------------
#     def _features_for_model(self, feats):
#         cols = ["RSI","MACD_Hist","MA50_Slope","OBV_Slope","Vol_Z","BB_Width","PctB","Price_Z","ATR_Pct","MA50_above_200"]
#         X = feats[cols].fillna(0).astype(float)
#         return X
#
#     def train_model(self, historical_df, save_path=None, use_lightgbm=True):
#         """
#         Train a model for predicting forward up (1) vs not (0).
#         historical_df must be a DataFrame with OHLCV and enough history.
#         """
#         feats = self._prepare_features(historical_df)
#         X = self._features_for_model(feats)
#         # target: next forward_horizon return > threshold
#         fh = self.cfg["forward_horizon"]
#         feats["target"] = (feats["FwdRet"] > 0.005).astype(int)  # threshold: 0.5% in fh days
#         y = feats["target"].fillna(0).astype(int)
#
#         # drop last fh rows that lack target
#         valid = ~y.isna()
#         Xv = X[valid]
#         yv = y[valid]
#
#         if len(Xv) < 50:
#             raise ValueError("Not enough training rows to train model")
#
#         X_train, X_test, y_train, y_test = train_test_split(Xv, yv, test_size=0.2, random_state=42, stratify=yv)
#
#         if use_lightgbm and LGB_INSTALLED:
#             params = {
#                 "objective": "binary",
#                 "metric": "auc",
#                 "verbosity": -1,
#                 "learning_rate": 0.01,
#                 "num_leaves": 31,
#                 "max_depth": -1,
#                 "n_estimators": 500
#             }
#             model = lgb.LGBMClassifier(**params)
#             model.fit(
#                 X_train,
#                 y_train,
#                 eval_set=[(X_test, y_test)],
#                 callbacks=[
#                     early_stopping(20),
#                     log_evaluation(-1)
#                 ]
#             )
#             self.model = model
#         else:
#             rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
#             rf.fit(X_train, y_train)
#             self.model = rf
#
#         # evaluate
#         probs = self.model.predict_proba(X_test)[:,1] if hasattr(self.model, "predict_proba") else self.model.predict(X_test)
#         auc = roc_auc_score(y_test, probs)
#         path = save_path or self.model_path
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         joblib.dump(self.model, path)
#         return {"auc": auc, "model_path": path}
#
#     # -------------------------
#     # Main evaluate method
#     # -------------------------
#     def evaluate(self, df_in):
#         """
#         Returns:
#           signal (str),
#           confidence (int 0..100),
#           trend (str),
#           score (float),
#           breakdown (dict of component contributions),
#           model_prob (float 0..1)  # probability from ML model if available else None
#         """
#         df = df_in.copy().reset_index(drop=True)
#         n = len(df)
#         if n < 30:
#             # fallback simple rule
#             last = df.iloc[-1]
#             score = 0.0
#             if last["Close"] > last["Close"].rolling(5).mean().iloc[-1]:
#                 score += 1.0
#             if last["Close"] > last["Close"].rolling(20).mean().iloc[-1]:
#                 score += 0.5
#             signal = "BUY" if score > 0.5 else "WAIT"
#             return signal, 50, "➡️ Side", round(score,3), {}, None
#
#         feats = self._prepare_features(df)
#         # Fill missing values from rolling indicators
#         feats = feats.fillna(method="ffill").fillna(method="bfill").fillna(0)
#
#         last = feats.iloc[-1]
#         idx = len(feats) - 1
#
#         # adaptive weights
#         weights = self._adaptive_weights(feats)
#
#         # compute indicator contributions
#         contrib = {}
#         # RSI contribution: lower RSI (below 35) positive
#         rsi_val = last.get("RSI", 50)
#         contrib["RSI"] = (35 - rsi_val) / 15.0
#
#         # MACD hist normalized
#         macdh = last.get("MACD_Hist", 0.0)
#         contrib["MACD_Hist"] = safe_div(macdh, (abs(macdh) + 1e-9), 1.0)
#
#         # MA50 slope
#         contrib["MA50_Slope"] = np.tanh(last.get("MA50_Slope", 0.0) * 5.0)
#
#         # OBV slope
#         contrib["OBV_Slope"] = np.tanh(last.get("OBV_Slope", 0.0) / (1 + abs(last.get("OBV_Slope", 0.0))))
#
#         # Vol Z
#         contrib["Vol_Z"] = safe_div(last.get("Vol_Z", 0.0), (1 + abs(last.get("Vol_Z", 0.0))), 0.0)
#
#         # BB width (lower => good for breakout)
#         contrib["BB_Width"] = -last.get("BB_Width", 0.1)
#
#         # PctB
#         contrib["PctB"] = (last.get("PctB", 0.5) - 0.5) * 2.0
#
#         # Price Z
#         contrib["Price_Z"] = -safe_div(last.get("Price_Z", 0.0), 1.0, 0.0)
#
#         # ATR pct
#         contrib["ATR_Pct"] = -safe_div(last.get("ATR_Pct", 0.0), 10.0, 0.0)
#
#         # combine with weights
#         score = 0.0
#         breakdown = {}
#         for k, v in contrib.items():
#             w = weights.get(k, 0.0)
#             part = float(v) * float(w)
#             score += part
#             breakdown[k] = {"value": float(v), "weight": float(w), "contribution": round(part,4)}
#
#         # pattern recognition boosters/penalties
#         squeeze = self._is_squeeze(last)
#         breakout = self._is_breakout(last)
#         divergence = self._rsi_divergence_score(feats, idx)
#
#         if breakout and last.get("MACD_Hist",0) > 0 and last.get("Vol_Z",0) > 0.8:
#             score += 1.5
#             breakdown["pattern_breakout"] = 1.5
#         if squeeze and last.get("Vol_Z",0) > 0.6:
#             score += 0.6
#             breakdown["pattern_squeeze"] = 0.6
#         if divergence == -1:
#             score -= 1.1
#             breakdown["pattern_divergence"] = -1.1
#         elif divergence == 1:
#             score += 1.1
#             breakdown["pattern_divergence"] = 1.1
#
#         # trend (MA50 vs MA200)
#         trend = "➡️ Side"
#         if last["50DMA"] > last["200DMA"]:
#             trend = "📈 Up"
#         elif last["50DMA"] < last["200DMA"]:
#             trend = "📉 Down"
#
#         # scaled score for mapping
#         abs_scale = max(0.5, sum(abs(v) for v in weights.values()))
#         scaled_score = score * 3.0 / (abs_scale + 1e-9)
#         breakdown["raw_score"] = round(float(score),4)
#         breakdown["scaled_score"] = round(float(scaled_score),4)
#
#         # ML model probability (if available)
#         model_prob = None
#         if self.model is not None:
#             try:
#                 X_last = self._features_for_model(feats.tail(200)).iloc[[-1]]
#                 if LGB_INSTALLED and isinstance(self.model, lgb.Booster):
#                     # predict from lgb
#                     model_prob = float(self.model.predict(X_last)[0])
#                 else:
#                     model_prob = float(self.model.predict_proba(X_last)[0,1])
#                 # incorporate model prob into score (small boost)
#                 # e.g., normalized contribution: (prob - 0.5)*2 -> [-1,1]
#                 model_boost = (model_prob - 0.5) * 1.5  # scale
#                 score += model_boost
#                 scaled_score = score * 3.0 / (abs_scale + 1e-9)
#                 breakdown["model_prob"] = round(float(model_prob),4)
#                 breakdown["model_boost"] = round(float(model_boost),4)
#             except Exception:
#                 model_prob = None
#
#         # map scaled_score to signal
#         if scaled_score >= 2.5:
#             signal = "STRONG BUY"
#         elif 1.0 <= scaled_score < 2.5:
#             signal = "BUY"
#         elif -0.75 <= scaled_score < 1.0:
#             signal = "WAIT"
#         elif -2.0 <= scaled_score < -0.75:
#             signal = "STRONG WAIT"
#         else:
#             signal = "AVOID"
#
#         # confidence
#         confidence = int(min(95, max(10, abs(scaled_score) / 3.0 * 100)))
#         print("Evaluated using SmartHybrid Strategy")
#
#         return signal, confidence, trend, round(float(scaled_score),4), breakdown, model_prob
