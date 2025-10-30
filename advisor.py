# advisor.py
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

class AdvisorEngine:
    """
    AdvisorEngineV2 - Enhanced rule-based advisory engine with:
      - context-aware reasoning (holding duration, volatility, drawdown)
      - confidence scoring
      - short-term price range projection (simple linear projection on recent closes)
      - actionable recommendation generation
      - portfolio-level risk & health metrics
      - scenario simulation (simple)
    """

    def __init__(self, config: Dict[str, Any] = None):
        defaults = {
            "rsi_buy": 35,
            "rsi_sell": 70,
            "zscore_buy": -1.5,
            "zscore_sell": 2.0,
            "adx_trend": 20,
            "macd_strength": 0.0,
            "golden_cross_bonus": 10,
            "volatility_threshold": 0.04,  # 4% daily std as high vol
            "beta_threshold": 1.2,
            "holding_days_threshold": 180,
            "drawdown_warning_pct": -10.0,
            "min_history_days_for_projection": 30,
            "projection_window_days": 7,
            "score_bias": 40,  # used to scale confidence
        }
        self.cfg = {**defaults, **(config or {})}

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _get_latest(self, df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series()
        return df.iloc[-1]

    def _pct_change_series(self, series: pd.Series) -> pd.Series:
        return series.pct_change().dropna()

    def _compute_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> Optional[float]:
        try:
            # match indices
            joined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
            if joined.shape[0] < 10:
                return None
            cov = np.cov(joined.iloc[:,0], joined.iloc[:,1])[0,1]
            var = np.var(joined.iloc[:,1])
            if var == 0:
                return None
            return float(cov / var)
        except Exception:
            return None

    # ---------------------------
    # Core scoring per position
    # ---------------------------
    def score_position(self, indicators: pd.DataFrame, portfolio_row: pd.Series,
                       benchmark_df: Optional[pd.DataFrame] = None,
                       sector_avg_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        indicators: DataFrame (Date-indexed or incremental) with Close, RSI, MACD, 50DMA, 200DMA, ADX, ZScore, VolRatio, Volume
        portfolio_row: Series containing at least Ticker, Qty, BuyPrice, CurrentPrice, PnL% (or PnL)
        benchmark_df: DataFrame of benchmark prices (e.g. NIFTY) for beta calc (Optional)
        sector_avg_df: DataFrame of sector index closes (Optional)
        Returns a dict with recommendation, confidence, reasons, action, score, next_7d_range, portfolio_impact_est
        """
        latest = self._get_latest(indicators)
        out = {
            "Ticker": portfolio_row.get("Ticker", "UNKNOWN"),
            "Recommendation": "HOLD",
            "Confidence": 50,
            "Score": 0.0,
            "Reasons": [],
            "Action": "Hold and monitor",
            "Next7DRange": None,
            "PortfolioImpactEst": None
        }

        # safety
        if latest.empty:
            out["Reasons"].append("No indicator data available")
            out["Confidence"] = 20
            return out

        # extract indicators safely
        def safe(name):
            try:
                val = latest.get(name, math.nan)
                return float(val) if val is not None else math.nan
            except Exception:
                return math.nan

        rsi = safe("RSI")
        macd = safe("MACD")
        adx = safe("ADX")
        z = safe("ZScore")
        dma50 = latest.get("50DMA", math.nan)
        dma200 = latest.get("200DMA", math.nan)
        vol_ratio = safe("VolRatio")
        close_series = indicators["Close"] if "Close" in indicators.columns else None
        vol_daily_std = float(close_series.pct_change().rolling(20).std().iloc[-1]) if (close_series is not None and len(close_series)>=20) else math.nan

        # timing / holding duration
        buy_date_str = portfolio_row.get("BuyDate")
        holding_days = None
        try:
            if buy_date_str and isinstance(buy_date_str, str):
                buy_date = pd.to_datetime(buy_date_str, dayfirst=True, errors="coerce")
                if not pd.isna(buy_date):
                    holding_days = (pd.Timestamp.now() - buy_date).days
            elif isinstance(buy_date_str, (pd.Timestamp,)):
                holding_days = (pd.Timestamp.now() - buy_date_str).days
        except Exception:
            holding_days = None

        # base scoring
        score = 0.0
        reasons = []

        # RSI
        if not math.isnan(rsi):
            if rsi <= self.cfg["rsi_buy"]:
                score += 18
                reasons.append(f"RSI low ({rsi:.0f}) → potential buy/oversold")
            elif rsi >= self.cfg["rsi_sell"]:
                score -= 20
                reasons.append(f"RSI high ({rsi:.0f}) → consider trimming")

        # Z-Score
        if not math.isnan(z):
            if z <= self.cfg["zscore_buy"]:
                score += 8
                reasons.append(f"Z-score {z:.2f} suggests undervaluation")
            elif z >= self.cfg["zscore_sell"]:
                score -= 10
                reasons.append(f"Z-score {z:.2f} suggests over-extension")

        # MACD momentum
        if not math.isnan(macd):
            if macd > self.cfg["macd_strength"]:
                score += 10
                reasons.append("MACD positive → bullish momentum")
            else:
                score -= 6
                reasons.append("MACD non-positive → weak momentum")

        # ADX (trend)
        if not math.isnan(adx):
            if adx >= self.cfg["adx_trend"]:
                score += 6
                reasons.append(f"ADX {adx:.0f} indicates trend strength")
            else:
                score -= 2
                reasons.append(f"ADX {adx:.0f} low → weak trend")

        # Moving average/golden cross
        if not math.isnan(dma50) and not math.isnan(dma200):
            try:
                if dma50 > dma200:
                    score += self.cfg["golden_cross_bonus"]
                    reasons.append("50DMA > 200DMA → bullish structure")
                else:
                    score -= 8
                    reasons.append("50DMA < 200DMA → bearish bias")
            except Exception:
                pass
        else:
            reasons.append("Insufficient history for 200DMA check")

        # volume confirmation
        if not math.isnan(vol_ratio):
            if vol_ratio > 1.2:
                score += 4
                reasons.append("Volume >20d average → move confirmed by volume")

        # volatility penalty (too choppy)
        if not math.isnan(vol_daily_std):
            if vol_daily_std > self.cfg["volatility_threshold"]:
                score -= 6
                reasons.append(f"High recent volatility (σ={vol_daily_std:.3f}) → manage risk")

        # beta vs benchmark (if provided)
        beta = None
        try:
            if benchmark_df is not None and close_series is not None:
                asset_rets = close_series.pct_change().dropna()
                bench_close = benchmark_df["Close"] if "Close" in benchmark_df.columns else None
                if bench_close is not None:
                    bench_rets = bench_close.pct_change().dropna()
                    beta = self._compute_beta(asset_rets, bench_rets)
                    if beta is not None:
                        if beta > self.cfg["beta_threshold"]:
                            score -= 4
                            reasons.append(f"Beta {beta:.2f} > {self.cfg['beta_threshold']} → higher market sensitivity")
                        else:
                            score += 2
                            reasons.append(f"Beta {beta:.2f} → acceptable market sensitivity")
        except Exception:
            beta = None

        # holding duration logic
        pnl = float(portfolio_row.get("PnL%", portfolio_row.get("PnL", 0.0)) or 0.0)
        if holding_days is not None:
            if holding_days >= self.cfg["holding_days_threshold"] and pnl < 5:
                score -= 6
                reasons.append(f"Held {holding_days} days with low return → re-evaluate")

        # drawdown detection (simple)
        try:
            highs = indicators["Close"].cummax()
            drawdown = (indicators["Close"].iloc[-1] - highs.max()) / highs.max() * 100
            if drawdown <= self.cfg["drawdown_warning_pct"]:
                score -= 6
                reasons.append(f"Significant drawdown ({drawdown:.2f}%) detected")
        except Exception:
            pass

        # compose numeric score
        out["Score"] = round(score, 2)

        # map score to recommendation
        if score >= 30:
            rec = "STRONG BUY"
        elif 12 <= score < 30:
            rec = "BUY"
        elif -10 < score < 12:
            rec = "HOLD"
        elif -30 < score <= -10:
            rec = "WAIT"
        else:
            rec = "SELL"

        out["Recommendation"] = rec

        # -------------------------
        # Confidence calculation
        # -------------------------
        # Combine normalized components into a 0..100 confidence
        # Normalize score roughly from [-50, 80] to 0..100
        norm_score = (score + self.cfg["score_bias"]) / (self.cfg["score_bias"] + 60)  # rough scaling
        conf = int(max(10, min(95, round(norm_score * 100))))
        out["Confidence"] = conf

        # -------------------------
        # Next 7-day range projection (simple linear regression on recent closes)
        # -------------------------
        next_range = None
        try:
            if close_series is not None and len(close_series.dropna()) >= self.cfg["min_history_days_for_projection"]:
                window = close_series.dropna().iloc[-60:]  # last 60 days
                x = np.arange(len(window))
                y = window.values.astype(float)
                if len(x) >= 10:
                    # linear fit
                    coef = np.polyfit(x, y, 1)
                    slope = coef[0]
                    # predict next 7 days
                    future_x = np.arange(len(x), len(x) + self.cfg["projection_window_days"])
                    preds = np.polyval(coef, future_x)
                    low = float(np.min(preds))
                    high = float(np.max(preds))
                    # add a small band (volatility)
                    band = np.std(np.diff(y)) * 5
                    next_range = (max(0.0, low - band), high + band)
        except Exception:
            next_range = None
        out["Next7DRange"] = None if next_range is None else (round(next_range[0],2), round(next_range[1],2))

        # -------------------------
        # Portfolio impact estimate (very rough)
        # -------------------------
        try:
            # compute weight if portfolio row includes current value / total value externally passable
            # We'll try to compute from Qty * CurrentPrice and assume total_invested passed separately in caller if available.
            cur_val = float(portfolio_row.get("Qty", 0) or 0) * float(portfolio_row.get("CurrentPrice", 0) or 0)
            # assume an average scenario: if price moves to mid of projection range, percent change:
            if next_range:
                mid = (next_range[0] + next_range[1]) / 2.0
                pct_move = (mid - float(portfolio_row.get("CurrentPrice", 0) or 0)) / max(1e-9, float(portfolio_row.get("CurrentPrice", 1)))
                # portfolio impact depends on caller providing portfolio total (we estimate small)
                out["PortfolioImpactEst"] = f"{pct_move * 100:.2f}% (per position)"
            else:
                out["PortfolioImpactEst"] = None
        except Exception:
            out["PortfolioImpactEst"] = None

        # -------------------------
        # Action suggestion (concrete)
        # -------------------------
        action = "Hold"
        # rules for action
        if rec in ["STRONG BUY", "BUY"]:
            if pnl < -10:
                action = "Accumulate gradually (average down) with size limit 5-10%"
            else:
                action = "Accumulate gradually — consider 5-10% additional allocation"
        elif rec == "HOLD":
            action = "Hold position; review if conditions worsen"
        elif rec == "WAIT":
            if pnl > 10:
                action = "Consider partial profit booking (e.g., 20%)"
            else:
                action = "Wait for clearer trend; consider trailing stop"
        elif rec == "SELL":
            action = "Prepare to exit or trim position; set stop or limit exit"

        # refine action with volatility/beta/holding context
        if beta is not None and beta > self.cfg["beta_threshold"]:
            action += " — position exhibits high market sensitivity (beta>1.2). Consider smaller size."

        if holding_days is not None and holding_days > 365:
            action += " — long-held position; reassess long-term thesis."

        out["Action"] = action
        out["Reasons"].extend(reasons)

        return out

    # ---------------------------------------------------------------------
    # 🔥 Main method — used in portfolio tracker
    # ---------------------------------------------------------------------
    def generate_advice(self, ticker, df, signal, confidence, trend, pnl, score):
        """
        Produce a detailed, human-readable advisory dictionary.
        Combines technical, statistical, and trend insights.
        """
        latest = self._get_latest(df)
        if latest.empty:
            return {
                "Ticker": ticker,
                "Signal": signal,
                "Confidence": confidence,
                "Trend": trend,
                "Advice": "No recent data available",
            }

        last_close = float(latest["Close"])
        volatility = latest.get("Volatility", 1.2)
        mean_price = df["Close"].rolling(20).mean().iloc[-1]
        projected_move = last_close * (volatility / 100) * 1.5
        next_range = (last_close - projected_move, last_close + projected_move)

        # Compute enriched reason
        rsi = latest.get("RSI", 50)
        macd = latest.get("MACD", 0)
        adx = latest.get("ADX", 20)
        zscore = latest.get("Z-Score", 0)

        reasons = []

        if signal == "BUY":
            if rsi < 40:
                reasons.append("RSI near oversold zone")
            if macd > latest.get("Signal_Line", 0):
                reasons.append("MACD bullish crossover")
            if adx > 25:
                reasons.append("Strong trend continuation expected")
            action = "Accumulate gradually — 10% additional allocation suggested."
        elif signal == "SELL":
            if rsi > 70:
                reasons.append("RSI overbought")
            if macd < latest.get("Signal_Line", 0):
                reasons.append("MACD bearish crossover")
            action = "Consider partial profit booking or tightening stop-loss."
        else:
            if abs(zscore) < 1:
                reasons.append("Price near mean — sideways consolidation likely.")
            action = "Hold position and monitor breakout levels."

        # Portfolio impact (hypothetical)
        impact = round((pnl / 100) * (confidence / 100), 2)

        # Final advice message
        reason_text = ", ".join(reasons) if reasons else self.get_reason(signal, trend, pnl)
        advice = (
            # f"Signal: {signal} (Confidence {confidence}%)\n"
            f"Reason: {reason_text}.\n"
            f"Action: {action}\n"
            # f"Next 7D Range: ₹{round(next_range[0],2)} – ₹{round(next_range[1],2)}\n"
            # f"Portfolio Impact: {impact:+.2f}% expected if trend continues."
        )

        return {
            "Ticker": ticker,
            "Signal": signal,
            "Confidence": confidence,
            "Trend": trend,
            "Score": score,
            "PnL%": round(pnl, 2),
            "Reason": reason_text,
            "Action": action,
            "NextRange": (round(next_range[0], 2), round(next_range[1], 2)),
            "PortfolioImpact": impact,
            "Advice": advice,
        }


    # ---------------------------
    # Portfolio-level aggregation
    # ---------------------------
    def portfolio_advice(self, portfolio_df: pd.DataFrame, indicators_map: Dict[str, pd.DataFrame],
                         benchmark_df: Optional[pd.DataFrame] = None,
                         sector_map: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        portfolio_df: DataFrame with at least Ticker, Qty, BuyPrice, CurrentPrice, PnL% (or PnL)
        indicators_map: dict[ticker] -> indicators DataFrame
        benchmark_df: optional for beta calculations
        sector_map: optional dict[ticker] -> sector average indicators (for sector comparison)
        Returns:
          {
            "positions": [per-ticker advice],
            "summary": {...}
          }
        """
        positions = []
        total_value = 0.0
        total_invested = 0.0

        for _, row in portfolio_df.iterrows():
            ticker = row.get("Ticker")
            ind = indicators_map.get(ticker)
            sector_df = None
            if sector_map and ticker in sector_map:
                sector_df = sector_map[ticker]
            advice = self.score_position(ind, row, benchmark_df=benchmark_df, sector_avg_df=sector_df)
            # add fields for display
            advice.update({
                "Qty": row.get("Qty"),
                "BuyPrice": row.get("BuyPrice"),
                "CurrentPrice": row.get("CurrentPrice"),
                "PnL%": round(float(row.get("PnL%", row.get("PnL", 0.0)) or 0.0), 2),
                "SuggestedAction": advice.get("Action"),
                "Score": advice.get("Score")
            })
            positions.append(advice)

            total_value += (row.get("Qty", 0) or 0) * (row.get("CurrentPrice", 0) or 0)
            total_invested += (row.get("Qty", 0) or 0) * (row.get("BuyPrice", 0) or 0)

        total_pnl_pct = ((total_value - total_invested) / total_invested * 100) if total_invested else 0.0

        # counts
        rec_counts = pd.Series([p["Recommendation"] for p in positions]).value_counts().to_dict()

        summary = {
            "total_value": total_value,
            "total_invested": total_invested,
            "total_pnl_pct": round(total_pnl_pct, 2),
            "recommendation_counts": rec_counts,
            "num_holdings": len(positions)
        }

        return {"positions": positions, "summary": summary}

    # ---------------------------
    # Risk & allocation helpers
    # ---------------------------
    def diversification_score(self, portfolio_df: pd.DataFrame, sector_lookup: Optional[Dict[str,str]] = None) -> Dict[str, Any]:
        """
        Returns simple diversification metrics:
        - concentration_by_ticker (largest holding %)
        - diversification_score (0-100)
        sector_lookup: optional mapping Ticker->Sector
        """
        rows = portfolio_df.copy()
        rows["CurrentValue"] = rows["Qty"] * rows["CurrentPrice"]
        total = rows["CurrentValue"].sum()
        if total == 0:
            return {"concentration_by_ticker": {}, "diversification_score": 0}

        rows["weight"] = rows["CurrentValue"] / total
        top = rows.sort_values("weight", ascending=False).iloc[0]
        top_pct = top["weight"] * 100

        # sector concentration
        sector_counts = {}
        if sector_lookup:
            rows["sector"] = rows["Ticker"].map(sector_lookup).fillna("Unknown")
            sector_weights = rows.groupby("sector")["CurrentValue"].sum() / total
            sector_top_pct = float(sector_weights.max() * 100)
        else:
            sector_top_pct = None

        # diversification score heuristic
        score = 100 - min(80, top_pct)  # penalize large single holdings
        if sector_top_pct:
            score -= min(30, (sector_top_pct - 30))  # penalty if sector too concentrated

        score = max(0, min(100, int(score)))
        return {"concentration_by_ticker": {top["Ticker"]: round(top_pct,2)}, "diversification_score": score, "top_sector_pct": sector_top_pct}

    def portfolio_health(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a health score combining trend, diversification and pnl
        """
        # Trend score: fraction of positions with Recommendation BUY/STRONG BUY
        # We assume portfolio_df has 'Recommendation' field; if not, user should call portfolio_advice first.
        recs = portfolio_df.get("Recommendation", None)
        if recs is None:
            # fallback: use PnL
            avg_pnl = (portfolio_df["Qty"] * portfolio_df["CurrentPrice"]).sum()
            return {"health_score": 50, "label": "Unknown (insufficient data)"}

        buys = (recs.isin(["BUY", "STRONG BUY"])).sum()
        total = len(recs)
        trend_component = (buys / total) * 100 if total else 50

        # diversification
        div = self.diversification_score(portfolio_df)
        div_component = div["diversification_score"]

        # pnl component (normalized)
        total_value = (portfolio_df["Qty"] * portfolio_df["CurrentPrice"]).sum()
        total_invested = (portfolio_df["Qty"] * portfolio_df["BuyPrice"]).sum()
        pnl_pct = ((total_value - total_invested) / total_invested * 100) if total_invested else 0.0
        pnl_component = max(-50, min(50, pnl_pct)) + 50  # map roughly to 0..100

        # weighted aggregate
        health = int((0.6 * trend_component) + (0.25 * div_component) + (0.15 * pnl_component))
        label = "Healthy" if health >= 70 else "Neutral" if health >= 50 else "At Risk"

        return {"health_score": health, "label": label, "trend_component": trend_component, "div_component": div_component, "pnl_component": pnl_component}

    # ---------------------------
    # Simple scenario simulation
    # ---------------------------
    def simulate_scenario(self, portfolio_df: pd.DataFrame, pct_move: float) -> Dict[str, Any]:
        """
        Simulate portfolio value if all holdings move by pct_move (e.g., -0.05 for -5%).
        Returns new total value and delta pct.
        """
        df = portfolio_df.copy()
        df["CurrentValue"] = df["Qty"] * df["CurrentPrice"] * (1 + pct_move)
        new_total = df["CurrentValue"].sum()
        old_total = (portfolio_df["Qty"] * portfolio_df["CurrentPrice"]).sum()
        delta_pct = ((new_total - old_total) / old_total * 100) if old_total else 0.0
        return {"new_total": new_total, "old_total": old_total, "delta_pct": delta_pct}