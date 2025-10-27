import math
import pandas as pd
from typing import Dict, Any, List

class AdvisorEngine:
    """
    AdvisorEngine - generate human-readable suggestions from indicator data + portfolio.
    Also provides tooltip reasoning and simple alert system.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Default thresholds (tweak these to taste)
        defaults = {
            "rsi_buy": 35,
            "rsi_sell": 70,
            "zscore_sell": 2.0,
            "zscore_buy": -1.5,
            "adx_trend": 20,
            "macd_strength": 0.0,
            "golden_cross": True,
            "min_data_points_for_200dma": 200
        }
        self.cfg = {**defaults, **(config or {})}

    def _get_latest(self, df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series()
        return df.iloc[-1]

    # ----------------------------------------------------------------------------------
    # 1️⃣ CORE POSITION SCORING ENGINE
    # ----------------------------------------------------------------------------------
    def score_position(self, indicators: pd.DataFrame, portfolio_row: pd.Series) -> Dict[str, Any]:
        latest = self._get_latest(indicators)
        ticker = portfolio_row.get("Ticker", "UNKNOWN")
        out = {
            "Ticker": ticker,
            "Recommendation": "HOLD",
            "Confidence": 50,
            "Score": 0.0,
            "Reasons": [],
            "CurrentPrice": portfolio_row.get("CurrentPrice"),
            "PnL%": portfolio_row.get("PnL%")
        }

        if latest.empty:
            out["Reasons"].append("No indicator data available")
            out["Confidence"] = 20
            return out

        # Extract indicators safely
        def safe_get(name):
            return float(latest.get(name, math.nan)) if name in latest.index else math.nan

        rsi, macd, adx, z, dma50, dma200, vol_ratio = (
            safe_get("RSI"), safe_get("MACD"), safe_get("ADX"),
            safe_get("ZScore"), latest.get("50DMA", math.nan),
            latest.get("200DMA", math.nan), safe_get("VolRatio")
        )

        score, reasons = 0.0, []

        # RSI rules
        if not math.isnan(rsi):
            if rsi <= self.cfg["rsi_buy"]:
                score += 20
                reasons.append(f"RSI low ({rsi:.0f}) → oversold/buy opportunity")
            elif rsi >= self.cfg["rsi_sell"]:
                score -= 25
                reasons.append(f"RSI high ({rsi:.0f}) → overbought/consider trimming")

        # Z-Score
        if not math.isnan(z):
            if z <= self.cfg["zscore_buy"]:
                score += 10
                reasons.append(f"ZScore {z:.2f} suggests undervaluation")
            elif z >= self.cfg["zscore_sell"]:
                score -= 12
                reasons.append(f"ZScore {z:.2f} suggests over-extension")

        # MACD
        if not math.isnan(macd):
            if macd > self.cfg["macd_strength"]:
                score += 12
                reasons.append("MACD positive → bullish momentum")
            else:
                score -= 6
                reasons.append("MACD non-positive → weak momentum")

        # ADX (trend)
        if not math.isnan(adx):
            if adx >= self.cfg["adx_trend"]:
                score += 6
                reasons.append(f"ADX {adx:.0f} indicates a trending market")
            else:
                score -= 2
                reasons.append(f"ADX {adx:.0f} low → weak trend")

        # Moving average cross
        if not math.isnan(dma50) and not math.isnan(dma200):
            if dma50 > dma200:
                score += 10
                reasons.append("50DMA above 200DMA → bullish (golden cross)")
            else:
                score -= 6
                reasons.append("50DMA below 200DMA → bearish")
        else:
            reasons.append("Insufficient history for 200DMA check")

        # Volume confirmation
        if not math.isnan(vol_ratio):
            if vol_ratio > 1.2:
                score += 4
                reasons.append("Volume above 20d average → confirmed by volume")

        # Final mapping
        out["Score"] = round(score, 2)
        if score >= 25:
            out["Recommendation"] = "STRONG BUY"
        elif 8 <= score < 25:
            out["Recommendation"] = "BUY"
        elif -8 < score < 8:
            out["Recommendation"] = "HOLD"
        elif -25 < score <= -8:
            out["Recommendation"] = "WAIT/TAKE PROFITS"
        else:
            out["Recommendation"] = "SELL"

        conf = int((score + 40) / 100 * 100)
        out["Confidence"] = max(10, min(95, conf))
        out["Reasons"] = reasons
        return out

    # ----------------------------------------------------------------------------------
    # 2️⃣ PORTFOLIO AGGREGATION
    # ----------------------------------------------------------------------------------
    def portfolio_advice(self, portfolio_df: pd.DataFrame, indicators_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        rows, total_value, total_invested = [], 0.0, 0.0

        for _, row in portfolio_df.iterrows():
            ticker = row["Ticker"]
            ind_df = indicators_map.get(ticker)
            advice = self.score_position(ind_df, row)
            advice.update({
                "Qty": row.get("Qty"),
                "BuyPrice": row.get("BuyPrice"),
                "CurrentPrice": row.get("CurrentPrice"),
                "PnL%": round(row.get("PnL%"), 2),
            })
            rows.append(advice)
            total_value += (row.get("Qty", 0) or 0) * (row.get("CurrentPrice", 0) or 0)
            total_invested += (row.get("Qty", 0) or 0) * (row.get("BuyPrice", 0) or 0)

        total_pnl_pct = ((total_value - total_invested) / total_invested * 100) if total_invested else 0.0
        rec_counts = pd.Series([p["Recommendation"] for p in rows]).value_counts().to_dict()

        summary = {
            "total_value": total_value,
            "total_invested": total_invested,
            "total_pnl_pct": round(total_pnl_pct, 2),
            "counts": rec_counts,
            "num_holdings": len(rows)
        }

        return {"positions": rows, "summary": summary}

    # ----------------------------------------------------------------------------------
    # 3️⃣ TOOLTIP REASONING (for UI hover)
    # ----------------------------------------------------------------------------------
    def get_reason(self, signal, trend, pnl):
        signal = (signal or "").upper()
        trend = (trend or "").lower()

        if "STRONG BUY" in signal:
            return "Momentum and technical indicators align — strong accumulation opportunity."
        elif signal == "BUY":
            if "up" in trend:
                return "Uptrend confirmed with improving momentum — can accumulate gradually."
            else:
                return "Indicators turning positive — early buy signal forming."
        elif "WAIT" in signal or "HOLD" in signal:
            if pnl > 10:
                return "Good profits already — consider partial booking."
            elif pnl < -10:
                return "Underperforming, wait for recovery before exiting."
            else:
                return "Mixed indicators — hold and wait for clearer trend."
        elif "SELL" in signal or "AVOID" in signal:
            if "down" in trend:
                return "Downtrend with weak technicals — avoid fresh entries."
            else:
                return "No strong signals — better opportunities elsewhere."
        else:
            return "Insufficient data for confident advice."

    # ----------------------------------------------------------------------------------
    # 4️⃣ ALERTS ENGINE (optional)
    # ----------------------------------------------------------------------------------
    def alerts(self, indicators_map: Dict[str, pd.DataFrame], rules: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        alerts = []
        r = rules or {}
        for ticker, df in indicators_map.items():
            latest = self._get_latest(df)
            if latest.empty:
                continue
            rsi = latest.get("RSI", None)
            macd = latest.get("MACD", None)
            if rsi is not None and rsi <= r.get("rsi_oversold", self.cfg["rsi_buy"]):
                alerts.append({
                    "ticker": ticker,
                    "type": "RSI_OVERSOLD",
                    "level": "info",
                    "message": f"{ticker} RSI {rsi:.0f} ≤ {self.cfg['rsi_buy']} (oversold). Consider averaging."
                })
            if macd is not None and macd > 0:
                alerts.append({
                    "ticker": ticker,
                    "type": "MACD_BULL",
                    "level": "info",
                    "message": f"{ticker} MACD positive ({macd:.4f}) — bullish momentum detected."
                })
        return alerts
