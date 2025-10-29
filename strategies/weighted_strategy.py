from .base_strategy import BaseStrategy
import numpy as np

class WeightedStrategy(BaseStrategy):

    def evaluate(self, df):
        last = df.iloc[-1]
        price, ma20, ma50, ma200 = last["Close"], df["Close"].rolling(20).mean().iloc[-1], last["50DMA"], last["200DMA"]
        rsi, zscore = last["RSI"], last["ZScore"]
        macd, macd_signal = last.get("MACD", 0), last.get("MACD_Signal", 0)
        adx, vol_ratio = last.get("ADX", 0), last.get("VolRatio", 1)

        # --- 1️⃣ Volatility (Bollinger Bands deviation) ---
        rolling_std = df["Close"].rolling(20).std().iloc[-1]
        bb_upper = ma20 + 2 * rolling_std
        bb_lower = ma20 - 2 * rolling_std
        volatility = (bb_upper - bb_lower) / ma20 * 100  # %
        vol_score = 1.0 * (1 if volatility < 4 else (-1 if volatility > 8 else 0))

        # --- 2️⃣ Trend Strength & Momentum ---
        slope_50dma = np.polyfit(range(10), df["50DMA"].tail(10), 1)[0]
        slope_trend = 1.0 * (1 if slope_50dma > 0 else -1)
        momentum = (price - ma50) / ma50 * 100
        momentum_score = 1.0 * (1 if momentum > 1 else (-1 if momentum < -1 else 0))

        # --- 3️⃣ Volume Surge Detection ---
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        latest_vol = last["Volume"]
        vol_surge = latest_vol / avg_vol if avg_vol > 0 else 1
        vol_surge_score = 1.5 * (1 if vol_surge > 1.5 else (-1 if vol_surge < 0.7 else 0))

        # --- 4️⃣ Breakout / Breakdown Detection ---
        recent_high = df["Close"].rolling(20).max().iloc[-1]
        recent_low = df["Close"].rolling(20).min().iloc[-1]
        breakout_score = 2.0 * (1 if price > recent_high * 1.01 else (-1 if price < recent_low * 0.99 else 0))

        # --- Existing Factors ---
        rsi_score = 1.5 * (2 if rsi < 40 else (-1 if rsi > 70 else 0))
        ma_score = 1.0 * (1 if ma50 > ma200 else -1)
        price_ma_score = 1.0 * (1 if price > ma50 * 1.02 else (-1 if price < ma50 * 0.98 else 0))
        zscore_score = 1.0 * (1 if zscore < -1 else (-1 if zscore > 2 else 0))
        macd_score = 1.0 * (1 if macd > macd_signal else -1)
        adx_score = 1.0 * (1 if adx > 25 else (-1 if adx < 15 else 0))

        # --- Combined Weighted Scoring ---
        score = (
                rsi_score + ma_score + price_ma_score + zscore_score + macd_score + adx_score +
                vol_score + slope_trend + momentum_score + vol_surge_score + breakout_score
        )

        confidence = int(min(abs(score) * 10, 100))

        # --- Trend Detection ---
        if ma50 > ma200 and slope_50dma > 0:
            trend = "📈 Up"
        elif ma50 < ma200 and slope_50dma < 0:
            trend = "📉 Down"
        else:
            trend = "➡️ Side"

        # --- Signal Decision ---
        if score >= 6:
            signal = "STRONG BUY"
        elif 3 <= score < 6:
            signal = "BUY"
        elif -2 < score < 3:
            signal = "WAIT"
        elif -5 <= score <= -2:
            signal = "STRONG WAIT"
        else:
            signal = "AVOID"

        return signal, confidence, trend, round(score, 2)
