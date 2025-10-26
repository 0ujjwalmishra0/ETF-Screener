from .base_strategy import BaseStrategy

class WeightedStrategy(BaseStrategy):
    def evaluate(self, df):
        last = df.iloc[-1]
        price, ma50, ma200 = last["Close"], last["50DMA"], last["200DMA"]
        rsi, zscore = last["RSI"], last["ZScore"]
        macd, macd_signal = last.get("MACD", 0), last.get("MACD_Signal", 0)
        adx, vol_ratio = last.get("ADX", 0), last.get("VolRatio", 1)

        rsi_score = 1.5 * (2 if rsi < 45 else (-1 if rsi > 70 else 0))
        ma_score = 1.0 * (1 if ma50 > ma200 else -1)
        price_ma_score = 1.0 * (1 if price > ma50 * 1.02 else (-1 if price < ma50 * 0.98 else 0))
        zscore_score = 1.0 * (1 if zscore < -1 else (-1 if zscore > 2 else 0))
        macd_score = 1.0 * (1 if macd > macd_signal else -1)
        adx_score = 1.0 * (1 if adx > 25 else (-1 if adx < 15 else 0))
        vol_score = 0.5 * (1 if vol_ratio > 1.2 else (-1 if vol_ratio < 0.8 else 0))

        score = rsi_score + ma_score + price_ma_score + zscore_score + macd_score + adx_score + vol_score
        confidence = int(min(abs(score) * 12.5, 100))

        if ma50 > ma200:
            trend = "📈 Up"
        elif ma50 < ma200:
            trend = "📉 Down"
        else:
            trend = "➡️ Side"

        if score >= 5:
            signal = "STRONG BUY"
        elif 3 <= score < 5:
            signal = "BUY"
        elif -2 < score < 3:
            signal = "WAIT"
        elif -4 <= score <= -2:
            signal = "STRONG WAIT"
        else:
            signal = "AVOID"

        return signal, confidence, trend, score
