from .base_strategy import BaseStrategy

class SimpleRuleStrategy(BaseStrategy):
    def evaluate(self, df):
        last = df.iloc[-1]
        rsi, ma50, ma200, price, zscore = (
            float(last.get("RSI", 50)),
            float(last.get("50DMA", last["Close"])),
            float(last.get("200DMA", last["Close"])),
            float(last["Close"]),
            float(last.get("ZScore", 0)),
        )

        score = 0
        # RSI
        if rsi < 40: score += 2
        elif 40 <= rsi < 50: score += 1
        elif rsi > 70: score -= 1

        # Trend
        score += 1 if ma50 > ma200 else -1

        # Price vs MA50
        score += 1 if price > ma50 else -1

        # ZScore
        if zscore < -2: score += 2
        elif -2 <= zscore < -1: score += 1
        elif zscore > 2: score -= 2
        elif 1 < zscore <= 2: score -= 1

        confidence = min(int((abs(score) / 6) * 100), 100)

        if price > ma50 > ma200:
            trend = "📈 Up"
        elif price < ma200 and ma50 < ma200:
            trend = "📉 Down"
        else:
            trend = "➡️ Side"

        if score >= 4:
            signal = "STRONG BUY"
        elif 2 <= score < 4:
            signal = "BUY"
        elif -1 <= score < 2:
            signal = "WAIT"
        elif -3 <= score < -1:
            signal = "STRONG WAIT"
        else:
            signal = "AVOID"

        return signal, confidence, trend, score
