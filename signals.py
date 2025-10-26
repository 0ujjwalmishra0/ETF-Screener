import pandas as pd, webbrowser
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from output_writer import write_outputs
from sparkline_generator import create_sparkline
from performance_charts import create_performance_chart

TICKERS = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "MID150BEES.NS",
    "BANKBEES.NS", "GOLDBEES.NS", "SILVERBEES.NS"
]

def evaluate_signal(rsi, ma50, ma200, price):
    """Return signal, confidence %, and trend label."""
    score, total = 0, 4
    if rsi < 40: score += 2
    elif rsi < 50: score += 1
    if ma50 > ma200: score += 1
    if price > ma50 * 1.01: score += 1

    confidence = int((score / total) * 100)

    if price > ma50 and ma50 > ma200:
        trend = "📈 Up"
    elif price < ma200 and ma50 < ma200:
        trend = "📉 Down"
    else:
        trend = "➡️ Side"

    if score >= 3:
        signal = "STRONG BUY"
    elif score == 2:
        signal = "BUY"
    elif score == 1:
        signal = "WAIT"
    else:
        signal = "STRONG WAIT"

    return signal, confidence, trend


def run_once():
    results = []

    for t in TICKERS:
        try:
            df = fetch_etf_data(t)
            if df.empty:
                print(f"⚠️  No price data for {t}")
                continue

            df = compute_basic_indicators(df)
            if df.empty:
                print(f"⚠️  Not enough history for {t}")
                continue

            last = df.iloc[-1]

            # Graceful fallback for missing indicators
            rsi = float(last.get("RSI", 50))
            ma50 = float(last.get("50DMA", last["Close"]))
            ma200 = float(last.get("200DMA", last["Close"]))
            price = float(last["Close"])
            zscore = float(last.get("ZScore", 0))

            signal, confidence, trend = evaluate_signal(rsi, ma50, ma200, price)
            trend_up = "Up" in trend

            # Sparkline (last 30 days)
            spark_path = create_sparkline(df["Close"].tail(30).tolist(), t.replace(".NS", ""), trend_up)
            # Full-year performance chart
            chart_path = create_performance_chart(df.tail(250), t)

            results.append({
                "ETF": t,
                "Trend": trend,
                "Sparkline": f"<img src='{spark_path}' width='80' height='25'>",
                "Chart": f"<img src='{chart_path}' width='200' height='100'>",
                "Close": round(price, 2),
                "RSI": round(rsi, 2),
                "50DMA": round(ma50, 2),
                "200DMA": round(ma200, 2),
                "ZScore": round(zscore, 2),
                "Signal": signal,
                "Confidence": f"{confidence}%"
            })

            print(f"{signal:>12} {trend:>8} → {t}: ₹{price:.2f} | RSI={rsi:.2f} | Conf={confidence}%")

        except Exception as e:
            print(f"❌ Error processing {t}: {e}")

    if not results:
        print("⚠️ No ETFs processed successfully.")
        return

    out = pd.DataFrame(results).sort_values(by="ZScore", ascending=True)
    html_file = write_outputs(out)

    # ✅ Auto-open the HTML dashboard
    webbrowser.open(html_file)

    print("\n✅ Screener complete. Dashboard auto-opened in browser.")
