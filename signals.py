import pandas as pd, webbrowser
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from output_writer import write_outputs
from sparkline_generator import create_sparkline
from performance_charts import create_performance_chart
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy

TICKERS= [
    'NIFTYBEES.NS',
    'BANKBEES.NS',
    'JUNIORBEES.NS',
    'GOLDBEES.NS',
    'SETFNIF50.NS',
    'MOM50.NS',
    'NIFTYIETF.NS',
    'SETFNIFBK.NS',
    'SETFNN50.NS',
    'ITBEES.NS',
    'INFRABEES.NS',
    'SHARIABEES.NS',
    'SMALLCAP.NS',
    'MIDCAPETF.NS',
    '0P0001IUFY.BO',
    'MONQ50.NS',
    'MAHKTECH.NS',
    'MON100.NS',
    'MAFANG.NS',
    'MASPTOP50.NS',
    'MODEFENCE.NS',
    'CPSEETF.NS',
    'PSUBANK.NS'
]



def run_once(strategy=None):
    strategy = strategy or WeightedStrategy()  # default
    results = []
    print(f"\n🚀 Running ETF Screener Pro 2026 using {strategy.__class__.__name__}...\n")
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

            # ✅ Strategy-based signal evaluation
            signal, confidence, trend, score = strategy.evaluate(df)
            trend_up = "Up" in trend

            # Sparkline (last 30 days)
            spark_path = create_sparkline(df["Close"].tail(30).tolist(), t.replace(".NS", ""), trend_up)
            # Full-year performance chart
            chart_path = create_performance_chart(df.tail(250), t)

            results.append({
                "Ticker": t,
                "Sparkline": f"<img src='{spark_path}' width='80' height='25'>",
                "Chart": f"<img src='{chart_path}' width='200' height='100'>",
                "Close": round(price, 2),
                "RSI": round(rsi, 2),
                "50DMA": round(ma50, 2),
                "200DMA": round(ma200, 2),
                "ZScore": round(zscore, 2),
                "Signal": signal,
                "Score": score,
                "Confidence": f"{confidence}%",
            })

            print(f"{signal:>12} {trend:>8} → {t}: ₹{price:.2f} | RSI={rsi:.2f} | Conf={confidence}%")

        except Exception as e:
            print(f"❌ Error processing {t}: {e}")

    if not results:
        print("⚠️ No ETFs processed successfully.")
        return

    # Convert results → DataFrame
    out = pd.DataFrame(results)

    # Ensure 'Score' column exists even if missing for some
    if "Score" not in out.columns:
        out["Score"] = 0
    if "Signal" not in out.columns:
        out["Signal"] = "WAIT"

    # Sorting by signal and score
    signal_order = {"STRONG BUY": 1, "BUY": 2, "WAIT": 3, "STRONG WAIT": 4, "AVOID": 5}
    out["SignalRank"] = out["Signal"].map(signal_order).fillna(99)
    out = out.sort_values(by=["SignalRank", "Score"], ascending=[True, False]).reset_index(drop=True)
    out = out.drop(columns=["SignalRank"])

    # Print summary
    print("\n📊 ETF Summary by Signal:")
    print(out["Signal"].value_counts().to_string())
    write_outputs(out)