import pandas as pd, webbrowser
from data_fetch import fetch_etf_data
from indicators import compute_basic_indicators
from output_writer import write_outputs
from sparkline_generator import create_sparkline
from performance_charts import create_performance_chart
from strategies.simple_rule_strategy import SimpleRuleStrategy
from strategies.weighted_strategy import WeightedStrategy
TICKERS = ['ARVINDPORT-SM.NS','UNIHEALTH-SM.NS','SPRL-SM.NS','NMDC.NS','UNIHEALTH.NS','TRENT.NS','LUPIN.NS','PIDILITIND.NS','NMDC.NS','INDUSTOWER.NS','CHAMBLFERT.NS','CIPLA.NS','ADANIGREEN.NS']
# TICKERS = ['ADANIGREEN.NS','PSUBNKIETF.NS','MAHKTECH.NS','METALIETF.NS','ABSLNN50ET.NS','SBIBPB.NS','METAL.NS','PSUBNKBEES.NS','INTERNET.NS','HDFCPSUBK.NS','PSUBANK.NS','BANKPSU.NS','MONQ50.NS','EBANKNIFTY.NS','ECAPINSURE.NS','MASPTOP50.NS','MIDSELIETF.NS','MAFANG.NS','NIFTY100EW.NS','HDFCNIF100.NS','HNGSNGBEES.NS','LICNMID100.NS','SML100CASE.NS','VAL30IETF.NS','SNXT30BEES.NS','HDFCPVTBAN.NS','AONENIFTY.NS','BANKETF.NS','PSUBANKADD.NS','NIFMID150.NS','BANKIETF.NS','IDFNIFTYET.NS','NIF5GETF.NS','CONSUMIETF.NS','INFRABEES.NS','HDFCNIFBAN.NS','UTISXN50.NS','INFRAIETF.NS','EMULTIMQ.NS','MOM50.NS','UTIBANKETF.NS','AXISBNKETF.NS','NIFTY1.NS','BANKNIFTY1.NS','BANKBEES.NS','FINIETF.NS','UTINIFTETF.NS','GSEC10YEAR.NS','MULTICAP.NS','IVZINNIFTY.NS','MOSMALL250.NS','LOWVOL1.NS','SETFNIFBK.NS','NETF.NS','HEALTHY.NS','GROWWLOVOL.NS','GROWWNIFTY.NS','GROWWN200.NS','MOMENTUM30.NS','HDFCSENSEX.NS','NEXT30ADD.NS','ALPHA.NS','SMALLCAP.NS','HDFCMID150.NS','ABSLBANETF.NS','BANKBETF.NS','SELECTIPO.NS','MIDCAP.NS','MOHEALTH.NS','TOP100CASE.NS','AONETOTAL.NS','UTINEXT50.NS','HDFCBSE500.NS','BSE500IETF.NS','LTGILTCASE.NS','GSEC10ABSL.NS','MOM100.NS','MANUFGBEES.NS','GILT5YBEES.NS','MAKEINDIA.NS','LICNETFSEN.NS','AXISNIFTY.NS','SENSEXADD.NS','MID150BEES.NS','MIDCAPETF.NS','MIDCAPIETF.NS','MID150.NS','MOALPHA50.NS','ELIQUID.NS','TECH.NS','QNIFTY.NS','PVTBANIETF.NS','NIFTYIETF.NS','LIQUIDSHRI.NS','GROWWLIQID.NS','AONELIQUID.NS','HDFCLIQUID.NS','LIQUIDCASE.NS','LIQGRWBEES.NS','HDFCSML250.NS','SENSEXIETF.NS','SBILIQETF.NS','GSEC10IETF.NS','LIQUIDBETF.NS','LIQUID1.NS','LIQUIDPLUS.NS','CASHIETF.NS','LIQUIDADD.NS','SENSEXETF.NS','LIQUIDETF.NS','LIQUIDBEES.NS','LTGILTBEES.NS','LIQUIDIETF.NS','ABSLLIQUID.NS','NIFTYCASE.NS','MONIFTY100.NS','GROWWPOWER.NS','GROWWMOM50.NS','CONSUMER.NS','AXISVALUE.NS','EQUAL200.NS','MON50EQUAL.NS','LIQUIDSBI.NS','MIDSMALL.NS','MID150CASE.NS','LIQUID.NS','DIVOPPBEES.NS','SETFNIF50.NS','BANKETFADD.NS','MOMIDMTM.NS','GSEC5IETF.NS','TNIDETF.NS','AXISHCETF.NS','MOMGF.NS','MONIFTY500.NS','EQUAL50.NS','ELM250.NS','NIFTYBEES.NS','HDFCNIFTY.NS','BFSI.NS','NIF10GETF.NS','PHARMABEES.NS','SHARIABEES.NS','SBIETFPB.NS','LICNETFN50.NS','MNC.NS','ABSLPSE.NS','ABGSEC.NS','NIF100IETF.NS','GROWWRAIL.NS','EBBETF0433.NS','ALPHAETF.NS','EVINDIA.NS','EBBETF0430.NS','NV20IETF.NS','EBBETF0431.NS','ICICIB22.NS','ALPL30IETF.NS','LICNETFGSC.NS','BSLSENETFG.NS','NIFTYBETF.NS','AXSENSEX.NS','SBINEQWETF.NS','MOMOMENTUM.NS','TOP10ADD.NS','BBETF0432.NS','HDFCMOMENT.NS','OILIETF.NS','AXISCETF.NS','NV20.NS','ESG.NS','HEALTHIETF.NS','NIFTYETF.NS','HDFCGROWTH.NS','SETF10GILT.NS','HDFCVALUE.NS','NEXT50.NS','BSLNIFTY.NS','IT.NS','MOTOUR.NS','MOLOWVOL.NS','NIFTY50ADD.NS','MOMENTUM50.NS','LOWVOLIETF.NS','MOGSEC.NS','PVTBANKADD.NS','MOM30IETF.NS','JUNIORBEES.NS','MOQUALITY.NS','MOMENTUM.NS','ITETFADD.NS','ITETF.NS','SBIETFIT.NS','MODEFENCE.NS','MOENERGY.NS','SDL26BEES.NS','EQUAL50ADD.NS','CONSUMBEES.NS','NEXT50IETF.NS','SBIETFCON.NS','EVIETF.NS','MON100.NS','MOINFRA.NS','FLEXIADD.NS','ITBEES.NS','NIFITETF.NS','UTISENSETF.NS','NIFTYQLITY.NS','NIF100BEES.NS','FMCGIETF.NS','SBIETFQLTY.NS','CONS.NS','SETFNN50.NS','TOP15IETF.NS','NV20BEES.NS','QUAL30IETF.NS','MSCIINDIA.NS','AUTOIETF.NS','LOWVOL.NS','MOVALUE.NS','COMMOIETF.NS','GROWWEV.NS','QUALITY30.NS','AUTOBEES.NS','HDFCNEXT50.NS','HEALTHADD.NS','LICNFNHGP.NS','AXISTECETF.NS','ITIETF.NS','MONEXT50.NS','BBNPNBETF.NS','HDFCQUAL.NS','HDFCNIFIT.NS','MIDQ50ADD.NS','CPSEETF.NS','AXISBPSETF.NS','MOPSE.NS','GROWWNXT50.NS','GROWWDEFNC.NS','NPBET.NS','MOCAPITAL.NS','HDFCLOWVOL.NS','SILVERETF.NS','GOLDBEES.NS','SBISILVER.NS','SILVERBEES.NS']

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

            # print(f"{signal:>12} {trend:>8} → {t}: ₹{price:.2f} | RSI={rsi:.2f} | Conf={confidence}%")

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