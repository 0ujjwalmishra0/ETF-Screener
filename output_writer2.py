import pandas as pd
import matplotlib.pyplot as plt
import os
import webbrowser

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "gray"
plt.rcParams["axes.linewidth"] = 0.6


# ===========================================================
# 🎨 Color coding for signals
# ===========================================================
def color_signal(val):
    if isinstance(val, str):
        if "STRONG BUY" in val:
            color = "#006400"  # dark green
        elif "BUY" in val:
            color = "#32CD32"  # lime green
        elif "STRONG WAIT" in val:
            color = "#8B0000"  # dark red
        elif "WAIT" in val:
            color = "#FFA500"  # orange
        else:
            color = "#808080"  # gray
        return f"background-color:{color}; color:white; font-weight:bold; text-align:center"
    return ""


# ===========================================================
# 🌈 Color coding for ETF categories
# ===========================================================
def color_category(val):
    if val == "Core India":
        bg = "#E3F2FD"  # light blue
    elif val == "Global":
        bg = "#FFF3E0"  # soft orange
    elif val == "Sectoral":
        bg = "#E8F5E9"  # pale green
    elif val == "Commodity":
        bg = "#FFFDE7"  # light gold
    else:
        bg = "white"
    return f"background-color:{bg}; text-align:center; font-weight:bold;"


# ===========================================================
# 📈 Sparkline generator for each ETF
# ===========================================================
def generate_sparkline(ticker, prices):
    """Creates a small sparkline and saves it as PNG"""
    try:
        os.makedirs("sparklines", exist_ok=True)
        fig, ax = plt.subplots(figsize=(2, 0.5))
        ax.plot(prices[-30:], color="blue", linewidth=1.5)
        ax.axis("off")
        filepath = f"sparklines/{ticker}.png"
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return filepath
    except Exception:
        return None


# ===========================================================
# 🧾 Write Excel + HTML Dashboard
# ===========================================================
def write_outputs(df):
    if df.empty:
        print("⚠️ No ETF data to write.")
        return

    # Generate sparklines
    sparkline_files = []
    for ticker in df["Ticker"]:
        try:
            import yfinance as yf
            data = yf.download(ticker, period="1mo", interval="1d", progress=False)
            if not data.empty:
                spark = generate_sparkline(ticker, data["Close"])
                sparkline_files.append(f'<img src="{spark}" width="80">')
            else:
                sparkline_files.append("—")
        except Exception:
            sparkline_files.append("—")

    df["Trend"] = sparkline_files

    # Save Excel
    excel_file = "ETF_Screener_Pro_2026.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"📘 Excel saved: {excel_file}")

    # Group by category
    df_sorted = df.sort_values(by=["Category", "ZScore"], ascending=[True, True])

    # Styling for HTML
    styled = (
        df_sorted.style
        .applymap(color_signal, subset=["Signal"])
        .applymap(color_category, subset=["Category"])
        .set_properties(**{"text-align": "center"})
        .hide(axis="index")
    )

    html_content = styled.to_html(escape=False)

    html_template = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>ETF Screener Pro 2026</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, sans-serif;
                background: #f5f7fa;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #1a237e;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                background: white;
            }}
            th {{
                background-color: #004d99;
                color: white;
                padding: 10px;
                text-align: center;
            }}
            td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .summary {{
                text-align: center;
                padding: 10px;
                background-color: #e3f2fd;
                border-radius: 8px;
                margin-bottom: 20px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>📊 ETF Screener Pro 2026 Dashboard</h1>
        <div class="summary">
            Total ETFs: {len(df_sorted)} |
            STRONG BUY: {(df_sorted["Signal"] == "💚 STRONG BUY").sum()} |
            BUY: {(df_sorted["Signal"] == "🟢 BUY").sum()} |
            WAIT: {(df_sorted["Signal"] == "🟠 WAIT").sum()} |
            STRONG WAIT: {(df_sorted["Signal"] == "🔴 STRONG WAIT").sum()}
        </div>
        {html_content}
        <p style="text-align:center; color:gray; font-size:12px;">Auto-updated at 8 AM IST daily</p>
    </body>
    </html>
    """

    html_file = "ETF_Screener_Pro_2026.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"🌐 HTML dashboard saved: {html_file}")

    # Auto-open dashboard
    webbrowser.open("file://" + os.path.abspath(html_file))