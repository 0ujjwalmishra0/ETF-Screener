import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def make_sparkline(prices):
    if not isinstance(prices, (list, pd.Series)) or len(prices) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(1.8, 0.4))
    ax.plot(prices, color="blue", linewidth=1)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_dashboard_html(df):
    if "PriceHistory" in df.columns:
        df["Sparkline"] = df["PriceHistory"].apply(make_sparkline)

    # Add Buy/Sell/Neutral tags
    def format_action(action):
        if action.lower() == "buy":
            return f"<span style='color:green;font-weight:bold;'>BUY</span>"
        elif action.lower() == "sell":
            return f"<span style='color:red;font-weight:bold;'>SELL</span>"
        return f"<span style='color:orange;'>HOLD</span>"

    df["Signal"] = df["Signal"].apply(format_action)

    html = f"""
    <html>
    <head>
        <title>ETF Screener Pro 2026</title>
        <style>
            body {{ font-family: Arial; background-color:#f5f5f5; padding:40px; }}
            h1 {{ color:#222; }}
            table {{ width:100%; border-collapse: collapse; background:white; }}
            th, td {{ border:1px solid #ddd; padding:8px; text-align:center; }}
            th {{ background:#4b6cb7; color:white; }}
            tr:hover {{ background:#f1f1f1; }}
            .button {{
                background-color:#007bff;
                color:white;
                padding:8px 15px;
                border-radius:6px;
                text-decoration:none;
                font-weight:bold;
            }}
        </style>
    </head>
    <body>
        <h1>📊 ETF Screener Pro 2026</h1>
        <a href="/portfolio" class="button">💰 View Portfolio</a>
        {df.to_html(escape=False, index=False)}
    </body>
    </html>
    """
    return html
