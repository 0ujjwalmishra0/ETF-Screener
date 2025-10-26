import matplotlib.pyplot as plt
import os

def create_performance_chart(df, name):
    """
    Generates a full historical performance chart (price + moving averages)
    for each ETF.
    """
    os.makedirs("charts", exist_ok=True)

    plt.figure(figsize=(6, 2))
    plt.plot(df["Close"], label="Price", color="#0078D7", linewidth=1.5)
    if "50DMA" in df:
        plt.plot(df["50DMA"], label="50DMA", color="#00b050", linestyle="--", linewidth=1)
    if "200DMA" in df:
        plt.plot(df["200DMA"], label="200DMA", color="#ff5050", linestyle="--", linewidth=1)

    plt.title(name.replace(".NS", ""), fontsize=9)
    plt.legend(fontsize=6, loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout(pad=0.5)
    path = f"charts/{name.replace('.NS','')}_chart.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path
