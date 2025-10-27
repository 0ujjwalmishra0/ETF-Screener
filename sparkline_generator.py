import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def create_sparkline(prices, name, trend_up=True):
    """Generate a small sparkline image for recent prices."""
    os.makedirs("sparklines", exist_ok=True)
    color = "#00b050" if trend_up else "#ff4d4d"

    plt.figure(figsize=(1.8, 0.4))
    plt.plot(prices, color=color, linewidth=1.8)
    plt.axis("off")
    plt.tight_layout(pad=0)
    path = f"sparklines/{name}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight", transparent=True)
    plt.close()
    return path
