import pandas as pd
import numpy as np

def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, 50DMA, 200DMA, and Z-Score indicators safely.
    """
    print(f"Computing basic indicators")
    # Ensure single-level columns (sometimes yfinance returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Ensure Close column exists
    if "Close" not in df.columns:
        raise ValueError("Missing 'Close' column in input DataFrame")
    # --- 1️⃣ 50DMA & 200DMA ---
    df["50DMA"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["200DMA"] = df["Close"].rolling(window=200, min_periods=1).mean()
    # --- 2️⃣ RSI (14-day) ---
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(14).mean()

    rs = roll_up / (roll_down.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- 3️⃣ Z-Score ---
    rolling_mean = df["Close"].rolling(window=50, min_periods=1).mean()
    rolling_std = df["Close"].rolling(window=50, min_periods=1).std()
    # avoid divide-by-zero issues
    df["ZScore"] = np.where(
        rolling_std == 0, 0, (df["Close"] - rolling_mean) / rolling_std
    )
    # --- 4️⃣ Clean up ---
    df = df.dropna(subset=["RSI", "50DMA", "200DMA", "ZScore"]).copy()

    return df