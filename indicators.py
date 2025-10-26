import pandas as pd

def compute_basic_indicators(df):
    # Ensure single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Compute 50DMA and 200DMA
    df["50DMA"] = df["Close"].rolling(50).mean()
    df["200DMA"] = df["Close"].rolling(200).mean()

    # RSI calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Fix for Z-Score calculation
    rolling_std = df["Close"].rolling(50).std()
    df["ZScore"] = (df["Close"] - df["50DMA"]) / rolling_std.squeeze()

    # Drop NaN rows from beginning
    df = df.dropna(subset=["RSI", "50DMA", "200DMA", "ZScore"])
    return df
