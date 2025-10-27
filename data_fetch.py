import yfinance as yf
import pandas as pd

def fetch_etf_data(ticker, start=None, end=None, period="1y", interval="1d"):
    """
    Fetch ETF price data from Yahoo Finance and flatten MultiIndex columns.
    """
    try:
        if start and end:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period=period, interval=interval, progress=False)

        if data.empty:
            print(f"⚠️ No data for {ticker}")
            return pd.DataFrame()

        # 🧠 Flatten MultiIndex columns (handles the new yfinance format)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] != "Date" else "Date" for col in data.columns]

        data.reset_index(inplace=True)

        # Keep only expected columns
        expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        cols = [c for c in expected_cols if c in data.columns]
        data = data[cols]

        # print(f"✅ Cleaned {ticker}: {len(data)} rows, columns={data.columns.tolist()}")
        return data

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
