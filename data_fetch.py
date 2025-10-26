import yfinance as yf
import pandas as pd

def fetch_etf_data(ticker, start=None, end=None, period="1y", interval="1d"):
    """
    Fetch ETF price data from Yahoo Finance.
    Allows both period-based and start-end date-based fetches.
    """
    try:
        if start and end:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period=period, interval=interval, progress=False)

        if data.empty:
            print(f"⚠️ No data for {ticker}")
            return pd.DataFrame()

        data.reset_index(inplace=True)
        return data

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
