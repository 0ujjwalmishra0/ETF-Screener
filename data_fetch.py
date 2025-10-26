import yfinance as yf
import pandas as pd

def fetch_etf_data(ticker: str) -> pd.DataFrame:
    """Download last year of data for a single ETF."""
    data = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
    return data