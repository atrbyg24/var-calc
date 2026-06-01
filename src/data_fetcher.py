import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pandas_market_calendars as mcal
from .config import PORTFOLIO_POSITIONS, FX_TICKERS, LOOKBACK_DAYS

def get_us_trading_days(num_days=LOOKBACK_DAYS + 1):
    """
    Get the last `num_days` of US trading days up to today.
    """
    nyse = mcal.get_calendar('NYSE')
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=int(num_days * 1.5) + 30)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index[-num_days:].tz_localize(None)

def fetch_data(num_days=None):
    """
    Fetch index data and FX data, align to US trading days, forward fill and back fill,
    convert to USD, and return the price dataframe and returns dataframe.
    """
    if num_days is None:
        num_days = LOOKBACK_DAYS + 1
    us_trading_days = get_us_trading_days(num_days)
    
    start_date = us_trading_days[0].strftime('%Y-%m-%d')
    # add 1 day to end_date to ensure yfinance gets the last day
    end_date = (us_trading_days[-1] + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    tickers = list(PORTFOLIO_POSITIONS.keys())
    fx_tickers = [fx for fx in FX_TICKERS.values() if fx is not None]
    
    all_tickers = tickers + fx_tickers
    
    print(f"Downloading data from {start_date} to {end_date}...")
    data = yf.download(all_tickers, start=start_date, end=end_date)['Close']
    
    # Ensure timezone naive for merging
    data.index = pd.to_datetime(data.index).tz_localize(None)
    
    # Reindex to US trading days
    df = pd.DataFrame(index=us_trading_days)
    df = df.join(data)
    
    # Forward fill then backfill for missing data (e.g. non-US holidays)
    df = df.ffill().bfill()
    
    # Convert local index values to USD using exchange rates
    usd_prices = pd.DataFrame(index=df.index)
    for ticker in tickers:
        fx_ticker = FX_TICKERS.get(ticker)
        if fx_ticker:
            # GBPUSD=X means 1 GBP = X USD.
            # So we multiply the local index by the FX rate
            usd_prices[ticker] = df[ticker] * df[fx_ticker]
        else:
            usd_prices[ticker] = df[ticker]
            
    # Calculate simple returns
    returns = usd_prices.pct_change().dropna()
    
    return usd_prices, returns
