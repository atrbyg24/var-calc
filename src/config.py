# config.py

# Portfolio Positions (in USD)
PORTFOLIO_POSITIONS = {
    "^DJI": 4_000_000,   # DJIA
    "^FTSE": 3_000_000,  # FTSE 100
    "^FCHI": 2_000_000,  # CAC 40
    "^N225": 1_000_000   # Nikkei 225
}

# Total Portfolio Value
TOTAL_PORTFOLIO_VALUE = sum(PORTFOLIO_POSITIONS.values())

# Confidence Levels for VaR and ES
CONFIDENCE_LEVELS = [0.95, 0.99]

# Number of trading days for historical data
LOOKBACK_DAYS = 252

# Target FX rates to convert local index closing prices to USD
# Yahoo Finance tickers for exchange rates
# GBPUSD=X (GBP to USD), EURUSD=X (EUR to USD), JPYUSD=X (JPY to USD)
FX_TICKERS = {
    "^FTSE": "GBPUSD=X",
    "^FCHI": "EURUSD=X",
    "^N225": "JPYUSD=X", 
    "^DJI": None # Already in USD
}
