import pandas as pd
import numpy as np

from src.config import CONFIDENCE_LEVELS, TOTAL_PORTFOLIO_VALUE
from src.data_fetcher import fetch_data
from src.historical_simulation import calculate_historical_var_es
from src.model_building import calculate_model_building_var_es
from src.volatility_models import (
    calculate_ewma_volatility, 
    calculate_garch_volatility,
    calculate_ewma_covariance,
    calculate_garch_covariance
)
from src.backtesting import run_backtest, print_backtest_report

def format_currency(value):
    return f"${value:,.2f}"

def format_percentage(value):
    return f"{value * 100:.2f}%"

def main():
    print("=" * 60)
    print("Value at Risk (VaR) and Expected Shortfall Calculator")
    print(f"Total Portfolio Value: {format_currency(TOTAL_PORTFOLIO_VALUE)}")
    print("=" * 60)
    
    # 1. Fetch Data
    print("\nFetching market data and calculating USD returns...")
    # Fetch 505 trading days of prices to get 504 days of returns
    # This allows a 252-day backtest with a 252-day estimation window
    prices, returns = fetch_data(num_days=505)
    print(f"Data fetched successfully. Start: {returns.index[0].date()}, End: {returns.index[-1].date()}")
    print(f"Number of observations: {len(returns)}")
    
    # Use the latest 252 returns for current forward-looking calculations
    latest_returns = returns.tail(252)
    print(f"Using latest {len(latest_returns)} observations for current VaR/ES estimation.")
    print(f"Estimation Start: {latest_returns.index[0].date()}, End: {latest_returns.index[-1].date()}")
    
    # 2. Historical Simulation
    print("\n--- 1. Historical Simulation (Latest 252 Days) ---")
    hist_results = calculate_historical_var_es(latest_returns, CONFIDENCE_LEVELS)
    for cl in CONFIDENCE_LEVELS:
        var, es = hist_results[cl]
        print(f"Confidence Level {cl*100:.2f}%:")
        print(f"  1-Day VaR:  {format_currency(var)} ({format_percentage(var / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  1-Day ES:   {format_currency(es)} ({format_percentage(es / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day VaR: {format_currency(var * np.sqrt(10))} ({format_percentage((var * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day ES:  {format_currency(es * np.sqrt(10))} ({format_percentage((es * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")

    # 3. Model Building (Historical Covariance)
    print("\n--- 2. Model Building (Historical Covariance, Latest 252 Days) ---")
    model_results = calculate_model_building_var_es(latest_returns, CONFIDENCE_LEVELS)
    for cl in CONFIDENCE_LEVELS:
        var, es = model_results[cl]
        print(f"Confidence Level {cl*100:.2f}%:")
        print(f"  1-Day VaR:  {format_currency(var)} ({format_percentage(var / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  1-Day ES:   {format_currency(es)} ({format_percentage(es / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day VaR: {format_currency(var * np.sqrt(10))} ({format_percentage((var * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day ES:  {format_currency(es * np.sqrt(10))} ({format_percentage((es * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")

    # 4. Model Building (EWMA Covariance)
    print("\n--- 3. Model Building (EWMA Covariance, Latest 252 Days) ---")
    ewma_cov = calculate_ewma_covariance(latest_returns)
    ewma_results = calculate_model_building_var_es(latest_returns, CONFIDENCE_LEVELS, cov_matrix=ewma_cov)
    for cl in CONFIDENCE_LEVELS:
        var, es = ewma_results[cl]
        print(f"Confidence Level {cl*100:.2f}%:")
        print(f"  1-Day VaR:  {format_currency(var)} ({format_percentage(var / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  1-Day ES:   {format_currency(es)} ({format_percentage(es / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day VaR: {format_currency(var * np.sqrt(10))} ({format_percentage((var * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day ES:  {format_currency(es * np.sqrt(10))} ({format_percentage((es * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")

    # 5. Model Building (GARCH Covariance)
    print("\n--- 4. Model Building (GARCH(1,1) Covariance, Latest 252 Days) ---")
    garch_cov = calculate_garch_covariance(latest_returns)
    garch_results = calculate_model_building_var_es(latest_returns, CONFIDENCE_LEVELS, cov_matrix=garch_cov)
    for cl in CONFIDENCE_LEVELS:
        var, es = garch_results[cl]
        print(f"Confidence Level {cl*100:.2f}%:")
        print(f"  1-Day VaR:  {format_currency(var)} ({format_percentage(var / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  1-Day ES:   {format_currency(es)} ({format_percentage(es / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day VaR: {format_currency(var * np.sqrt(10))} ({format_percentage((var * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")
        print(f"  10-Day ES:  {format_currency(es * np.sqrt(10))} ({format_percentage((es * np.sqrt(10)) / TOTAL_PORTFOLIO_VALUE)})")

    # 6. Future Volatility Forecasts
    print("\n--- 5. Future Volatility Forecasts (Next Day) ---")
    
    print("\nEWMA (lambda=0.94):")
    ewma_vols = calculate_ewma_volatility(latest_returns)
    for ticker, vol in ewma_vols.items():
        print(f"  {ticker:^6}: {format_percentage(vol)} daily ({format_percentage(vol * np.sqrt(252))} annualized)")
        
    print("\nGARCH(1,1):")
    garch_vols = calculate_garch_volatility(latest_returns)
    for ticker, vol in garch_vols.items():
        print(f"  {ticker:^6}: {format_percentage(vol)} daily ({format_percentage(vol * np.sqrt(252))} annualized)")

    # 7. VaR Backtesting
    print("\n--- 6. VaR Backtesting (1-Year / 252-Day Rolling Window) ---")
    backtest_results = run_backtest(returns, CONFIDENCE_LEVELS, backtest_days=252, lookback_window=252)
    print_backtest_report(backtest_results, CONFIDENCE_LEVELS)

    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Disable warnings for clean output
    import warnings
    warnings.filterwarnings('ignore')
    
    main()
