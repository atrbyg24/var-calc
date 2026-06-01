import numpy as np
from .config import PORTFOLIO_POSITIONS, TOTAL_PORTFOLIO_VALUE

def calculate_historical_var_es(returns, confidence_levels):
    """
    Calculate 1-day VaR and Expected Shortfall using Historical Simulation.
    
    :param returns: DataFrame of asset returns (250 days)
    :param confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
    :return: dict mapping confidence level to (VaR, ES) in USD
    """
    weights = np.array([PORTFOLIO_POSITIONS[ticker] / TOTAL_PORTFOLIO_VALUE for ticker in returns.columns])
    
    # Portfolio daily returns
    portfolio_returns = returns.dot(weights)
    
    # Portfolio PnL for 1 day
    portfolio_pnl = portfolio_returns * TOTAL_PORTFOLIO_VALUE
    
    results = {}
    for cl in confidence_levels:
        alpha = 1 - cl
        # Historical VaR is the negative of the alpha-quantile of PnL
        var = -np.percentile(portfolio_pnl, alpha * 100, method='linear')
        
        # Expected shortfall is the negative average of PnL below the negative VaR
        es = -portfolio_pnl[portfolio_pnl <= -var].mean()
        
        results[cl] = (var, es)
        
    return results
