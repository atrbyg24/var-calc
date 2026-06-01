import numpy as np
from scipy.stats import norm
from .config import PORTFOLIO_POSITIONS, TOTAL_PORTFOLIO_VALUE

def calculate_model_building_var_es(returns, confidence_levels, cov_matrix=None):
    """
    Calculate 1-day VaR and Expected Shortfall using the Variance-Covariance method.
    Assumes normal distribution of returns.
    
    :param returns: DataFrame of asset returns (250 days)
    :param confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
    :param cov_matrix: Optional. Provide a custom covariance matrix (e.g., from EWMA or GARCH). If None, uses sample covariance.
    :return: dict mapping confidence level to (VaR, ES) in USD
    """
    weights = np.array([PORTFOLIO_POSITIONS[ticker] / TOTAL_PORTFOLIO_VALUE for ticker in returns.columns])
    
    # Calculate covariance matrix if not provided
    if cov_matrix is None:
        cov_matrix = returns.cov().values
        
    # Check if the covariance matrix is positive semi-definite
    # We use a small tolerance for floating point inaccuracies
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigenvalues < -1e-8):
        raise ValueError("The provided covariance matrix is not positive semi-definite.")
    
    # Calculate portfolio variance and volatility (standard deviation)
    port_variance = weights.T @ cov_matrix @ weights
    port_volatility = np.sqrt(port_variance)
    
    results = {}
    for cl in confidence_levels:
        alpha = 1 - cl
        # z-score corresponding to the confidence level
        z_score = norm.ppf(cl)
        
        # VaR under normal distribution
        var = TOTAL_PORTFOLIO_VALUE * z_score * port_volatility
        
        # ES under normal distribution
        # ES = Value * Volatility * phi(z_score) / alpha
        es = TOTAL_PORTFOLIO_VALUE * port_volatility * norm.pdf(z_score) / alpha
        
        results[cl] = (var, es)
        
    return results
