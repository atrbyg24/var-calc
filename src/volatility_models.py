import pandas as pd
import numpy as np
from arch import arch_model

def calculate_ewma_volatility(returns, decay=0.94):
    """
    Calculate future 1-day volatility for each asset using EWMA.
    Uses the RiskMetrics standard lambda = 0.94.
    
    :param returns: DataFrame of asset returns
    :param decay: Lambda decay factor
    :return: Series of next day volatility forecasts
    """
    # Using pandas EWM to calculate variance
    # alpha = 1 - decay
    ewm_variance = returns.ewm(alpha=1 - decay).var()
    
    # The forecast for next day is the last available EWMA variance
    future_variance = ewm_variance.iloc[-1]
    
    return np.sqrt(future_variance)

def calculate_garch_volatility(returns):
    """
    Calculate future 1-day volatility for each asset using GARCH(1,1).
    
    :param returns: DataFrame of asset returns
    :return: Series of next day volatility forecasts
    """
    future_vols = {}
    
    # We fit a separate GARCH model for each asset
    for col in returns.columns:
        # Rescale returns to help optimizer convergence (multiply by 100)
        rescaled_returns = returns[col] * 100
        
        am = arch_model(rescaled_returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        res = am.fit(disp='off')
        
        # Forecast 1 step ahead
        forecasts = res.forecast(horizon=1)
        future_var_rescaled = forecasts.variance.iloc[-1, 0]
        
        # Convert back to original scale
        future_var = future_var_rescaled / 10000
        future_vols[col] = np.sqrt(future_var)
        
    return pd.Series(future_vols)

def calculate_ewma_covariance(returns, decay=0.94):
    """
    Calculate the next day covariance matrix using EWMA.
    
    :param returns: DataFrame of asset returns
    :param decay: Lambda decay factor
    :return: 2D numpy array of covariance forecasts
    """
    return returns.ewm(alpha=1 - decay).cov().loc[returns.index[-1]].values

def calculate_garch_covariance(returns):
    """
    Calculate the next day covariance matrix using CCC-GARCH.
    Constant Conditional Correlation (CCC) GARCH uses individual GARCH(1,1)
    volatility forecasts combined with the sample correlation matrix.
    
    :param returns: DataFrame of asset returns
    :return: 2D numpy array of covariance forecasts
    """
    # Get GARCH(1,1) volatility forecasts
    garch_vols = calculate_garch_volatility(returns)
    
    # Calculate sample correlation matrix
    corr_matrix = returns.corr().values
    
    # Create diagonal matrix of GARCH volatilities
    D = np.diag(garch_vols.values)
    
    # Calculate predicted covariance matrix: D * R * D
    cov_matrix = D @ corr_matrix @ D
    
    return cov_matrix
