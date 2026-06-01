import numpy as np
import pandas as pd
from scipy.stats import chi2
from .config import PORTFOLIO_POSITIONS, TOTAL_PORTFOLIO_VALUE
from .historical_simulation import calculate_historical_var_es
from .model_building import calculate_model_building_var_es
from .volatility_models import calculate_ewma_covariance, calculate_garch_covariance

def perform_kupiec_pof_test(exceptions, cl):
    """
    Perform Kupiec's Proportion of Failure (POF) test (Likelihood Ratio test).
    Null hypothesis: The true exception rate is equal to the expected exception rate (1 - cl).
    """
    N = len(exceptions)
    x = int(np.sum(exceptions))
    p = 1.0 - cl
    p_hat = x / N
    
    # Handle boundary conditions to avoid log(0)
    if x == 0:
        num = N * np.log(1.0 - p)
        den = 0.0
    elif x == N:
        num = N * np.log(p)
        den = 0.0
    else:
        num = (N - x) * np.log(1.0 - p) + x * np.log(p)
        den = (N - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
        
    lr_stat = -2.0 * (num - den)
    lr_stat = max(0.0, lr_stat) # Ensure non-negative
    
    p_value = chi2.sf(lr_stat, df=1)
    reject = p_value < 0.05
    
    return {
        'lr_stat': lr_stat,
        'p_value': p_value,
        'reject': reject,
        'exceptions': x,
        'n_obs': N
    }

def perform_christoffersen_independence_test(exceptions):
    """
    Perform Christoffersen's Independence test.
    Null hypothesis: The exceptions are independent over time (no clustering).
    """
    N = len(exceptions)
    if N < 2:
        return {
            'lr_stat': 0.0,
            'p_value': 1.0,
            'reject': False,
            't00': 0,
            't01': 0,
            't10': 0,
            't11': 0
        }
        
    t00 = t01 = t10 = t11 = 0
    for i in range(1, N):
        prev = exceptions[i - 1]
        curr = exceptions[i]
        if prev == 0 and curr == 0:
            t00 += 1
        elif prev == 0 and curr == 1:
            t01 += 1
        elif prev == 1 and curr == 0:
            t10 += 1
        elif prev == 1 and curr == 1:
            t11 += 1
            
    total_ex = t01 + t11
    # If there are no exceptions or no transitions, independence cannot be rejected
    if total_ex == 0 or (t00 + t01) == 0 or (t10 + t11) == 0:
        return {
            'lr_stat': 0.0,
            'p_value': 1.0,
            'reject': False,
            't00': t00,
            't01': t01,
            't10': t10,
            't11': t11
        }
        
    pi01 = t01 / (t00 + t01)
    pi11 = t11 / (t10 + t11)
    pi = (t01 + t11) / (t00 + t01 + t10 + t11)
    
    # Calculate likelihood under null (independence) and alternative (first-order Markov)
    ln_null = 0.0
    if pi > 0:
        ln_null += (t01 + t11) * np.log(pi)
    if 1 - pi > 0:
        ln_null += (t00 + t10) * np.log(1.0 - pi)
        
    ln_alt = 0.0
    if pi01 > 0:
        ln_alt += t01 * np.log(pi01)
    if 1 - pi01 > 0:
        ln_alt += t00 * np.log(1.0 - pi01)
    if pi11 > 0:
        ln_alt += t11 * np.log(pi11)
    if 1 - pi11 > 0:
        ln_alt += t10 * np.log(1.0 - pi11)
        
    lr_stat = -2.0 * (ln_null - ln_alt)
    lr_stat = max(0.0, lr_stat)
    
    p_value = chi2.sf(lr_stat, df=1)
    reject = p_value < 0.05
    
    return {
        'lr_stat': lr_stat,
        'p_value': p_value,
        'reject': reject,
        't00': t00,
        't01': t01,
        't10': t10,
        't11': t11
    }

def perform_conditional_coverage_test(exceptions, cl):
    """
    Perform Christoffersen's Conditional Coverage test.
    Combines the POF test and the Independence test (LR_cc = LR_pof + LR_ind).
    """
    pof_res = perform_kupiec_pof_test(exceptions, cl)
    ind_res = perform_christoffersen_independence_test(exceptions)
    
    lr_stat = pof_res['lr_stat'] + ind_res['lr_stat']
    p_value = chi2.sf(lr_stat, df=2)
    reject = p_value < 0.05
    
    return {
        'lr_stat': lr_stat,
        'p_value': p_value,
        'reject': reject,
        'pof_reject': pof_res['reject'],
        'ind_reject': ind_res['reject']
    }

def determine_basel_zone(exceptions, cl):
    """
    Determine the Basel traffic light zone and penalty factor.
    Strictly defined for 99% VaR and a 250-day backtesting window.
    """
    x = int(np.sum(exceptions))
    
    # Basel framework zones for 1-sided 99% VaR with N=250 observations:
    if x <= 4:
        zone = 'Green'
        penalty = 0.0
    elif x <= 9:
        zone = 'Yellow'
        # Multipliers / penalty additions for yellow zone:
        # 5 -> 0.40, 6 -> 0.50, 7 -> 0.65, 8 -> 0.75, 9 -> 0.85
        penalties = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
        penalty = penalties.get(x, 0.40)
    else:
        zone = 'Red'
        penalty = 1.0
        
    return {
        'zone': zone,
        'penalty': penalty,
        'exceptions': x
    }

def run_backtest(returns, confidence_levels, backtest_days=252, lookback_window=252):
    """
    Run a rolling backtest over the last `backtest_days` of returns.
    Estimates models dynamically using `lookback_window` returns at each step.
    """
    weights = np.array([PORTFOLIO_POSITIONS[ticker] / TOTAL_PORTFOLIO_VALUE for ticker in returns.columns])
    
    methods = [
        'Historical Simulation',
        'Parametric (Historical Cov)',
        'Parametric (EWMA)',
        'Parametric (GARCH)'
    ]
    
    # Initialize dictionary structure for recording results
    raw_results = {
        cl: {
            method: {'exceptions': [], 'vars': []} for method in methods
        } for cl in confidence_levels
    }
    
    total_len = len(returns)
    start_idx = total_len - backtest_days
    
    for idx in range(start_idx, total_len):
        # Periodic progress updates since GARCH fitting can take some time
        if (idx - start_idx) % 50 == 0 or (idx - start_idx) == backtest_days - 1:
            print(f"  Backtesting progress: {idx - start_idx + 1}/{backtest_days} days...")
            
        train_returns = returns.iloc[idx - lookback_window : idx]
        actual_ret = returns.iloc[idx].dot(weights)
        actual_pnl = actual_ret * TOTAL_PORTFOLIO_VALUE
        
        # Calculate VaR forecasts for each model
        # 1. Historical Simulation
        hist_res = calculate_historical_var_es(train_returns, confidence_levels)
        
        # 2. Parametric (Historical Covariance)
        param_res = calculate_model_building_var_es(train_returns, confidence_levels)
        
        # 3. Parametric (EWMA)
        ewma_cov = calculate_ewma_covariance(train_returns)
        ewma_res = calculate_model_building_var_es(train_returns, confidence_levels, cov_matrix=ewma_cov)
        
        # 4. Parametric (GARCH)
        garch_cov = calculate_garch_covariance(train_returns)
        garch_res = calculate_model_building_var_es(train_returns, confidence_levels, cov_matrix=garch_cov)
        
        for cl in confidence_levels:
            # 1-day VaRs (positive values in USD)
            var_hist = hist_res[cl][0]
            var_param = param_res[cl][0]
            var_ewma = ewma_res[cl][0]
            var_garch = garch_res[cl][0]
            
            # Record VaR forecasts
            raw_results[cl]['Historical Simulation']['vars'].append(var_hist)
            raw_results[cl]['Parametric (Historical Cov)']['vars'].append(var_param)
            raw_results[cl]['Parametric (EWMA)']['vars'].append(var_ewma)
            raw_results[cl]['Parametric (GARCH)']['vars'].append(var_garch)
            
            # Record whether loss exceeded VaR (actual_pnl < -var)
            raw_results[cl]['Historical Simulation']['exceptions'].append(1 if actual_pnl < -var_hist else 0)
            raw_results[cl]['Parametric (Historical Cov)']['exceptions'].append(1 if actual_pnl < -var_param else 0)
            raw_results[cl]['Parametric (EWMA)']['exceptions'].append(1 if actual_pnl < -var_ewma else 0)
            raw_results[cl]['Parametric (GARCH)']['exceptions'].append(1 if actual_pnl < -var_garch else 0)
            
    # Process raw results and perform tests
    results = {}
    for cl in confidence_levels:
        results[cl] = {}
        for method in methods:
            exceptions = np.array(raw_results[cl][method]['exceptions'])
            vars_array = np.array(raw_results[cl][method]['vars'])
            
            results[cl][method] = {
                'exceptions': exceptions,
                'vars': vars_array,
                'num_exceptions': int(np.sum(exceptions)),
                'kupiec': perform_kupiec_pof_test(exceptions, cl),
                'christoffersen': perform_christoffersen_independence_test(exceptions),
                'cc': perform_conditional_coverage_test(exceptions, cl),
                'basel': determine_basel_zone(exceptions, cl)
            }
            
    return results

def print_backtest_report(results, confidence_levels):
    """
    Print a detailed summary table of the VaR backtest results.
    """
    print("\n" + "=" * 115)
    print(f"{'VaR Backtesting Report (252-day Rolling Window)':^115}")
    print("=" * 115)
    
    headers = f"{'Method':<28} | {'CL':<4} | {'Expected':<8} | {'Actual':<6} | {'Kupiec p-val':<12} | {'Indep p-val':<11} | {'CondCov p-val':<13} | {'Basel Zone':<13}"
    print(headers)
    print("-" * 115)
    
    for cl in confidence_levels:
        expected_exceptions = len(next(iter(results[cl].values()))['exceptions']) * (1.0 - cl)
        for method, res in results[cl].items():
            num_ex = res['num_exceptions']
            
            kupiec_p = res['kupiec']['p_value']
            kupiec_rej = "REJECT" if res['kupiec']['reject'] else "PASS"
            
            ind_p = res['christoffersen']['p_value']
            ind_rej = "REJECT" if res['christoffersen']['reject'] else "PASS"
            
            cc_p = res['cc']['p_value']
            cc_rej = "REJECT" if res['cc']['reject'] else "PASS"
            
            basel_zone = res['basel']['zone']
            basel_penalty = res['basel']['penalty']
            basel_str = f"{basel_zone} (+{basel_penalty:.2f})"
            
            # Format p-values and results
            kp_str = f"{kupiec_p:6.4f} ({kupiec_rej})"
            ip_str = f"{ind_p:6.4f} ({ind_rej})"
            cp_str = f"{cc_p:6.4f} ({cc_rej})"
            
            row = f"{method:<28} | {cl*100:2.0f}% | {expected_exceptions:<8.1f} | {num_ex:<6d} | {kp_str:<12} | {ip_str:<11} | {cp_str:<13} | {basel_str:<13}"
            print(row)
        print("-" * 115)
