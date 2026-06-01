# Value at Risk (VaR) and Expected Shortfall (ES) Calculator

A Python program for calculating, forecasting, and backtesting **Value at Risk (VaR)** and **Expected Shortfall (ES)** for multi-asset portfolios. 

The project fetches historical pricing data dynamically via Yahoo Finance, aligns trading days across international markets, adjusts for exchange rate fluctuations to compute portfolio returns in USD, and implements multiple advanced risk modeling techniques alongside regulatory backtesting suites.

---

## Portfolio Overview

The default portfolio is configured in USD with a total value of **\$10,000,000.00**, split across major global equity indices:

| Ticker | Index | Market | Currency | Allocation (USD) | Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `^DJI` | Dow Jones Industrial Average | United States | USD | \$4,000,000.00 | 40% |
| `^FTSE` | FTSE 100 | United Kingdom | GBP | \$3,000,000.00 | 30% |
| `^FCHI` | CAC 40 | France | EUR | \$2,000,000.00 | 20% |
| `^N225` | Nikkei 225 | Japan | JPY | \$1,000,000.00 | 10% |

All asset prices are converted to USD on a daily basis using contemporaneous exchange rates (`GBPUSD=X`, `EURUSD=X`, `JPYUSD=X`) fetched from Yahoo Finance.

---

## Methodologies & Mathematics

### 1. Historical Simulation (Non-Parametric)
Historical Simulation makes no assumptions about the distribution of asset returns. It computes the portfolio's historical P&L distribution using the asset weights:

$$
\text{Portfolio Return at } t \quad R_{p,t} = \sum_{i=1}^{n} w_i R_{i,t}
$$

$$
\text{Portfolio PnL at } t \quad \text{PnL}_t = R_{p,t} \times V_0
$$

Where $V_0$ is the total portfolio value (\$10,000,000).

- **Value at Risk** ($\text{VaR}_\alpha$): The negative of the $(1-\alpha)$-quantile of the portfolio P&L distribution:
  $$
  \text{VaR}_\alpha = -\text{Percentile}(\{\text{PnL}_t\}, (1-\alpha) \times 100)
  $$
- **Expected Shortfall** ($\text{ES}_\alpha$): The average loss given that the loss exceeds the $\text{VaR}_\alpha$ threshold:
  $$
  \text{ES}_\alpha = -E[\text{PnL}_t \mid \text{PnL}_t \le -\text{VaR}_\alpha]
  $$

---

### 2. Model Building (Parametric / Variance-Covariance)
The parametric approach assumes that portfolio returns follow a multivariate normal distribution:
$$
R_p \sim N(0, \sigma_p^2)
$$

$$
\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}
$$

Where $\mathbf{w}$ is the vector of asset weights and $\mathbf{\Sigma}$ is the covariance matrix.

- **Value at Risk** ($\text{VaR}_\alpha$):
  $$
  \text{VaR}_\alpha = V_0 \times z_\alpha \times \sigma_p
  $$
- **Expected Shortfall** ($\text{ES}_\alpha$):
  $$
  \text{ES}_\alpha = V_0 \times \sigma_p \times \frac{\phi(z_\alpha)}{1 - \alpha}
  $$

Where $z_\alpha = \Phi^{-1}(\alpha)$ is the inverse cumulative distribution function (CDF) of the standard normal distribution, and $\phi(x)$ is the standard normal probability density function (PDF).

---

### 3. Volatility & Covariance Forecasting Models
To capture volatility clustering and time-varying correlations, the library implements two advanced dynamic covariance models:

#### A. Exponentially Weighted Moving Average (EWMA)
Following the **RiskMetrics** standard, EWMA assigns higher weights to recent returns using a decay factor $\lambda$ (default $\lambda = 0.94$):
$$
\sigma_{i,j,t}^2 = \lambda \sigma_{i,j,t-1}^2 + (1 - \lambda) R_{i,t-1} R_{j,t-1}
$$

#### B. Constant Conditional Correlation GARCH (CCC-GARCH(1,1))
To capture mean-reverting volatility, individual univariate GARCH(1,1) models are fitted to each asset's return series:
$$
\sigma_{i,t}^2 = \omega_i + \alpha_i R_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2
$$

The next-day covariance matrix is then computed by combining the GARCH(1,1) dynamic volatility forecasts with the constant sample correlation matrix $\mathbf{R}$:
$$
\mathbf{\Sigma}_t = \mathbf{D}_t \mathbf{R} \mathbf{D}_t
$$

Where $\mathbf{D}_t = \text{diag}(\sigma_{1,t}, \dots, \sigma_{n,t})$ is the diagonal matrix of dynamic standard deviations.

---

## Statistical Backtesting Suite

To validate the model accuracy, the library implements a rolling backtesting system (default: 252-day window) and evaluates exceptions (days where the actual portfolio loss exceeds the predicted 1-day VaR) using standard quantitative diagnostics:

### 1. Kupiec's Proportion of Failures (POF) Test
Checks whether the observed frequency of exceptions differs statistically from the expected exception rate $p = 1 - \alpha$. It is formulated as a Likelihood Ratio (LR) test:
$$
LR_{POF} = -2 \ln \left( \frac{(1-p)^{N-x} p^x}{(1-\hat{p})^{N-x} \hat{p}^x} \right) \sim \chi^2(1)
$$

Where $N$ is the number of backtesting observations, $x$ is the number of exceptions, and $\hat{p} = x/N$.

### 2. Christoffersen's Independence Test
Tests whether exceptions are independent over time (i.e. checking for exception clustering). It models the transition of exceptions as a first-order Markov chain:
$$
LR_{ind} = -2 \ln \left( \frac{(1-\pi)^{T_{00}+T_{10}} \pi^{T_{01}+T_{11}}}{(1-\pi_{01})^{T_{00}} \pi_{01}^{T_{01}} (1-\pi_{11})^{T_{10}} \pi_{11}^{T_{11}} \right) \sim \chi^2(1)
$$

Where $T_{ij}$ is the count of transitions from state $i$ to state $j$ ($0$ = no exception, $1$ = exception), $\pi_{01} = \frac{T_{01}}{T_{00}+T_{01}}$, $\pi_{11} = \frac{T_{11}}{T_{10}+T_{11}}$, and $\pi = \frac{T_{01}+T_{11}}{T_{00}+T_{01}+T_{10}+T_{11}}$.

### 3. Conditional Coverage Test
Combines the POF test and the Independence test to evaluate both frequency and independence simultaneously:
$$
LR_{cc} = LR_{POF} + LR_{ind} \sim \chi^2(2)
$$

### 4. Basel Traffic Light Zones
In accordance with Basel Accord regulations, models are categorized into three color zones based on the number of exceptions over a 250-day period at the 99% confidence level:

- **Green Zone** ($\le 4$ exceptions): Model is considered accurate. Penalty factor = 0.00.
- **Yellow Zone** (5 to 9 exceptions): Model is monitored. A penalty factor between 0.40 and 0.85 is added to the market risk capital multiplier.
- **Red Zone** ($\ge 10$ exceptions): Model is deemed inaccurate. Capital penalty factor = 1.00 (automatic supervisor intervention).

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- A virtual environment tool (optional but recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/atrbyg24/var-calc.git
   cd var-calc
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
To run the full suite, including historical pricing downloads, currency conversion, VaR/ES calculations for all models, and the 252-day rolling backtest report:
```bash
python main.py
```

### Running Tests
To run the automated mathematical test suite verifying the backtesting formulas, Kupiec POF, Christoffersen independence, and Basel zones:
```bash
python -m unittest tests/verify_backtesting.py
```

---

## Project Structure

```
var-calc/
│
├── main.py                    # Entry point; runs data fetch, VaR/ES calculations, and backtests
├── requirements.txt           # External Python dependencies (yfinance, arch, pandas, scipy, etc.)
│
├── src/
│   ├── __init__.py            # Packages the source files
│   ├── config.py              # Configuration values (allocations, tickers, confidence levels)
│   ├── data_fetcher.py        # Fetches and aligns index data and currency rates from Yahoo Finance
│   ├── historical_simulation.py  # Calculates historical simulation VaR and ES
│   ├── model_building.py      # Calculates parametric variance-covariance VaR and ES
│   ├── volatility_models.py   # Computes dynamic forecasts using EWMA and CCC-GARCH(1,1)
│   └── backtesting.py         # Implements statistical backtests (Kupiec, Christoffersen, Basel zones)
│
└── tests/
    └── verify_backtesting.py  # Unit tests for the statistical backtesting formulas
```
