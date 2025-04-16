import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA

# -----------------------
# 1) Download price data 
# -----------------------
tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL",
    "TSLA", "GOOG", "BRK-B", "META", "UNH", "SPY"
]

end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=3*365)  # approx 3 years

# Download daily data from Yahoo Finance
#   Note: 'BRK.B' in yfinance is often 'BRK-B'
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# We’ll focus on the 'Adj Close' for returns if available, else fall back to 'Close'
if 'Adj Close' in data.columns.get_level_values(0):
    adj_close = data['Adj Close'].copy()
else:
    adj_close = data['Close'].copy()

# Sometimes, ticker naming can be inconsistent for BRK-B / BRK.B.
# Make sure your adj_close has the correct columns:
#   If "BRK-B" doesn't work on your system, try "BRK.B".
#   For this script, let's attempt to rename if needed:
if 'BRK-B' not in adj_close.columns and 'BRK.B' in adj_close.columns:
    adj_close.rename(columns={'BRK.B': 'BRK-B'}, inplace=True)

# Drop rows with all NaNs (leading days with no data)
adj_close.dropna(how='all', inplace=True)

# -----------------------
# 2) Compute daily returns
# -----------------------
# Log returns
daily_returns = np.log(adj_close).diff().dropna()

# We will exclude SPY from the factor model, but keep it in the daily_returns DataFrame
# for reference and the final data merges, etc.
eq_tickers = [t for t in tickers if t != "SPY"]  # for factor modeling

# Construct the return matrix for PCA
returns_for_pca = daily_returns[eq_tickers].copy()

# --------------------------------
# 3) Perform PCA (4 factors total)
# --------------------------------
pca_model = PCA(n_components=4)
pca_factors = pca_model.fit_transform(returns_for_pca.fillna(0))
# shape: (num_days, 4)

# Factor loadings (sometimes called eigenvectors or components)
# shape: (4, number_of_stocks)
loadings = pca_model.components_

# Let's put the factor time series into a DataFrame for convenience
factors_df = pd.DataFrame(
    pca_factors,
    index=returns_for_pca.index,
    columns=[f"Factor{i+1}" for i in range(4)]
)

# For each stock, compute the idiosyncratic (residual) return with 4 factors
# Reconstruct the portion of returns explained by the first 4 factors, and subtract.
# Note: Reconstructed_return = sum_{k=1..4} [Factor_k(t) * loading(k, stock)]
# We'll do a matrix multiplication for convenience.
# loadings: shape (4, n_stocks), pca_factors: shape (n_days, 4)
# So reconstruction: (n_days, 4) @ (4, n_stocks) => (n_days, n_stocks)
recon_returns = pca_factors @ loadings
# We need to transform recon_returns into a DataFrame with same columns as eq_tickers
recon_df = pd.DataFrame(
    recon_returns,
    index=returns_for_pca.index,
    columns=eq_tickers
)

# Idiosyncratic = actual - reconstructed
idio_returns = returns_for_pca - recon_df

# -----------------------------------
# 4) Create signals with desired ICs
# -----------------------------------
#
# We want:
# - For each stock: a daily signal that correlates with *next day’s idio return*
#   with an implied Sharpe ~ 1.
# - For Factor2, Factor3: a signal that correlates with *next day’s factor return*
#   with Sharpe ~ 0.5.

# First, shift the idio_returns by 1 day to get "future" returns we want to predict
future_idio = idio_returns.shift(-1)

# Similarly for factors 2 and 3
future_factors_2 = factors_df["Factor2"].shift(-1)
future_factors_3 = factors_df["Factor3"].shift(-1)

# Helper function to create a random signal that has a specified correlation
# with a target series. This is a simplistic approach: we generate random noise
# and combine it with the target to yield the desired correlation.
def make_correlated_signal(target, desired_corr=0.2, seed=42):
    np.random.seed(seed)
    rnd_noise = np.random.randn(len(target))
    # Normalize them
    t_std = (target - target.mean()) / (target.std() + 1e-8)
    r_std = (rnd_noise - rnd_noise.mean()) / (rnd_noise.std() + 1e-8)
    # Combine with a weighting to achieve the desired correlation
    # Cor(X, aX + bY) can be tuned. For simplicity: signal = alpha * t_std + beta * r_std
    alpha = desired_corr
    beta = np.sqrt(1 - alpha**2)
    signal = alpha * t_std + beta * r_std
    # Scale signal to have the same std as the target, or any arbitrary scale
    signal = signal * target.std() + target.mean()
    return pd.Series(signal, index=target.index)

# We also want a certain "Sharpe" for the strategy, which depends on the
# standard deviation of the signal vs. the standard deviation of the returns
# and the correlation. For demonstration, we'll just ensure the correlation
# is in the ballpark and let the "Sharpe" alignment be approximate.

# Create signals for idiosyncratic returns
stock_signals = {}
corr_target = 0.2  # or some value that might produce ~1 Sharpe if you scale it
for stock in eq_tickers:
    # future_idio[stock] is the next-day idiosyncratic return
    target = future_idio[stock].dropna()
    sig = make_correlated_signal(target, desired_corr=corr_target, seed=42)
    # Reindex to align with full date range, fill with NaN
    stock_signals[stock] = sig.reindex(future_idio.index)

stock_signals_df = pd.DataFrame(stock_signals)

# Create signals for factor2 and factor3
# We'll aim for correlation ~ 0.3 or so (toy example).
# You can tune to attempt an "IC" that might produce Sharpe 0.5
factor2_signal = make_correlated_signal(future_factors_2.dropna(), desired_corr=0.3, seed=42)
factor3_signal = make_correlated_signal(future_factors_3.dropna(), desired_corr=0.3, seed=42)

factor_signals_df = pd.DataFrame({
    "Factor2_signal": factor2_signal.reindex(future_factors_2.index),
    "Factor3_signal": factor3_signal.reindex(future_factors_3.index)
})

# -------------------------------------
# 5) Create a copy with 1% missing data
# -------------------------------------
# We ensure SPY is never missing. We'll apply random masking to everything else.
combined_data = pd.concat([
    daily_returns,
    idio_returns.add_prefix("IDIO_"),
    factors_df.add_prefix("FCT_"),
    stock_signals_df.add_prefix("SIG_"),
    factor_signals_df
], axis=1)

# Create a copy
combined_data_missing = combined_data.copy()

# Identify columns that can have missing data (exclude SPY columns)
can_have_nan_cols = [col for col in combined_data.columns if "SPY" not in col]
# Randomly set 1% of the data to NaN in those columns
np.random.seed(999)
mask_size = int(0.01 * combined_data_missing[can_have_nan_cols].size)
# Random row/col indices
row_indices = np.random.choice(combined_data_missing.index, mask_size)
col_indices = np.random.choice(can_have_nan_cols, mask_size)
combined_data_missing.values[
    tuple(zip(*[
        (combined_data_missing.index.get_loc(r), combined_data_missing.columns.get_loc(c))
        for r, c in zip(row_indices, col_indices)
    ]))
] = np.nan

# ------------------------------------------------
# 6) Collect earnings dates for each stock (non-SPY)
# ------------------------------------------------
earnings_data = {}
for ticker in eq_tickers:
    try:
        t = yf.Ticker(ticker)
        # Some versions of yfinance have get_earnings_dates, some do not.
        # This call typically allows limiting number of entries or specifying a timeframe.
        # We try without arguments or limit=16 etc.
        # May return a DataFrame or None depending on availability.
        ed = t.get_earnings_dates()
        earnings_data[ticker] = ed
    except Exception as e:
        print(f"Could not retrieve earnings for {ticker}: {e}")
        earnings_data[ticker] = None

# ------------------------------------------------
# Print or return final results
# ------------------------------------------------
print("==== Original Combined Data (head) ====")
print(combined_data.head())

# print("\n==== Combined Data with Missing (head) ====")
# print(combined_data_missing.head())

# print("\n==== Sample Earnings Data ====")
# for ticker in eq_tickers:
#     print(f"\nEarnings for {ticker}:")
#     print(earnings_data[ticker])

# If you wanted to store these as member variables, return them, or write to CSV:
# combined_data.to_csv('combined_data.csv')
# combined_data_missing.to_csv('combined_data_with_missing.csv')
# etc.

# ================================
# Rolling Analysis: Annualized Volatility for Selected Stocks
# ================================

import matplotlib.pyplot as plt

# (This section uses variables already defined in the previous code block:
#  - 'daily_returns' computed as log returns from 'adj_close')
#
# Choose a subset of stocks for the rolling analysis:
rolling_tickers = ["AAPL", "MSFT", "AMZN", "GOOG"]

# Subset the daily_returns DataFrame to the chosen stocks.
rolling_returns = daily_returns[rolling_tickers]

# Define rolling window sizes (in trading days): 5, 20, and 40.
windows = [5, 20, 40]

# Annualization factor: for daily data we typically use 252 trading days per year.
ANNUALIZATION_FACTOR = 252

# Compute rolling variances and then annualized volatilities.
rolling_vols = {}
for w in windows:
    # Compute the rolling variance (using the default sample variance, i.e. ddof=1)
    # Note: For log returns (assumed mean ~ 0), this is acceptable.
    rolling_var = rolling_returns.rolling(window=w).var() * ANNUALIZATION_FACTOR
    # Annualized volatility is the square root of the annualized variance.
    rolling_vol = np.sqrt(rolling_var)
    rolling_vols[w] = rolling_vol

# --------------------------
# Plot Rolling Annualized Volatility
# --------------------------
fig, axs = plt.subplots(len(rolling_tickers), 1, figsize=(10, 10), sharex=True)
for i, ticker in enumerate(rolling_tickers):
    for w in windows:
        axs[i].plot(rolling_vols[w].index, rolling_vols[w][ticker], label=f"Window = {w}")
    axs[i].set_title(f"{ticker} - Rolling Annualized Volatility")
    axs[i].legend()
plt.tight_layout()
plt.show()

# =========================
# 2) Basic HAR Regression
# =========================

import statsmodels.api as sm
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA

# Define realized variance as the square of daily log returns
# (daily_returns is computed earlier as np.log(adj_close).diff().dropna())
rv = daily_returns**2  # realized variance each day for each stock

def make_har_features(series, max_lag=22):
    """
    Given a pandas Series of daily realized variance, returns a DataFrame containing
    HAR features:
      - daily_lag: the value at t-1
      - weekly_lag: average of past 5 days (t-1 to t-5)
      - monthly_lag: average of past 22 days (t-1 to t-22)
    The features are shifted so that each row represents the data used to predict future variance.
    """
    # daily: lagged series (lag of 1 day)
    daily_lag = series.shift(1)
    # weekly: average of the past 5 days
    weekly_lag = series.rolling(window=5).mean().shift(1)
    # monthly: average of the past 22 days
    monthly_lag = series.rolling(window=22).mean().shift(1)

    df = pd.DataFrame({
        "daily_lag": daily_lag,
        "weekly_lag": weekly_lag,
        "monthly_lag": monthly_lag
    }, index=series.index)

    return df

har_results = {}
horizon = 10  # Predict 10 days ahead

# Loop over tickers (or a subset; here we use all tickers from the earlier block)
for ticker in tickers:
    # Check if the ticker exists in daily_returns (it should, given the download)
    if ticker not in daily_returns.columns:
        continue

    # Create HAR features from the realized variance series for this ticker.
    features_df = make_har_features(rv[ticker])
    # Define the target as the realized variance 10 days ahead.
    target = rv[ticker].shift(-horizon)

    # Combine features and target into one DataFrame and drop any rows with NaN values.
    df_har = pd.concat([features_df, target.rename("target")], axis=1).dropna()
    X = df_har[["daily_lag", "weekly_lag", "monthly_lag"]]
    y = df_har["target"]

    # Add a constant term (intercept) and fit a simple linear regression.
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    har_results[ticker] = model

    print(f"\n=== HAR Model for {ticker} (predicting {horizon}-day ahead realized variance) ===")
    print(model.summary())
