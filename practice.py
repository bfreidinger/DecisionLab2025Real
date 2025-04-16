import numpy as np
import pandas as pd

np.random.seed(42)

def simulate_mean_reverting_correlated_processes(
    n=10_000,
    # Mean-reversion speeds (small => slower mean reversion)
    alpha_x=0.001,    # for Asset X
    alpha_y=0.001,    # for Asset Y
    alpha_corr=0.01,  # for correlation
    # Volatilities chosen so that long-run var(X)=1, var(Y)=2
    # in discrete-time AR(1)-like models X[t] = (1-α)*X[t-1] + σ * e[t].
    # Approx stationary var ≈ σ² / (2α) for small α.
    # => for var(X)=1 => σ_x² = 2*α_x
    # => for var(Y)=2 => σ_y² = 4*α_y
    # Adjust them as desired.
    sigma_x=None,
    sigma_y=None,
    # Correlation process parameters
    sigma_corr=0.002,  # noise strength in correlation’s mean-reversion
    corr_init=0.5,     # initial correlation
    corr_min=0.4,
    corr_max=0.6,
    corr_target=0.5    # long-run mean correlation
):
    """
    Returns a pandas DataFrame with two columns: 'Asset1' and 'Asset2'.
    """

    # If not specified, derive sigmas so that each AR(1) process
    # has the desired approximate stationary variance.
    if sigma_x is None:
        sigma_x = np.sqrt(2 * alpha_x)  # ensures var(X)~1
    if sigma_y is None:
        sigma_y = np.sqrt(4 * alpha_y)  # ensures var(Y)~2

    # Prepare arrays
    x = np.zeros(n)
    y = np.zeros(n)
    corr = np.zeros(n)
    corr[0] = corr_init

    # Simulate
    for t in range(1, n):
        # 1) Update correlation with mean-reversion to corr_target
        c_next = (
            corr[t - 1]
            + alpha_corr * (corr_target - corr[t - 1])
            + sigma_corr * np.random.randn()
        )
        # 2) Clamp correlation between corr_min and corr_max
        c_next = max(corr_min, min(corr_max, c_next))
        corr[t] = c_next

        # 3) Generate correlated noise (e1, e2)
        e1 = np.random.randn()
        # e2 has correlation = c_next with e1
        e2 = c_next * e1 + np.sqrt(max(0.0, 1.0 - c_next**2)) * np.random.randn()

        # 4) Update each mean-reverting series
        #    X[t] = (1 - alpha_x)*X[t-1] + sigma_x*e1
        x[t] = (1 - alpha_x) * x[t - 1] + sigma_x * e1
        y[t] = (1 - alpha_y) * y[t - 1] + sigma_y * e2

    # Create a DataFrame of the results
    df = pd.DataFrame({'Asset1': x, 'Asset2': y})
    return df

df_data = simulate_mean_reverting_correlated_processes(n=10000)
print("Returns:")
print(df_data.head())
print("Correlation matrix:")
print(df_data.corr())

#Correlation Modeling
#Given windows of size [20, 50, 100] predict the average realized variance over the next 50 periods. Write two models, one for each asset.

#Asset 1

