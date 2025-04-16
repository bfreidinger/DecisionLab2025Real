import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define dimensions
T = 10000  # Number of time periods
N = 6      # Number of assets

# Define a predetermined correlation matrix (symmetric, positive definite)
corr = np.array([
    [1.00, 0.30, 0.20, 0.10, 0.05, 0.00],
    [0.30, 1.00, 0.30, 0.20, 0.10, 0.05],
    [0.20, 0.30, 1.00, 0.30, 0.20, 0.10],
    [0.10, 0.20, 0.30, 1.00, 0.30, 0.20],
    [0.05, 0.10, 0.20, 0.30, 1.00, 0.30],
    [0.00, 0.05, 0.10, 0.20, 0.30, 1.00]
])

# Define different standard deviations for each asset (variances = std^2)
std_devs = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

# Compute the covariance matrix from the correlation matrix and standard deviations
cov = np.outer(std_devs, std_devs) * corr

# Define mean returns (zero for simplicity)
mu = np.zeros(N)

# Generate T samples from the multivariate normal distribution
# The generated data will have shape (T, N)
data = np.random.multivariate_normal(mu, cov, size=T)

# Transpose the data to get shape (N, T)
returns = data.T
print(returns)