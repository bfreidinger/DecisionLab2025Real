import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mosek
import os
import expectedreturns as exp

# Create the environment
env = mosek.Env()

# Set the license path manually, ensuring the path is correct and accessible.
license_path = "/Users/benfreidinger/downloads/mosek/mosek.lic"  # Update with your license file path

# Check if the file exists at the specified path.
if not os.path.exists(license_path):
    raise FileNotFoundError(f"MOSEK license file not found at: {license_path}")

env.putlicensepath(license_path)

# Create a task (this implicitly initializes the environment)
task = env.Task(0, 0)  # Create a simple task with 0 variables and 0 constraints

from mosek.fusion import *

# --------------------------------------------------
# 1) Data and Parameters
# --------------------------------------------------
#mu = np.array([0.0720, 0.1552, 0.1754, 0.0898, 0.4290, 0.3929, 0.3217, 0.1838])
S = np.array([
    [0.0946, 0.0374, 0.0349, 0.0348, 0.0542, 0.0368, 0.0321, 0.0327],
    [0.0374, 0.0775, 0.0387, 0.0367, 0.0382, 0.0363, 0.0356, 0.0342],
    [0.0349, 0.0387, 0.0624, 0.0336, 0.0395, 0.0369, 0.0338, 0.0243],
    [0.0348, 0.0367, 0.0336, 0.0682, 0.0402, 0.0335, 0.0436, 0.0371],
    [0.0542, 0.0382, 0.0395, 0.0402, 0.1724, 0.0789, 0.0700, 0.0501],
    [0.0368, 0.0363, 0.0369, 0.0335, 0.0789, 0.0909, 0.0536, 0.0449],
    [0.0321, 0.0356, 0.0338, 0.0436, 0.0700, 0.0536, 0.0965, 0.0442],
    [0.0327, 0.0342, 0.0243, 0.0371, 0.0501, 0.0449, 0.0442, 0.0816]
])

# Choose stocks to build portfolio from
security_names = ["AAPL", "MSFT", "AMZN", "NVDA", "UAL", "T", "CRSP", "CMCSA"]
expected_returns_dict = exp.expected_returns(security_names,market_return=0.8)
mu = np.array([expected_returns_dict[stock] for stock in security_names])
print(mu)

N = mu.shape[0]
gamma2 = 0.05  # Risk limit

# Compute the Cholesky factor of S
G = np.linalg.cholesky(S)
# Wrap G (or G^T) as a Fusion Matrix object for multiplication with x
G_mat = Matrix.dense(G)  # shape NxN

# --------------------------------------------------
# 2) MarkowitzReturn Model
# --------------------------------------------------
with Model("MarkowitzReturn") as M:
    # Variables: fraction of holdings in each security (no short-selling)
    x = M.variable("x", N, Domain.greaterThan(0.0))

    # Budget constraint: sum(x) == 1
    M.constraint("budget", Expr.sum(x), Domain.equalsTo(1.0))

    # Objective: maximize expected return using dot(x, m)
    M.objective("obj", ObjectiveSense.Maximize, Expr.dot(x, mu))

    # Risk constraint using the rotated QCone
    # We want: vstack(gamma2, 0.5, G^T x) in that cone
    # But we must do: Expr.mul(G_mat, x) if G_mat = G or Matrix.dense(G)
    # If you specifically need G^T x, use Matrix.dense(G.T) instead.
    # Below we do G^T x by using G_mat for G, then we transpose it:
    G_matT = Matrix.dense(G.T)  # NxN
    M.constraint("risk", Expr.vstack(gamma2, 0.5, Expr.mul(G_matT, x)),
                 Domain.inRotatedQCone())

    # Solve
    M.solve()
    returns = M.primalObjValue()
    portfolio = x.level()

print("=== MarkowitzReturn Portfolio ===")
for name, w in zip(security_names, portfolio):
    print(f"{name}: {w:.4f}")
print("Objective (expected return):", returns)


# --------------------------------------------------
# 3) MarkowitzFrontier Model
# --------------------------------------------------
with Model("MarkowitzFrontier") as M:
    # Variables
    x = M.variable("x", N, Domain.greaterThan(0.0))
    s = M.variable("s", 1, Domain.unbounded())

    # Budget constraint: sum(x) == 1
    M.constraint("budget", Expr.sum(x), Domain.equalsTo(1.0))

    # Parameter for risk aversion
    delta = M.parameter()

    # Objective: maximize (dot(x,m) - delta*s)
    M.objective("obj", ObjectiveSense.Maximize,
                Expr.sub(Expr.dot(x, mu), Expr.mul(delta, s)))

    # Conic constraint for the portfolio variance
    # If s >= ||G x||2, we use the standard QCone: vstack(s, G*x)
    M.constraint("risk", Expr.vstack(s, Expr.mul(G_mat, x)), Domain.inQCone())

    # Sweep over a range of delta to build the efficient frontier
    deltas = np.logspace(start=-1, stop=1.5, num=20)[::-1]

    columns = ["delta", "obj", "return", "risk"] + security_names
    df_result = pd.DataFrame(columns=columns)

    for d in deltas:
        # Update the parameter
        delta.setValue(d)
        # Solve
        M.solve()

        # Portfolio stats
        portfolio_return = mu @ x.level()
        portfolio_risk = s.level()[0]  # standard deviation
        row = pd.Series(
            [d, M.primalObjValue(), portfolio_return, portfolio_risk] + list(x.level()),
            index=columns
        )
        df_result.loc[len(df_result)] = row


# --------------------------------------------------
# 4) Plotting the Results
# --------------------------------------------------
# Efficient frontier
df_result.plot(x="risk", y="return", style="-o",
               xlabel="Portfolio risk (std. dev.)",
               ylabel="Portfolio return", grid=True)

# Portfolio composition
my_cmap = LinearSegmentedColormap.from_list("non-extreme gray",
    ["#0000FF", "#ffffff"], N=256, gamma=1.0)

df_result.set_index('risk').iloc[:, 3:].abs().plot.area(colormap=my_cmap,
    xlabel='portfolio risk (std. dev.)', ylabel="x")

plt.show()
