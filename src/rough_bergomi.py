"""
Rough Bergomi model: path simulation via direct discretized Volterra kernel sum,
with antithetic variates for variance reduction at extreme strikes.
Reference: Bayer, Friz, Gatheral (2016) "Pricing under rough volatility"
"""

import numpy as np


def simulate_rough_bergomi(H, eta, rho, xi0, n_steps, n_paths, T=1.0, seed=None,
                            antithetic=True):
    """
    Simulate rough Bergomi price and variance paths.

    Parameters
    ----------
    H : float, Hurst parameter (roughness), typically ~0.1 for equities
    eta : float, vol-of-vol parameter
    rho : float, correlation between price and variance drivers, typically ~-0.7
    xi0 : float, flat forward variance level (constant curve for now)
    n_steps : int, number of time steps
    n_paths : int, number of Monte Carlo paths (rounded up to even if antithetic=True)
    T : float, time horizon in years
    seed : int, random seed for reproducibility
    antithetic : bool, if True pairs each draw with its negation, halving the
        independent draws needed and materially reducing variance at extreme
        strikes/tails without extra compute cost.

    Returns
    -------
    dict with keys: 'S' (price paths), 'V' (variance paths), 't' (time grid)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)

    if antithetic:
        n_half = (n_paths + 1) // 2
        base1 = np.random.normal(0.0, np.sqrt(dt), size=(n_half, n_steps))
        base_perp = np.random.normal(0.0, np.sqrt(dt), size=(n_half, n_steps))
        dW1 = np.concatenate([base1, -base1], axis=0)[:n_paths]
        dW1_perp = np.concatenate([base_perp, -base_perp], axis=0)[:n_paths]
    else:
        dW1 = np.random.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        dW1_perp = np.random.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

    # Volterra process: Y_{t_i} = sqrt(2H) * sum_{j<i} ((t_i - t_j))^(H-0.5) * dW1_j
    norm = np.sqrt(2 * H)
    Y = np.zeros((n_paths, n_steps + 1))
    for i in range(1, n_steps + 1):
        lags = (np.arange(i, 0, -1)) * dt
        kernel = lags ** (H - 0.5)
        Y[:, i] = norm * (dW1[:, :i] @ kernel)

    # Variance process
    t_safe = np.maximum(t_grid, 1e-12)
    V = xi0 * np.exp(eta * Y - 0.5 * eta ** 2 * t_safe ** (2 * H))

    # Correlated price driver, Euler-discretized log-price
    dW_S = rho * dW1 + np.sqrt(1 - rho ** 2) * dW1_perp
    logS = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        logS[:, i + 1] = logS[:, i] - 0.5 * V[:, i] * dt + np.sqrt(V[:, i]) * dW_S[:, i]

    S = np.exp(logS)

    return {"S": S, "V": V, "t": t_grid}


if __name__ == "__main__":
    out = simulate_rough_bergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04,
                                  n_steps=252, n_paths=1000, seed=42)
    print("S shape:", out["S"].shape)
    print("V min/max:", out["V"].min(), out["V"].max())
    print("V always positive:", (out["V"] > 0).all())
    print("mean terminal S:", out["S"][:, -1].mean(), "(should be ~1.0, martingale check)")