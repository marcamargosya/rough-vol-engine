"""
Calibrate rough Bergomi parameters (H, eta, rho) to a market implied vol surface
via Monte Carlo pricing + least-squares fit.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
from rough_bergomi import simulate_rough_bergomi


def bs_call_price(S, K, T, sigma, r=0.0):
    """Black-Scholes call price, used to convert MC prices back to implied vol."""
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_from_price(price, S, K, T, r=0.0):
    """Invert BS price -> implied vol via bisection."""
    if price <= max(S - K, 0):
        return 0.0
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, mid, r) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def model_iv_surface(H, eta, rho, xi0, strikes, T, spot, n_paths=4000, seed=0):
    """Simulate rough Bergomi and back out implied vols at given strikes for maturity T."""
    n_steps = max(int(T * 252), 10)
    out = simulate_rough_bergomi(H, eta, rho, xi0, n_steps, n_paths, T=T, seed=seed)
    ST = out["S"][:, -1] * spot  # rescale since simulate_rough_bergomi starts at S0=1

    ivs = []
    for K in strikes:
        payoff = np.maximum(ST - K, 0.0)
        price = payoff.mean()
        iv = implied_vol_from_price(price, spot, K, T)
        ivs.append(iv)
    return np.array(ivs)


def calibrate(market_df, n_paths=8000, n_seeds=3, n_restarts=4):
    """
    Fit H, eta, rho, xi0 to minimize squared error between model and market IVs
    for a single expiry slice of market_df (must have columns: strike, impliedVolatility, T, spot).

    Averages over multiple MC seeds per evaluation to smooth the noisy objective,
    and runs multiple optimizer restarts to avoid bad local minima.
    Now calibrates rho jointly instead of fixing it -- a fixed rho was forcing
    eta to pin against its bound trying to compensate.
    """
    strikes = market_df["strike"].values
    market_ivs = market_df["impliedVolatility"].values
    T = market_df["T"].iloc[0]
    spot = market_df["spot"].iloc[0]

    def objective(params):
        H, eta, rho, xi0 = params
        if not (0.02 < H < 0.49 and 0.1 < eta < 5.0 and -0.95 < rho < -0.1 and 0.001 < xi0 < 1.0):
            return 1e3
        errs = []
        for s in range(n_seeds):
            model_ivs = model_iv_surface(H, eta, rho, xi0, strikes, T, spot,
                                          n_paths=n_paths, seed=s)
            errs.append(np.sum((model_ivs - market_ivs) ** 2))
        return np.mean(errs)

    starts = [
        [0.10, 1.5, -0.7, market_ivs.mean() ** 2],
        [0.20, 2.0, -0.6, market_ivs.mean() ** 2],
        [0.30, 1.0, -0.8, market_ivs.mean() ** 2],
        [0.15, 2.5, -0.5, market_ivs.mean() ** 2],
    ][:n_restarts]

    best = None
    for x0 in starts:
        result = minimize(objective, x0, method="Nelder-Mead",
                           options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-5})
        if best is None or result.fun < best.fun:
            best = result

    return {
        "H": best.x[0],
        "eta": best.x[1],
        "rho": best.x[2],
        "xi0": best.x[3],
        "loss": best.fun,
        "success": best.success,
    }


if __name__ == "__main__":
    df = pd.read_csv("../data/spy_options.csv")
    expiry = df["expiry"].iloc[0]
    slice_df = df[(df["expiry"] == expiry) & (df["type"] == "call")].sort_values("moneyness")
    slice_df = slice_df[(slice_df["moneyness"] > 0.9) & (slice_df["moneyness"] < 1.1)]

    print(f"Calibrating to {len(slice_df)} quotes, expiry {expiry}")
    fit = calibrate(slice_df)
    print("Calibrated H:", fit["H"])
    print("Calibrated eta:", fit["eta"])
    print("Calibrated rho:", fit["rho"])
    print("Calibrated xi0:", fit["xi0"])
    print("Loss:", fit["loss"])