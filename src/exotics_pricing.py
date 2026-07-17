"""
Price exotic derivatives under the calibrated rough Bergomi model vs Black-Scholes,
and quantify the divergence -- the headline result of the project.

Instruments:
  1. Down-and-out barrier call (path-dependent, sensitive to vol path roughness)
  2. Variance swap (direct exposure to realized variance path)
"""

import sys
sys.path.append("../src")
import numpy as np
import pandas as pd
from scipy.stats import norm
from rough_bergomi import simulate_rough_bergomi
from calibration import calibrate, bs_call_price, implied_vol_from_price


def price_barrier_rough_bergomi(H, eta, rho, xi0, spot, K, barrier, T,
                                 n_paths=50000, n_steps=252, seed=0):
    """
    Down-and-out call: pays max(S_T - K, 0) at T, unless S ever touches
    or crosses 'barrier' from above -- then it pays 0.
    barrier must be below spot (knock-out on the downside).
    """
    out = simulate_rough_bergomi(H, eta, rho, xi0, n_steps, n_paths, T=T, seed=seed)
    S = out["S"] * spot  # rescale, since simulator starts at S0=1

    knocked_out = (S <= barrier).any(axis=1)
    payoff = np.where(knocked_out, 0.0, np.maximum(S[:, -1] - K, 0.0))
    price = payoff.mean()
    stderr = payoff.std() / np.sqrt(n_paths)
    return price, stderr


def price_barrier_bs_flat_vol(spot, K, barrier, T, sigma, r=0.0, n_paths=50000,
                               n_steps=252, seed=0):
    """
    Same down-and-out call priced via Monte Carlo under flat-vol GBM (i.e. what
    a BSM-based desk would use), for an apples-to-apples MC comparison rather
    than the closed-form BSM barrier formula, so both prices share the same
    simulation methodology and only the vol dynamics differ.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    logS = np.zeros((n_paths, n_steps + 1))
    for i in range(n_steps):
        logS[:, i + 1] = logS[:, i] + (r - 0.5 * sigma ** 2) * dt + sigma * dW[:, i]
    S = spot * np.exp(logS)

    knocked_out = (S <= barrier).any(axis=1)
    payoff = np.where(knocked_out, 0.0, np.maximum(S[:, -1] - K, 0.0))
    price = payoff.mean()
    stderr = payoff.std() / np.sqrt(n_paths)
    return price, stderr


def price_variance_swap_rough_bergomi(H, eta, rho, xi0, T, n_paths=50000,
                                       n_steps=252, seed=0):
    """
    Fair variance swap strike = E[realized annualized variance] under the model.
    """
    out = simulate_rough_bergomi(H, eta, rho, xi0, n_steps, n_paths, T=T, seed=seed)
    log_returns = np.diff(np.log(out["S"]), axis=1)
    # sum of squared log returns over the path, annualized by dividing by T
    realized_var = np.sum(log_returns ** 2, axis=1) / T
    return realized_var.mean(), realized_var.std() / np.sqrt(n_paths)


def price_variance_swap_bs_flat_vol(sigma):
    """Under flat vol GBM, fair variance swap strike is just sigma^2 (no smile effect)."""
    return sigma ** 2


if __name__ == "__main__":
    df = pd.read_csv("../data/spy_options.csv")
    expiry = sorted(df["expiry"].unique())[2]  # third-nearest expiry, avoids degenerate near-0DTE tenor
    slice_df = df[(df["expiry"] == expiry) & (df["type"] == "call")].sort_values("moneyness")
    slice_df = slice_df[(slice_df["moneyness"] > 0.9) & (slice_df["moneyness"] < 1.08)]

    fit = calibrate(slice_df)
    print(f"Using calibrated params: H={fit['H']:.4f}, eta={fit['eta']:.4f}, "
          f"rho={fit['rho']:.4f}, xi0={fit['xi0']:.5f}")

    spot = slice_df["spot"].iloc[0]
    T = slice_df["T"].iloc[0]
    K = spot  # ATM strike
    barrier = spot * 0.85  # 15% down-and-out barrier
    atm_vol = slice_df.iloc[(slice_df["moneyness"] - 1).abs().argsort()[:1]]["impliedVolatility"].values[0]

    print(f"\nSpot: {spot:.2f}, Strike (ATM): {K:.2f}, Barrier: {barrier:.2f}, T: {T:.3f}")
    print(f"BSM flat vol used (ATM implied vol): {atm_vol:.4f}")
    # Sanity check: is xi0 (the model's flat variance level) actually consistent
    # with the market's own ATM vol? If these are far apart, the variance swap
    # divergence below is likely an artifact of xi0 being a poor proxy for
    # realized variance, not a genuine finding about rough vol pricing.
    print(f"Sanity check: ATM vol from smile = {atm_vol:.4f}, "
          f"sqrt(xi0) = {np.sqrt(fit['xi0']):.4f}")

    print("\n--- Down-and-out barrier call ---")
    rb_price, rb_err = price_barrier_rough_bergomi(
        fit["H"], fit["eta"], fit["rho"], fit["xi0"], spot, K, barrier, T)
    bs_price, bs_err = price_barrier_bs_flat_vol(spot, K, barrier, T, atm_vol)
    print(f"Rough Bergomi price: {rb_price:.4f} (+/- {rb_err:.4f})")
    print(f"BSM flat-vol price:  {bs_price:.4f} (+/- {bs_err:.4f})")
    pct_diff = 100 * (rb_price - bs_price) / bs_price if bs_price > 0 else float("nan")
    print(f"Divergence: {pct_diff:.2f}%")

    print("\n--- Variance swap (fair strike) ---")
    rb_varswap, rb_vs_err = price_variance_swap_rough_bergomi(
        fit["H"], fit["eta"], fit["rho"], fit["xi0"], T)
    bs_varswap = price_variance_swap_bs_flat_vol(atm_vol)
    print(f"Rough Bergomi fair variance: {rb_varswap:.5f} (+/- {rb_vs_err:.5f})  "
          f"[implied vol equiv: {np.sqrt(rb_varswap):.4f}]")
    print(f"BSM flat-vol fair variance:  {bs_varswap:.5f}  "
          f"[implied vol equiv: {np.sqrt(bs_varswap):.4f}]")
    pct_diff_vs = 100 * (rb_varswap - bs_varswap) / bs_varswap
    print(f"Divergence: {pct_diff_vs:.2f}%")