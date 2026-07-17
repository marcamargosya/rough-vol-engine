"""
Fetch SPX (via SPY as a liquid proxy) option chain data for calibration.
Uses yfinance -- free, no API key needed.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_option_chain(ticker="SPY", max_expiries=6):
    """
    Fetch option chain data across several expiries.

    Returns a DataFrame with columns:
    strike, expiry, T (years to expiry), type (call/put), impliedVolatility,
    bid, ask, spot (underlying price at fetch time)
    """
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]

    expiries = tk.options[:max_expiries]
    rows = []

    for exp in expiries:
        chain = tk.option_chain(exp)
        T = (pd.Timestamp(exp) - pd.Timestamp.today()).days / 365.0
        if T <= 0:
            continue

        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                rows.append({
                    "strike": row["strike"],
                    "expiry": exp,
                    "T": T,
                    "type": opt_type,
                    "impliedVolatility": row["impliedVolatility"],
                    "bid": row["bid"],
                    "ask": row["ask"],
                    "spot": spot,
                })

    data = pd.DataFrame(rows)
    data = data[(data["impliedVolatility"] > 0.01) & (data["bid"] > 0)]
    data["moneyness"] = data["strike"] / data["spot"]

    return data


if __name__ == "__main__":
    df = fetch_option_chain("SPY", max_expiries=6)
    print(f"Fetched {len(df)} option quotes across {df['expiry'].nunique()} expiries")
    print(df[["strike", "expiry", "T", "type", "impliedVolatility", "moneyness"]].head(10))
    df.to_csv("../data/spy_options.csv", index=False)
    print("Saved to data/spy_options.csv")