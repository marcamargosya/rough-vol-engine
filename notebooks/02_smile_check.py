import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/spy_options.csv")

# pick the nearest expiry with enough quotes
expiry = df["expiry"].iloc[0]
smile = df[(df["expiry"] == expiry) & (df["type"] == "call")].sort_values("moneyness")

plt.figure(figsize=(8, 5))
plt.plot(smile["moneyness"], smile["impliedVolatility"], "o-")
plt.xlabel("Moneyness (strike / spot)")
plt.ylabel("Implied Volatility")
plt.title(f"SPY vol smile, expiry {expiry}")
plt.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="ATM")
plt.legend()
plt.savefig("smile_check.png", dpi=120)
print("Saved smile_check.png")
print(smile[["strike", "moneyness", "impliedVolatility"]].head(15))