import sys
sys.path.append("../src")
import pandas as pd
import matplotlib.pyplot as plt
from calibration import calibrate, model_iv_surface

# Excludes the single deepest-OTM quote (moneyness > 1.08): even with antithetic
# variates, MC pricing can't reliably resolve this far into the tail at reasonable
# path counts. This is a documented limitation, not a workaround.

df = pd.read_csv("../data/spy_options.csv")
expiry = df["expiry"].iloc[0]
slice_df = df[(df["expiry"] == expiry) & (df["type"] == "call")].sort_values("moneyness")
slice_df = slice_df[(slice_df["moneyness"] > 0.9) & (slice_df["moneyness"] < 1.08)]

fit = calibrate(slice_df)
print("H:", fit["H"], "eta:", fit["eta"], "rho:", fit["rho"], "xi0:", fit["xi0"])

strikes = slice_df["strike"].values
T = slice_df["T"].iloc[0]
spot = slice_df["spot"].iloc[0]

# Bumped to 100k paths -- far OTM/ITM strikes need many more simulated paths
# to get a non-zero payoff count, or the IV solver's intrinsic-value guard
# kills them and the smile shows a flat-zero region at the wings.
model_ivs = model_iv_surface(fit["H"], fit["eta"], fit["rho"], fit["xi0"],
                              strikes, T, spot, n_paths=100000, seed=99)

# Diagnostic: flag any strike where the IV solver still failed
n_failed = 0
for K, iv in zip(strikes, model_ivs):
    if iv == 0.0:
        n_failed += 1
        print(f"WARNING: strike {K} (moneyness {K/spot:.3f}) still returned IV=0")
if n_failed == 0:
    print("All strikes returned non-zero IV -- fix worked.")
else:
    print(f"{n_failed} strikes still failing -- needs variance reduction, not just more paths.")

plt.figure(figsize=(8, 5))
plt.plot(slice_df["moneyness"], slice_df["impliedVolatility"], "o-", label="Market")
plt.plot(strikes / spot, model_ivs, "s--", label="Rough Bergomi (calibrated)")
plt.xlabel("Moneyness (strike / spot)")
plt.ylabel("Implied Volatility")
plt.title(f"Calibration fit, expiry {expiry}\nH={fit['H']:.3f}, eta={fit['eta']:.2f}, rho={fit['rho']:.2f}")
plt.legend()
plt.savefig("calibration_fit.png", dpi=120)
print("Saved calibration_fit.png")