import sys
sys.path.append("../src")
import numpy as np
import matplotlib.pyplot as plt
from rough_bergomi import simulate_rough_bergomi

# 1. Simulate rough paths (H=0.1, genuinely rough) vs a near-Brownian case (H=0.5)
rough = simulate_rough_bergomi(H=0.1, eta=1.5, rho=-0.7, xi0=0.04,
                                n_steps=1000, n_paths=3, seed=1)
smooth = simulate_rough_bergomi(H=0.5, eta=1.5, rho=-0.7, xi0=0.04,
                                 n_steps=1000, n_paths=3, seed=1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(rough["t"], rough["S"].T)
axes[0, 0].set_title("Price paths, H=0.1 (rough)")

axes[0, 1].plot(smooth["t"], smooth["S"].T)
axes[0, 1].set_title("Price paths, H=0.5 (Brownian-like)")

axes[1, 0].plot(rough["t"], rough["V"].T)
axes[1, 0].set_title("Variance paths, H=0.1 (rough)")

axes[1, 1].plot(smooth["t"], smooth["V"].T)
axes[1, 1].set_title("Variance paths, H=0.5")

plt.tight_layout()
plt.savefig("path_comparison.png", dpi=120)
print("Saved path_comparison.png")

# 2. Quantitative roughness check
dt = rough["t"][1] - rough["t"][0]
dV_rough = np.diff(rough["V"], axis=1)
dV_smooth = np.diff(smooth["V"], axis=1)
print("Mean |dV| rough (H=0.1):", np.abs(dV_rough).mean())
print("Mean |dV| smooth (H=0.5):", np.abs(dV_smooth).mean())