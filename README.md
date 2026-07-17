## Rough Volatility Surface Calibration & Exotic Derivatives Pricing Engine

This project implements a rough Bergomi stochastic volatility model, calibrated to live SPY options data, and compares its pricing of exotic derivatives against standard Black-Scholes. Unlike traditional models, rough volatility captures the empirically observed roughness of the volatility process (Hurst parameter H << 0.5), which better reflects real market dynamics.

The model was calibrated to real market data using Monte Carlo simulation with antithetic variates for variance reduction, yielding a Hurst parameter of H ≈ 0.15, vol-of-vol eta ≈ 2.9–3.6, and correlation rho ≈ -0.47 — consistent with published rough volatility research on equity indices.

Using these calibrated parameters, a down-and-out barrier call priced under rough Bergomi diverges by approximately 50% from the same option priced under flat-vol Black-Scholes, demonstrating that rough volatility dynamics have a material, practically significant impact on path-dependent option pricing.