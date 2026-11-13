# Population Gaussian retraining benchmark

**Purpose.** Eliminate finite-sample regression noise, optimization error, and neural nonconvexity from the market-learning loop, then assess how endogenous retraining changes private-information aggregation. This analytic finite-lag Gaussian model is distinct from the finite neural-market experiments.

## Economic model and identification of information

The economic primitives match the aligned eight-state market (`marketlab/scenarios.py`): eight latent factors, two noisy public macro signals, private noisy signals, two CARA investor groups, an aligned one-period payoff, and competitive clearing. The forecaster is a globally optimal population Gaussian regression of the next dividend on current macro signals, selected lags of public prices, dividends, and macro signals, its own private signal, and the current auction price. The larger predictor class contains the smaller class.

A clearing profile is represented by `q=[A,C,Delta]` with price `p_t=(A theta_t + C'z_t - zeta_t)/Delta`. A known-policy observer's transformed fresh-price likelihood precision is `J=A²/Var(zeta)`. **This is not unconditional mutual information.** `cycles.py` additionally evaluates conditional mutual information given the **same three-lag public observation set** in every regime.

Under a frozen profile, the augmented linear market is a Gaussian state-space model. Its invariant covariance `P` solves a *discrete Lyapunov equation*, not a Monte Carlo approximation. Population regression coefficients and predictive variances follow exactly from covariance OLS. Retraining is the deterministic population-best-response map, and its Jacobian is computed from differentiated Lyapunov equations and implicit OLS derivatives (`sensitivity.py`).

## Main findings

At supply noise SD **0.90**, upgrading type 1's public memory from 0 to 3 lag blocks while leaving type 2 at 0:

- Fresh-price precision `J`: **2.469507 (baseline) -> 2.262529 (impact) -> 2.073514 (retrained stable fixed point)**.
- Information changes: **-8.381%** impact and **-16.035%** long-run. Fixed point Jacobian spectral radius `0.723580` and fast market radius `0.94`.
- At adoption fraction 0.5, exact implicit differentiation yields initial derivative `-0.127920`, equilibrium derivative `-0.366678`, feedback amplification `2.866x`.
- With 1-to-3 or 2-to-3 history upgrades, the corresponding effects are smaller but remain negative.

At supply noise SD **0.715**, the upgraded fixed point becomes unstable (radius `1.011628`), while population optimal retraining has a **locally stable period-two cycle** `J=3.145212/2.837061` (two-step Jacobian radius `0.956315`; two-cycle residual < 7e-15), both below the low-capacity baseline `J=3.528594`. The dominant eigenvalue crosses -1 near supply SD `0.720896` (numeric bifurcation witness). Each frozen fast market remains stationary (`rho=0.94`).

A **bounded price-head sensitivity** `|chi| <= 0.45`, matching the bound used in the neural experiments, has another stable period-two population orbit at supply noise SD **0.62** with cycle `J=4.093490/3.604803`, baseline `4.932512`, and two-step radius `0.841848`. Its numerically located flip threshold is near `0.647130`. This does *not* establish a cycle of a finite neural architecture at supply SD .90, and the coefficient bound changes the assumptions of the general initial-deployment comparison.

## Validation

- Nine unit tests pass (`python -m pytest -q tests`).
- Analytical Jacobian versus independent centered differences: maximum entrywise difference `3.6e-10`.
- Independent phase-1 competitive clearing engine, 26,000 periods: maximum demand residual `2.7e-15`, exact target information loading, price variance within 0.27% of analytic moment.
- Independent 85,000-period Gaussian time-series moment check; 12 local initializations converge to the same stable fixed point or stable cycle in their tested basins.
- Observed fixed points and cycles are **numerical witnesses**, not formally interval-certified existence proofs. All examples use the constant-affine Gaussian forecasting head, not general state-dependent variance forecasts.

## Run

Requires Python 3.10+, numpy, scipy, matplotlib, pytest (and no PyTorch). From this directory:

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests
python benchmark.py                   # direct population baseline demonstration
python analysis.py                    # capacity and adoption comparisons; sensitivity / plots
python cycles.py                      # flip threshold, period-two roots, exact-moment MI
python bounded.py                     # bounded current-price coefficient case
python independent_engine_check.py    # independent market-engine validation
python multistart.py                  # neighborhood basin test
pdflatex -interaction=nonstopmode POPULATION_RETRAINING_NOTE.tex
```

## Files

`benchmark.py`: exact stationary Lyapunov covariance, Gaussian log-score minimization, price-clearing update.
`sensitivity.py`: analytic Jacobian of the entire population retraining operator.
`analysis.py`: impact vs equilibrium crowd-out, adoption feedback derivative, moment checks.
`cycles.py`: numerical population flip and stable 2-cycle, plus conditional mutual information.
`bounded.py`: price-slope constraint ablation.
`independent_engine_check.py`: independent simulation verification on the shared market engine.
`multistart.py`: neighborhood numerical basins.
`marketlab`: minimal market modules used by this benchmark.
`tests`: unit tests.
`results.json`, `cycles.json`, `bounded.json`: detailed reusable numerical outputs.
`*.png`: locally generated figures.
