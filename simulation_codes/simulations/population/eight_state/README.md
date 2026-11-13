# Eight-state Gaussian-CARA market calculations

This package evaluates an eight-dimensional aligned-payoff competitive-clearing economy. It distinguishes conditional comparative statics from numerical calculations in a selected set of convergent calibrations.

## Files

- `baseline_finiteK3.csv`: exact population conditional variances and private upgrade values for 21 adoption shares under two protocols.
- `aligned_condition_audit.csv`: endpoint finite differences for 147 prior converged aligned calibrations, for x: 0 -> 1 and .25 -> .75.
- `aligned_local_derivative_audit.csv`: local x=.5 derivative signs, 147 x 2 protocols.
- `baseline_derivative_decomposition.json`: local sharp-formula contributions for the baseline case.
- `audit_theorem2.py`: re-computes the 147 aligned conditions.
- `audit_local_conditions.py`: re-computes 147 local derivative signs.
- `local_decomp.py`: baseline derivative decomposition.
- `verify_jacobian.py`: IFT/Lyapunov/OLS Jacobian vs independent finite differences.
- `plot_baseline.py`: generates baseline CSV and vector figure.
- `prior/vendor/`: population market implementation and deterministic inputs for the 147 aligned calibrations.

## Reproduce (Python 3.11+, numpy, scipy, matplotlib)

From the package root, run:

```
OPENBLAS_NUM_THREADS=1 python audit_theorem2.py
OPENBLAS_NUM_THREADS=1 python audit_local_conditions.py
OPENBLAS_NUM_THREADS=1 python local_decomp.py
OPENBLAS_NUM_THREADS=1 python verify_jacobian.py
OPENBLAS_NUM_THREADS=1 python plot_baseline.py
python -m unittest discover -s tests -v
```

All results are exact Gaussian population moment calculations. There is no Monte Carlo. Fixed incumbent forecasts and fully retrained population equilibria differ, but the *marginal capacity-upgrade valuation* in both paths is for a candidate reoptimized under the law currently being compared.

## Conventions and limits

- **The low class** sees contemporaneous macro and current price. **The high class** has the *same private signal* and also the **actual feasible three-lag public history**. Do not substitute a 40-lag oracle in the private value theorem.
- The 147 aligned cases are the convergent subset of 180 specified parameter draws. Their signs do not prove a generic or universal market-sign theorem.
- Nonaligned calibrations do not satisfy the aligned-payoff reduction to conditional fundamental variance and are therefore not pooled in this calculation.
- Positive information spillovers for external downstream users are not assertions of higher equilibrium profits of low-capacity traders or higher aggregate welfare.
- The 8d equilibrium point derivatives are numerical evaluations of regular population optimum/market clearing, checked against the analytical derivative of the price training map. They are not interval certificates uniformly over all x.
