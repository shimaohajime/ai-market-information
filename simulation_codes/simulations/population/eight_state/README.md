# Theorem 2: eight-state Gaussian-CARA equilibrium audit

This research package evaluates the *same* 8-dimensional E1 aligned-payoff competitive-clearing economy used in the Science AI-trading project. It distinguishes a dimension-free **conditional** comparative statics theorem from selected-model numerical verification.

## Files

- `THEOREM2_EIGHT_STATE_AUDIT.pdf` / `.tex`: 5-page internal theoretical research note with proofs and economic scope.
- `figure_bridge_8d.pdf` / `.png`: baseline population finite-3-history information, variance, and computation-value curves.
- `baseline_finiteK3.csv`: exact population conditional variances and private upgrade value for 21 adoption shares, both protocols.
- `aligned_condition_audit.csv`: endpoint finite differences for 147 prior converged aligned calibrations, for x: 0 -> 1 and .25 -> .75.
- `aligned_local_derivative_audit.csv`: local x=.5 derivative signs, 147 x 2 protocols.
- `baseline_derivative_decomposition.json`: local sharp-formula contributions for the baseline case.
- `audit_theorem2.py`: re-computes the 147 aligned conditions.
- `audit_local_conditions.py`: re-computes 147 local derivative signs.
- `local_decomp.py`: baseline derivative decomposition.
- `verify_jacobian.py`: IFT/Lyapunov/OLS Jacobian vs independent finite differences.
- `plot_baseline.py`: generates baseline CSV and vector figure.
- `prior/vendor/`: previously verified microfounded population market + deterministic seeds of 147 prior aligned economy calibrations.

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
- All 147 aligned cases were **preselected** for convergence from 180 previously explored parameter draws. Their signs do not prove any generic or universal market sign theorem.
- The 316 previously converged nonaligned calibrations **do not satisfy the aligned-payoff reduction to conditional fundamental variance**; they are therefore not pooled in the theorem validation.
- Positive information spillovers for external downstream users are not assertions of higher equilibrium profits of low-capacity traders or higher aggregate welfare.
- The 8d equilibrium point derivatives are numerical evaluations of regular population optimum/market clearing, checked against the analytical derivative of the price training map. They are not interval certificates uniformly over all x.
