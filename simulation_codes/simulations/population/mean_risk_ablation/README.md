# Forecast-mean, forecast-risk, and retraining ablations

## Executive summary

Three separate designs must NOT be conflated:

1. **Population-aligned (E1)**: exact Gaussian covariance of the 8-factor stationary market (solved by discrete Lyapunov). At the same inherited learning law, the upgraded type's optimal NLL mean/private slope and optimal variance are mixed in a 2x2 head factorial. Fresh-price `J = A^2 / supply_variance` is exact conditional private innovation precision. Four partial-retraining arms solve population-learning fixed points with designated types held at original learned policy. A signal-noise ratio grid varies heterogeneity. A robustness check enforces the bounded `abs(price_coef) <= 0.45` specification.
2. **State-dependent E3 inherited market**: 4-state low and 12-state high pretrained head-B SSM/GRU for type 0. 2x2 mean/variance policy recombinations are applied to *exactly the same old public histories*. Type 1 and 2 retain their original model policies. Forecast heads and competitive auction prices are recomputed from the saved models. Four independent train/test innovation-bank pairs are used. Scoring is **OOS linear decoder MSE improvement (3 lags)**, *not* exact mutual information.
3. **E3 frozen-policy endogenous history**: separate endogenous price histories are simulated for all four cross-swapped head policies (no agent retraining), using two independent train/test bank pairs for each SSM and GRU. Score is again the external OOS linear decoder, not population equilibrium.

Cross-swapped hybrid forecasting beliefs are **not reoptimized subject to holding their other components fixed**. Therefore, the factorial estimates are mechanical policy-component effects, not comparisons of optimal independently trained mean-only or variance-only AI agents. A separate restricted-head calculation is provided in `../reoptimized_ablation/`.

## Execution

From this directory:

```bash
python population_ablation.py
python bounded_population_check.py
python neural_head_ablation.py
python frozen_market_ablation.py
python make_figures.py
python -m unittest discover -s tests -v
pdflatex -interaction=nonstopmode -halt-on-error TASK37_ABLATION_NOTE.tex
```

On some BLAS installations, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` makes performance more reproducible. Tested with Python >=3.11, NumPy, SciPy, pandas, PyTorch, Numba, Matplotlib. `numba` JIT compilation occurs the first time an encoder executes.

## File index

- `population_ablation.py`, `population_results.json` — Gaussian population factorial, investor-retraining restrictions, heterogeneity grid.
- `bounded_population_check.py`, `population_bounded_results.json` — price-head coefficient bounded counterpart.
- `neural_head_ablation.py`, `neural_head_arms.csv`, `neural_factorial_effects.csv`, `neural_results.json` — E3 inherited-history head swaps, four independent shock-bank pairs per architecture.
- `frozen_market_ablation.py`, `frozen_stationary_arms.csv` — endogenously evolving frozen-policy market, two independent pairs per architecture.
- `figures/` — five plots.
- `tests/test_ablation.py` — 10 tests covering clearing, factorial identities, learned predictor checkpoints, partial population fixed points, and data integrity.
- `vendor/population_benchmark` — 8-dimensional analytical Gaussian model with the population OLS learner.
- `vendor/phase3_long_validation` — competitive-clearing, forecasting, and inference components used by this calculation.
- `vendor/task36_src/features.py` — causal external decoder public-history features.

## Key caveats

- E1 `J` means newly aggregated **conditional** private information, not full-price mutual information or social welfare.
- E3 OOS risk reduction is an external **linear** decoder metric and generally not `J` or conditional mutual information.
- The population ablation permits an unconstrained current-price coefficient in its main table; the 0.45-bound reanalysis confirms the direction of the results.
- Four independent test banks do not constitute four independent trained AI models; both architectures use one stored training seed. Across-market generalization has not yet been tested.
- Relearning by only one investor type is a partial-update equilibrium diagnostic; outcomes are not additive causal components.
- Normalization uses the training-law data, and test time information respects `u_t=[p_{t-1},D_t,macro_t]`.
