# Phase 2: Resource-bounded forecasting AI in a microfounded asset market

**Scope:** Step 03, Phase 2 (Tasks 3.3-3.4), October 2026. Standalone, CPU-only Python reference package extending the tested Phase 1 market engine. No external data or network are required to reproduce these experiments.

## Quick reproduction

```bash
python -m pip install -r requirements.txt
python -m pytest -q                   # 17 deterministic tests
python run_phase2.py                  # 36 fitted models; overwrites results JSON/plots
python validate_phase2.py            # independent law-seed and dynamic-policy checks
```

`python run_phase2.py --quick` runs the fast 14-condition smoke demonstration. `--no-retrain` reuses existing fitted model checkpoints. The full run uses 36 models and can take several minutes on a CPU. The package already includes all 36 checkpoint files, 8d Phase 1 data panels, and experiment results.

**Important:** A quick run overwrites the JSON output with *quick-run* results; rerun the full command to regenerate the provided 36-condition results. Do not mix the two outputs.

## Study design and time ordering

1. The Phase 1 frozen incumbent market generates training data under law **Q0**. At the auction at t, public inputs are exactly `U_t=[p_(t-1),D_t,macro_t]`. Investor i's private information is `s_i,t`. The candidate auction price `p_t` enters only the affine head; realized `D_(t+1)` is the training target and is never passed as an input.
2. Each architecture learns a bounded-resource Gaussian forecast `N[a(z)+beta(z)*s+chi(z)*p,v(z)]`, optimizing Gaussian negative log likelihood. This is a subjective belief: conditional normality in the objective world is not assumed. Mean and variance both enter optimal CARA demand.
3. For first deployment, **do not regenerate the public-history trajectory**. Feed the inherited `U_t` into a newly trained forecasting model, replace investor type 1's demand schedule, retain other investors' inherited schedules, and re-clear the auction using the exact microfounded price equation. This is a one-shot inherited-law intervention, not a new stationary equilibrium.
4. Compare small/large within the same architecture using independent frozen-law holdouts and identical latent/supply shock paths. Do not interpret nonnested cross-architecture comparisons causally. Relearning of all agents and the induced new stationary market are for Phase 3.

## Architectures and resource definitions

| Component | Details |
|---|---|
| Public-history encoder | Stable diagonal linear SSM or finite-context GRU |
| Memory levels | 4 and 12 hidden states per architecture (not identical FLOP costs across architectures) |
| History | Fixed 48-market-period rolling context in **both training and inference** |
| Constant head A | Flexible nonlinear public-history mean, constant private-signal and price coefficients, constant Gaussian variance |
| State-dependent head B | Public-state-dependent mean, private/price coefficients, **and variance** |
| Price-affinity restriction | `chi(z)=0.45*tanh(raw(z))` and `R_f>=1`, guaranteeing downward-sloping aggregate demand for all learned models |
| Training | Gaussian NLL, Adam, validation early stopping, normalized training data, gradient clipping |
| Budget accounting | parameter count, hidden-state bytes, 48-period public-input rolling buffer, *approximate* inference MACs; no claim of hardware-neutral FLOP equivalence |

State size is nested **within** each architecture: GRU(4) is functionally embeddable in GRU(12), and likewise SSM(4) in SSM(12). Test verifies this. Nevertheless, SGD optimization and finite training data need not order the achieved losses. GRU versus SSM is not a nested comparison.

## Data and experiments

* `E1_aligned` (8d): dividends aligned with a common privately observed target; Model A, SSM/GRU, 4/12 states, 3 training seeds, 12 runs.
* `E2_nonaligned` (8d): general dividend loading and a third investor observing a different factor; Model A, 12 runs; Model B (12-state) 4 supplementary runs.
* `E3_state_risk` (8d): genuinely state-dependent conditional fundamental risk; Model A/B (12-state) 8 supplementary runs.
* 36 runs total. Chronological splits of the inherited market data are 65% training, 17% validation, 18% test. All models are evaluated additionally on an independently drawn inherited-market shock sequence of 6,000 periods.

## Measures

* **Forecast NLL/MSE:** heldout individual payoff-forecast accuracy under inherited market Q0.
* **E1 fresh-price likelihood precision:** `J = mean_t[(A_t)^2]/Var(supply shock)`, where all private signals load the same target and `A_t` is the aggregate fresh target loading (pre-auction public history conditioned upon). This is a conditional likelihood coefficient, **not mutual information**.
* **E2/E3 OOS linear decoder gain:** reduction in holdout MSE of a linear public observer predicting the target from `[U_t,p_t]`, compared with `[U_t]`. This is not nonparametric information or mutual information.
* **Market clearing:** maximum absolute aggregate demand minus exogenous net supply, recomputed also using the original Phase 1 `auction_price` routine at chosen dates.
* **Stability:** numerical initial-history coupling under a selected, fixed learned policy, not a global stationarity proof and not a retraining stability result.

## Main files

* `marketlab/learning.py`: trained neural and state-space encoders; normalized Gaussian heads; training and checkpointing.
* `marketlab/deployment.py`: inherited-history inference, one-shot competitive clearing and heldout decoders, optional endogenous-price deployment.
* `marketlab/economy.py`, `policies.py`, `simulation.py`, `scenarios.py`: verbatim Phase 1 microfounded economy specification.
* `run_phase2.py`: 36-condition fitting and evaluation grid, JSON/CSV export, figures.
* `validate_phase2.py`: independent market-shock verification and dynamic fixed-policy check.
* `tests/`: 17 deterministic unit/integration tests including nestedness, price-head equivalence, causal timing, auction clearing, and checkpoint replay.
* `models/`: 36 checkpoint files that contain learned parameters, model architecture, normalization and training metrics.
* `results/`: inherited 8d panels, JSON/CSV outputs, 4 figures, run log, secondary-seed validation.
* `docs/PHASE2_REPORT.pdf`, `.tex`, `.md`: concise technical writeup of the experiment, findings, and limitations.

## Main scientific caveats

This is a **first-deployment simulation pilot**, not a full test of the theorem's population-optimum assumptions. Other incumbent types continue to use **Phase 1 diagnostic policies**, which were not optimal fitted beliefs. The new AI is fitted only under inherited Q0 and its model is not rationally re-estimated after its demand changes price. Comparisons of two separately optimized finite-data models need not be monotone even for nested hypothesis families. Price information is measured only by the clearly defined E1 likelihood precision or by heldout linear decoder performance, and should not be labeled conditional mutual information.

When these tests are complete, Phase 3 can iterate learned policy -> endogenous-price market law -> retraining, and introduce larger/independently evaluated forecasting classes. No transformer, equilibrium retraining, computability theorem, or general claim of price-information degradation is established by Phase 2.
