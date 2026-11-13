# Market-simulation code

This directory contains the simulation evidence for *The Opposing Informational Effects of AI Trading*. It corresponds to the paper's theoretical and numerical sequence: an analytically tractable competitive market, population Gaussian calculations, and finite neural-market experiments.

The package contains executable source, configurations, tests, experiment-specific methodological notes, and small aggregate reference results. It does **not** contain financial data, model/API records, neural checkpoints, full simulated trajectories, or credentials.

## Connection to the paper

| Paper component | Reproduction material |
| --- | --- |
| Theory and competitive-market environment (S1--S6) | Population market primitives and numerical checks in `simulations/population/` |
| Population Gaussian markets (S7) | `population/gaussian/`, `population/eight_state/`, and the population ablations |
| Initial deployment and finite retraining (S8) | `neural/first_deployment/`, `neural/slow_fast/`, `neural/e5_retraining/`, and `neural/e6_e7_group_retraining/` |
| Price usefulness and the value of further computation (S9) | `neural/e3_e4_information_value/` and its aggregate reference tables |
| Learning, readout, and payout-law checks (S10) | `neural/e9_e10_extensions/` and its aggregate reference tables |

The directories use compact experiment names because they match the computational protocols. The table above is the paper-facing map.

## Quick start

The population benchmarks are the lightest fully runnable entry points:

~~~bash
cd simulation_codes/simulations/population/gaussian
python -m pytest -q tests
python benchmark.py
python analysis.py

cd ../eight_state
python -m pytest -q tests
~~~

For a syntax check from the repository root:

~~~bash
python -m compileall -q simulation_codes/simulations simulation_codes/tools/check_repository.py
python simulation_codes/tools/check_repository.py
~~~

Finite neural-market experiments may require the experiment-specific dependencies in their `requirements.txt` files and substantial CPU time. Their protocol notes state the training, evaluation, and numerical-validation conventions. Small reference summaries for checks and plotting are in [`results/aggregate/`](results/aggregate/).

## Reproduction boundaries

- The population calculations solve stationary covariance and regression objects directly. They are distinct from the finite-sample neural experiments.
- The neural experiments retain their market generator, forecasting architecture, training procedure, clearing logic, and evaluation code. Full reruns can require excluded checkpoints or synthetic trajectory banks.
- The included JSON and CSV summaries are synthetic aggregate reference outputs. They are neither licensed financial data nor individual-level records.
- `neural/e8_robustness_partial/` is a partial implementation record, not a standalone reproducible experiment. It is not a primary reproduction entry point for the paper's results.
- Supplementary-figure utilities are in [`tools/si_figures/`](tools/si_figures/). They document figure construction but need archived path-level inputs that are not part of this source-and-summary release.

## Citation and license

Citation metadata are in [`CITATION.cff`](CITATION.cff). This package does not yet have an explicit open-source license. Select one only after confirming rights for all included and vendored code, and before making the repository public.
