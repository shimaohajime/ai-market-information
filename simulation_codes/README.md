# Simulation code

This directory contains the simulation component of the research-code release for *The Opposing Informational Effects of AI Trading*. It covers analytic Gaussian population benchmarks and finite neural-market experiments on first deployment, retraining, adoption, information measurement, and robustness.

The directory contains executable source, configurations, tests, experiment-specific documentation, and small aggregate simulation summaries. It does **not** contain proprietary financial data, raw model/API records, neural checkpoints, full simulated trajectories, or any external credential.

## Components

| Directory | Contents |
| --- | --- |
| `simulations/population/gaussian` | Analytic finite-lag Gaussian retraining benchmark and information measures |
| `simulations/population/eight_state` | Eight-state information and local-condition benchmark |
| `simulations/population/mean_risk_ablation` | Mean/risk and neural-head ablations |
| `simulations/population/reoptimized_ablation` | Independent restricted-head ablation |
| `simulations/neural/first_deployment` | Finite SSM/GRU initial-deployment simulations |
| `simulations/neural/slow_fast` | Slow/fast retraining simulations |
| `simulations/neural/e3_e4_information_value` | Simulation measurements of price-sharing and computation value |
| `simulations/neural/e5_retraining` | Retraining continuation |
| `simulations/neural/e6_e7_group_retraining` | Group retraining and adoption studies |
| `simulations/neural/e8_robustness_partial` | Available E8 robustness source; see limitation below |
| `simulations/neural/e9_e10_extensions` | Learning-dynamics and extension studies |
| `results/aggregate` | Small aggregate synthetic summaries used for checking and plotting |
| `tools` | Repository validation and simulation-figure utilities |

The `tools/si_figures/` directory retains the scripts used to construct and audit supplementary figures. Those scripts document the calculations but require archived path-level inputs that are deliberately excluded from this source-and-aggregate-results release; see its README before using them.

## Quick start

Use the dependency file and release note in the specific experiment directory. The Gaussian population benchmark is the lightest entry point:

~~~bash
cd simulation_codes/simulations/population/gaussian
python -m pytest -q tests
python benchmark.py
python analysis.py
~~~

For a whole-source syntax check from the repository root:

~~~bash
python -m compileall -q simulation_codes/simulations simulation_codes/tools/check_repository.py
python simulation_codes/tools/check_repository.py
~~~

Neural experiments can require substantial compute to recreate trained policies. Their source is released for transparency and reruns, but model checkpoints and full trajectory banks are intentionally not distributed.

## Reproducibility boundaries

- The population Gaussian benchmark solves its stationary covariance and population regressions analytically rather than using a Monte Carlo approximation for its main moments.
- The neural studies retain executable architectures, training, clearing, and evaluation logic. Some full reruns require excluded initial-policy checkpoints or synthetic trajectory banks.
- Small synthetic aggregate JSON and CSV files are included for numerical checking and figure generation. They are not licensed financial data or individual-level records.
- The recovered E8 subset is incomplete: 45 source-archive members are unavailable. It must not be described as a complete E8 reproduction.
- The empirical analysis code lives at the repository root (`src/`, `config/`, and `prompt_templates/`), rather than in this subdirectory.

## Citation and license

Citation metadata are in `CITATION.cff`. This package currently has no explicit open-source license; select an appropriate license after confirming rights for all vendored and archived code, and before changing the GitHub repository to public.
