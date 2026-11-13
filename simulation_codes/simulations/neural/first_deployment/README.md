# Initial-deployment neural-market experiment

This directory implements the initial-deployment comparison in Supplementary Section S8. It trains bounded-resource linear state-space models (SSMs) and GRUs on a reference competitive market, replaces one investor type's demand schedule, and re-clears the auction on the inherited public history. No external data or network access are required.

## Run

~~~bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_phase2.py
python validate_phase2.py
~~~

`python run_phase2.py --quick` is a small smoke test and overwrites generated result files. A full run fits the configured model grid and can take several CPU minutes. `--no-retrain` is appropriate only when compatible local checkpoints already exist.

## Design

At each auction, the public input is `U_t = [p_(t-1), D_t, macro_t]`; an investor also observes its own private signal `s_i,t`. Each model learns a Gaussian forecasting rule of the form `N[a(z) + beta(z)s + chi(z)p, v(z)]` by minimizing Gaussian negative log likelihood. Both the forecast mean and forecast variance enter the investor's CARA demand.

The comparison holds the inherited public-history trajectory fixed. A newly trained model replaces type 1's demand schedule, the other schedules remain fixed, and the price is recomputed from the competitive clearing equation. This is an immediate deployment comparison, not a stationary retrained equilibrium. Capacity contrasts are interpreted within architecture: state sizes 4 and 12 are nested within SSMs and within GRUs, whereas an SSM and a GRU are not a nested pair.

## Measures

- **Forecast NLL and MSE:** out-of-sample individual payoff-forecast accuracy under the inherited market law.
- **Fresh-price precision:** `J = mean_t[A_t^2] / Var(supply shock)`, where `A_t` is the aggregate loading on the fresh privately observed target. This is a conditional likelihood coefficient, not mutual information.
- **External decoder gain:** held-out MSE reduction from adding the current price to a linear public observer. This is a specified prediction metric, not a nonparametric information measure.
- **Clearing and numerical checks:** aggregate-demand residuals, causal timing, nestedness, and checkpoint replay.

## Files

- `marketlab/`: market, forecasting, clearing, and evaluation implementation.
- `run_phase2.py`: model fitting and initial-deployment evaluation.
- `validate_phase2.py`: independent law-seed and fixed-policy checks.
- `tests/`: deterministic unit and integration tests.
- `results/`: locally generated diagnostics and summaries.

## Interpretation

The experiment evaluates fitted finite-data models under a specified inherited market law. It does not establish a population optimum, a retraining fixed point, or a general theorem about nonlinear models. The paper keeps the initial-deployment, fixed-policy, and retraining comparisons distinct; this directory implements the first of those objects.
