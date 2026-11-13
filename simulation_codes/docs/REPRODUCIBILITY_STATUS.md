# Simulation reproducibility status

This is a lightweight simulation-source release. It contains executable source, configurations, tests, and small aggregate numerical summaries; it is not an archive of trained neural weights or full synthetic market trajectories.

## Validation commands

From the repository root:

~~~bash
python -m compileall -q simulation_codes/simulations simulation_codes/tools/check_repository.py
python simulation_codes/tools/check_repository.py
python -m pytest -q simulation_codes/simulations/population/gaussian/tests
python -m pytest -q simulation_codes/simulations/population/eight_state/tests
python -m pytest -q simulation_codes/simulations/neural/first_deployment/tests
~~~

The experiment-specific `README`, `PROTOCOL`, and `requirements.txt` files document additional commands and dependencies.

## Interpretation of a successful check

A syntax or unit-test pass verifies the released source and the listed local checks. It does not by itself regenerate every expensive neural training run. Several neural experiments require excluded checkpoints or synthetic training banks to reproduce archived estimates from scratch. The release preserves the associated training, market-clearing, and evaluation logic and documents those dependencies at the experiment level.

## Known boundary

`e8_robustness_partial` is a partial implementation record and is not a standalone reproduction target. The paper-facing simulation map and primary reproduction entry points are in `simulation_codes/README.md`.
