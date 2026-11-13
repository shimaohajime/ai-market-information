# Finite neural retraining paths

This experiment extends the neural-market learning paths used in Supplementary Section S8. Starting from common initial profiles, investor groups iteratively generate public price histories, refit their forecasting rules, and deploy the resulting rules in the next market update.

Run with Python 3.12, CPU PyTorch, and the dependencies in `requirements.txt`:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python -m unittest discover -s . -p 'test_engine.py' -v
    python run.py --workers 5
    python analyze.py
    python audit_baseline.py --round 12 --reps 4 --workers 3
    python baseline_shock.py --round 12
    python summarize_e5.py

`e5_summary/` contains the endpoint summaries, numerical checks, and diagnostics. The baseline audit uses independently generated one-step refits and is not substituted into the learning paths. The fixed-policy shock calculations and the retraining paths answer different counterfactual questions.

The results describe a finite sequence of fitted policies under the stated training procedure. They do not certify a stationary retraining equilibrium, global population optimum, welfare comparison, or mutual-information measure.
