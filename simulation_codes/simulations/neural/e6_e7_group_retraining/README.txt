# Group retraining and forecast-component decomposition

This experiment supports Supplementary Section S8 by separating the retraining of two investor groups and decomposing the roles of forecast mean and forecast risk. All four update arms begin from identical parameters within each seed and adoption environment; each active arm then generates its own public history and training data.

Run with Python 3.12, CPU PyTorch, and the dependencies in `requirements.txt`:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python -m unittest discover -s . -p 'test_*.py' -v
    python run_e6.py --workers 7
    python e7_adoption.py
    python analyze_e6.py
    python make_report.py

`summary/` contains the factorial effects, paired descriptive intervals, per-update component decompositions, and numerical validations. Common primitive streams support paired comparisons; they do not force prices or training data to coincide across market laws.

The decomposition uses post-fit substitutions, not constrained reoptimization. It therefore reports mechanical forecast-component effects within the fitted models, not general causal effects of separately optimized AI systems. The experiment is a finite-learning calculation rather than a certified retraining equilibrium.
