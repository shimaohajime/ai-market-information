# Finite neural retraining implementation

This directory implements an iterative neural-market retraining calculation. At each update, deployed policies are held fixed while the market generates a full training bank; candidate policies are then fitted on that bank and deployed simultaneously in the next update. No model is updated during the generation of its own training data.

Run with Python 3.12, NumPy, SciPy, pandas, and CPU PyTorch:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python -m unittest test_engine -v
    python run.py --workers 3
    python run.py --summarize
    python check_training.py
    python analyze.py

The adoption variable is the upgraded share of the first investor group. Low- and high-capacity candidates are fitted on the same market data at each adoption value, including the zero-adoption comparison. The generated results are seed-indexed and resumable; use a separate copy of the directory for a fresh regeneration.

The calculation is a finite-sample warm-start retraining exercise. Stability of selected price moments does not establish convergence of complete neural policies or a global population optimum.
