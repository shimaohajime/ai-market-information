# Neural learning dynamics and payout-law robustness

This directory implements the learning-dynamics and payout-law extensions in Supplementary Section S10. The learning calculation follows the same finite neural-market protocol as the earlier retraining experiments; the payout extensions change the payoff mapping while preserving the stated market and evaluation conventions.

Run with Python 3.12, CPU PyTorch, and the dependencies in `requirements.txt`:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python -m unittest discover -s . -p 'test_experiments.py' -v
    python run_e9.py --workers 6
    python functional_check.py --workers 1
    python run_e10.py --phase initial --workers 1
    python run_e10.py --phase measure --workers 7
    python audit_inherited_laws.py
    python analyze.py

The learning trajectories distinguish stability of selected aggregate moments from equality of complete fitted policies or convergence to a population learning equilibrium. The payout extensions evaluate specified finite-learning and fixed-policy comparisons; they do not supply a general equilibrium certification.
