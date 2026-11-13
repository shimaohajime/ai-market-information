# Price usefulness and the value of additional computation

This experiment supplies the neural-market evidence in Supplementary Section S9. It evaluates a low- and a high-capacity observer on the same fixed neural market profiles at different adoption shares. The two outcomes are (i) the held-out dividend-MSE reduction from observing the current price and (ii) the high-minus-low certainty-equivalent value of the forecasting upgrade.

Run with Python 3.12, CPU PyTorch, and the dependencies in `requirements.txt`:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python -m unittest discover -s . -p 'test_*.py' -v
    python run_measurement.py --workers 7
    python analyze.py
    python make_report.py

The `summary/` directory contains seed-level results, paired descriptive intervals, and numerical diagnostics. Evaluation paths are separate from the model-training paths. The three seed-level contrasts are paired within a market profile; they are not independent market replications.

The MSE comparison is a specified predictive metric rather than conditional mutual information. The certainty-equivalent calculation evaluates the fitted strategies under the true simulated payout distribution, is gross of computational cost, and is not a population-optimal utility comparison.
