Slow-fast AI market retraining: implementation and numerical pilot

Read PROTOCOL.txt first. Existing accepted manuscript and prior experiments
are unchanged. The copied reference_capacity code and initial checkpoints
make this archive self-contained. source_hashes.json preserves their hashes.

Run from this directory with Python 3.12, numpy, scipy, pandas, torch:
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python -m unittest test_engine -v
  python run.py --workers 3
  python run.py --summarize
  python check_training.py
  python analyze.py
  python make_report.py
  python package_results.py
The report additionally uses matplotlib and reportlab.

Each slow iteration freezes all policies, generates the full data bank from
the endogenous fast market, and only then fits all candidate policies on that
bank. They are deployed simultaneously at the next slow iteration. This is
neither online learning inside the auction nor resampling prices from the
original exogenous benchmark. The next period receives the just-cleared price.
Separate paths never share rolling windows. All model parameters and initial
normalization stay frozen during price generation.

Adoption x is the upgraded share of the .55-mass first group. The second
group's information class remains no-public-encoder, but its private/price
coefficients and residual variance update every slow epoch. Both neural
capacity cells are fitted even at zero market share, on the same market data.
This keeps a well-defined counterfactual low/high comparison at x=0 and x=1.

results/seed*_x*/round*/ contains deployed and next model checkpoints,
coefficients for group 2, data-generation provenance, held-out model metrics,
refitting metrics on the OLD law, learning curves, and fast-market block
diagnostics. The market's deployed metrics use the newly generated law.
Do not compare prediction errors across different laws as a pure model effect.

The six-update run begins with the existing trained models. Its round-zero
market is the frozen-policy stationary approximation for those models at x.
It is not an independently solved all-low retraining equilibrium. Comparison
across x therefore describes matched learning trajectories from a common
initial profile. Fixed-point and actual adoption-shock studies can follow.

Completed stages are reused. --seed, --x and --updates allow a selected run
to be extended. A previously terminal stage is regenerated for fitting when
extended; its old profile and deterministic shock bank are retained. A
completed one-update smoke run is reused in the full run. For a fresh full
reproduction, remove generated results in a COPY of the directory.

Temporary generated data are discarded after a complete stage is saved.
Every data bank is reconstructible from the retained deployed checkpoints,
profile hashes, seeds, path lengths, burn-in and generator version. No model
is updated during its data generation. The archive excludes temporary data
and incomplete stages, and verifies every included file against a manifest.

Interpretation: this is finite-sample warm-start neural retraining. Empirical
settling of J or price moments is not proof that the neural models reached
global population optima. The first three V2 seeds are reused; this pilot is
not a new ten-seed independent confirmation. Independent validation,
calibration and evaluation shocks are separated from all optimizer inputs.
