E8: learning and measurement robustness (29 October 2026)
Read deliverables/E8_Learning_and_Measurement_Robustness_20261029.pdf and
PROTOCOL.txt first. The economic calibration and9 frozen E6 laws are fixed.
This package includes the verified prior E3/E4 package in e3_e4_measurement/.
That prior package's source, models, outputs and manifest are retained unchanged.

Reproduce using Python3.12, CPU PyTorch2.5.1 and base requirements.txt:
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  python -m unittest discover -s . -p 'test_e8.py' -v
  python run_e8.py --phase train --workers 7
  python run_e8.py --phase evaluate --workers 7
  python decoder.py --workers 7
  python analyze_e8.py
  python make_report.py
Seven training workers and one decoder worker can run simultaneously. Fewer workers
reduce memory requirements without changing indexed random streams. Completed jobs
are resumed. For fresh regeneration move results/ to another location and create
empty results/ and data/ directories. Do not modify the bundled prior package.
Training panels are reproducible from logged simulation keys and are removed after
training. New models, heads, training curves, path-level evaluation scores and logs
are retained. Exogenous shocks are paired across adoption and model comparisons.

The new1x replications vary training samples and optimizer shuffling, not the
pre-adoption parameter initialization or the market profile. Continued2x/4x fits
restart the original learning-rate schedule on fresh equal-size data blocks.
This tests finite learning procedures, not attained population optima.

Primary tables:
summary/learning_changes.csv: adoption changes by learning condition,3-seed intervals.
summary/decoder_changes.csv: same for matched-sample affine and RFF64/128 decoders.
summary/contrast_sensitivity.csv: changes in adoption contrasts vs original1x.
summary/CE_diagnostics.csv and CE_evaluation_uncertainty.csv: tail weights and
conditional evaluation uncertainty; these are not independent market replications.
summary/decoder_selection.csv: all validation-selected ridge strengths and budgets.
summary/VALIDATION.json and TEST_RESULTS.txt: audit and measurement checks.
