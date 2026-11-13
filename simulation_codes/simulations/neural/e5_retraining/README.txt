E5: continued slow-fast retraining and all-low baseline diagnostics
29 October 2026

Start with deliverables/E5_Retraining_and_Baseline_20261029.pdf and
E5_PROTOCOL.txt. The authoritative outcome is e5_summary/BASELINE_STATUS.json.
A saved baseline checkpoint is NOT automatically a certified equilibrium.

Reproducing the main continuation:
  python -m unittest discover -s . -p test_engine.py -v
  python run.py --workers 5
  python analyze.py
  python audit_baseline.py --round 12 --reps 4 --workers 3
  python baseline_shock.py --round 12
  python summarize_e5.py
  python make_report_e5.py

Install requirements.txt in a suitable Python environment. CPU PyTorch 2.5.1
was used. Each worker uses one torch/BLAS thread. Set OPENBLAS_NUM_THREADS=1
and OMP_NUM_THREADS=1 before running. No GPU is required. Data banks are large;
allow at least 8 GB of RAM for five workers. Fewer workers produce the same
seed-indexed experiments. Temporary training data are regenerated, used only
after generation completes, and deleted after each successful update.

Existing completed stages are resumed. To reproduce rather than resume, move
results to a different directory and recreate an empty results directory.
The original inherited models are in initial/. This regenerates updates 0-12
from the original inherited law, not just the extension. Some existing
training_adequacy files describe the earlier experiment and are labeled by
their original scope; they are not fresh E5 audit evidence.

results/seed*_x*/round*/ contains every deployed checkpoint and next fit,
generation fingerprints, model metrics, price moments and timing metadata.
config.json determines the original-path endpoint. analyze.py independently
verifies price clearing, model fingerprints, chronology and exact continuity.

baseline_audit/ contains four independent same-law one-step refits for each
all-low endpoint. Their newly fitted coefficients are NOT deployed in the
continued trajectory or substituted for the baseline. Each repetition uses
fresh training shocks but a common, separate held-out evaluation bank.

baselines/ contains the three saved candidate all-low profiles and source
metadata. See BASELINE_STATUS.json to determine whether they passed the
practical mechanism-stability check. High models are counterfactual at x=0.

baseline_shocks/ compares x=0,.25,.5,.75,1 using each saved all-low profile:
first auction with inherited histories, then recursively adjusted histories
with policies frozen. These branches have NOT been retrained after the shock.
The original x=.5/1 continuation paths are different counterfactuals.

e5_summary/ holds individual and aggregated results, uncertainty intervals,
late-block drift, fixed-law audit residuals, and baseline eligibility.
The public precision statistic J is not mutual information or welfare.

The old Main Text V5, S1-S3 V2 and theorem contract are unchanged. The source
archive is AI_Trading_Slow_Fast_Retraining_v1_20261028.zip (version 2).
This package is a new experiment deliverable, preserving the old archive.
No new independent cohort of initial training seeds is claimed.
