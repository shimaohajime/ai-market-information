E9: retraining dynamics and noise. E10: departures from dividend assumptions.
29 October2026. Read PROTOCOL.txt and the report in deliverables/ first.

Python3.12; CPU PyTorch2.5.1; dependencies in requirements.txt.
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  python -m unittest discover -s . -p 'test_experiments.py' -v
  python run_e9.py --workers 6
  python functional_check.py --workers 1
  python run_e10.py --phase initial --workers 1
  python run_e10.py --phase measure --workers 7
  python audit_inherited_laws.py
  python analyze.py
  python make_report.py
The first E10 phase can run concurrently with E9 (total7 workers). Only run
functional_check after E9, and E10 measure after E10 initial has completed.
Completed jobs resume. For a fresh run move results/ elsewhere and create empty
results/ and data/ directories. Intermediate training panels are regenerated from
saved keys and removed after fitting. All new models and fitted heads are retained.

Inputs: immutable E5 candidate baselines in baselines/; E6 both-group update6 models
and update0..6 analytical state metrics in source_e6/. Source SHA256s retained.
engine.py and reference_capacity/ are unchanged core E6 market and learning code.
common.py defines shared fitting and evaluation; economies.py contains the only
changes to exogenous dividends. Control draws reproduce the inherited generator.

E9 continues market participants, including group2. Common-shock trajectories use
identical primitive innovations and optimizer order every iteration, but prices and
histories are recomputed. Fixed-law audit refits independently reset to one saved
profile. Stability of aggregate metrics, equality of whole policies, convergence
of a finite-sample algorithm, and a population learning equilibrium are distinct.

E10 retains constant-affine ModelA demands. State-dependent true payoff noise does
not introduce a state-dependent forecasting head. Nonalignment changes the true
payoff target; the fresh-information precision J continues to refer to theta, not
the full dividend. E3 targets dividends; E4 evaluates actual ex-ante exponential
utility with the mode-specific conditional payoff mean and noise variance.
The E10 matched control uses the same first-deployment/frozen-history timeline as
its extensions; it is not the old E6 six-update market. No equilibrium certification.
