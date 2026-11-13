E6 group retraining and E7 mean/variance channels
29 October 2026

Read deliverables/E6_E7_Group_Retraining_20261029.pdf and PROTOCOL.txt first.
This is a finite six-update factorial experiment from the E5 update-12 all-low
candidate baselines. It does not certify a retraining equilibrium.

Reproduction (Python 3.12.14; CPU PyTorch 2.5.1):
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  python -m unittest discover -s . -p 'test_*.py' -v
  python run_e6.py --workers 7
  python e7_adoption.py
  python analyze_e6.py
  python make_report.py

Install requirements.txt in a suitable environment. The report fonts and their
license are bundled. Seven CPU workers were used; fewer workers run the same
seed-indexed design with less memory. This can require substantial CPU time.
Existing COMPLETE stages are resumed for the configured six-update experiment.
For a fresh regeneration, move results/ elsewhere, create an empty results/
and data/ directory, then run the commands above. Do not alter baselines/.

Four arms: none, group1, group2, both. Group 1 comprises both low/high GRUs;
group 2 uses private signal and current price only. All four arms begin with
identical parameters within each seed/adoption case. Each updating arm creates
its own market history and training data. Matching primitive streams provide
paired comparisons; they do not impose identical prices across different laws.

Data banks are generated before learning and removed after a successful
update. They can be regenerated from saved generation.json keys. Every deployed
model and next fitted model is retained. All disabled-group parameters are
checked for exact constancy. Other groups can still affect their forecasts by
changing the public price history.

Primary results:
  summary/E6_effects.csv: raw and x=0-adjusted learning contrasts.
  summary/E6_factorial.csv: direct adoption, separate group effects,
      interaction and total; also conditional marginal group effects.
  summary/E6_intervals.csv: intervals over three paired initial seeds.
  summary/E7_steps.csv: all per-update swaps and exact decompositions.
  summary/E7_cumulative.csv: telescoping structural mean/risk contributions.
  summary/E7_adoption_adjusted.csv: initial adoption and extra retraining
      contributions, expressed in percent of pre-adoption J.
  summary/model_metrics.csv: beta, v, iota, MSE and price coefficient by type.
  summary/same_law_learning.csv: before/after fit improvement on the same law.
  summary/VALIDATION.json: chronology, fingerprints and accounting checks.

Counts: 36 seed/adoption/arm paths; 27 updating paths with seven states plus
nine frozen paths simulated once = 198 simulated states. The 252 reported
arm/horizon states include 54 explicitly labeled reuses of the unchanged law.
Training panels contain 1,048,576 observations per permitted slow update.
The optional independent head-calibration bank in E5 is not used for E6/E7.

E7 uses post-fit swaps, not constrained reoptimization. Its mean/risk effects
sum exactly after a symmetric allocation of the interaction. Only structural
J, A and iota telescope across slow updates. Price-variance swap diagnostics
are on inherited laws and must not be summed into a stationary price-variance
change. No mutual-information, utility, welfare or global-optimality claim is
made by this accounting.

Source: AI_Trading_E5_Continuation_20261029.zip, Library item
libfile_67bb55e00fac8191b13d728caf78986a. The baseline files are copied unchanged.
The original initial/ models are retained only for the inherited engine tests;
production runs load baselines/, not initial/. E5 and manuscript files remain
unchanged. E3/E4 evaluation and E8-E10 extensions are separate tasks.

summary/E3_E4_candidate_markets.json identifies the nine frozen markets after
six both-group updates, with their profile fingerprints, for the next planned
E3/E4 evaluation. It is a handoff manifest, not an E3/E4 result.
