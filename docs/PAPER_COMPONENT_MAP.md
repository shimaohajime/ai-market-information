# Paper component map

Use the paper-facing commands below when reproducing the empirical analyses.
The source paths are listed only for implementation traceability: their stable
filenames preserve compatibility with the public configurations. They are not
names of additional paper studies.

| Paper component | Public command | Stable implementation |
| --- | --- | --- |
| E1 public-data forecast comparison | `python3 scripts/run_component.py e1_forecasts run` | `src/step1_pipeline.py` |
| E1 inference and information-set audit | `python3 scripts/run_component.py e1_inference run`; `python3 scripts/run_component.py e1_information_audit run` | `src/step1_inference.py`; `src/step1_information_set_audit.py` |
| E1 earnings-announcement panel and diagnostics | `python3 scripts/run_component.py e1_event_panel build`, followed by the E1 diagnostic commands in the root README | `src/step2_*.py` and `src/make_si_results.py` |
| E2 synthetic Gaussian benchmark | `python3 scripts/run_component.py e2_synthetic_benchmark build` | `src/step4_synthetic_v1.py` |
| E2 anonymous dossier construction | `python3 scripts/run_component.py e2_dossiers build` | `src/step4_phase0.py` |
| E2 public financial context and replication sample | The E2 construction commands in the root README | `src/step4_price_v2.py`, `src/step4_v3_task12_build.py`, and `src/step4_holdout_*.py` |
| E2 behavioral replication analysis | `python3 scripts/run_component.py e2_behavioral_analysis` | `src/step4_holdout_behavioral_replication_analysis.py` |

There is no omitted empirical component between E1 and E2. The simulation
package is organized separately under [`simulation_codes/`](../simulation_codes/README.md)
and is mapped to Supplementary Sections S1--S10 there.
