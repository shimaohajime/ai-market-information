# The Opposing Informational Effects of AI Trading: Research Code

This repository accompanies *The Opposing Informational Effects of AI Trading*. It contains the numerical market experiments, the observed-market analysis, and the controlled language-model analysis used in the paper and its Supplementary Information.

Only public source code, configurations, prompt templates, and small aggregate simulation summaries are versioned. The repository does **not** contain licensed financial data, raw API responses, firm identifiers, anonymous dossiers, outcome keys, model checkpoints, full simulated trajectories, internal notes, or credentials.

## Research design and paper map

The repository follows the paper's logic rather than the order in which the analyses were built.

1. **Simulation evidence (theory and Supplementary Sections S1--S10).** The population Gaussian benchmarks and finite neural-market experiments ask how more capable public-data processing changes private-information aggregation, the usefulness of prices to observers with different capabilities, and the value of further computation. The source is in [`simulation_codes/`](simulation_codes/README.md).
2. **E1: market evidence on public-data processing.** Two pre-specified processors receive the same historical firm-quarter records. The analysis measures how much a current market-information snapshot improves each processor's out-of-sample earnings-surprise forecasts, then relates the resulting forecast components to earnings-announcement returns. This is a counterfactual processing comparison, not an AI-adoption event study.
3. **E2: controlled AI-forecasting behavior.** Language models receive identical anonymous financial dossiers; only a labeled synthetic private clue and the public-context condition vary. The study measures the change in the reported forecast, not trading behavior, issuer-specific prediction, or investor behavior. Synthetic Gaussian exercises and researcher-side public-context audits state the limits of this behavioral evidence.

The empirical analyses do not measure actual asset-manager AI adoption or identify a causal effect of AI adoption on market prices.

## Licensed inputs

E1 and the real-data portion of E2 require CRSP, Compustat, and I/B/E/S data. These inputs are commercially licensed and are not redistributed. Obtain them directly through an authorized provider and comply with the applicable terms.

By default, the code expects this ignored local layout:

~~~text
data/raw/
├── crsp/
│   ├── stock_monthly_data.csv.gz
│   ├── compustat_annual.csv.gz
│   ├── compustat_quaterly.csv.gz
│   ├── compustat_crsp_link.csv
│   └── crsp_daily_stock_2010_2025.csv.gz
└── ibes/
    ├── IBES_idsum.csv.gz
    ├── IBES_surpsumu.csv.gz
    └── IBES_statsumu_epsus.csv.gz
~~~

`compustat_quaterly.csv.gz` retains the spelling of the original extract. Set `AI_MARKET_DATA_DIR` to use a different root, or set `AI_MARKET_CRSP_ROOT` and `AI_MARKET_IBES_ROOT` separately.

## Installation

Python 3.11 or later is recommended.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Neural simulations may additionally require the experiment-specific packages listed in their `requirements.txt` files. The controlled-model scripts use standard environment-variable authentication:

~~~bash
export OPENAI_API_KEY="your-key"
~~~

Never commit a key. Any command that sends a paid model request requires an explicit acknowledgement flag; all request and response records remain local.

## Reproducing the simulation evidence

Start with the lightweight population benchmarks:

~~~bash
cd simulation_codes/simulations/population/gaussian
python -m pytest -q tests
python benchmark.py
python analysis.py

cd ../eight_state
python -m pytest -q tests
~~~

The remaining directories in [`simulation_codes/simulations/`](simulation_codes/simulations/) implement the finite neural-market comparisons, retraining exercises, and robustness calculations reported in the Supplementary Information. Their directory-level protocol notes specify dependencies, computational requirements, and reproduction boundaries. Aggregate reference outputs are under [`simulation_codes/results/aggregate/`](simulation_codes/results/aggregate/).

## Reproducing E1: market evidence

The commands below construct the two empirical components described in Supplementary Section E1.

Build the expanding-window public-data forecast comparison and its inference:

~~~bash
python3 scripts/run_component.py e1_forecasts run
python3 scripts/run_component.py e1_inference run
python3 scripts/run_component.py e1_information_audit run
~~~

The quarterly-accounting robustness uses point-in-time Compustat Quarterly inputs:

~~~bash
python3 scripts/run_component.py e1_forecasts run \
  --accounting-frequency quarterly \
  --panel data/interim/e1_quarterly_firm_quarter_panel.parquet \
  --output-dir output/e1_quarterly
python3 scripts/run_component.py e1_inference run \
  --scores output/e1_quarterly/oos_scores.parquet \
  --folds output/e1_quarterly/oos_folds.csv \
  --output-dir output/e1_quarterly
python3 scripts/run_component.py e1_information_audit run \
  --panel data/interim/e1_quarterly_firm_quarter_panel.parquet \
  --output-dir output/e1_quarterly
~~~

Construct the earnings-announcement panel and diagnostic analyses:

~~~bash
python3 scripts/run_component.py e1_event_panel build
python3 scripts/run_component.py e1_rdq_validation run
python3 scripts/run_component.py e1_market_expectations run
python3 scripts/run_component.py e1_signal_decomposition run
python3 scripts/run_component.py e1_preannouncement_returns run
python3 scripts/run_component.py e1_preannouncement_analysis run
python3 scripts/run_component.py e1_earnings_persistence run
python3 scripts/run_component.py e1_make_si run
~~~

All inputs, intermediate panels, figures, and aggregate outputs remain local under ignored `data/` and `output/` directories.

## Reproducing E2: controlled AI-forecasting behavior

The synthetic Gaussian benchmark can be built and inspected without an API call:

~~~bash
python3 scripts/run_component.py e2_synthetic_benchmark build
python3 scripts/run_component.py e2_synthetic_benchmark dry-run
~~~

The real-data workflow first creates anonymous, time-safe dossiers and local audit keys. These commands do not call an external model:

~~~bash
python3 scripts/run_component.py e2_dossiers build
python3 scripts/run_component.py e2_dossiers check
python3 scripts/run_component.py e2_price_context build
python3 scripts/run_component.py e2_development_context build
python3 scripts/run_component.py e2_replication_sample build
python3 scripts/run_component.py e2_replication_context build
python3 scripts/run_component.py e2_public_context_audit run
python3 scripts/run_component.py e2_analyst_context_audit run
~~~

The behavioral analysis writes aggregate-only outputs; it does not expose dossiers, issuer identities, dates, outcomes, or raw model responses:

~~~bash
python3 scripts/run_component.py e2_behavioral_analysis
~~~

## Repository layout

~~~text
simulation_codes/   Population and neural-market simulations (S1--S10)
src/                Empirical construction, inference, and controlled-model analysis (E1--E2)
config/             Public analysis configurations
prompt_templates/   Versioned response-format and prompt contracts for E2
data/               Ignored licensed inputs and local intermediate files
output/             Ignored local results, figures, and audit records
~~~

All matching, model training, and dossier construction are designed to be point in time. In E2, model-visible payloads exclude issuer identifiers, dates, realized outcomes, and price levels. The real-data component measures a model's response to an experimental synthetic clue; it does not test whether a target firm's price history predicts that firm's next earnings surprise.

The paper-facing component names, commands, and implementation paths are listed in [`docs/PAPER_COMPONENT_MAP.md`](docs/PAPER_COMPONENT_MAP.md). Please cite the accompanying paper when using this code; citation metadata are in [`CITATION.cff`](CITATION.cff). No open-source license is granted yet, pending a rights review of the included vendor directories.
