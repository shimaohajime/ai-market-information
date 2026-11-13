# AI Market Information: Research Code

This repository is the public replication package for the accompanying research paper. It contains the code for counterfactual public-information processing, an earnings-announcement event study, and controlled LLM experiments.

Only public code, configurations, and prompt templates are versioned. The repository does **not** contain licensed financial data, raw API responses, firm identifiers, anonymous dossiers, outcome keys, internal notes, or credentials.

## Scope

- **Step 1 — Counterfactual public-information processing.** Compares limited and richer processors of the same public financial records on common expanding-window out-of-sample firm-quarter observations.
- **Step 2 — Earnings-announcement event study.** Compares the announcement-return loading of pre-specified forecast increments with residual earnings-surprise news. It includes RDQ date validation, an independent-basis decomposition, alternative event windows, and diagnostic tests.
- **Step 4 — Controlled LLM experiment.** Provides synthetic benchmarks and anonymous, time-safe real-data contexts. API calls are explicitly gated and all request/response records remain local.

These analyses do not measure actual asset-manager AI adoption or identify a causal effect of AI adoption on market prices.

## Licensed data

The required CRSP, Compustat, and I/B/E/S inputs are commercially licensed and not redistributed. Obtain them directly through an authorized provider and comply with its terms.

By default, Steps 1–2 expect this ignored local layout:

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

`compustat_quaterly.csv.gz` keeps the spelling of the original extract. For Steps 1–2, the data location may instead be specified with:

~~~bash
export AI_MARKET_DATA_DIR=/absolute/path/to/data/raw
# Or set AI_MARKET_CRSP_ROOT and AI_MARKET_IBES_ROOT separately.
~~~

The Step 4 public configurations refer to `data/raw/`. To use another location, copy the relevant configuration file outside version control and change its input paths.

## Installation

Python 3.11 or later is recommended.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

LLM runners use the OpenAI SDK's standard environment-variable authentication:

~~~bash
export OPENAI_API_KEY="your-key"
~~~

Never commit a key. The API runners use `store: false` and require `--acknowledge-paid-api-calls` before any paid request is sent.

## Steps 1–2

Build the annual-accounting baseline and its inference:

~~~bash
python3 src/step1_pipeline.py run
python3 src/step1_inference.py run
python3 src/step1_information_set_audit.py run
~~~

Build the point-in-time Quarterly-accounting robustness:

~~~bash
python3 src/step1_pipeline.py run \
  --accounting-frequency quarterly \
  --panel data/interim/step1_quarterly_firm_quarter_panel.parquet \
  --output-dir output/step1_quarterly
python3 src/step1_inference.py run \
  --scores output/step1_quarterly/oos_scores.parquet \
  --folds output/step1_quarterly/oos_folds.csv \
  --output-dir output/step1_quarterly
python3 src/step1_information_set_audit.py run \
  --panel data/interim/step1_quarterly_firm_quarter_panel.parquet \
  --output-dir output/step1_quarterly
~~~

Construct and analyze the annual-accounting event panel:

~~~bash
python3 src/step2_event_panel.py build
python3 src/step2_rdq_validation.py run
python3 src/step2_market_expectations.py run
python3 src/step2_market_expectations.py run \
  --panel data/interim/step2_earnings_event_panel_rdq.parquet \
  --sample-flag rdq_match_within_1_calendar_day \
  --prefix market_expectations_rdq_within_1
python3 src/step2_integrated_decomposition.py run
python3 src/step2_preannouncement_returns.py run
python3 src/step2_preannouncement_analysis.py run
python3 src/step2_earnings_persistence.py run
python3 src/make_si_results.py run
~~~

All inputs, intermediates, figures, and aggregate results are local artifacts under ignored `data/` and `output/` folders.

## Step 4

The synthetic benchmark can be built and inspected without an API call:

~~~bash
python3 src/step4_synthetic_v1.py build
python3 src/step4_synthetic_v1.py dry-run
~~~

The real-data workflow first constructs time-safe anonymous dossiers and local audit keys:

~~~bash
python3 src/step4_phase0.py build
python3 src/step4_phase0.py check
python3 src/step4_price_v2.py build
python3 src/step4_v3_task12_build.py build
~~~

The holdout and researcher-side audit workflow makes no API call:

~~~bash
python3 src/step4_holdout_build.py build
python3 src/step4_price_v2.py build --config config/step4_holdout_price_v1.json
python3 src/step4_holdout_treatments.py build
python3 src/step4_holdout_public_signal_audit.py run
python3 src/step4_holdout_analyst_signal_audit.py run
~~~

Any `execute` subcommand makes paid API calls. Review its versioned configuration, cost ceiling, and prompt contract before adding the required acknowledgement flag.

## Layout

~~~text
src/                Construction, estimation, inference, and figure code
config/             Public analysis configurations
prompt_templates/   Versioned prompt and strict-JSON response contracts
data/               Ignored licensed inputs and local intermediate files
output/             Ignored local results, figures, and audit records
~~~

All matching, model training, and dossier construction are designed to be point in time. In Step 4, model-visible payloads exclude issuer identifiers, dates, realized outcomes, and price levels. Its real-data component measures response to an experimental synthetic clue, not issuer-specific inference or investor behavior.

Please cite the accompanying paper when using this code. A formal citation record will be added with the paper's final bibliographic details.
