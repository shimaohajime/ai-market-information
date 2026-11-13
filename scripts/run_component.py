#!/usr/bin/env python3
"""Run a paper-facing empirical component through its stable implementation.

The public commands use the paper's component names. Implementation filenames
remain stable for compatibility with the public configurations and are mapped
here rather than exposed as the primary interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {
    "e1_forecasts": "src/step1_pipeline.py",
    "e1_inference": "src/step1_inference.py",
    "e1_information_audit": "src/step1_information_set_audit.py",
    "e1_event_panel": "src/step2_event_panel.py",
    "e1_rdq_validation": "src/step2_rdq_validation.py",
    "e1_market_expectations": "src/step2_market_expectations.py",
    "e1_signal_decomposition": "src/step2_integrated_decomposition.py",
    "e1_preannouncement_returns": "src/step2_preannouncement_returns.py",
    "e1_preannouncement_analysis": "src/step2_preannouncement_analysis.py",
    "e1_earnings_persistence": "src/step2_earnings_persistence.py",
    "e1_make_si": "src/make_si_results.py",
    "e2_synthetic_benchmark": "src/step4_synthetic_v1.py",
    "e2_dossiers": "src/step4_phase0.py",
    "e2_price_context": "src/step4_price_v2.py",
    "e2_development_context": "src/step4_v3_task12_build.py",
    "e2_replication_sample": "src/step4_holdout_build.py",
    "e2_replication_context": "src/step4_holdout_treatments.py",
    "e2_public_context_audit": "src/step4_holdout_public_signal_audit.py",
    "e2_analyst_context_audit": "src/step4_holdout_analyst_signal_audit.py",
    "e2_behavioral_analysis": "src/step4_holdout_behavioral_replication_analysis.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available components:\n  " + "\n  ".join(sorted(COMPONENTS)),
    )
    parser.add_argument("component", choices=sorted(COMPONENTS))
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected component; use -- before a leading option.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = ROOT / COMPONENTS[args.component]
    forwarded = args.arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *forwarded]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
