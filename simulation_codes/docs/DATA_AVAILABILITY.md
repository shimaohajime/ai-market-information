# Data and model availability

## Financial and experimental data

The empirical analyses use licensed CRSP, Compustat, and I/B/E/S data. Those sources, firm-level extracts, mappings, and intermediate panels are not distributed. Authorized researchers should obtain data directly from the original providers and follow the instructions in the repository-root `README.md`.

The controlled LLM experiments use local anonymous dossier payloads, outcome keys, and raw API responses. These files are not distributed. The released code contains the construction and analysis logic, prompt contracts, and public configurations only.

## Simulation materials

The simulation source is included under `simulation_codes/simulations/`. Small aggregate synthetic outputs are included under `simulation_codes/results/aggregate/` for numerical checking and plotting. The release excludes neural checkpoints, full training datasets, and full simulated market trajectories; full reruns of some neural studies may therefore require substantial compute and locally regenerated artifacts.

The E8 source subset is incomplete: 45 original archive members were unavailable. It is released for transparency but is insufficient for a claim of complete E8 reproducibility.

## Credentials and licensing

No API keys, WRDS credentials, individual identifiers, licensed data, or private mapping keys are part of this release. The code has no explicit open-source license yet; select one only after confirming rights for every included dependency and vendored component.
