# LLMs as Mobile App Recommenders — Replication Package

This repository contains the code, input datasets, and output data of an empirical study on the behaviour of Large Language Models (LLMs) as mobile app recommenders. The study covers 10 LLMs (4 proprietary, 6 open-source) and is organized around three research questions:

- **RQ1 — Ranking criteria elicitation.** Which ranking criteria do LLMs report when recommending mobile apps? We collect *blind* (unconditioned) app recommendations together with the self-reported ranking criteria, and consolidate the latter into a taxonomy of 16 criteria via filtering and embedding-based clustering.
- **RQ2 — Recommendation consistency.** How consistent are blind LLM app recommendations? We measure *internal* consistency (run-to-run, within a model) and *external* consistency (cross-model agreement) using Rank-Biased Overlap (RBO, `k=20`, `p=0.9`).
- **RQ3 — Ranking criteria impact.** How does conditioning recommendations on an explicit ranking criterion (*guided* recommendations) affect convergence? We compare the internal and external consistency of guided recommendations against the blind baselines from RQ2, and quantify how far guided lists are displaced from blind ones.

## Models

| Key | Cohort | Model |
|-----|--------|-------|
| `openai` | proprietary | `gpt-5.3-chat-latest` |
| `gemini` | proprietary | `gemini-3-flash-preview` |
| `anthropic` | proprietary | `claude-opus-4-6` |
| `mistral` | proprietary | `mistral-large-latest` |
| `llama31_8b` | open (Ollama) | `llama4:scout` |
| `gemma3_4b` | open (Ollama) | `gemma3:27b` |
| `qwen3_8b` | open (Ollama) | `qwen3:30b-a3b` |
| `gptoss20b` | open (Ollama) | `gpt-oss:20b` |
| `mistral_open` | open (Ollama) | `mistral-small3.1:24b` |
| `deepseekr1_8b` | open (Ollama) | `deepseek-r1:8b` |

Keys are internal identifiers used in output file names (some predate model upgrades and are kept for backwards compatibility with existing bundles). Additional provider adapters (`deepseek`, `perplexity`) are available via `--model-keys` but are not part of the default study set.

## Repository structure

```
├── run_experiments.py            # RQ1/RQ3 recommendation experiments; RQ2 shortcut
├── run_criteria_elicitation.py   # RQ1 ranking criteria pipeline (extract → filter → consolidate)
├── run_consistency_analysis.py   # RQ2 RBO consistency analysis
├── check_experiment_coverage.py  # Report missing cells in bundled runs
├── code/
│   ├── experiments/              # Experiment framework (config, runner, schemas, providers)
│   ├── elicitation/              # RQ1 criteria extraction, filtering, consolidation, RC merge
│   └── consistency/              # RQ2 metrics/plots + RQ3 convergence analysis
│       ├── metrics.py            # RBO and Jaccard implementations
│       ├── runner.py             # RQ2 orchestration
│       ├── publication_figures.py# RQ2 publication figures
│       └── rq3_convergence.py    # RQ3 convergence analysis + publication figures
└── data/
    ├── input/
    │   ├── prompts/              # System and user prompts (RQ1/RQ3)
    │   ├── schema/               # JSON Schemas enforcing k ranked apps
    │   └── use-case/             # features_small.csv (16), features_large.csv (100), k.csv
    └── output/
        ├── features/rq1/apps/    # Blind recommendation bundles (per suite)
        ├── features/rq1/rc/      # Ranking criteria artifacts + merged taxonomy
        ├── features/rq2/         # Consistency CSVs, heatmaps, publication figures
        └── features/rq3/         # Guided bundles, convergence analysis, publication figures
```

Dataset *suites* combine cohort and scale: `open_small`, `open_large`, `proprietary_small`, `proprietary_large`, plus `proprietary_small_wo_websearch` for the knowledge-only proprietary ablation.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

API credentials are read from a `.env` file at the repository root (see `python-dotenv`):

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
```

Open-source models run through a local [Ollama](https://ollama.com/) server (`OLLAMA_BASE_URL`, default `http://localhost:11434`). Pull the models listed above before running the open cohort.

## Replicating the study

All experiments use structured JSON output (schemas in `data/input/schema/`) with exactly `k=20` ranked apps per response.

### RQ1 — Blind recommendations and criteria elicitation

Collect blind recommendations (10 runs per feature):

```bash
# Open cohort, small (16 features) and large (100 features) datasets
python run_experiments.py --rq rq1 --families open --features-csv data/input/use-case/features_small.csv
python run_experiments.py --rq rq1 --families open --features-csv data/input/use-case/features_large.csv

# Proprietary cohort, knowledge-only (primary configuration)
python run_experiments.py --rq rq1 --families proprietary \
  --features-csv data/input/use-case/features_small.csv \
  --dataset-suite proprietary_small_wo_websearch

# Proprietary cohort, web-augmented ablation (Gemini Google Search, Mistral web_search)
python run_experiments.py --rq rq1 --families proprietary \
  --features-csv data/input/use-case/features_small.csv --web-search
```

Bundles are written to `data/output/features/rq1/apps/<suite>/`. Use `--dry-run` to preview planned cells and `--continue-on-error` / re-runs to fill failed cells. Verify completeness with:

```bash
python check_experiment_coverage.py --rq rq1 --families open --k 20
```

Elicit the ranking criteria taxonomy from the collected bundles (Steps: extraction, rule-based filtering, embedding-based consolidation with cluster-count validation):

```bash
python run_criteria_elicitation.py \
  --input-folders data/output/features/rq1/apps/open_large \
  --output-folder data/output/features/rq1/rc/open_large \
  --steps all
```

The consolidation output is manually reviewed into `rc_wo_id_<suite>.csv`. The per-suite criteria lists are then aligned into the unified 16-criteria taxonomy used by RQ3:

```bash
python -m code.elicitation.merge_rc_lists
# -> data/output/features/rq1/rc/merge/rc_merge_unified.csv
```

### RQ2 — Consistency analysis

Compute internal, intra-family external, and cross-family external RBO on the RQ1 bundles:

```bash
python run_consistency_analysis.py --dataset-scale large
python run_consistency_analysis.py --dataset-scale small

# RBO persistence-parameter sensitivity (reported in the paper)
python run_consistency_analysis.py --dataset-scale large --rbo-p-values 0.8,0.9,1.0
```

Outputs (CSVs and heatmaps) are written to `data/output/features/rq2/<suite>/` and `data/output/features/rq2/cross_<scale>/`. The publication figures are regenerated with:

```bash
python -m code.consistency.publication_figures
# -> data/output/features/rq2/publication/
```

### RQ3 — Guided recommendations and convergence analysis

Collect guided recommendations (5 runs per feature × criterion, 16 criteria from the unified taxonomy):

```bash
python run_experiments.py --rq rq3 --families open,proprietary \
  --features-csv data/input/use-case/features_small.csv
```

Bundles are written to `data/output/features/rq3/apps/<suite>/`. The convergence analysis compares guided recommendations against the blind baselines (internal convergence per model × criterion, external convergence per criterion with paired statistics, cohort breakdown, and guided-vs-blind displacement):

```bash
python -m code.consistency.rq3_convergence
# CSV tables  -> data/output/features/rq3/analysis/
# Figures     -> data/output/features/rq3/publication/
```

## Metrics

Ranking similarity uses Rank-Biased Overlap (`code/consistency/metrics.py`) at depth `k=20` with persistence `p=0.9`; app names are matched by normalized exact string comparison. Criteria similarity in the elicitation pipeline uses sentence-embedding cosine similarity.

## License

This project is released under the GNU General Public License v3.0 (see `LICENSE`).
