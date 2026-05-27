# Empirical Study: LLM Behavior as System Recommenders in Mobile App Domain

An empirical research study that investigates how Large Language Models (LLMs) behave when deployed as system recommenders in the mobile app domain. This study systematically evaluates multiple LLM providers (OpenAI, Google Gemini, and Mistral) to understand their recommendation patterns, consistency, and behavior when generating app rankings based on specific features and categories.

## 📋 Study Overview

This empirical investigation examines LLM behavior in mobile app recommendation scenarios across different AI-powered categories and features. The study focuses on:

- **Multi-LLM Behavioral Analysis**: Comparing recommendation patterns across OpenAI, Google Gemini, Anthropic Claude, Mistral, and Perplexity models
- **Feature-Based Recommendation Studies**: Analyzing how LLMs generate app recommendations for specific app features (e.g., "Photo effects", "Go Live", "Collaborate with others")
- **Category-Based Behavioral Analysis**: Examining LLM behavior when evaluating apps within AI-powered categories (e.g., "AI-powered entertainment", "AI-powered productivity")
- **Consistency Measurement**: Quantifying ranking consistency both within and across different LLM models
- **Behavioral Visualization**: Creating comprehensive visualizations of LLM recommendation patterns and criteria

## 📁 Project Structure

```
llm-recommender-system/
├── code/
│   ├── experiments/              # Unified experiment framework
│   │   ├── config.py             # RQ definitions, providers, default models
│   │   ├── runner.py             # Orchestration (k × feature × provider × mode)
│   │   ├── schema.py             # JSON Schema + k-sized array constraints
│   │   └── providers/            # Provider adapters (structured + prompt modes)
│   ├── consistency/              # Ranking consistency analysis
│   ├── correlation/
│   ├── elicitation/
│   └── visualization/
├── data/
│   ├── input/
│   │   ├── prompts/              # system-prompt-* and user-prompt-* variants
│   │   ├── schema/               # rq1.base.json, rq3.base.json (+ OpenAI wrappers)
│   │   └── use-case/             # k.csv, features.csv
│   └── output/                   # Experiment results (generated)
├── run_experiments.py            # Single entry point for all providers/RQs
└── requirements.txt
```

## 🚀 Features

### Supported model families (default models)
- **Proprietary**
  - `openai`: `gpt-5.3-chat-latest`
  - `gemini`: `gemini-3-flash-preview`
  - `anthropic`: `claude-opus-4-6-thinking`
  - `mistral`: `mistral-large-latest`
  - `perplexity`: `perplexity/sonar`
- **Open (Hugging Face)**
  - `llama4scout`: `meta-llama/Llama-4-Scout-17B-16E`
  - `gemma4`: `google/gemma-4-31B`
  - `qwen3`: `Qwen/Qwen3-30B-A3B`
  - `gptoss20b`: `openai/gpt-oss-20b`
  - `mistralsmall31`: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
  - `deepseekv3`: `deepseek-ai/DeepSeek-V3`

Each experiment runs in two **output modes**:
- **`structured`**: API JSON Schema enforcement (`system-prompt-rq*.txt`)
- **`prompt`**: JSON shape described in the system prompt (`system-prompt-output-rq*.txt`)

Schemas enforce exactly **k** ranked apps via `minItems` / `maxItems` on the `a` array.

### Analysis Capabilities
- **Multi-Model Comparison**: Evaluate consistency across different LLMs
- **Feature-Specific Rankings**: Generate recommendations for specific app features
- **Category-Based Analysis**: Analyze apps within AI-powered categories
- **Consistency Metrics**: 
  - Rank-Biased Overlap (RBO)
  - Jaccard Similarity
  - Internal consistency (within model)
  - External consistency (across models)

### Advanced Analytics
- **Semantic Clustering**: Group similar ranking criteria using embeddings
- **Active Learning**: Interactive threshold optimization for criteria deduplication
- **Visualization**: Heatmaps, dendrograms, and comprehensive charts
- **Data Processing**: Automated merging and cleaning of recommendation data

## 🎯 Study Features

The empirical study examines LLM behavior when generating recommendations for 16 specific app features:
- Broadcast messages to multiple contacts
- Send files
- Watch streams
- Go Live
- Play playlist on shuffle mode
- Access to podcasts
- Build photo collage
- Photo effects
- Access to movies
- Rate movies
- Keeping up with friends
- Play games
- Collaborate with others
- Write notes
- Search for offer on item
- List items for sale

## 🛠️ Experimental Setup

### Prerequisites
- Python 3.8+
- Required API keys for LLM providers

### Environment Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd llm-recommender-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
MISTRAL_API_KEY=your_mistral_key
PERPLEXITY_API_KEY=your_perplexity_key
HF_TOKEN=your_huggingface_token
```

## 🔬 Running Experiments

All feature-based experiments use **`run_experiments.py`**. Outputs are written as flat JSON files under:

`data/output/features/{rq}/{family}/`

Filename pattern (underscore-separated):

| RQ | Pattern | Example |
|----|---------|---------|
| RQ1 | `{family}_{provider}_{modelKey}_{mode}_k{k}_{FeatureCamelCase}_{run}.json` | `proprietary_openai_openai_structured_k20_PhotoEffects_0.json` |
| RQ3 | `{family}_{provider}_{modelKey}_{mode}_k{k}_{FeatureCamelCase}_{CriterionCamelCase}_{run}.json` | `open_huggingface_gemma4_prompt_k20_PhotoEffects_CustomerSupport_0.json` |

where `mode` is `structured` or `prompt`, and multi-word features use CamelCase (e.g. `Photo effects` → `PhotoEffects`).

### Feature-based ranking with self-elicited criteria (RQ1)

10 runs per (family, model, mode, k, feature) by default.

```bash
# All model families, both output modes
python run_experiments.py --rq rq1

# Proprietary only
python run_experiments.py --rq rq1 --families proprietary

# Open models only
python run_experiments.py --rq rq1 --families open

# Single model key, structured output only
python run_experiments.py --rq rq1 --model-keys openai --modes structured

# Smoke test (no API calls)
python run_experiments.py --rq rq1 --model-keys openai --modes structured \
    --search "Photo effects" --k 20 --dry-run
```

### Ranking with fixed criteria from RQ1 pipeline (RQ3)

4 runs per criteria row by default. Requires `data/output/features/rq1/rc_wo_id.csv` (from the RQ1 criteria consolidation pipeline) unless you pass `--criteria-csv`.

```bash
python run_experiments.py --rq rq3

python run_experiments.py --rq rq3 --criteria-csv path/to/criteria.csv
```

### CLI reference

| Flag | Description |
|------|-------------|
| `--rq` | `rq1` or `rq3` (required) |
| `--families` | `both`, `proprietary`, `open`, or comma list |
| `--model-keys` | `all` or comma list of keys (`openai,gemini,...,gemma4,qwen3,...`) |
| `--modes` | `both`, `structured`, `prompt`, or comma list |
| `--models` | Overrides by model key, e.g. `gemma4=google/gemma-4-31B` |
| `--k` | Override k values (default from `data/input/use-case/k.csv`) |
| `--search` | Override features (default from `features.csv`) |
| `--n` | Runs per item (default: 10 for rq1, 4 for rq3) |
| `--sleep` | Seconds between API calls (default: 10) |
| `--max-attempts` | Retry attempts per run when JSON/schema validation fails |
| `--dry-run` | Print planned output paths only |

### Consistency Analysis

#### App Ranking Consistency
```bash
python -m code.consistency.app_consistency \
    --input data/output/evaluation/app_rankings.csv \
    --output data/output/evaluation/consistency
```

#### Ranking Criteria Consistency
```bash
python -m code.consistency.ranking_criteria_consistency \
    --input data/output/evaluation/app_ranking_criteria.csv \
    --output data/output/evaluation/consistency/ranking_criteria
```

### Visualization

#### Criteria Visualization
```bash
python -m code.visualization.criteria_visualization \
    --input data/output/features/rq1/rc_extracted.csv \
    --output data/output/features/rq1/ \
    --similarity-threshold 0.72
```

#### Source Visualization
```bash
python -m code.visualization.source_visualization \
    --input data/output/features/rq1/rc_extracted.csv \
    --output data/output/features/rq1/
```

## 📊 Experimental Output Structure

### Generated Data
- **JSON Responses**: Individual LLM responses for each experimental trial
- **CSV Rankings**: Consolidated app rankings across all trials and models
- **Consistency Metrics**: RBO and Jaccard similarity calculations
- **Visualization Files**: Heatmaps, dendrograms, and analysis charts
- **Evaluation Reports**: Comprehensive analysis of LLM behavior patterns

### Data Organization
```
data/output/features/
├── rq1/
│   ├── proprietary/
│   │   ├── proprietary_openai_openai_structured_k20_PhotoEffects_0.json
│   │   └── ...
│   └── open/
│       ├── open_huggingface_gemma4_structured_k20_PhotoEffects_0.json
│       └── ...
└── rq3/
    ├── proprietary/
    │   └── proprietary_openai_openai_structured_k20_PhotoEffects_CustomerSupport_0.json
    └── open/
        └── open_huggingface_qwen3_prompt_k20_PhotoEffects_CustomerSupport_0.json
```

## 📈 Results and Analysis

### Key Findings
- **Model Consistency**: Analysis of ranking consistency within and across LLM models
- **Feature Sensitivity**: How different app features affect recommendation patterns
- **Category Behavior**: LLM behavior variations across AI-powered app categories
- **Ranking Criteria**: Semantic analysis of ranking criteria used by different models

### Visualization Examples
- **Heatmaps**: Model comparison matrices showing ranking similarities
- **Dendrograms**: Hierarchical clustering of ranking criteria
- **Consistency Charts**: RBO and Jaccard similarity visualizations
- **Correlation Plots**: Cross-model correlation analysis

## 📚 References

### Research Papers
- ...

### Technical Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Mistral AI API Documentation](https://docs.mistral.ai/)
- [Perplexity API Documentation](https://docs.perplexity.ai/)

### Related Work
- ....
## 📄 License

This project is licensed under the GPL version 3 - see the [LICENSE](LICENSE) file for details.

## 📄 Acknowledgments

- ...