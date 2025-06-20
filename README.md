# Empirical Study: LLM Behavior as System Recommenders in Mobile App Domain

An empirical research study that investigates how Large Language Models (LLMs) behave when deployed as system recommenders in the mobile app domain. This study systematically evaluates multiple LLM providers (OpenAI, Google Gemini, and Mistral) to understand their recommendation patterns, consistency, and behavior when generating app rankings based on specific features and categories.

## 📋 Study Overview

This empirical investigation examines LLM behavior in mobile app recommendation scenarios across different AI-powered categories and features. The study focuses on:

- **Multi-LLM Behavioral Analysis**: Comparing recommendation patterns across OpenAI GPT-4, Google Gemini, and Mistral models
- **Feature-Based Recommendation Studies**: Analyzing how LLMs generate app recommendations for specific app features (e.g., "Photo effects", "Go Live", "Collaborate with others")
- **Category-Based Behavioral Analysis**: Examining LLM behavior when evaluating apps within AI-powered categories (e.g., "AI-powered entertainment", "AI-powered productivity")
- **Consistency Measurement**: Quantifying ranking consistency both within and across different LLM models
- **Behavioral Visualization**: Creating comprehensive visualizations of LLM recommendation patterns and criteria

## 📁 Project Structure

```
llm-recommender-system/
├── code/                          # Main source code
│   ├── llm/                      # LLM integration modules
│   │   ├── google/               # Google Gemini implementation
│   │   ├── mistral/              # Mistral AI implementation
│   │   ├── openai/               # OpenAI implementation
│   │   ├── create_assistant.py   # Abstract assistant creation
│   │   └── use_assistant.py      # Assistant usage utilities
│   ├── consistency/              # Ranking consistency analysis
│   │   ├── app_consistency.py    # App ranking consistency
│   │   ├── app_internal_consistency.py
│   │   └── ranking_criteria_consistency.py
│   ├── correlation/              # Correlation analysis tools
│   ├── data-processor/           # Data processing utilities
│   └── visualization/            # Visualization modules
│       ├── criteria_visualization.py
│       └── source_visualization.py
├── data/                         # Data directory
│   ├── input/                    # Input data and configurations
│   │   ├── prompts/              # System and user prompts
│   │   ├── schema/               # JSON schemas for responses
│   │   └── use-case/             # Categories and features data
│   ├── output/                   # Generated outputs
│   │   ├── category/             # Category-based results
│   │   ├── features/             # Feature-based results
│   │   ├── evaluation/           # Evaluation metrics
│   │   └── search/               # Search results
│   └── assistants/               # Stored assistant IDs
├── experiments-*.py              # Experiment runner scripts
└── hot-fix.py                    # Utility scripts
```

## 🚀 Features

### Supported LLM Providers
- **OpenAI GPT-4**: Advanced reasoning and ranking capabilities
- **Google Gemini**: Web search integration for real-time data
- **Mistral AI**: Cost-effective alternative with strong performance

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
MISTRAL_API_KEY=your_mistral_key
```

## 🔬 Running Experiments

### Feature-Based Behavioral Analysis (RQ1)
```bash
# Run experiments for all LLM providers
python experiments-gemini-rq1.py
python experiments-mistral-rq1.py
python experiments-openai-rq1.py
```

### Category-Based Behavioral Analysis (RQ3)
```bash
# Run category-based experiments
python experiments-gemini-rq3.py
python experiments-mistral-rq3.py
python experiments-openai-rq3.py
```

### Individual LLM Searches

#### Google Gemini
```bash
python -m code.llm.google.search_gemini_rq1 \
    --output ./data/output/features/rq1/gemini/k20_Photo_effects \
    --k 20 \
    --search "Photo effects" \
    --n 10 \
    --model "gemini-2.0-flash" \
    --system-prompt "data/input/prompts/system-prompt-output-rq1.txt"
```

#### OpenAI
```bash
python -m code.llm.openai.search_openai_rq1 \
    --output ./data/output/features/rq1/openai/k20_Photo_effects \
    --k 20 \
    --search "Photo effects" \
    --n 10 \
    --model "gpt-4o" \
    --system-prompt "data/input/prompts/system-prompt-output-rq1.txt"
```

#### Mistral
```bash
python -m code.llm.mistral.search_mistral_rq1 \
    --output ./data/output/features/rq1/mistral/k20_Photo_effects \
    --k 20 \
    --search "Photo effects" \
    --n 10 \
    --model "mistral-large-latest" \
    --system-prompt "data/input/prompts/system-prompt-output-rq1.txt"
```

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
    --input data/output/features/rq1/gemini/all_criteria.csv \
    --output data/output/features/rq1/gemini/ \
    --similarity-threshold 0.72
```

#### Source Visualization
```bash
python -m code.visualization.source_visualization \
    --input data/output/features/rq1/gemini/all_criteria.csv \
    --output data/output/features/rq1/gemini/
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
data/output/
├── features/rq1/           # Feature-based analysis results
│   ├── gemini/            # Google Gemini results
│   ├── mistral/           # Mistral AI results
│   └── openai/            # OpenAI results
├── category/rq1/          # Category-based analysis results
│   ├── gemini/            # Google Gemini results
│   ├── mistral/           # Mistral AI results
│   └── openai/            # OpenAI results
├── evaluation/            # Consistency and correlation analysis
│   ├── consistency/       # Ranking consistency metrics
│   └── correlation/       # Cross-model correlation analysis
└── search/               # Search functionality results
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
- [Mistral AI API Documentation](https://docs.mistral.ai/)

### Related Work
- ....
## 📄 License

This project is licensed under the GPL version 3 - see the [LICENSE](LICENSE) file for details.

## 📄 Acknowledgments

- ...