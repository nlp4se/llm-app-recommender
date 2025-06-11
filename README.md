## How to use

Create assistant

```python -m code.llm.openai.create_assistant_openai --system_prompt ./data/input/prompts/system-prompt.txt --model gpt-4o```

Use assitant (UC1)

```python -m code.llm.openai.use_assistant_openai_uc1 --output ./data/output/uc1/openai --k 10 --category "sports tracking" --n 10 --model gpt-4o --sleep 10```

Process rankings

```python -m code.data-processor.ranking_matrix --input_folder ./data/output/uc1/openai --output_folder ./data/output/uc1/openai --experiment_name user-prompt-uc1```

Process recommendation ranking criteria

```python -m code.data-processor.ranking_criteria --input_folder ./data/output/uc1/openai --output_folder ./data/output/uc1/openai --experiment_name user-prompt-uc1```

After normalization, generate json criteria file

```python -m code.data-processor.ranking_criteria_csv_to_json --input ./data/output/uc1/openai/ranking_criteria/ranking_criteria_normalized.xlsx --output ./data/output/uc1/openai/ranking_criteria/ranking_criteria_normalized.json```

Use assitant (UC2)

```python -m code.llm.openai.use_assistant_openai_uc2 --output ./data/output/uc2/openai --k 10 --category "sports tracking" --ranking_criteria ./data/output/uc1/openai/ranking_criteria/ranking_criteria.json --n 10 --model gpt-4o --sleep 10```

Evaluate internal consistency

```python -m code.evaluation.internal_consistency --uc1_file ./data/output/uc1/openai/user-prompt-uc1_rankings.csv --uc2_file ./data/output/uc2/openai/user-prompt-uc2_rankings.csv --output_path ./data/output/evaluation/openai/internal-consistency/all-criteria```

## Search-based

Use search OpenAI model

```python -m code.llm.openai.search_openai_uc1 --output ./data/output/search/uc1/openai --k 10 --category "sports tracking" --n 10 --model gpt-4o-search-preview --sleep 10```

Extract ranking criteria

```python -m code.data-processor.ranking_criteria_to_csv --input-folder .\data\output\search\uc1\gemini\ --output-file-all .\data\output\search\uc1\gemini\ranking-criteria-all.csv --output-file-grouped .\data\output\search\uc1\gemini\ranking-criteria-grouped.csv```

Internal consistency

```python -m code.data-processor.ranking_matrix --input_folder .\data\output\search\uc1\gemini\k20_AI-powered_education\ --experiment_name user-prompt-uc1 --output_folder .\data\output\search\uc1\gemini\k20_AI-powered_education\ --category_name "AI-powered education" --max_rank 10```

