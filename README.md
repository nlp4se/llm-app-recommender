## How to use

Create assistant

```python .\code\llm\openai\create_assistant_openai.py --system_prompt .\data\input\prompts\system-prompt.txt --model gpt-4o```

Use assitant (UC1)

```python.exe .\code\llm\openai\use_assistant_openai_uc1.py --output .\data\output\uc1\openai\temperature-10 --k 10 --category "sports tracking" --n 10 --model gpt-4o --sleep 10```

Use assitant (UC2)

```python.exe .\code\llm\openai\use_assistant_openai_uc2.py --output .\data\output\uc2\openai\temperature-10 --k 10 --category "sports tracking" --ranking_criteria .\data\output\uc1\openai\temperature-10\ranking_criteria\ranking_criteria.json --n 10 --model gpt-4o --sleep 10```

Process rankings

```python.exe .\code\data-processor\ranking_matrix.py --input_folder .\data\output\uc1\openai\temperature-10 --output_folder .\data\output\uc1\openai\temperature-10 --experiment_name user-prompt-uc1```

Process recommendation ranking criteria

```python.exe .\code\data-processor\ranking_criteria.py --input_folder .\data\output\uc1\openai\temperature-10 --output_folder .\data\output\uc1\openai\temperature-10 --experiment_name user-prompt-uc1```
