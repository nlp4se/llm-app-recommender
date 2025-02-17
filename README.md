## How to use

Create assistant

```python.exe .\code\openai\create_assistant.py --system_prompt .\data\input\prompts\system-prompt.txt --model gpt-4o```

Use assitant

```python.exe .\code\openai\use_assistant.py --input .\data\input\prompts\user-prompt-uc1.txt --output .\data\output\uc1 --k 10 --category "sports tracking" --n 10 --model gpt-4o```

Process rankings

```python.exe .\code\data-processor\ranking_matrix.py --input_folder .\data\output\uc1\ --output_folder .\data\output\uc1\ --experiment_name user-prompt-uc1```

Process recommendation reasons
