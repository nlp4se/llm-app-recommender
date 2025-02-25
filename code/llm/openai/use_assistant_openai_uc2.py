import json
import os
import time
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from code.llm.openai.use_assistant_openai_uc1 import OpenAIAssistant

# Load environment variables
load_dotenv()

class OpenAIAssistantUC2(OpenAIAssistant):
    def read_and_format_prompt(self, k: int, category: str, ranking_criteria_file: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        # Read ranking criteria
        with open(ranking_criteria_file, 'r') as file:
            ranking_criteria = file.read().strip()

        # Read prompt template from hardcoded path
        input_file = "data/input/prompts/user-prompt-uc2.txt"
        with open(input_file, 'r') as file:
            prompt = file.read()
        
        return prompt.replace('{k}', str(k)) \
                     .replace('{category}', category) \
                     .replace('{ranking_criteria}', ranking_criteria)

    def save_response(self, response: str, output_folder: str, base_name: str, run_number: int):
        """Save the assistant's response to a JSON file."""
        output_file = os.path.join(output_folder, f'{base_name}_{run_number}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)

    def run_prompt(self, output_folder: str, k: int, category: str, ranking_criteria_file: str, n: int = 1, sleep_time: int = 10):
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        user_prompt = self.read_and_format_prompt(k, category, ranking_criteria_file)
        assistant_id = self.get_assistant()
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename("user-prompt-uc2.txt"))[0]

        for i in range(n):
            print(f"Processing run {i+1}/{n}")
            start_time = time.time()
            
            thread_id = self.create_thread()

            try:
                response = self.send_message_and_wait(thread_id, assistant_id, user_prompt)
                self.save_response(response, output_folder, base_name, i)

                process_time = time.time() - start_time
                print(f"Run {i+1} completed (Time: {process_time:.2f}s)")
                
                time.sleep(sleep_time)  # Rate limiting between runs

            except Exception as e:
                print(f"Error in run {i}: {e}")

        print("Process complete.")

def main(output_folder: str, k: int, category: str, ranking_criteria_file: str, n: int = 1, model: str = "gpt-4o", sleep_time: int = 10):
    assistant = OpenAIAssistantUC2(model)
    assistant.run_prompt(output_folder, k, category, ranking_criteria_file, n, sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Assistants API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--category', type=str, required=True, help='Category parameter')
    parser.add_argument('--ranking_criteria', type=str, required=True, help='Ranking criteria file')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name for assistant')
    parser.add_argument('--sleep_time', type=int, default=10, help='Sleep time between runs in seconds')
    args = parser.parse_args()
    
    main(args.output, args.k, args.category, args.ranking_criteria, args.n, args.model, args.sleep_time)
