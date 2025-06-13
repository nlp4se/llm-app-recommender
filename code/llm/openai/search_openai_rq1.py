import os
import argparse
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import json
import logging

# Load environment variables
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAISearch:
    def __init__(self, model: str = "gpt-4o-search-preview"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def read_and_format_prompt(self, k: int, search: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        input_file = "data/input/prompts/user-prompt-feature-rq1.txt"
        with open(input_file, 'r') as file:
            prompt = file.read()
        
        return prompt.replace('{k}', str(k)).replace('{search}', search)

    def read_system_prompt(self) -> str:
        """Read the system prompt from a file."""
        system_prompt_file = "data/input/prompts/system-prompt-rq1.txt"
        with open(system_prompt_file, 'r') as file:
            return file.read()

    def read_json_schema(self, schema_file: str) -> dict:
        """Read the JSON schema from a file."""
        with open(schema_file, 'r') as file:
            return json.load(file)

    def save_response(self, response: str, output_folder: str, base_name: str, run_number: int):
        """Save the response to a JSON file."""
        output_file = os.path.join(output_folder, f'{base_name}_{run_number}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)

    def run_prompt(self, output_folder: str, k: int, search: str, n: int = 1, sleep_time: int = 10, schema_file: str = "data/input/schema/rank_apps_schema.json"):
        """Run the prompt n times and save responses."""
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        user_prompt = self.read_and_format_prompt(k, search)
        system_prompt = self.read_system_prompt()
        json_schema = self.read_json_schema(schema_file)
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename("user-prompt-feature-rq1.txt"))[0]

        for i in range(n):
            print(f"Processing run {i+1}/{n}")
            start_time = time.time()

            try:
                # Create chat completion
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    response_format=json_schema
                )

                # Extract the response content
                response_content = response.choices[0].message.content
                self.save_response(response_content, output_folder, base_name, i)

                process_time = time.time() - start_time
                print(f"Run {i+1} completed (Time: {process_time:.2f}s)")
                
                time.sleep(sleep_time)  # Rate limiting between runs

            except Exception as e:
                logger.error(f"Error in run {i}: {e}")
                raise

        print("Process complete.")

def main(output_folder: str, k: int, search: str, n: int = 1, model: str = "gpt-4o-search-preview", sleep_time: int = 10, schema_file: str = "data/input/schema/rank_apps_schema.json"):
    searchOpenAI = OpenAISearch(model)
    searchOpenAI.run_prompt(output_folder, k, search, n, sleep_time, schema_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Chat Completions API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--search', type=str, required=True, help='search parameter')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='gpt-4o-search-preview', help='Model name')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep time between runs in seconds')
    parser.add_argument('--schema', type=str, default='data/input/schema/rq1.json', help='Path to JSON schema file')
    args = parser.parse_args()
    
    main(args.output, args.k, args.search, args.n, args.model, args.sleep, args.schema)
