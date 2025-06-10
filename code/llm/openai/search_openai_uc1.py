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

    def read_and_format_prompt(self, k: int, category: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        input_file = "data/input/prompts/user-prompt-uc1.txt"
        with open(input_file, 'r') as file:
            prompt = file.read()
        
        return prompt.replace('{k}', str(k)).replace('{category}', category)

    def read_system_prompt(self) -> str:
        """Read the system prompt from a file."""
        system_prompt_file = "data/input/prompts/system-prompt.txt"
        with open(system_prompt_file, 'r') as file:
            return file.read()

    def save_response(self, response: str, output_folder: str, base_name: str, run_number: int):
        """Save the response to a JSON file."""
        output_file = os.path.join(output_folder, f'{base_name}_{run_number}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)

    def run_prompt(self, output_folder: str, k: int, category: str, n: int = 1, sleep_time: int = 10):
        """Run the prompt n times and save responses."""
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        user_prompt = self.read_and_format_prompt(k, category)
        system_prompt = self.read_system_prompt()
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename("user-prompt-uc1.txt"))[0]

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
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "rank_apps",
                            "description": "Ranks apps based on specified criteria and provides detailed ranking information",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "apps": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "rank": {"type": "integer", "description": "Position in the ranking"},
                                                "name": {"type": "string", "description": "Name of the app"}
                                            },
                                            "required": ["rank", "name"]
                                        }
                                    },
                                    "criteria": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string", "description": "Name of the ranking criterion"},
                                                "description": {"type": "string", "description": "Description of the ranking criterion"},
                                                "type": {"type": "string", "description": "Data type(e.g., Integer, Float, Boolean, Text, Media...) and cardinality (e.g., [1], [5], [*]...)"},
                                                "sources": {"type": "string", "description": "Sources of the criterion"},
                                            },
                                            "required": ["name", "description", "type", "sources"]
                                        }
                                    }
                                },
                                "required": ["apps", "criteria"]
                            }
                        }
                    }
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

def main(output_folder: str, k: int, category: str, n: int = 1, model: str = "gpt-4o-search-preview", sleep_time: int = 10):
    search = OpenAISearch(model)
    search.run_prompt(output_folder, k, category, n, sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Chat Completions API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--category', type=str, required=True, help='Category parameter')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='gpt-4o-search-preview', help='Model name')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep time between runs in seconds')
    args = parser.parse_args()
    
    main(args.output, args.k, args.category, args.n, args.model, args.sleep)
