import os
import argparse
import time
from datetime import datetime
import requests
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

class PerplexitySearch:
    def __init__(self, model: str = "sonar"):
        # Initialize the API key from environment
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not found")
        self.model = model
        self.base_url = "https://api.perplexity.ai/chat/completions"
        logger.info(f"Using model: {self.model}")

    def read_and_format_prompt(self, k: int, search: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        input_file = "data/input/prompts/user-prompt-feature-rq1.txt"
        with open(input_file, 'r') as file:
            prompt = file.read()
        
        return prompt.replace('{k}', str(k)).replace('{search}', search)

    def read_system_prompt(self, system_prompt_path: str) -> str:
        """Read the system prompt from the specified file."""
        with open(system_prompt_path, 'r') as file:
            return file.read()

    def save_response(self, response: str, output_folder: str, base_name: str, run_number: int):
        """Save the response to a JSON file."""
        output_file = os.path.join(output_folder, f'{base_name}_{run_number}.json')
        
        # Remove ```json prefix and ``` suffix if present
        if response.startswith('```json'):
            response = response[7:]  # Remove ```json prefix
        if response.endswith('```'):
            response = response[:-3]  # Remove ``` suffix
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)

    def run_prompt(self, output_folder: str, k: int, search: str, system_prompt_path: str, n: int = 1, sleep_time: int = 10):
        """Run the prompt n times and save responses."""
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        user_prompt = self.read_and_format_prompt(k, search)
        system_instruction = self.read_system_prompt(system_prompt_path)
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename("user-prompt-feature-rq1.txt"))[0]

        for i in range(n):
            print(f"Processing run {i+1}/{n}")
            start_time = time.time()

            try:
                # Log the request parameters
                logger.info(f"Making request with model: {self.model}")
                logger.info(f"User prompt: {user_prompt[:200]}...")  # Log first 200 chars of prompt
                logger.info(f"System prompt: {system_instruction[:200]}...")  # Log first 200 chars of system prompt

                # Prepare the payload for Perplexity API
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_instruction
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    "max_tokens": 20000
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                # Make the API request
                response = requests.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Extract the response content
                response_data = response.json()
                response_content = response_data['choices'][0]['message']['content']
                self.save_response(response_content, output_folder, base_name, i)

                process_time = time.time() - start_time
                print(f"Run {i+1} completed (Time: {process_time:.2f}s)")
                
                time.sleep(sleep_time)  # Rate limiting between runs

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error in run {i}: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error in run {i}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                raise

        print("Process complete.")

def main(output_folder: str, k: int, search: str, system_prompt_path: str, n: int = 1, model: str = "sonar", sleep_time: int = 10):
    searchPerplexity = PerplexitySearch(model)
    searchPerplexity.run_prompt(output_folder, k, search, system_prompt_path, n, sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Perplexity API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--search', type=str, required=True, help='search parameter')
    parser.add_argument('--system-prompt', required=True, help='Path to the system prompt file')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='sonar', help='Model name')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep time between runs in seconds')
    args = parser.parse_args()
    
    main(args.output, args.k, args.search, args.system_prompt, args.n, args.model, args.sleep)
