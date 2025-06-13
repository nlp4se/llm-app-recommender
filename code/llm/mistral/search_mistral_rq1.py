import os
import argparse
import time
from datetime import datetime
from mistralai import Mistral
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

def load_assistant_id(model: str) -> str:
    """Load the assistant ID from a model-specific file if it exists."""
    assistant_id_file = f"data/assistants/assistant_id_{model}.txt"
    if os.path.exists(assistant_id_file):
        with open(assistant_id_file, 'r') as file:
            return file.read().strip()
    return None

def get_assistant(model: str) -> str:
    """Check for an existing assistant ID or raise an error if not found."""
    assistant_id = load_assistant_id(model)
    
    if assistant_id:
        print(f"Reusing existing assistant ID for {model}: {assistant_id}")
        return assistant_id
    
    raise ValueError(f"Assistant ID not found for {model}. Run create_assistant.py first.")

class MistralSearch:
    def __init__(self, model: str = "mistral-large-latest"):
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.model = model
        self.assistant_id = get_assistant(model)
        
        # Add validation
        if not self.assistant_id:
            raise ValueError(f"No assistant ID found for model {model}")
        logger.info(f"Using assistant ID: {self.assistant_id}")

    def read_and_format_prompt(self, k: int, search: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        input_file = "data/input/prompts/user-prompt-feature-rq1.txt"
        with open(input_file, 'r') as file:
            prompt = file.read()
        
        return prompt.replace('{k}', str(k)).replace('{search}', search)

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

    def run_prompt(self, output_folder: str, k: int, search: str, n: int = 1, sleep_time: int = 10):
        """Run the prompt n times and save responses."""
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        user_prompt = self.read_and_format_prompt(k, search)
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename("user-prompt-feature-rq1.txt"))[0]

        for i in range(n):
            print(f"Processing run {i+1}/{n}")
            start_time = time.time()

            try:
                # Log the request parameters
                logger.info(f"Making request with agent_id: {self.assistant_id}")
                logger.info(f"User prompt: {user_prompt[:200]}...")  # Log first 200 chars of prompt
                
                # Create chat completion using Mistral API
                response = self.client.agents.complete(
                    messages=[{"role": "user","content": user_prompt}],
                    agent_id=self.assistant_id
                )

                # Extract the response content
                response_content = response.choices[0].message.content
                self.save_response(response_content, output_folder, base_name, i)

                process_time = time.time() - start_time
                print(f"Run {i+1} completed (Time: {process_time:.2f}s)")
                
                time.sleep(sleep_time)  # Rate limiting between runs

            except Exception as e:
                logger.error(f"Error in run {i}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                if hasattr(e, 'response'):
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                raise

        print("Process complete.")

def main(output_folder: str, k: int, search: str, n: int = 1, model: str = "mistral-large-latest", sleep_time: int = 10):
    searchMistral = MistralSearch(model)
    searchMistral.run_prompt(output_folder, k, search, n, sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Mistral Chat Completions API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--search', type=str, required=True, help='search parameter')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='mistral-large-latest', help='Model name')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep time between runs in seconds')
    args = parser.parse_args()
    
    main(args.output, args.k, args.search, args.n, args.model, args.sleep)
