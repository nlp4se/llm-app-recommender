import os
import argparse
import time
from datetime import datetime
from google import genai
from google.genai.types import Tool, GoogleSearch
from google.genai.types import GenerateContentConfig
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

class GeminiSearch:
    def __init__(self, model: str = "gemini-2.0-flash"):
        # Initialize the client with API key from environment
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model
        logger.info(f"Using model: {self.model}")

    def read_and_format_prompt(self, k: int, category: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        input_file = "data/input/prompts/user-prompt-uc1.txt"
        with open(input_file, 'r') as file:
            prompt = file.read()
        
        return prompt.replace('{k}', str(k)).replace('{category}', category)

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

    def run_prompt(self, output_folder: str, k: int, category: str, system_prompt_path: str, n: int = 1, sleep_time: int = 10):
        """Run the prompt n times and save responses."""
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        user_prompt = self.read_and_format_prompt(k, category)
        system_instruction = self.read_system_prompt(system_prompt_path)
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename("user-prompt-uc1.txt"))[0]

        for i in range(n):
            print(f"Processing run {i+1}/{n}")
            start_time = time.time()

            try:
                # Log the request parameters
                logger.info(f"Making request with model: {self.model}")
                logger.info(f"User prompt: {user_prompt[:200]}...")  # Log first 200 chars of prompt
                logger.info(f"System prompt: {system_instruction[:200]}...")  # Log first 200 chars of system prompt

                search_tool = Tool(google_search=GoogleSearch())
                config = GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[search_tool],
                    response_modalities=["TEXT"],
                    candidate_count=1
                )
                
                # Create chat completion using Gemini API with web search enabled
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config
                )

                # Extract the response content
                response_content = response.text
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

def main(output_folder: str, k: int, category: str, system_prompt_path: str, n: int = 1, model: str = "gemini-2.0-flash", sleep_time: int = 10):
    search = GeminiSearch(model)
    search.run_prompt(output_folder, k, category, system_prompt_path, n, sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Google Gemini API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--category', type=str, required=True, help='Category parameter')
    parser.add_argument('--system-prompt', required=True, help='Path to the system prompt file')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', help='Model name')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep time between runs in seconds')
    args = parser.parse_args()
    
    main(args.output, args.k, args.category, args.system_prompt, args.n, args.model, args.sleep)
