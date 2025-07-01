import os
import argparse
import time
from datetime import datetime
import anthropic
from dotenv import load_dotenv
import json
import logging
import pandas as pd  # Add pandas import

# Load environment variables
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnthropicSearch:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        # Initialize the client with API key from environment
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model
        logger.info(f"Using model: {self.model}")

    def read_and_format_prompt(self, k: int, search: str, ranking_criteria: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        input_file = "data/input/prompts/user-prompt-feature-rq3.txt"  # Updated to rq3
        with open(input_file, 'r') as file:
            prompt = file.read()
        
        return prompt.replace('{k}', str(k)).replace('{search}', search).replace('{ranking_criteria}', ranking_criteria)

    def read_system_prompt(self) -> str:
        """Read the system prompt from a file."""
        system_prompt_file = "data/input/prompts/system-prompt-output-rq3.txt"  # Updated to rq3
        with open(system_prompt_file, 'r') as file:
            return file.read()

    def save_response(self, response: str, output_folder: str, ranking_criteria_name: str, run_number: int):
        """Save the response to a JSON file."""
        # Create filename with ranking criteria name and run number
        output_file = os.path.join(output_folder, f'{ranking_criteria_name}_{run_number}.json')
        
        # Remove ```json prefix and ``` suffix if present
        if response.startswith('```json'):
            response = response[7:]  # Remove ```json prefix
        if response.endswith('```'):
            response = response[:-3]  # Remove ``` suffix
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)

    def run_prompt(self, output_folder: str, k: int, input_csv: str, search: str, n: int = 1, sleep_time: int = 10):
        """Run the prompt n times for each feature-criteria pair from the CSV file and save responses."""
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        system_instruction = self.read_system_prompt()
        
        # Create the main output folder
        os.makedirs(output_folder, exist_ok=True)

        try:
            criteria_df = pd.read_csv(input_csv, encoding='utf-8', sep=";")
        except FileNotFoundError:
            logger.error(f"Input CSV file not found: {input_csv}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"UTF-8 encoding error reading CSV file {input_csv}: {e}")
            logger.error("Please ensure the CSV file is saved with UTF-8 encoding")
            raise

        for index, row in criteria_df.iterrows():
            print(row)
            
            ranking_criteria_dict = [row.to_dict()]
            ranking_criteria = json.dumps(ranking_criteria_dict, indent=2)
            
            user_prompt = self.read_and_format_prompt(k, search, ranking_criteria)

            # Get ranking criteria name from the row (assuming it's in a column, adjust as needed)
            # You may need to adjust this based on your CSV structure
            ranking_criteria_name = "".join(c if c.isalnum() else "_" for c in str(row.iloc[0]))  # Assuming first column contains criteria name

            for i in range(n):
                print(f"Processing criteria '{search}', run {i+1}/{n}")
                start_time = time.time()

                try:
                    # Log the request parameters
                    logger.info(f"Making request with model: {self.model}")
                    logger.info(f"User prompt: {user_prompt[:200]}...")  # Log first 200 chars of prompt
                    logger.info(f"System prompt: {system_instruction[:200]}...")  # Log first 200 chars of system prompt

                    # Create message using Anthropic API with web search enabled
                    message = self.client.beta.messages.create(
                        model=self.model,
                        max_tokens=20000,
                        system=system_instruction,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": user_prompt
                                    }
                                ]
                            }
                        ]
                    )

                    # Extract the response content
                    response_content = message.content[0].text
                    self.save_response(response_content, output_folder, ranking_criteria_name, i)

                    process_time = time.time() - start_time
                    print(f"Run {i+1} completed (Time: {process_time:.2f}s)")
                    
                    time.sleep(sleep_time)  # Rate limiting between runs

                except Exception as e:
                    logger.error(f"Error in run {i+1} for criteria '{search}': {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    if hasattr(e, 'response'):
                        logger.error(f"Response status: {e.response.status_code}")
                        logger.error(f"Response body: {e.response.text}")
                    raise

        print("Process complete.")

def main(output_folder: str, k: int, input_csv: str, search: str, n: int = 1, model: str = "claude-sonnet-4-20250514", sleep_time: int = 10):
    searchAnthropic = AnthropicSearch(model)
    searchAnthropic.run_prompt(output_folder, k, input_csv, search, n, sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Anthropic Claude API based on a CSV input.')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--input-csv', required=True, help='Input CSV file with ranking criteria')
    parser.add_argument('--search', type=str, required=True, help='search parameter')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform for each row')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514', help='Model name')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep time between runs in seconds')
    args = parser.parse_args()
    
    main(args.output, args.k, args.input_csv, args.search, args.n, args.model, args.sleep)
