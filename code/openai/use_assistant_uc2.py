import json
import os
import time
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_assistant_id(model: str) -> str:
    """Load the assistant ID from a model-specific file if it exists."""
    assistant_id_file = f"assistant_id_{model}.txt"
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

def create_thread() -> str:
    """Create a new thread for conversation persistence."""
    return client.beta.threads.create().id

def read_and_format_prompt(input_file: str, k: int, category: str, ranking_criteria_file: str) -> str:
    """Read the prompt template and replace placeholders with parameters."""
    # Read ranking criteria
    with open(ranking_criteria_file, 'r') as file:
        ranking_criteria = file.read().strip()

    # Read prompt template
    with open(input_file, 'r') as file:
        prompt = file.read()
    
    return prompt.replace('{k}', str(k)) \
                 .replace('{category}', category) \
                 .replace('{ranking_criteria}', ranking_criteria)

def save_response(response: str, output_folder: str, base_name: str, run_number: int):
    """Save the assistant's response to a JSON file."""
    output_file = os.path.join(output_folder, f'{base_name}_{run_number}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response)

def main(output_folder: str, k: int, category: str, ranking_criteria_file: str, n: int = 1, model: str = "gpt-4o"):
    print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Use hardcoded prompt file path
    input_file = "data/input/prompts/user-prompt-uc2.txt"
    user_prompt = read_and_format_prompt(input_file, k, category, ranking_criteria_file)
    
    assistant_id = get_assistant(model)
    os.makedirs(output_folder, exist_ok=True)

    # Generate base output filename from input filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Run n iterations
    for i in range(n):
        print(f"Processing run {i+1}/{n}")
        start_time = time.time()
        
        thread_id = create_thread()

        try:
            # Send prompt to assistant
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_prompt
            )

            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )

            while True:
                run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                if run_status.status == "completed":
                    break
                elif run_status.status != "in_progress" and run_status.status != "queued":
                    print(run_status)
                time.sleep(2)

            messages = client.beta.threads.messages.list(thread_id=thread_id)
            response = messages.data[0].content[0].text.value

            # Save response
            save_response(response, output_folder, base_name, i)

            process_time = time.time() - start_time
            print(f"Run {i+1} completed (Time: {process_time:.2f}s)")
            
            time.sleep(10)  # Rate limiting between runs

        except Exception as e:
            print(f"Error in run {i}: {e}")

    print("Process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Assistants API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--category', type=str, required=True, help='Category parameter')
    parser.add_argument('--ranking_criteria', type=str, required=True, help='Ranking criteria file')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name for assistant')
    args = parser.parse_args()
    
    main(args.output, args.k, args.category, args.ranking_criteria, args.n, args.model)
