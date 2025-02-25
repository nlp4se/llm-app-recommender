from abc import ABC, abstractmethod
import os
from datetime import datetime
import time

class BaseAssistant(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def load_assistant_id(self) -> str:
        """Load the assistant ID from a model-specific file if it exists."""
        pass

    @abstractmethod
    def get_assistant(self) -> str:
        """Check for an existing assistant ID or create a new one."""
        pass

    @abstractmethod
    def create_thread(self) -> str:
        """Create a new thread for conversation persistence."""
        pass

    @abstractmethod
    def send_message_and_wait(self, thread_id: str, assistant_id: str, content: str) -> str:
        """Send a message and wait for the response."""
        pass

    def read_and_format_prompt(self, input_file: str, k: int, category: str) -> str:
        """Read the prompt template and replace placeholders with parameters."""
        with open(input_file, 'r') as file:
            prompt = file.read()
        return prompt.replace('{k}', str(k)).replace('{category}', category)

    def save_response(self, response: str, output_folder: str, base_name: str, run_number: int):
        """Save the assistant's response to a JSON file."""
        output_file = os.path.join(output_folder, f'{base_name}_{run_number}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)

    def run_prompt(self, output_folder: str, k: int, category: str, n: int = 1):
        """Run the prompt n times and save responses."""
        print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        input_file = "data/input/prompts/user-prompt-uc1.txt"
        user_prompt = self.read_and_format_prompt(input_file, k, category)
        
        assistant_id = self.get_assistant()
        os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_file))[0]

        for i in range(n):
            print(f"Processing run {i+1}/{n}")
            start_time = time.time()
            
            thread_id = self.create_thread()

            try:
                response = self.send_message_and_wait(thread_id, assistant_id, user_prompt)
                self.save_response(response, output_folder, base_name, i)

                process_time = time.time() - start_time
                print(f"Run {i+1} completed (Time: {process_time:.2f}s)")
                
                time.sleep(10)  # Rate limiting between runs

            except Exception as e:
                print(f"Error in run {i}: {e}")

        print("Process complete.") 