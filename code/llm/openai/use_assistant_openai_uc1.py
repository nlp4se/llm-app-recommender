import os
import argparse
import time
from openai import OpenAI
from dotenv import load_dotenv
from code.llm.use_assistant import UseAssistant

# Load environment variables
load_dotenv()

class OpenAIAssistant(UseAssistant):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load_assistant_id(self) -> str:
        assistant_id_file = f"data/assistants/assistant_id_{self.model}.txt"
        if os.path.exists(assistant_id_file):
            with open(assistant_id_file, 'r') as file:
                return file.read().strip()
        return None

    def get_assistant(self) -> str:
        assistant_id = self.load_assistant_id()
        
        if assistant_id:
            print(f"Reusing existing assistant ID for {self.model}: {assistant_id}")
            return assistant_id
        
        raise ValueError(f"Assistant ID not found for {self.model}. Run create_assistant.py first.")

    def create_thread(self) -> str:
        return self.client.beta.threads.create().id

    def send_message_and_wait(self, thread_id: str, assistant_id: str, content: str) -> str:
        # Send prompt to assistant
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        while True:
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id, 
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            elif run_status.status != "in_progress" and run_status.status != "queued":
                print(run_status)
            time.sleep(2)

        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value

def main(output_folder: str, k: int, category: str, n: int = 1, model: str = "gpt-4o"):
    assistant = OpenAIAssistant(model)
    assistant.run_prompt(output_folder, k, category, n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run prompts with Assistants API')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--k', type=int, required=True, help='Value for k parameter')
    parser.add_argument('--category', type=str, required=True, help='Category parameter')
    parser.add_argument('--n', type=int, default=1, help='Number of runs to perform')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name for assistant')
    args = parser.parse_args()
    
    main(args.output, args.k, args.category, args.n, args.model)
