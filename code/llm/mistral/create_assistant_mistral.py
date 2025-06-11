import os
import argparse
from mistralai import Mistral
from typing import Any
from dotenv import load_dotenv
import logging
from code.llm.create_assistant import CreateAssistant

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MistralAssistantCreator(CreateAssistant):
    def __init__(self, model: str = "mistral-large-2411", temperature: float = 0.3, top_p: float = 0.95):
        super().__init__(model)
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.temperature = temperature
        self.top_p = top_p
        self.client = self._initialize_client()

    def _initialize_client(self) -> Mistral:
        """Initialize the Mistral client."""
        return Mistral(api_key=self.api_key)

    def create_assistant(self, guidelines_file: str) -> Any:
        """Create a Mistral assistant with guidelines stored in memory."""
        self.logger.info(f"Creating new assistant with model {self.model}")
        system_prompt = self.load_system_prompt(guidelines_file)

        try:
            print("Creating a new assistant...")

            # Create Assistant using the beta agents API format
            assistant = self.client.beta.agents.create(
                model=self.model,
                name="App Ranking Assistant",
                description="App Ranking Assistant",
                instructions=system_prompt,
                tools=[{"type": "web_search"}],
                #completion_args={
                #    "temperature": self.temperature,
                #    "top_p": self.top_p
                #}
            )

            self.save_assistant_id(assistant.id)

            print("Assistant created successfully.")
            self.logger.info(f"Assistant created successfully with ID: {assistant.id}")
            return assistant
        except Exception as e:
            self.logger.error(f"Error creating assistant: {str(e)}")
            raise

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Create a Mistral assistant for app ranking.')
    parser.add_argument('--system_prompt', required=True, help='Path to the system prompt file')
    parser.add_argument('--model', default="mistral-large-2411", help='Mistral model to use')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature parameter for the model')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p parameter for the model')

    args = parser.parse_args()
    try:
        creator = MistralAssistantCreator(args.model, args.temperature, args.top_p)
        creator.create_assistant(args.system_prompt)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
