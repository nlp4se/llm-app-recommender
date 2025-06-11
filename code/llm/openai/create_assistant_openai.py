from openai import OpenAI
import argparse
from dotenv import load_dotenv
import os
import logging
from code.llm.create_assistant import CreateAssistant
import json

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OpenAIAssistantCreator(CreateAssistant):
    def __init__(self, model: str = "gpt-4o-search-preview"):
        super().__init__(model)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load_json_schema(self, schema_file: str) -> dict:
        """Load JSON schema from a file."""
        self.logger.info(f"Loading JSON schema from {schema_file}")
        try:
            with open(schema_file, 'r', encoding='utf-8') as file:
                schema = json.loads(file.read())
                self.logger.info("JSON schema loaded successfully")
                return schema
        except Exception as e:
            self.logger.error(f"Error loading JSON schema: {str(e)}")
            raise

    def create_assistant(self, guidelines_file: str, schema_file: str):
        """Create an OpenAI assistant with guidelines stored in memory."""
        self.logger.info(f"Creating new assistant with model {self.model}")
        system_prompt = self.load_system_prompt(guidelines_file)
        response_format = self.load_json_schema(schema_file)

        try:
            print("Creating a new assistant...")

            # Create Assistant
            assistant = self.client.beta.assistants.create(
                name="App Ranking Assistant",
                instructions=system_prompt,
                model=self.model,
                temperature=0.7,
                response_format=response_format
            )

            self.save_assistant_id(assistant.id)

            print("Assistant created successfully.")
            self.logger.info(f"Assistant created successfully with ID: {assistant.id}")
            return assistant
        except Exception as e:
            self.logger.error(f"Error creating assistant: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an OpenAI assistant for emotion annotation.')
    parser.add_argument('--system_prompt', required=True, help='Path to the system prompt file')
    parser.add_argument('--schema_file', required=True, help='Path to the JSON schema file')
    parser.add_argument('--model', default="gpt-4o", help='OpenAI model to use')

    args = parser.parse_args()
    try:
        creator = OpenAIAssistantCreator(args.model)
        creator.create_assistant(args.system_prompt, args.schema_file)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise
