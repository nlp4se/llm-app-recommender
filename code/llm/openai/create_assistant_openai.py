from openai import OpenAI
import argparse
from dotenv import load_dotenv
import os
import logging
from code.llm.create_assistant import CreateAssistant

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OpenAIAssistantCreator(CreateAssistant):
    def __init__(self, model: str = "gpt-4"):
        super().__init__(model)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def create_assistant(self, guidelines_file: str):
        """Create an OpenAI assistant with guidelines stored in memory."""
        self.logger.info(f"Creating new assistant with model {self.model}")
        system_prompt = self.load_system_prompt(guidelines_file)

        try:
            print("Creating a new assistant...")

            # Create Assistant
            assistant = self.client.beta.assistants.create(
                name="App Ranking Assistant",
                instructions=system_prompt,
                model=self.model,
                temperature=0.7,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rank_apps",
                        "description": "Ranks apps based on specified criteria and provides detailed ranking information",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "apps_ranked": {
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
                                "ranking_criteria": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string", "description": "Name of the ranking criterion"},
                                            "metrics": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "List of metrics used"
                                            },
                                            "data_sources": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "List of data sources"
                                            }
                                        },
                                        "required": ["name", "metrics", "data_sources"]
                                    }
                                }
                            },
                            "required": ["apps_ranked", "ranking_criteria"]
                        }
                    }
                }
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
    parser.add_argument('--model', default="gpt-4o", help='OpenAI model to use')

    args = parser.parse_args()
    try:
        creator = OpenAIAssistantCreator(args.model)
        creator.create_assistant(args.system_prompt)
    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise
