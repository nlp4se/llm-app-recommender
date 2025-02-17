from openai import OpenAI
import argparse
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_system_prompt(system_prompt_path: str) -> str:
    """Load system prompt from a text file."""
    logger.info(f"Loading system prompt from {system_prompt_path}")
    try:
        with open(system_prompt_path, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.info("System prompt loaded successfully")
            return content
    except Exception as e:
        logger.error(f"Error loading system prompt: {str(e)}")
        raise

def save_assistant_id(assistant_id: str, model: str):
    """Save the assistant ID to a file for reuse."""
    filename = f"assistant_id_{model}.txt"
    logger.info(f"Saving assistant ID to {filename}")
    try:
        with open(filename, 'w') as file:
            file.write(assistant_id)
            logger.info("Assistant ID saved successfully")
    except Exception as e:
        logger.error(f"Error saving assistant ID: {str(e)}")
        raise

def create_assistant(guidelines_file: str, model: str = "gpt-4o"):
    """Create an OpenAI assistant with guidelines stored in memory."""
    logger.info(f"Creating new assistant with model {model}")
    system_prompt = load_system_prompt(guidelines_file)

    try:
        print("Creating a new assistant...")

        # Create Assistant
        assistant = client.beta.assistants.create(
            name="App Ranking Assistant",
            instructions=system_prompt,
            model=model,
            temperature=0.0,
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
                                        "name": {"type": "string", "description": "Name of the app"},
                                        "features": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "List of notable features"
                                        }
                                    },
                                    "required": ["rank", "name", "features"]
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

        save_assistant_id(assistant.id, model)

        print("Assistant created successfully.")
        logger.info(f"Assistant created successfully with ID: {assistant.id}")
        return assistant
    except Exception as e:
        logger.error(f"Error creating assistant: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an OpenAI assistant for emotion annotation.')
    parser.add_argument('--system_prompt', required=True, help='Path to the system prompt file')
    parser.add_argument('--model', default="gpt-4o", help='OpenAI model to use')

    args = parser.parse_args()
    try:
        create_assistant(args.system_prompt, args.model)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise
