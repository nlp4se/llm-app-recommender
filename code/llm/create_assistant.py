from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Optional

class CreateAssistant(ABC):
    def __init__(self, model: str):
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_system_prompt(self, system_prompt_path: str) -> str:
        """Load system prompt from a text file."""
        self.logger.info(f"Loading system prompt from {system_prompt_path}")
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.logger.info("System prompt loaded successfully")
                return content
        except Exception as e:
            self.logger.error(f"Error loading system prompt: {str(e)}")
            raise

    def save_assistant_id(self, assistant_id: str):
        """Save the assistant ID to a file for reuse."""
        filename = f"data/assistants/assistant_id_{self.model}.txt"
        self.logger.info(f"Saving assistant ID to {filename}")
        try:
            with open(filename, 'w') as file:
                file.write(assistant_id)
                self.logger.info("Assistant ID saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving assistant ID: {str(e)}")
            raise

    @abstractmethod
    def create_assistant(self, guidelines_file: str) -> Any:
        """Create an assistant with the specified guidelines."""
        pass 