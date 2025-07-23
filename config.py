"""
Configuration module for Hivey project.
Centralizes all configuration settings and environment variables.
"""

import os
from typing import Optional

from dotenv import load_dotenv


class Config:
    """Centralized configuration class for the Hivey application."""

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        # Load environment variables from .env file if it exists
        load_dotenv()

    # API Configuration
    SWARM_API_KEY: Optional[str] = os.getenv("SWARM_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    XAI_API_KEY: Optional[str] = os.getenv("XAI_API_KEY")

    # Service Configuration
    HOST: str = os.getenv("HOST", "localhost")
    PORT: int = int(os.getenv("PORT", "8001"))
    SWARM_BASE_URL: str = os.getenv("SWARM_BASE_URL", f"http://{HOST}:{PORT}")

    # Database Configuration
    DB_NAME: str = os.getenv("DB_NAME", "swarmmind.db")

    # LLM Configuration
    DEFAULT_LLM_MODEL: str = os.getenv(
        "DEFAULT_LLM_MODEL", "xai/grok-3-latest"
    )
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "text-embedding-ada-002"
    )
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "xai/grok-3-latest")

    # External API Configuration
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434/api"
    )
    XAI_API_BASE_URL: str = os.getenv(
        "XAI_API_BASE_URL", "https://api.x.ai/v1"
    )

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "swarm_service.log")

    # Validation Configuration
    MAX_TASK_DESCRIPTION_LENGTH: int = int(
        os.getenv("MAX_TASK_DESCRIPTION_LENGTH", "10000")
    )
    MIN_TASK_DESCRIPTION_LENGTH: int = int(
        os.getenv("MIN_TASK_DESCRIPTION_LENGTH", "1")
    )

    @classmethod
    def validate_required_config(cls) -> None:
        """Validate that required configuration is present."""
        required_configs = {
            "SWARM_API_KEY": cls.SWARM_API_KEY,
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
        }

        missing_configs = [
            key for key, value in required_configs.items() if not value
        ]

        if missing_configs:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing_configs)}"
            )

    @classmethod
    def get_api_key_header_name(cls) -> str:
        """Get the standardized API key header name."""
        return "X-API-Key"


# Create a global configuration instance
config = Config()
