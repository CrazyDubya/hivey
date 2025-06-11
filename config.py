# Centralized configuration for the SwarmMind project

# --- Database Configuration ---
DB_NAME = "swarmmind.db"

# --- Model Configuration ---
# Consolidated to use text-embedding-3-small as it's more recent and was used in utils.py
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "xai/grok-3-latest"
CHAT_MODEL = "xai/grok-3-latest" # Centralized as a core config

# Add other shared constants here as the project grows.
