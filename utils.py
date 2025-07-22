import logging
import os
import sqlite3

import numpy as np
from openai import OpenAI, OpenAIError

DB_NAME = "swarmmind.db"

# Configure logging
logger = logging.getLogger(__name__)


def configure_logging():
    """Configures basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Logging configured.")


def initialize_database():
    """Initializes the SQLite database and creates tables if they don't exist."""  # noqa: E501
    db_path = os.path.abspath(DB_NAME)
    conn = None
    try:
        logger.info(f"Initializing database at: {db_path}")
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Create experiences table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT,
                agent_name TEXT,
                content TEXT,
                confidence_score REAL,
                feedback TEXT,
                embedding TEXT,
                timestamp TEXT
            )
            """
        )
        logger.info("'experiences' table checked/created.")

        # Create indexes for performance optimization
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiences_agent_name ON experiences(agent_name)"  # noqa: E501
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON experiences(timestamp)"  # noqa: E501
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiences_task ON experiences(task)"  # noqa: E501
        )
        logger.info("Database indexes created/verified for experiences table.")

        # Create memories table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content_text TEXT NOT NULL,
                content_embedding TEXT,
                metadata_json TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        logger.info("'memories' table checked/created.")

        # Create indexes for memories table
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_agent_name ON memories(agent_name)"  # noqa: E501
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)"  # noqa: E501
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"  # noqa: E501
        )
        logger.info("Database indexes created/verified for memories table.")

        # Create tasks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL, -- e.g., queued, running, completed, failed  # noqa: E501
                result TEXT,          -- Store JSON or text result
                error_message TEXT,   -- Store error if status is 'failed'
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        logger.info("'tasks' table checked/created.")

        conn.commit()
        logger.info("Database initialized and tables created successfully.")

    except sqlite3.Error as e:
        logger.error(
            f"SQLite error during database initialization (path: {db_path}): {e}"  # noqa: E501
        )
        raise  # Re-raise the exception to signal failure
    finally:
        if conn:
            conn.close()
            logger.info(f"Database connection to {db_path} closed.")


# Initialize OpenAI client lazily
client = None


def get_openai_client():
    """Get or initialize the OpenAI client."""
    global client
    if client is None:
        try:
            # Ensure the API key is available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OPENAI_API_KEY environment variable not set.")
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set."
                )

            # Initialize client
            client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized successfully.")

        except OpenAIError as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            # Depending on the application's needs, you might want to exit or handle this differently  # noqa: E501
            raise
        except ValueError as e:
            logger.error(e)
            # Handle missing API key error appropriately
            raise
    return client


def get_embedding(text, model="text-embedding-3-small"):
    """Generates an embedding for the given text using the specified OpenAI model."""  # noqa: E501
    if not text or not isinstance(text, str):
        logger.error("Invalid text input for embedding.")
        return None
    try:
        text = text.replace("\n", " ")
        client = get_openai_client()
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        # logger.debug(f"Generated embedding for text snippet: {text[:50]}...")
        return embedding
    except OpenAIError as e:
        logger.error(f"OpenAI API error during embedding: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during embedding: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        logger.error("Cannot calculate similarity with None vector(s).")
        return 0.0

    # Ensure vectors are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.shape != vec2.shape:
        logger.error(
            f"Cannot calculate similarity between vectors of different shapes: {vec1.shape} vs {vec2.shape}"  # noqa: E501
        )
        return 0.0

    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        # logger.warning("Cannot calculate similarity with zero vector(s).")
        return 0.0

    similarity = dot_product / (norm_vec1 * norm_vec2)
    # logger.debug(f"Calculated cosine similarity: {similarity}")
    return similarity
