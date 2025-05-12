# Utility functions for the Hive Mind project

import logging
import json
import os
import openai
import numpy as np
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (if not already loaded by main script, good to have here too)
load_dotenv()

# Constants for embeddings (similar to swarms.py)
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize OpenAI client
# This assumes OPENAI_API_KEY is set in the environment
# If it might not be, add error handling or a check
if not openai.api_key:
    openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Configure logging (example, can be expanded)
def configure_logging(level=logging.INFO, log_file=None):
    logging.basicConfig(level=level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    return logging.getLogger(__name__) # Return a logger instance for the calling module

# Experience loading/saving (as mentioned in the original world_builder.py)
EXPERIENCES_FILE = "data/experiences.json"

def save_experiences(experiences, filepath=EXPERIENCES_FILE):
    try:
        with open(filepath, 'w') as f:
            json.dump(experiences, f, indent=4)
        logging.info(f"Experiences saved to {filepath}")
    except IOError as e:
        logging.error(f"Error saving experiences to {filepath}: {e}")

def load_experiences(filepath=EXPERIENCES_FILE):
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            logging.info(f"No experiences file found at {filepath}, starting fresh.")
            return []
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error loading experiences from {filepath}: {e}")
        return []

# Placeholder for database initialization (if using SQLite later from plan)
def initialize_database():
    # This would set up SQLite tables if used.
    # For now, ensure data directory exists for experiences.json
    data_dir = os.path.dirname(EXPERIENCES_FILE)
    try:
        os.makedirs(data_dir, exist_ok=True) # Ensure directory is created, exist_ok=True prevents error if it exists
        logging.info(f"Ensured data directory exists: {data_dir}")
    except OSError as e:
        logging.error(f"Error creating data directory {data_dir}: {e}")
    pass

# Embedding and Similarity Utilities

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embedding for text using OpenAI API."""
    # Get a logger instance, possibly the root logger if configure_logging wasn't called from here
    logger = logging.getLogger(__name__) 
    try:
        response = client.embeddings.create(
            input=text.replace("\n", " "),
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Return zero vector matching the expected dimension for ada-002
        # Ensure this dimension is correct for the model used.
        # For "text-embedding-ada-002", it's 1536.
        return [0.0] * 1536

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not a or not b or len(a) == 0 or len(b) == 0 or len(a) != len(b):
        # Ensure vectors are not empty and have the same dimension
        return 0.0
    
    # Convert to numpy arrays for efficient calculation
    vec_a = np.array(a)
    vec_b = np.array(b)
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        # Avoid division by zero if one of the vectors is a zero vector
        return 0.0
        
    return dot_product / (norm_a * norm_b)
