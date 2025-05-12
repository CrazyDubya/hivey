import os
import json
import requests
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OLLAMA_BASE_URL = "http://localhost:11434/api"
XAI_API_BASE_URL = "https://api.x.ai/v1"

def get_ollama_local_tags():
    """Queries Ollama API to list locally available models (tags)."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags")
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama API to list tags: {e}")
        return None

def call_ollama_chat(model_name: str, messages: list, stream: bool = False, options: dict = None):
    """
    Sends a chat completion request to the Ollama API.

    Args:
        model_name (str): The name of the Ollama model to use (e.g., 'long-gemma').
        messages (list): A list of message objects, e.g., [{'role': 'user', 'content': 'Hello'}].
        stream (bool): Whether to stream the response. Defaults to False.
        options (dict): Additional model parameters (e.g., temperature).

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": stream
    }
    if options:
        payload["options"] = options

    logger.info(f"Sending request to Ollama model: {model_name} with {len(messages)} messages.")
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/chat", json=payload)
        response.raise_for_status()
        # If stream is False, Ollama returns a single JSON object
        # If stream is True, it returns a stream of JSON objects separated by newlines
        # For simplicity with stream=False, we parse directly.
        # For streaming, this would need to be handled differently (iter_lines).
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Ollama chat API for model {model_name}: {e}")
        logger.error(f"Response content: {response.content if 'response' in locals() else 'No response object'}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from Ollama for model {model_name}: {e}")
        logger.error(f"Response content: {response.content if 'response' in locals() else 'No response object'}")
        return None

def call_xai_chat(model_name: str, messages: list, api_key: str = None, base_url: str = XAI_API_BASE_URL, temperature: float = None, max_tokens: int = None):
    """
    Sends a chat completion request to the X.AI API (OpenAI compatible).

    Args:
        model_name (str): The name of the X.AI model (e.g., 'grok-3-latest').
        messages (list): A list of message objects.
        api_key (str, optional): X.AI API key. Defaults to os.getenv("XAI_API_KEY").
        base_url (str, optional): The base URL for the X.AI API. Defaults to XAI_API_BASE_URL.
        temperature (float, optional): Sampling temperature. Defaults to None.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to None.

    Returns:
        dict: The API response object, or None if an error occurs.
    """
    if not api_key:
        api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        logger.error("XAI_API_KEY not found in environment variables.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    logger.info(f"Sending request to X.AI model: {model_name} with {len(messages)} messages.")
    try:
        params = {
            "model": model_name,
            "messages": messages
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
            
        completion = client.chat.completions.create(**params)
        # The response object from openai v1.x.x is not a direct dict, 
        # but can be easily converted or accessed.
        # For consistency, let's aim to return a serializable structure if possible, e.g. model_dump()
        return completion.model_dump() 
    except Exception as e:
        logger.error(f"Error calling X.AI API for model {model_name}: {e}")
        return None

if __name__ == '__main__':
    # Example usage (requires Ollama server running with 'long-gemma' and XAI_API_KEY set)
    print("Listing local Ollama models:")
    local_models = get_ollama_local_tags()
    if local_models:
        print(json.dumps(local_models, indent=2))
        custom_model_exists = any(m.get('name') and 'long-gemma' in m['name'] for m in local_models)
        print(f"Custom model 'long-gemma' found: {custom_model_exists}")

    print("\n--- Testing Ollama long-gemma ---")
    ollama_messages = [
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    # Assuming 'long-gemma:latest' or just 'long-gemma' is the tag in Ollama
    ollama_response = call_ollama_chat("long-gemma", ollama_messages)
    if ollama_response and ollama_response.get("message"):
        print(f"Ollama response: {ollama_response['message']['content']}")
    else:
        print(f"Ollama call failed or returned unexpected structure: {ollama_response}")

    print("\n--- Testing X.AI grok-3-latest ---")
    xai_messages = [
        {"role": "user", "content": "What is the future of AI according to Grok?"}
    ]
    xai_response = call_xai_chat("grok-3-latest", xai_messages)
    if xai_response and xai_response.get("choices") and xai_response["choices"]:
        print(f"X.AI response: {xai_response['choices'][0]['message']['content']}")
    else:
        print(f"X.AI call failed or returned unexpected structure: {xai_response}")
