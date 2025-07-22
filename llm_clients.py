import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Removed this line

OLLAMA_BASE_URL = "http://localhost:11434/api"
XAI_API_BASE_URL = "https://api.x.ai/v1"


def get_ollama_local_tags() -> Optional[List[Dict[str, Any]]]:
    """Queries Ollama API to list locally available models (tags)."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags")
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)  # noqa: E501
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama API to list tags: {e}")
        return None


def call_ollama_chat(
    model_name: str,
    messages: List[Dict[str, Any]],
    stream: bool = False,
    options: Dict[str, Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Sends a chat completion request to the Ollama API.

    Args:
        model_name (str): The name of the Ollama model to use (e.g.,
            'long-gemma').
        messages (List[Dict[str, Any]]): A list of message objects, e.g.,
            [{'role': 'user', 'content': 'Hello'}].
        stream (bool): Whether to stream the response. Defaults to False.
        options (Dict[str, Any]): Additional model parameters (e.g.,
            temperature).

    Returns:
        Dict[str, Any]: The JSON response from the API, or None if an error
            occurs.
    """
    payload = {"model": model_name, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options

    logger.info(
        f"Sending request to Ollama model: {model_name} with "
        f"{len(messages)} messages."
    )
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/chat", json=payload, timeout=60.0
        )
        response.raise_for_status()
        # If stream is False, Ollama returns a single JSON object
        # If stream is True, it returns a stream of JSON objects separated by newlines  # noqa: E501
        # For simplicity with stream=False, we parse directly.
        # For streaming, this would need to be handled differently (iter_lines).  # noqa: E501
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Error calling Ollama chat API for model {model_name}: {e}"
        )
        logger.error(
            f"Response content: {response.content if 'response' in locals() else 'No response'}"  # noqa: E501
        )
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON response from Ollama for model {model_name}: {e}"  # noqa: E501
        )
        logger.error(
            f"Response content: {response.content if 'response' in locals() else 'No response object'}"  # noqa: E501
        )
        return None


def call_xai_chat(
    model_name: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: str = XAI_API_BASE_URL,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Sends a chat completion request to the X.AI API using requests.

    Args:
        model_name (str): The name of the X.AI model (e.g., 'grok-3-latest').
        messages (List[Dict[str, Any]]): A list of message objects.
        api_key (Optional[str], optional): X.AI API key. Defaults to os.getenv("XAI_API_KEY").  # noqa: E501
        base_url (str, optional): The base URL for the X.AI API. Defaults to XAI_API_BASE_URL.  # noqa: E501
        temperature (Optional[float], optional): Sampling temperature. Defaults to None.  # noqa: E501
        max_tokens (Optional[int], optional): Maximum number of tokens to generate. Defaults to None.  # noqa: E501

    Returns:
        Dict[str, Any]: The API response object (JSON parsed), or None if an error occurs.  # noqa: E501
    """
    if not api_key:
        api_key = os.getenv("XAI_API_KEY")

    if not api_key:
        logger.error("XAI_API_KEY not found in environment variables.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {"model": model_name, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    # X.AI docs mention 'stream_options': {'include_usage': True} for usage data with streaming  # noqa: E501
    # For non-streaming, usage might be included by default or not available this way.  # noqa: E501
    # We are not streaming here for simplicity in this function.

    logger.info(
        f"Sending request to X.AI model: {model_name} with {len(messages)} messages via requests."  # noqa: E501
    )
    request_url = f"{base_url}/chat/completions"

    try:
        response = requests.post(
            request_url, headers=headers, json=payload, timeout=60.0
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)  # noqa: E501
        return response.json()  # Parse JSON response
    except requests.exceptions.Timeout:
        logger.error(
            f"Timeout error calling X.AI API for model {model_name} at {request_url}"  # noqa: E501
        )
        return None
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Error calling X.AI API for model {model_name} at {request_url}: {e}"  # noqa: E501
        )
        if "response" in locals() and response is not None:
            logger.error(f"Response content: {response.content}")
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON response from X.AI for model {model_name} at {request_url}: {e}"  # noqa: E501
        )
        if "response" in locals() and response is not None:
            logger.error(f"Response content: {response.content}")
        return None


if __name__ == "__main__":
    # Example usage (requires Ollama server running with 'long-gemma' and XAI_API_KEY set)  # noqa: E501
    print("Listing local Ollama models:")
    local_models = get_ollama_local_tags()
    if local_models:
        print(json.dumps(local_models, indent=2))
        custom_model_exists = any(
            m.get("name") and "long-gemma" in m["name"] for m in local_models
        )
        print(f"Custom model 'long-gemma' found: {custom_model_exists}")

    print("\n--- Testing Ollama long-gemma ---")
    ollama_messages = [
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms.",
        }
    ]
    # Assuming 'long-gemma:latest' or just 'long-gemma' is the tag in Ollama
    ollama_response = call_ollama_chat("long-gemma", ollama_messages)
    if ollama_response and ollama_response.get("message"):
        print(f"Ollama response: {ollama_response['message']['content']}")
    else:
        print(
            f"Ollama call failed or returned unexpected structure: {ollama_response}"  # noqa: E501
        )

    print("\n--- Testing X.AI grok-3-latest ---")
    xai_messages = [
        {
            "role": "user",
            "content": "What is the future of AI according to Grok?",
        }
    ]
    xai_response = call_xai_chat("grok-3-latest", xai_messages)
    if (
        xai_response
        and xai_response.get("choices")
        and xai_response["choices"]
    ):
        print(
            f"X.AI response: {xai_response['choices'][0]['message']['content']}"  # noqa: E501
        )
    else:
        print(
            f"X.AI call failed or returned unexpected structure: {xai_response}"  # noqa: E501
        )
