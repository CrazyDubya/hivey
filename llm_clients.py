import os
import json
import requests
from openai import OpenAI # Ensure this is OpenAI v1.x+
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Removed this line

OLLAMA_BASE_URL = "http://localhost:11434/api"
XAI_API_BASE_URL = "https://api.x.ai/v1"


def get_ollama_local_tags() -> Optional[List[Dict[str, Any]]]:
    """Queries Ollama API to list locally available models (tags)."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags")
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json().get("models", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama API to list tags: {e}")
        return None


def call_ollama_chat(
    model_name: str, messages: List[Dict[str, Any]], stream: bool = False, options: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Sends a chat completion request to the Ollama API.

    Args:
        model_name (str): The name of the Ollama model to use (e.g., 'long-gemma').
        messages (List[Dict[str, Any]]): A list of message objects, e.g., [{'role': 'user', 'content': 'Hello'}].
        stream (bool): Whether to stream the response. Defaults to False.
        options (Dict[str, Any]): Additional model parameters (e.g., temperature).

    Returns:
        Dict[str, Any]: The JSON response from the API, or None if an error occurs.
    """
    payload = {"model": model_name, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options

    logger.info(
        f"Sending request to Ollama model: {model_name} with {len(messages)} messages. Options: {options}"
    )
    response = None  # Initialize response to None
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/chat", json=payload, timeout=60.0) # Standard timeout 60s
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX responses

        # Attempt to parse JSON
        try:
            ollama_response = response.json()
        except json.JSONDecodeError as jd_err:
            logger.error(
                f"Ollama JSONDecodeError for model {model_name}: {jd_err}. Status: {response.status_code if response else 'N/A'}. Response text (first 500 chars): '{response.text[:500] if response else ''}'"
            )
            return {"error": f"JSON decode error: {jd_err}", "status_code": response.status_code if response else None, "content": response.text if response else None}

        # Check response structure
        if not isinstance(ollama_response, dict) or ollama_response.get("message") is None:
            logger.warning(
                f"Ollama response for model {model_name} missing 'message' field or is not a dict. Response: {str(ollama_response)[:500]}"
            )
            # Return the response as is, but add an error key for easier upstream detection
            if isinstance(ollama_response, dict):
                 ollama_response["error"] = "Unexpected response structure: 'message' field missing or invalid."
                 return ollama_response
            return {"error": "Unexpected response structure", "details": str(ollama_response)[:500]}

        # Specific check for message content being a dict (as expected by current consumers)
        if not isinstance(ollama_response.get("message"), dict):
            logger.warning(
                f"Ollama response 'message' field is not a dictionary for model {model_name}. Message: {str(ollama_response.get('message'))[:200]}"
            )
            # Modify the response to include an error, or return a new error dict
            ollama_response["error"] = "Unexpected 'message' field type: should be a dictionary."
            return ollama_response

        return ollama_response

    except requests.exceptions.Timeout as t_err:
        logger.error(f"Ollama request timed out for model {model_name}: {t_err}")
        return {"error": f"Request timed out: {t_err}"}
    except requests.exceptions.RequestException as req_err:
        status_code = response.status_code if response is not None else "N/A"
        content_preview = response.content[:500].decode('utf-8', errors='replace') if response is not None and response.content else "No content"
        logger.error(
            f"Ollama RequestException for model {model_name}: {req_err}. Status: {status_code}. Response content (first 500 bytes): '{content_preview}'"
        )
        return {"error": f"RequestException: {req_err}", "status_code": status_code, "content_preview": content_preview}
    except Exception as e: # Catch-all for any other unexpected errors
        logger.error(f"Unexpected error in call_ollama_chat for model {model_name}: {e}", exc_info=True)
        return {"error": f"Unexpected error: {str(e)}"}


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
        api_key (Optional[str], optional): X.AI API key. Defaults to os.getenv("XAI_API_KEY").
        base_url (str, optional): The base URL for the X.AI API. Defaults to XAI_API_BASE_URL.
        temperature (Optional[float], optional): Sampling temperature. Defaults to None.
        max_tokens (Optional[int], optional): Maximum number of tokens to generate. Defaults to None.

    Returns:
        Dict[str, Any]: The API response object (JSON parsed), or None if an error occurs.
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
    # X.AI docs mention 'stream_options': {'include_usage': True} for usage data with streaming
    # For non-streaming, usage might be included by default or not available this way.
    # We are not streaming here for simplicity in this function.

    logger.info(
        f"Sending request to X.AI model: {model_name} with {len(messages)} messages. Temp: {temperature}, MaxTokens: {max_tokens}."
    )
    request_url = f"{base_url}/chat/completions"
    response = None # Initialize response to None

    try:
        response = requests.post(request_url, headers=headers, json=payload, timeout=60.0) # Standard timeout 60s
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        try:
            xai_response = response.json()
        except json.JSONDecodeError as jd_err:
            logger.error(
                f"X.AI JSONDecodeError for model {model_name} at {request_url}: {jd_err}. Status: {response.status_code if response else 'N/A'}. Response text (first 500 chars): '{response.text[:500] if response else ''}'"
            )
            return {"error": f"JSON decode error: {jd_err}", "status_code": response.status_code if response else None, "content": response.text if response else None}

        # Check response structure - X.AI specific
        if not isinstance(xai_response, dict) or not xai_response.get("choices"):
            logger.warning(
                f"X.AI response for model {model_name} missing 'choices' field or is not a dict. Response: {str(xai_response)[:500]}"
            )
            if isinstance(xai_response, dict):
                xai_response["error"] = "Unexpected response structure: 'choices' field missing or invalid."
                return xai_response
            return {"error": "Unexpected response structure", "details": str(xai_response)[:500]}

        if not isinstance(xai_response["choices"], list) or len(xai_response["choices"]) == 0:
            logger.warning(
                f"X.AI response 'choices' field is not a list or is empty for model {model_name}. Response: {str(xai_response)[:500]}"
            )
            xai_response["error"] = "Unexpected response structure: 'choices' field is not a list or is empty."
            return xai_response

        first_choice = xai_response["choices"][0]
        if not isinstance(first_choice, dict) or first_choice.get("message") is None:
            logger.warning(
                f"X.AI response first choice missing 'message' field for model {model_name}. Choice: {str(first_choice)[:200]}"
            )
            xai_response["error"] = "Unexpected response structure: first choice missing 'message' field."
            return xai_response

        message_content = first_choice["message"].get("content")
        if message_content is None: # Allow empty string, but not None
            logger.warning(
                f"X.AI response message missing 'content' for model {model_name}. Message: {str(first_choice['message'])[:200]}"
            )
            xai_response["error"] = "Unexpected response structure: message missing 'content'."
            return xai_response

        return xai_response

    except requests.exceptions.Timeout as t_err:
        logger.error(f"X.AI request timed out for model {model_name} at {request_url}: {t_err}")
        return {"error": f"Request timed out: {t_err}"}
    except requests.exceptions.RequestException as req_err:
        status_code = response.status_code if response is not None else "N/A"
        # API Key is not in payload or response.content, so logging snippet is okay.
        content_preview = response.content[:500].decode('utf-8', errors='replace') if response is not None and response.content else "No content"
        logger.error(
            f"X.AI RequestException for model {model_name} at {request_url}: {req_err}. Status: {status_code}. Response (first 500 bytes): '{content_preview}'"
        )
        return {"error": f"RequestException: {req_err}", "status_code": status_code, "content_preview": content_preview}
    except Exception as e: # Catch-all for any other unexpected errors
        logger.error(f"Unexpected error in call_xai_chat for model {model_name}: {e}", exc_info=True)
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    # Example usage (requires Ollama server running with 'long-gemma' and XAI_API_KEY set)
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
