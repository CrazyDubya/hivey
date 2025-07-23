"""
Asynchronous LLM client implementations for improved performance.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/api"
XAI_API_BASE_URL = "https://api.x.ai/v1"


async def call_ollama_chat_async(
    model_name: str,
    messages: List[Dict[str, Any]],
    stream: bool = False,
    options: Dict[str, Any] = None,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Async version of Ollama chat completion.

    Args:
        model_name: The name of the Ollama model to use
        messages: List of message objects
        stream: Whether to stream the response
        options: Additional model parameters
        timeout: Request timeout in seconds

    Returns:
        JSON response from the API or None if error occurs
    """
    payload = {"model": model_name, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options

    logger.info(
        f"Async request to Ollama model: {model_name} with "
        f"{len(messages)} messages"
    )

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/chat", json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                logger.info(
                    f"Ollama async call successful for model: {model_name}"
                )
                return result

    except aiohttp.ClientError as e:
        logger.error(
            f"Async HTTP error calling Ollama for model {model_name}: {e}"
        )
        return None
    except asyncio.TimeoutError:
        logger.error(f"Async timeout calling Ollama for model {model_name}")
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Async JSON decode error for Ollama model {model_name}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected async error calling Ollama model {model_name}: {e}"
        )
        return None


async def call_xai_chat_async(
    model_name: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    timeout: int = 60,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Async version of X.AI chat completion.

    Args:
        model_name: The X.AI model name
        messages: List of message objects
        api_key: X.AI API key
        timeout: Request timeout in seconds
        **kwargs: Additional parameters

    Returns:
        JSON response from the API or None if error occurs
    """
    import os

    if not api_key:
        api_key = os.getenv("XAI_API_KEY")

    if not api_key:
        logger.error("XAI_API_KEY not found for async call")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {"model": model_name, "messages": messages, **kwargs}

    logger.info(f"Async request to X.AI model: {model_name}")

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(
                f"{XAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                logger.info(
                    f"X.AI async call successful for model: {model_name}"
                )
                return result

    except aiohttp.ClientError as e:
        logger.error(
            f"Async HTTP error calling X.AI for model {model_name}: {e}"
        )
        return None
    except asyncio.TimeoutError:
        logger.error(f"Async timeout calling X.AI for model {model_name}")
        return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Async JSON decode error for X.AI model {model_name}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected async error calling X.AI model {model_name}: {e}"
        )
        return None


async def batch_llm_calls(
    calls: List[Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    """
    Execute multiple LLM calls concurrently for improved performance.

    Args:
        calls: List of call specifications, each containing:
            - provider: "ollama" or "xai"
            - model_name: Model identifier
            - messages: Message list
            - options: Additional parameters (optional)

    Returns:
        List of results in the same order as input calls
    """
    tasks = []

    for call in calls:
        provider = call.get("provider")
        model_name = call.get("model_name")
        messages = call.get("messages")
        options = call.get("options", {})

        if provider == "ollama":
            task = call_ollama_chat_async(
                model_name, messages, options=options
            )
        elif provider == "xai":
            task = call_xai_chat_async(model_name, messages, **options)
        else:
            logger.warning(f"Unknown provider: {provider}")
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task
            continue

        tasks.append(asyncio.create_task(task))

    logger.info(f"Executing {len(tasks)} concurrent LLM calls")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to None for consistency
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Batch call resulted in exception: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results
