import openai
import sqlite3
import logging
import os
import json
import re
import datetime
import inspect
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,  # General type hint
)

from dotenv import load_dotenv

from utils import get_embedding, cosine_similarity, client as openai_client
from llm_clients import call_ollama_chat, call_xai_chat

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger("SwarmMind")

# Configure OpenAI API (This global openai.api_key might be redundant if client from utils is used consistently)
# It's good practice to ensure it's set for any direct openai legacy calls if they exist elsewhere, but new calls should use the client.
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    logger.warning("OPENAI_API_KEY not found in environment variables.")

# Removed redundant logging.basicConfig - will be handled by utils.configure_logging()
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("SwarmMind") # Original position

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "xai/grok-3-latest"
DB_NAME = "swarmmind.db"
DEFAULT_LLM_MODEL = "xai/grok-3-latest"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Result:
    value: str
    context_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Function:
    name: str
    description: str
    handler: Callable


class KnowledgeBase:
    def __init__(self, connection: Optional[sqlite3.Connection] = None):
        self.shared_connection = connection is not None
        if self.shared_connection:
            self.conn = connection
        else:
            db_path = os.path.abspath(DB_NAME)
            logger.warning(
                f"KnowledgeBase creating NEW non-shared DB connection for path: {db_path}"
            )
            try:
                self.conn = sqlite3.connect(
                    DB_NAME
                )  # Tables are now initialized by utils.initialize_database()
            except sqlite3.Error as e:
                logger.error(
                    f"SQLite error creating connection in KB __init__ (path: {db_path}): {e}"
                )
                raise

        if not self.conn:
            raise Exception("Failed to establish DB connection in KnowledgeBase")

        # self.cursor = self.conn.cursor() # Removed: Cursors should be method-scoped
        # Table creation logic moved to utils.initialize_database()
        # try:
        #     # self.cursor.execute( # Example of old usage
        #         """
        #         CREATE TABLE IF NOT EXISTS experiences (
        #             id INTEGER PRIMARY KEY AUTOINCREMENT,
        #             task TEXT,
        #             agent_name TEXT,
        #             content TEXT,
        #             confidence_score REAL,
        #             feedback TEXT,
        #             embedding TEXT,
        #             timestamp TEXT
        #         )
        #     """
        #     )

        #     self.cursor.execute(
        #         """
        #         CREATE TABLE IF NOT EXISTS memories (
        #             id INTEGER PRIMARY KEY AUTOINCREMENT,
        #             agent_name TEXT NOT NULL,
        #             memory_type TEXT NOT NULL,
        #             content_text TEXT NOT NULL,
        #             content_embedding TEXT,
        #             metadata_json TEXT,
        #             timestamp TEXT NOT NULL
        #         )
        #     """
        #     )

        #     # --- Add Tasks Table ---
        #     self.cursor.execute(
        #         """
        #         CREATE TABLE IF NOT EXISTS tasks (
        #             task_id TEXT PRIMARY KEY,
        #             description TEXT NOT NULL,
        #             status TEXT NOT NULL, -- e.g., queued, running, completed, failed
        #             result TEXT,          -- Store JSON or text result
        #             error_message TEXT,   -- Store error if status is 'failed'
        #             created_at TEXT NOT NULL,
        #             updated_at TEXT NOT NULL,
        #             parent_task_id TEXT,          -- Added for subtask relationship
        #             is_subtask BOOLEAN DEFAULT 0  -- Added to identify subtasks
        #         )
        #     """
        #     )
        #     # Add index for parent_task_id for faster querying
        #     self.cursor.execute(
        #         "CREATE INDEX IF NOT EXISTS idx_parent_task_id ON tasks(parent_task_id)"
        #     )
        #     self.conn.commit()

        # except sqlite3.Error as e:
        #     logger.error(f"SQLite error during KB table init/commit: {e}")
        #     if not self.shared_connection and hasattr(self, "conn") and self.conn:
        #         self.conn.close()
        #     raise

    def add_memory(
        self,
        agent_name: str,
        memory_type: str,
        content_text: str,
        content_embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        embedding_json = json.dumps(content_embedding) if content_embedding else None
        metadata_json_str = json.dumps(metadata) if metadata else None
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO memories (agent_name, memory_type, content_text, content_embedding, metadata_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    agent_name,
                    memory_type,
                    content_text,
                    embedding_json,
                    metadata_json_str,
                    datetime.datetime.now().isoformat(),
                ),
            )
            if self.conn:
                self.conn.commit()
            logger.info(f"KB: Added memory for agent '{agent_name}', type '{memory_type}'. Content preview: '{content_text[:30]}...'. Metadata: {metadata is not None}")
        except sqlite3.Error as e:
            logger.error(f"KB: SQLite error in add_memory for agent '{agent_name}', type '{memory_type}': {e}", exc_info=True)
            raise

    def save_experience(
        self,
        task: str,
        agent_name: str,
        content: str,
        confidence_score: float,
        feedback: str,
    ) -> None:
        content_embedding = get_embedding(content)
        embedding_json = json.dumps(content_embedding)
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO experiences (task, agent_name, content, confidence_score, feedback, timestamp, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task,
                    agent_name,
                    content,
                    confidence_score,
                    feedback,
                    datetime.datetime.now().isoformat(),
                    embedding_json,
                ),
            )
            if self.conn:
                self.conn.commit()
            logger.info(f"KB: Saved experience for agent '{agent_name}', task '{task[:30]}...'. Content preview: '{content[:50]}...'. Score: {confidence_score:.2f}")
        except sqlite3.Error as e:
            logger.error(f"KB: SQLite error in save_experience for agent '{agent_name}', task '{task[:30]}...': {e}", exc_info=True)
            raise

    def query_experiences_by_similarity(
        self,
        task_embedding: List[float],
        agent_name_filter: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        logger.info(f"KB: Querying experiences by similarity. Agent filter: '{agent_name_filter}', Limit: {limit}. Embedding provided: {task_embedding is not None and len(task_embedding) > 0}")
        if not task_embedding:
            logger.warning(
                "KB: Task embedding is empty for query_experiences_by_similarity. Cannot perform semantic search."
            )
            return []

        query = "SELECT task, agent_name, content, confidence_score, feedback, embedding, timestamp FROM experiences"
        params = []
        if agent_name_filter:
            query += " WHERE agent_name = ?"
            params.append(agent_name_filter)
            logger.debug(f"KB: Applying agent_name_filter: {agent_name_filter}")

        cursor = self.conn.cursor() # Get a local cursor
        try:
            cursor.execute(query, tuple(params))
            all_experiences = cursor.fetchall()
            logger.debug(f"KB: Fetched {len(all_experiences)} experiences from DB before similarity calculation.")
        except sqlite3.Error as e:
            logger.error(
                f"KB: SQLite error during query_experiences_by_similarity fetch (agent: {agent_name_filter}): {e}", exc_info=True
            )
            return []

        experiences_with_similarity = []
        for (
            exp_task,
            exp_agent,
            exp_content,
            exp_score,
            exp_feedback,
            exp_embedding_json,
            exp_timestamp,
        ) in all_experiences:
            if not exp_embedding_json:
                continue
            try:
                exp_embedding = json.loads(exp_embedding_json)
                if (
                    not exp_embedding
                ):  # Handle case where embedding string is 'null' or empty list
                    continue
                similarity = cosine_similarity(task_embedding, exp_embedding)
                experiences_with_similarity.append(
                    {
                        "task": exp_task,
                        "agent_name": exp_agent,
                        "content": exp_content,
                        "confidence_score": exp_score,
                        "feedback": exp_feedback,
                        "timestamp": exp_timestamp,
                        "similarity": similarity,  # Store for sorting
                    }
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to decode embedding JSON for experience: {exp_content[:50]}..."
                )
            except Exception as e:  # Catch other errors during similarity calculation
                logger.error(
                    f"Error calculating similarity for experience '{exp_content[:50]}...': {e}"
                )

        # Sort by similarity in descending order
        experiences_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)

        top_n_experiences = experiences_with_similarity[:limit]
        logger.info(f"KB: Returning {len(top_n_experiences)} experiences after similarity scoring and limit. Agent filter: '{agent_name_filter}'.")
        return top_n_experiences

    def semantic_search(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Performs a semantic search for memories based on query embedding."""
        logger.info(f"KB: Semantic search called. Limit: {limit}. Embedding provided: {query_embedding is not None and len(query_embedding) > 0}. Currently a stub.")
        # This is a placeholder. Actual implementation will query 'memories' table.
        # Example structure it might return:
        # return [
        #     {
        #         "id": 1,
        #         "agent_name": "some_agent",
        #         "memory_type": "observation",
        #         "content_text": "Found relevant info...",
        #         "similarity": 0.85 # Calculated similarity score
        #     }
        # ]
        return []

    def close(self) -> None:
        if not self.shared_connection and hasattr(self, "conn") and self.conn:
            self.conn.close()
            self.conn = None


class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        agent_type: str,
        llm_model_identifier: str,
        db_connection: Optional[sqlite3.Connection],
    ):
        self.name = name
        self.instructions = instructions
        self.agent_type = agent_type
        self.llm_model_identifier = llm_model_identifier
        self.knowledge_base = KnowledgeBase(connection=db_connection)
        self.task_history: List[Dict[str, Any]] = []
        self.creation_time = datetime.datetime.now().isoformat()

        # LLM Provider Handlers Dispatch Table
        self.LLM_PROVIDER_HANDLERS: Dict[str, Callable] = {
            "ollama": self._ollama_llm_handler,
            "xai": self._xai_llm_handler,
            "openai": self._openai_llm_handler,
            # Add other providers here, e.g., "anthropic": self._anthropic_llm_handler
        }

    def _ollama_llm_handler(self, model_name: str, messages: List[Dict[str, Any]], model_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handles LLM calls to Ollama models via llm_clients.py."""
        logger.debug(f"Agent {self.name} using Ollama handler for model {model_name}.")
        raw_response = call_ollama_chat(
            model_name=model_name,
            messages=messages,
            options=model_params
        )
        if raw_response and raw_response.get("error"):
            logger.error(f"Ollama API error for agent {self.name}, model {model_name}: {raw_response.get('error')}")
            return {"error": str(raw_response.get("error"))}
        elif raw_response and raw_response.get("message") and isinstance(raw_response["message"], dict) and "content" in raw_response["message"]:
            return {"content": raw_response["message"].get("content")}
        else:
            logger.warning(f"Unexpected Ollama response structure for agent {self.name}, model {model_name}: {str(raw_response)[:200]}")
            return {"error": "Unexpected response structure from Ollama."}

    def _xai_llm_handler(self, model_name: str, messages: List[Dict[str, Any]], model_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handles LLM calls to X.AI models via llm_clients.py."""
        logger.debug(f"Agent {self.name} using X.AI handler for model {model_name}.")
        # Pass relevant model_params if call_xai_chat supports them (e.g., temperature, max_tokens)
        # For now, call_xai_chat takes temperature and max_tokens directly.
        # We'll assume model_params might contain these.
        temperature = model_params.get("temperature") if model_params else None
        max_tokens = model_params.get("max_tokens") if model_params else None

        raw_response = call_xai_chat(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if raw_response and raw_response.get("error"):
            logger.error(f"X.AI API error for agent {self.name}, model {model_name}: {raw_response.get('error')}")
            return {"error": str(raw_response.get("error"))}
        elif raw_response and raw_response.get("choices") and isinstance(raw_response["choices"], list) and len(raw_response["choices"]) > 0:
            first_choice = raw_response["choices"][0]
            if isinstance(first_choice, dict) and first_choice.get("message") and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]:
                return {"content": first_choice["message"].get("content")}

        logger.warning(f"Unexpected X.AI response structure for agent {self.name}, model {model_name}: {str(raw_response)[:200]}")
        return {"error": "Unexpected response structure from X.AI."}

    def _openai_llm_handler(self, model_name: str, messages: List[Dict[str, Any]], model_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handles LLM calls to OpenAI models."""
        logger.debug(f"Agent {self.name} using OpenAI handler for model {model_name}.")
        try:
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=model_params.get("temperature", 0.7) if model_params else 0.7,
                max_tokens=model_params.get("max_tokens", 1024) if model_params else 1024,
            ) # type: ignore[arg-type]
            if completion.choices and completion.choices[0].message and isinstance(completion.choices[0].message.content, str):
                return {"content": completion.choices[0].message.content}
            else:
                logger.warning(f"Unexpected OpenAI response structure for agent {self.name}, model {model_name}: No content in choices.")
                return {"error": "Unexpected response structure from OpenAI: No content."}
        except openai.APIError as e:
            logger.error(f"OpenAI API Error for agent {self.name}, model {model_name}: {str(e)}")
            return {"error": f"OpenAI API Error: {str(e)}"}
        except Exception as e:
            logger.error(f"General error in OpenAI handler for agent {self.name}, model {model_name}: {str(e)}", exc_info=True)
            return {"error": f"An unexpected error occurred with OpenAI model: {str(e)}"}

    def remember(self, info: Dict[str, Any], long_term: bool = False) -> None:
        info["timestamp"] = datetime.datetime.now().isoformat()

        if long_term:
            self.knowledge_base.save_experience(
                task=info.get("task", ""),
                agent_name=self.name,
                content=info.get("content", ""),
                confidence_score=info.get("confidence_score", 0.0),
                feedback=info.get("feedback", ""),
            )
        else:
            self.task_history.append(info)
            if len(self.task_history) > 10:
                self.task_history.pop(0)

    def recall(
        self, query: Optional[str] = None, long_term: bool = False, limit: int = 5
    ) -> List[Dict[str, Any]]:
        if long_term:
            if not query:
                logger.warning("Long-term recall attempted without a query.")
                # Decide what to return: empty list, or perhaps all memories up to limit?
                # For now, let's assume if no query, it implies no specific search, maybe return generic memories?
                # This behavior might need refinement based on desired functionality.
                # Returning empty for now if no query for long_term.
                return []
            query_embedding = get_embedding(query)
            if query_embedding is None:
                logger.error(f"Could not generate embedding for query: {query}")
                return []
            # semantic_search expects query_embedding and limit.
            # Agent name filtering would need to be part of semantic_search implementation itself if desired.
            return self.knowledge_base.semantic_search(query_embedding, limit=limit)
        else:
            if not query or not self.task_history:
                return self.task_history[:limit]

            query_embedding = get_embedding(query)
            scores = []
            for i, entry in enumerate(self.task_history):
                content = entry.get("content", "")
                if not content:
                    scores.append(0)
                    continue

                entry_embedding = get_embedding(content)
                similarity = cosine_similarity(query_embedding, entry_embedding)
                scores.append(similarity)

            sorted_pairs = sorted(
                zip(scores, self.task_history), key=lambda x: x[0], reverse=True
            )
            return [item for _, item in sorted_pairs[:limit]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "instructions": self.instructions,
            "tier": self.agent_type,
            "functions": [],
            "creation_time": self.creation_time,
            "task_count": 0,
            "success_rate": 0.0,
            "last_active": None,
        }

    def _get_llm_response(
        self, task_description: str, model_params: Optional[Dict[str, Any]] = None
    ) -> str:
        model_params = model_params or {}
        system_prompt = (
            f"You are {self.name}, a specialized agent in the SwarmMind collective intelligence system.\n\n"
            f"Your instructions: {self.instructions}\n\n"
            f"Task: {task_description}\n\n"
            "Respond with clear, concise, and detailed information related to your specialization. "
            "Focus on producing high-quality output that will contribute to the collective task."
        )

        # Use specific types for OpenAI compatibility
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ]

        # Standardized variable for extracted text content
        extracted_text_content: str | None = None
        llm_response_data: Dict[str, Any] = {}

        # Parse provider and model_name from llm_model_identifier
        parts = self.llm_model_identifier.split("/", 1)
        if len(parts) == 2:
            provider_prefix, actual_model_name = parts[0].lower(), parts[1]
        else: # Default to openai if no prefix
            provider_prefix, actual_model_name = "openai", self.llm_model_identifier
            logger.warning(f"Agent {self.name}: LLM identifier '{self.llm_model_identifier}' has no clear provider prefix, defaulting to 'openai'.")

        logger.info(
            f"Agent {self.name} routing task to LLM provider: '{provider_prefix}' with model: '{actual_model_name}' for task: {task_description[:100]}..."
        )

        handler = self.LLM_PROVIDER_HANDLERS.get(provider_prefix)

        if handler:
            try:
                llm_response_data = handler(
                    model_name=actual_model_name,
                    messages=messages,
                    model_params=model_params
                )
            except Exception as e: # Catch errors from within the handler itself
                logger.error(f"Agent {self.name}: Exception during LLM handler execution for provider '{provider_prefix}', model '{actual_model_name}': {e}", exc_info=True)
                llm_response_data = {"error": f"Handler exception: {str(e)}"}
        else:
            logger.error(f"Agent {self.name}: No LLM handler found for provider prefix '{provider_prefix}'. Check LLM_PROVIDER_HANDLERS configuration.")
            llm_response_data = {"error": f"No handler for provider '{provider_prefix}'"}

        # Extract content or error from the standardized handler response
        if llm_response_data.get("error"):
            logger.error(
                f"LLM call failed for agent {self.name} using {provider_prefix}/{actual_model_name}: {llm_response_data['error']}"
            )
            # Ensure the error message is a string for the return value
            return str(llm_response_data["error"])

        extracted_text_content = llm_response_data.get("content")

        if isinstance(extracted_text_content, str):
            self.remember({"role": "assistant", "content": extracted_text_content})
            response_length = len(extracted_text_content)
            logger.info(
                f"Agent {self.name} received response from {provider_prefix}/{actual_model_name}. Length: {response_length}. Preview: {extracted_text_content[:100]}..."
            )
            final_content = extracted_text_content.strip()
            logger.info(f"LLM call successful for agent {self.name} using {provider_prefix}/{actual_model_name}. Response length: {len(final_content)}.")
            return final_content
        else:
            logger.error(
                f"LLM call failed for agent {self.name} using {provider_prefix}/{actual_model_name}: No string content received from handler. Handler response: {str(llm_response_data)[:200]}"
            )
            return "Error: No or invalid content received from LLM handler."

    def run(
        self, task_description: str, context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        logger.info(f"Agent {self.name} starting task: {task_description[:100]}...")
        self.task_history.append({"task": task_description})
        return self._get_llm_response(task_description)

    def _evaluate_output(
        self, task: str, output: str
    ) -> Dict[str, Any]:
        judge = self.swarm.meta_agents.get("JudgeAgent") if hasattr(self, 'swarm') else None
        if not judge:
            logger.warning("JudgeAgent not found, cannot evaluate output.")
            return {
                "confidence_score": 0.75,
                "feedback": "Evaluation not available (JudgeAgent not found)",
            }

        prompt = (
            f"Task given to {self.name}: {task}\n\n"
            f"Output from {self.name}:\n\n{output}\n\n"
            "Evaluate this output's quality, relevance, and coherence. "
            "Provide a confidence score between 0.001 (poor) and 0.999 (excellent) "
            "and specific feedback including strengths and areas for improvement. "
            "Format your response clearly, ensuring the confidence score is easily parsable (e.g., 'Confidence Score: 0.85')."
        )

        evaluation_text: str | None = None

        judge_model_identifier = judge.llm_model_identifier
        # Use specific types for OpenAI compatibility
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": judge.instructions},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info(
                f"JudgeAgent ({judge_model_identifier}) evaluating output from {self.name} for task: {task[:50]}..."
            )

            if judge_model_identifier.startswith("ollama/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                evaluation_text = call_ollama_chat(
                    model_name=model_name, messages=messages, options={"temperature": 0.3}
                ) 
                if (
                    evaluation_text
                    and evaluation_text.get("message")
                    and isinstance(evaluation_text["message"], dict)
                ):
                    evaluation_text = evaluation_text["message"].get("content")
                elif evaluation_text and evaluation_text.get("error"):
                    logger.error(
                        f"Ollama API error for JudgeAgent ({model_name}): {str(evaluation_text.get('error')) if isinstance(evaluation_text, dict) else 'Unknown error'}"
                    )
                else:
                    logger.warning(
                        f"Unexpected Ollama response structure for JudgeAgent ({model_name}): {str(evaluation_text)}"
                    )

            elif judge_model_identifier.startswith("xai/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                evaluation_text = call_xai_chat(
                    model_name=model_name, messages=messages, temperature=0.3
                ) 
                if (
                    evaluation_text
                    and evaluation_text.get("choices")
                    and evaluation_text["choices"]
                ):
                    message = evaluation_text["choices"][0].get("message")
                    if message and isinstance(message, dict):
                        evaluation_text = message.get("content")

            else:
                openai_model_name = judge_model_identifier
                if openai_model_name.startswith("openai/"):
                    openai_model_name = openai_model_name.split("/", 1)[1]

                completion = openai_client.chat.completions.create(
                    model=openai_model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500,
                )  # type: ignore[arg-type]
                if completion.choices and completion.choices[0].message and isinstance(completion.choices[0].message.content, str):
                    evaluation_text = completion.choices[0].message.content
                # else evaluation_text remains None, addressing old L587 error.

            if not evaluation_text: # evaluation_text is str | None here
                logger.error(
                    f"JudgeAgent ({judge_model_identifier}) received no content."
                )
                raise ValueError("No content received from LLM for evaluation.")

            confidence_score = 0.75
            feedback = evaluation_text

            if isinstance(evaluation_text, str):
                score_match = re.search(
                    r"(?:confidence\s*score|score):?\s*(\d(?:\.\d+)?)", evaluation_text, re.IGNORECASE
                ) 
            else:
                score_match = None
            if score_match:
                try:
                    confidence_score = float(score_match.group(1))
                    confidence_score = max(0.001, min(0.999, confidence_score)) 
                except ValueError:
                    logger.warning(
                        f"Could not parse confidence score from: {score_match.group(1) if score_match else 'N/A'}"
                    )
                    pass
            else:
                logger.warning(
                    f"No confidence score found in JudgeAgent output for agent {self.name}, task '{task[:50]}...'. Raw output: {evaluation_text[:100]}..." if isinstance(evaluation_text, str) else "No confidence score found."
                )

            logger.info(f"Evaluation completed for agent {self.name}, task '{task[:50]}...'. Score: {confidence_score:.3f}.")
            return {"confidence_score": confidence_score, "feedback": evaluation_text if isinstance(evaluation_text, str) else "Evaluation error occurred."}

        except Exception as e:
            logger.error(
                f"Error evaluating output with JudgeAgent ({judge_model_identifier}) for agent {self.name}, task '{task[:50]}...': {e}", exc_info=True
            )
            return {
                "confidence_score": 0.5, # Default score on error
                "feedback": f"Error in evaluation: {str(e)}",
            }


class Swarm:
    def __init__(self, db_connection: Optional[sqlite3.Connection]):
        """Initialize the swarm with a shared database connection and load agents"""
        self.db_connection = db_connection  # Store the shared connection
        self.agents: Dict[str, Agent] = {}
        self.supervisors: Dict[str, Agent] = {}
        self.meta_agents: Dict[str, Agent] = {}
        self.context_variables: Dict[str, Any] = {}
        self.global_memory: List[Dict[str, Any]] = []
        # self.cursor attribute is removed. Cursors will be method-scoped.
        self._init_db() # Ensures tables are created if they don't exist
        self.knowledge_base = KnowledgeBase(connection=self.db_connection) # KnowledgeBase manages its own cursors
        self._initialize_essential_agents() # Load/create initial agents

    def _init_db(self):
        if not self.db_connection:
            logger.error("Database connection is not initialized for Swarm._init_db")
            raise ConnectionError("Database connection not initialized")
        
        cursor = self.db_connection.cursor() # Use a local cursor
        
        # Ensure 'agents' table exists
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                name TEXT PRIMARY KEY,
                instructions TEXT,
                agent_type TEXT,
                llm_model_identifier TEXT,
                functions_json TEXT
            )
        """
        )
        # Ensure 'tasks' table exists with new fields
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL, -- e.g., queued, running, completed, failed
                result TEXT,          -- Store JSON or text result
                error_message TEXT,   -- Store error if status is 'failed'
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                parent_task_id TEXT,          -- Added for subtask relationship
                is_subtask BOOLEAN DEFAULT 0,  -- Added to identify subtasks
                assigned_agent_name TEXT      -- Added to specify which agent should run a subtask
            )
        """
        )

        # Schema migration for existing 'tasks' table
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [info[1] for info in cursor.fetchall()]

        if "parent_task_id" not in columns:
            logger.info("Migrating 'tasks' table: Adding column 'parent_task_id'")
            cursor.execute("ALTER TABLE tasks ADD COLUMN parent_task_id TEXT")
        if "is_subtask" not in columns:
            logger.info("Migrating 'tasks' table: Adding column 'is_subtask'")
            cursor.execute("ALTER TABLE tasks ADD COLUMN is_subtask BOOLEAN DEFAULT 0")
        if "assigned_agent_name" not in columns:
            logger.info("Migrating 'tasks' table: Adding column 'assigned_agent_name'")
            cursor.execute("ALTER TABLE tasks ADD COLUMN assigned_agent_name TEXT")

        # Add index for parent_task_id for faster querying (now columns are guaranteed to exist)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_parent_task_id ON tasks(parent_task_id)"
        )
        self.db_connection.commit() # Commit changes made with the local cursor
        logger.info("Swarm database initialized: 'agents' and 'tasks' tables ensured with new fields.")

    def _initialize_essential_agents(self):
        # Calls to self.add_agent will use self.knowledge_base.conn (shared) via Agent constructor
        # Meta-agents (highest tier)
        self.add_agent(
            name="OrganizerAgent",
            instructions=(
                "You are the Organizer Agent responsible for coordinating the swarm. "
                "You analyze tasks, delegate to appropriate agents, and ensure coherent outputs. "
                "You can propose new agents when needed based on task requirements."
            ),
            agent_type="meta",
        )

        self.add_agent(
            name="JudgeAgent",
            instructions=(
                "You evaluate the outputs of other agents, providing confidence scores and feedback. "
                "You ensure the quality, relevance, and coherence of content generated by the swarm. "
                "Score from 0.001 to 0.999 and provide specific feedback including strengths and areas for improvement. "
                "Format your response clearly, ensuring the confidence score is easily parsable (e.g., 'Confidence Score: 0.85')."
            ),
            agent_type="meta",
        )

        self.add_agent(
            name="InspiratorAgent",
            instructions=(
                "You are a highly creative and analytical AI. Your role is to identify gaps in the swarm's capabilities "
                "and propose new, specialized agents that would enhance the collective intelligence. When proposing, "
                "suggest a suitable name, detailed instructions, an agent_type (worker/supervisor). For the LLM model, "
                "you must choose between 'low' (for simpler tasks, maps to ollama/long-gemma) or 'high' (for complex tasks, maps to xai/grok-3-latest). "
                "Provide a clear rationale for your proposal, including your choice of 'low' or 'high' for the model."
            ),
            agent_type="meta",
            llm_model_identifier=DEFAULT_LLM_MODEL,
        )

        # Supervisor agents (middle tier)
        self.add_agent(
            name="WorldSupervisor",
            instructions=(
                "You supervise agents involved in world-building tasks. "
                "You coordinate their efforts and ensure consistency across geographical, cultural, historical, "
                "and technological aspects of created worlds."
            ),
            agent_type="supervisor",
        )

        self.add_agent(
            name="NarrativeSupervisor",
            instructions=(
                "You supervise agents involved in narrative creation. "
                "You ensure coherent storylines, character development, and plot progression. "
                "You coordinate between character, plot, and dialogue agents."
            ),
            agent_type="supervisor",
        )

        # Worker agents (base tier)
        self.add_agent(
            name="GeographyAgent",
            instructions=(
                "Create detailed geographical aspects of worlds including continents, climates, "
                "terrain features, natural resources, and ecosystems. Consider how geography "
                "influences other aspects of the world such as culture and politics."
            ),
            llm_model_identifier="ollama/long-gemma",
            agent_type="worker",
        )

        self.add_agent(
            name="CultureAgent",
            instructions=(
                "Develop rich cultural elements including customs, traditions, languages, "
                "arts, religions, social structures, and values. Create distinct cultural groups "
                "and explain how they interact with each other and their environment."
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker",
        )

        self.add_agent(
            name="HistoryAgent",
            instructions=(
                "Craft detailed historical timelines including major events, wars, discoveries, "
                "technological advancements, political shifts, and cultural developments. "
                "Create a sense of how the past shapes the present world state."
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker",
        )

        self.add_agent(
            name="CharacterAgent",
            instructions=(
                "Create compelling characters with distinct personalities, motivations, backgrounds, "
                "relationships, strengths, and flaws. Ensure characters feel authentic and reflect "
                "their cultural and historical context."
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker",
        )

    def add_agent(
        self,
        name: str,
        instructions: str,
        agent_type: str = "worker",
        llm_model_identifier: str = DEFAULT_LLM_MODEL,
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Agent:
        agent = Agent(
            name=name,
            instructions=instructions,
            agent_type=agent_type,
            llm_model_identifier=llm_model_identifier,
            db_connection=self.db_connection,
        )

        if agent_type == "meta":
            self.meta_agents[name] = agent
        elif agent_type == "supervisor":
            self.supervisors[name] = agent
        else:
            self.agents[name] = agent

        # Use a local cursor for this specific database write operation
        cursor = self.db_connection.cursor()
        try:
            # Ensure all columns from the 'agents' table schema in _init_db are covered.
            # The schema was: name, instructions, agent_type, llm_model_identifier, functions_json
            # The old INSERT was: name, instructions, tier (agent_type), creation_time, task_count, success_rate, last_active
            # This suggests the table schema used by add_agent was different from _init_db or incomplete.
            # Assuming the schema from _init_db is the source of truth:
            # (name, instructions, agent_type, llm_model_identifier, functions_json)
            # And agent stats (task_count, success_rate, last_active, creation_time) might be separate or part of it.
            # The INSERT OR REPLACE below matches the old logic more closely, using 'tier' for agent_type.
            # Let's assume 'agents' table needs all these: name, instructions, agent_type (tier), llm_model_identifier, creation_time, task_count, success_rate, last_active.
            # functions_json was in _init_db's CREATE TABLE for agents.
            cursor.execute(
                """
                INSERT OR REPLACE INTO agents
                (name, instructions, agent_type, llm_model_identifier, functions_json, creation_time, task_count, success_rate, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    instructions,
                    agent_type, # Maps to 'tier' in some old logs/inserts
                    llm_model_identifier,
                    json.dumps(functions) if functions else None, # functions_json
                    agent.creation_time,
                    0, # task_count
                    0.0, # success_rate
                    None # last_active
                )
            )
            self.db_connection.commit()
            logger.info(f"Swarm: Added/Replaced {agent_type} agent '{name}' in DB.")
        except sqlite3.Error as e:
            logger.error(f"Swarm: SQLite error in add_agent for '{name}': {e}", exc_info=True)
            # Not re-raising, as the agent is added in memory. DB error is logged.

        logger.info(f"Swarm: Agent {name} ({agent_type}) added to in-memory dictionary.") # Log for in-memory
        return agent

    def run_agent(
        self, agent_name: str, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        agent = self._get_agent(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return {"error": f"Agent {agent_name} not found"}

        logger.info(f"Swarm.run_agent: Running agent '{agent_name}' on task: '{task[:50]}...'. Context (keys): {list(context.keys()) if context else 'None'}")

        full_context = self.context_variables.copy()
        if context:
            full_context.update(context)

        logger.debug(f"Swarm.run_agent: Recalling experiences for agent '{agent_name}', task '{task[:50]}...'")
        relevant_experiences = self._get_relevant_experiences(task, agent_name, limit=3)
        experiences_text = self._format_experiences(relevant_experiences)
        if relevant_experiences:
            logger.info(f"Swarm.run_agent: Recalled {len(relevant_experiences)} experiences for agent '{agent_name}'.")

        logger.debug(f"Swarm.run_agent: Recalling memories for agent '{agent_name}', query '{task[:50]}...'")
        relevant_memories = agent.recall(query=task, limit=3)
        memories_text = self._format_memories(relevant_memories)

        system_prompt = (
            f"You are {agent_name}, a specialized agent in the SwarmMind collective intelligence system.\n\n"
            f"Your instructions: {agent.instructions}\n\n"
            f"Task: {task}\n\n"
        )

        if experiences_text:
            system_prompt += "Relevant past experiences:\n" + experiences_text + "\n\n"
        if memories_text:
            logger.info(f"Swarm.run_agent: Recalled {len(relevant_memories)} memories for agent '{agent_name}'.")
            system_prompt += "Your relevant memories:\n" + memories_text + "\n\n"

        if full_context:
            # Avoid logging full context if it's too verbose or contains sensitive data by default
            logger.debug(f"Swarm.run_agent: Providing full_context with keys: {list(full_context.keys())} to agent '{agent_name}'.")
            system_prompt += (
                "Context variables:\n" + json.dumps(full_context, indent=2) + "\n\n"
            )

        system_prompt += (
            "Respond with clear, concise, and detailed information related to your specialization. "
            "Focus on producing high-quality output that will contribute to the collective task."
        )

        try:
            response = agent.run(task) # Agent.run now logs the task execution

            output: str = response

            logger.debug(f"Swarm.run_agent: Agent '{agent_name}' completed task. Output preview: {output[:100]}...")

            logger.info(f"Swarm.run_agent: Saving experience for agent '{agent_name}' after task '{task[:50]}...'.")
            agent.remember({"task": task, "content": output}, long_term=True) # This calls knowledge_base.save_experience

            evaluation = self._evaluate_output(agent_name, output, task) # _evaluate_output now logs score

            # --- Update Agent Stats in DB ---
            try:
                # Use a local cursor for updating agent stats
                stat_cursor = self.db_connection.cursor()
                # Fetch current stats
                stat_cursor.execute(
                    "SELECT task_count, success_rate FROM agents WHERE name = ?",
                    (agent_name,),
                )
                result = stat_cursor.fetchone()
                if result:
                    current_task_count, current_success_rate = result
                    current_task_count = current_task_count or 0  # Handle None
                    current_success_rate = current_success_rate or 0.0  # Handle None
                else:
                    # Agent might not be in DB if add_agent failed at DB level but agent was used.
                    logger.warning(
                        f"Swarm.run_agent: Could not fetch existing stats for agent '{agent_name}' to update. Assuming new agent for stat purposes."
                    )
                    current_task_count, current_success_rate = 0, 0.0

                # Calculate new stats
                new_task_count = current_task_count + 1
                evaluation_score = evaluation.get("confidence_score", 0.0)

                new_success_rate = ((current_success_rate * current_task_count) + evaluation_score) / new_task_count if new_task_count > 0 else evaluation_score


                # Update DB
                stat_cursor.execute(
                    "UPDATE agents SET task_count = ?, success_rate = ?, last_active = ? WHERE name = ?",
                    (
                        new_task_count,
                        new_success_rate,
                        datetime.datetime.now().isoformat(),
                        agent_name,
                    ),
                )
                self.db_connection.commit() # Commit the stat update
                logger.debug(
                    f"Swarm.run_agent: Updated stats for agent '{agent_name}': tasks={new_task_count}, success_rate={new_success_rate:.3f}"
                )
            except sqlite3.Error as db_err:
                logger.error(
                    f"Database error updating agent stats for {agent_name}: {db_err}"
                )
            except (
                Exception
            ) as calc_err:  # Catch potential calculation errors (e.g., division by zero if logic flawed)
                logger.error(
                    f"Error calculating agent stats for {agent_name}: {calc_err}"
                )
            # --- End Update Agent Stats ---

            # self.knowledge_base.save_experience is called within agent.remember(long_term=True)
            # So, no need to call it directly here again.

            self.global_memory.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "agent": agent_name,
                    "task": task,
                    "output": output,
                    "evaluation": evaluation,
                }
            )

            if len(self.global_memory) > 20:
                self.global_memory.pop(0)

            return {
                "agent_name": agent_name,
                "output": output,
                "evaluation": evaluation,
            }

        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {e}")
            return {"error": str(e), "agent_name": agent_name}

    def _get_agent(self, name: str) -> Optional[Agent]:
        if name in self.agents:
            return self.agents[name]
        elif name in self.supervisors:
            return self.supervisors[name]
        elif name in self.meta_agents:
            return self.meta_agents[name]
        return None

    def _evaluate_output(
        self, agent_name: str, output: str, task: str
    ) -> Dict[str, Any]:
        judge = self.meta_agents.get("JudgeAgent")
        if not judge:
            logger.warning("JudgeAgent not found, cannot evaluate output.")
            return {
                "confidence_score": 0.75,
                "feedback": "Evaluation not available (JudgeAgent not found)",
            }

        prompt = (
            f"Task given to {agent_name}: {task}\n\n"
            f"Output from {agent_name}:\n\n{output}\n\n"
            "Evaluate this output's quality, relevance, and coherence. "
            "Provide a confidence score between 0.001 (poor) and 0.999 (excellent) "
            "and specific feedback including strengths and areas for improvement. "
            "Format your response clearly, ensuring the confidence score is easily parsable (e.g., 'Confidence Score: 0.85')."
        )

        evaluation_text: str | None = None

        judge_model_identifier = judge.llm_model_identifier
        # Use specific types for OpenAI compatibility
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": judge.instructions},
            {"role": "user", "content": prompt},
        ]

        try:
            logger.info(
                f"JudgeAgent ({judge_model_identifier}) evaluating output from {agent_name} for task: {task[:50]}..."
            )

            if judge_model_identifier.startswith("ollama/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                evaluation_text = call_ollama_chat(
                    model_name=model_name, messages=messages, options={"temperature": 0.3}
                ) 
                if (
                    evaluation_text
                    and evaluation_text.get("message")
                    and isinstance(evaluation_text["message"], dict)
                ):
                    evaluation_text = evaluation_text["message"].get("content")
                elif evaluation_text and evaluation_text.get("error"):
                    logger.error(
                        f"Ollama API error for JudgeAgent ({model_name}): {str(evaluation_text.get('error')) if isinstance(evaluation_text, dict) else 'Unknown error'}"
                    )
                else:
                    logger.warning(
                        f"Unexpected Ollama response structure for JudgeAgent ({model_name}): {str(evaluation_text)}"
                    )

            elif judge_model_identifier.startswith("xai/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                evaluation_text = call_xai_chat(
                    model_name=model_name, messages=messages, temperature=0.3
                ) 
                if (
                    evaluation_text
                    and evaluation_text.get("choices")
                    and evaluation_text["choices"]
                ):
                    message = evaluation_text["choices"][0].get("message")
                    if message and isinstance(message, dict):
                        evaluation_text = message.get("content")

            else:
                openai_model_name = judge_model_identifier
                if openai_model_name.startswith("openai/"):
                    openai_model_name = openai_model_name.split("/", 1)[1]

                completion = openai_client.chat.completions.create(
                    model=openai_model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500,
                )  # type: ignore[arg-type]
                if completion.choices and completion.choices[0].message and isinstance(completion.choices[0].message.content, str):
                    evaluation_text = completion.choices[0].message.content
                # else evaluation_text remains None, addressing old L1041 error.

            if not evaluation_text: # evaluation_text is str | None here
                logger.error(
                    f"JudgeAgent ({judge_model_identifier}) received no content."
                )
                raise ValueError("No content received from LLM for evaluation.")

            confidence_score = 0.75
            feedback = evaluation_text

            if isinstance(evaluation_text, str):
                score_match = re.search(
                    r"(?:confidence\s*score|score):?\s*(\d(?:\.\d+)?)", evaluation_text, re.IGNORECASE
                ) 
            else:
                score_match = None
            if score_match:
                try:
                    confidence_score = float(score_match.group(1))
                    confidence_score = max(0.001, min(0.999, confidence_score)) 
                except ValueError:
                    logger.warning(
                        f"Could not parse confidence score from: {score_match.group(1) if score_match else 'N/A'}"
                    )
                    pass
            else:
                logger.warning(
                    f"No confidence score found in JudgeAgent output: {evaluation_text[:100]}..." if isinstance(evaluation_text, str) else "No confidence score found."
                )

            return {"confidence_score": confidence_score, "feedback": evaluation_text if isinstance(evaluation_text, str) else "Evaluation error occurred."}

        except Exception as e:
            logger.error(
                f"Error evaluating output with JudgeAgent ({judge_model_identifier}): {e}"
            )
            return {
                "confidence_score": 0.5,
                "feedback": f"Error in evaluation: {str(e)}",
            }

    def _get_relevant_experiences(
        self, task: str, agent_name: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        logger.debug(
            f"Getting relevant experiences for task: '{task[:50]}...' for agent: {agent_name}"
        )
        try:
            task_embedding = get_embedding(task)
            if not task_embedding:
                logger.warning(f"Could not generate embedding for task: {task}")
                return []

            relevant_experiences = self.knowledge_base.query_experiences_by_similarity(
                task_embedding=task_embedding,
                agent_name_filter=agent_name,  # Filter by agent_name
                limit=limit,
            )
            logger.debug(f"Found {len(relevant_experiences)} relevant experiences.")
            return relevant_experiences
        except Exception as e:
            logger.error(
                f"Error getting relevant experiences for task '{task}': {e}",
                exc_info=True,
            )
            return []

    def _format_experiences(self, experiences: List[Dict[str, Any]]) -> str:
        if not experiences:
            return ""

        formatted = []
        for exp in experiences:
            if isinstance(exp.get('content'), str):
                formatted.append(
                    f"Agent: {exp['agent_name']}\nTask: {exp['task']}\nOutput: {exp['content'][:100]}..."
                )
            else:
                formatted.append(
                    f"Agent: {exp['agent_name']}\nTask: {exp['task']}\nOutput: {str(exp.get('content', ''))[:100]}..."
                )

        return "\n".join(formatted)

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""

        formatted = []
        for mem in memories:
            formatted.append(
                f"Task: {mem.get('task', 'Unknown')}\nContent: {mem.get('content', '')[:100]}...\nTimestamp: {mem.get('timestamp', 'Unknown')}\n"
            )

        return "\n".join(formatted)

    def propose_new_agent(self, task: str) -> Dict[str, Any]:
        inspirator = self.meta_agents.get("InspiratorAgent")
        if not inspirator:
            logger.error("InspirationAgent not found")
            return {"error": "InspirationAgent not found"}

        prompt = (
            f"Analyze this task and propose a new specialized agent that would enhance the swarm's capabilities:\n\n"
            f"Task: {task}\n\n"
            f"Current agents: {', '.join(list(self.agents.keys()) + list(self.supervisors.keys()))}\n\n"
            "Provide your response in this format:\n"
            "Agent Name: [name]\n"
            "Instructions: [detailed instructions]\n"
            "Agent Type: [worker/supervisor]\n"
            "LLM Model Identifier: [low/high]\n"
            "Rationale: [why this agent would be valuable]"
        )

        response_text = inspirator._get_llm_response(prompt)

        if response_text.startswith("Error:"):
            logger.error(
                f"InspiratorAgent failed to generate proposal: {response_text}"
            )
            return {
                "status": "error",
                "message": f"InspiratorAgent error: {response_text}",
            }

        proposal = response_text

        name_match = re.search(r"Agent Name:\s*(.+)", proposal, re.IGNORECASE)
        instructions_match = re.search(
            r"Instructions:\s*(.+?)(?=Agent Type:|LLM Model Identifier:|Rationale:|$)",
            proposal,
            re.DOTALL | re.IGNORECASE,
        )
        agent_type_match = re.search(
            r"Agent Type:\s*(worker|supervisor)", proposal, re.IGNORECASE
        )
        llm_identifier_match = re.search(
            r"LLM Model Identifier:\s*([\w\-\/\.]+)", proposal, re.IGNORECASE
        )
        rationale_match = re.search(
            r"Rationale:\s*(.+)", proposal, re.DOTALL | re.IGNORECASE
        )

        if name_match and instructions_match and agent_type_match:
            suggested_model_tier = (
                llm_identifier_match.group(1).strip().lower()
                if llm_identifier_match
                else "high"
            )

            if suggested_model_tier == "low":
                final_llm_model_identifier = "ollama/long-gemma"
            else:
                final_llm_model_identifier = "xai/grok-3-latest"
                if suggested_model_tier != "high" and llm_identifier_match:
                    logger.warning(
                        f"InspiratorAgent suggested LLM tier '{suggested_model_tier}', defaulting to 'high' (xai/grok-3-latest)."
                    )

            agent_proposal = {
                "name": name_match.group(1).strip(),
                "instructions": instructions_match.group(1).strip(),
                "agent_type": agent_type_match.group(1).strip().lower(),
                "llm_model_identifier": final_llm_model_identifier,
                "rationale": (
                    rationale_match.group(1).strip()
                    if rationale_match
                    else "No rationale provided"
                ),
            }

            self.pending_agents.append(agent_proposal)

            return {"status": "success", "proposal": agent_proposal}
        else:
            logger.error(f"Failed to parse agent proposal. Raw: {proposal}")
            return {
                "status": "error",
                "message": "Failed to parse agent proposal",
                "raw_proposal": proposal,
            }

    def approve_agent(self, agent_index: int) -> Dict[str, Any]:
        if agent_index < 0 or agent_index >= len(self.pending_agents):
            return {"error": "Invalid agent index"}

        agent_spec = self.pending_agents[agent_index]
        new_agent = self.add_agent(
            name=agent_spec["name"],
            instructions=agent_spec["instructions"],
            agent_type=agent_spec["agent_type"],
            llm_model_identifier=agent_spec.get(
                "llm_model_identifier", DEFAULT_LLM_MODEL
            ),
        )

        self.pending_agents.pop(agent_index)

        return {
            "status": "success",
            "message": f"Agent {new_agent.name} created successfully",
            "agent": new_agent.to_dict(),
        }

    def execute_subtask(self, agent_name: str, task_description: str, subtask_id: str) -> Dict[str, Any]:
        logger.info(f"[{subtask_id}] Executing subtask by agent '{agent_name}': {task_description[:50]}...")
        agent = self._get_agent(agent_name)

        if not agent:
            logger.error(f"[{subtask_id}] Swarm.execute_subtask: Agent '{agent_name}' not found.")
            error_message = f"Agent '{agent_name}' not found."
            cursor = self.db_connection.cursor() # Local cursor
            try:
                cursor.execute(
                    "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",
                    ("failed", error_message, datetime.datetime.now().isoformat(), subtask_id)
                )
                self.db_connection.commit()
            except sqlite3.Error as db_err:
                logger.error(f"Swarm: DB error in execute_subtask updating task {subtask_id} to failed (agent not found): {db_err}", exc_info=True)
            return {"error": error_message, "subtask_id": subtask_id}

        try:
            # Agent._get_llm_response returns a string (content or error message)
            llm_output_or_error = agent._get_llm_response(task_description)
            logger.info(f"[{subtask_id}] Swarm.execute_subtask: LLM call by '{agent_name}' completed. Output preview: {llm_output_or_error[:100]}...")
            
            cursor = self.db_connection.cursor() # Local cursor for this transaction
            if llm_output_or_error.startswith("Error:"):
                error_from_agent = llm_output_or_error
                logger.error(f"[{subtask_id}] Swarm.execute_subtask: Agent '{agent_name}' LLM call resulted in an error: {error_from_agent}")
                cursor.execute(
                    "UPDATE tasks SET status = ?, result = ?, error_message = ?, updated_at = ? WHERE task_id = ?",
                    ("failed", json.dumps({"output": llm_output_or_error}), error_from_agent, datetime.datetime.now().isoformat(), subtask_id)
                )
                final_status = {"error": error_from_agent, "subtask_id": subtask_id, "output": llm_output_or_error}
            else:
                # Success case
                cursor.execute(
                    "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE task_id = ?",
                    ("completed", json.dumps({"output": llm_output_or_error}), datetime.datetime.now().isoformat(), subtask_id)
                )
                final_status = {"success": True, "subtask_id": subtask_id, "output": llm_output_or_error}
            
            self.db_connection.commit()
            return final_status

        except Exception as e: # Catch any other unexpected error during the process
            logger.error(f"[{subtask_id}] Swarm.execute_subtask: Outer error for agent '{agent_name}': {e}", exc_info=True)
            error_message = f"Outer execution error by agent '{agent_name}': {str(e)}"
            # Use a new local cursor for this specific error update
            error_cursor = self.db_connection.cursor()
            try:
                error_cursor.execute(
                    "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",
                    ("failed", error_message, datetime.datetime.now().isoformat(), subtask_id)
                )
                self.db_connection.commit()
            except sqlite3.Error as db_err:
                logger.error(f"Swarm: DB error in execute_subtask attempting to mark task {subtask_id} as failed after outer exception: {db_err}", exc_info=True)
            return {"error": error_message, "subtask_id": subtask_id}

    def organize_task(self, task: str) -> Dict[str, Any]:
        # This method is now a legacy method or can be deprecated/refactored.
        organizer = self.meta_agents.get("OrganizerAgent")
        if not organizer:
            logger.error("OrganizerAgent not found")
            return {"error": "OrganizerAgent not found"}

        available_agents = {
            "workers": [a.name for a in self.agents.values()],
            "supervisors": [a.name for a in self.supervisors.values()],
        }

        prompt = (
            f"Analyze this task and organize a workflow to accomplish it effectively:\n\n"
            f"Task: {task}\n\n"
            f"Available agents: {', '.join(list(self.agents.keys()) + list(self.supervisors.keys()))}\n\n"
            "Break down the task into subtasks and assign them to appropriate agents. "
            "For each subtask, specify:\n"
            "1. The agent to handle it\n"
            "2. The specific subtask description\n"
            "3. The order of execution\n\n"
            "Provide your workflow plan in JSON format."
        )

        try:
            logger.info(f"[{self.__class__.__name__}] About to call OrganizerAgent ({organizer.llm_model_identifier}) _get_llm_response for task: {task[:50]}...")
            workflow_plan = organizer._get_llm_response(prompt)
            logger.info(f"[{self.__class__.__name__}] OrganizerAgent call returned. Workflow plan received (first 100 chars): {str(workflow_plan)[:100]}")

            # Handle potential non-string responses from _get_llm_response if necessary
            if not isinstance(workflow_plan, str):
                logger.warning(
                    f"Unexpected response type from organizer._get_llm_response: {type(workflow_plan)}. Converting to string."
                )
                workflow_plan = str(workflow_plan)

            json_match = re.search(r"```json\n(.*?)\n```", workflow_plan, re.DOTALL)
            if json_match:
                workflow_json = json_match.group(1)
            else:
                json_match = re.search(r"(\{.*\})", workflow_plan, re.DOTALL)
                if json_match:
                    workflow_json = json_match.group(1)
                else:
                    workflow_json = workflow_plan

            try:
                workflow = json.loads(workflow_json)
            except json.JSONDecodeError:
                logger.error(
                    f"Could not parse workflow JSON: {workflow_json}. Falling back."
                )
                workflow = {
                    "workflow": [
                        {
                            "step": 1,
                            "agent": next(
                                iter(self.agents), "GeographyAgent"
                            ),  # Fallback agent
                            "subtask": task,
                        }
                    ]
                }

            actual_steps = None
            if isinstance(parsed_workflow, dict):
                # Try common keys where the list of steps might be nested
                if "subtasks" in parsed_workflow and isinstance(parsed_workflow["subtasks"], list):
                    actual_steps = parsed_workflow["subtasks"]
                elif "steps" in parsed_workflow and isinstance(parsed_workflow["steps"], list):
                    actual_steps = parsed_workflow["steps"]
                elif "workflow" in parsed_workflow:
                    # If 'workflow' key exists, check if IT is the list or contains the list
                    if isinstance(parsed_workflow["workflow"], list):
                        actual_steps = parsed_workflow["workflow"]
                    elif isinstance(parsed_workflow["workflow"], dict):
                        if "subtasks" in parsed_workflow["workflow"] and isinstance(
                            parsed_workflow["workflow"]["subtasks"], list
                        ):
                            actual_steps = parsed_workflow["workflow"]["subtasks"]
                        elif "steps" in parsed_workflow["workflow"] and isinstance(
                            parsed_workflow["workflow"]["steps"], list
                        ):
                            actual_steps = parsed_workflow["workflow"]["steps"]
            elif isinstance(parsed_workflow, list):
                # Handle case where the root JSON object IS the list of steps
                actual_steps = parsed_workflow

            if not actual_steps:
                logger.warning(
                    f"[{parent_task_id}] Could not extract a list of steps from the parsed workflow: {parsed_workflow}. Falling back to single step execution."
                )
                # Fallback: Treat the original task as a single step for the first available agent
                first_agent_name = next(
                    iter(self.agents), "GeographyAgent"
                )  # Default fallback agent
                actual_steps = [{"step": 1, "agent": first_agent_name, "subtask": task}]
                parsed_workflow = {
                    "subtasks": actual_steps
                }  # Ensure parsed_workflow variable holds the steps for later return
            # --- End Improved Extraction ---

            # Now execute the extracted steps
            logger.info(f"Executing workflow with {len(actual_steps)} steps extracted.")
            results: List[Dict[str, Any]] = self._execute_workflow(actual_steps, task)
            # Combine results (using the corrected _combine_results method)
            combined_result = self._combine_results(results, task)
            # Return success case inside the try block
            return {
                "status": "success",
                "workflow": workflow,  # Return the workflow structure used (might be the fallback)
                "results": results,
                "combined_result": combined_result,
            }
        except Exception as e:
            logger.error(
                f"Error organizing task: {e}", exc_info=True
            )  # Add exc_info for traceback
            return {"error": str(e)}

    # Correctly placed except block for the main try block

    # --- Methods for workflow execution and result combination ---

    def _execute_workflow(
        self, steps: List[Dict[str, Any]], task: str
    ) -> List[Dict[str, Any]]:
        """Executes the steps defined in the workflow."""
        results: List[Dict[str, Any]] = []

        # Sort steps by 'order' or 'step' key if present
        sort_key = None
        if steps and isinstance(steps[0], dict):
            if "order" in steps[0]:
                sort_key = "order"
            elif "step" in steps[0]:
                sort_key = "step"
            else:
                logger.warning(
                    "No 'order' or 'step' key found in steps. Assuming order as given."
                )

        if sort_key:
            steps.sort(key=lambda x: x.get(sort_key, 0))

        for step in steps:
            agent_name = step.get("agent", "")
            subtask = step.get("subtask", task)

            if not self._get_agent(agent_name):
                logger.warning(f"Agent {agent_name} not found, skipping step")
                continue

            context = {
                "original_task": task,
                "previous_results": results,
                "current_step": step,
            }

            result = self.run_agent(agent_name, subtask, context)
            results.append(result)

            time.sleep(0.5)

        return results

    def _combine_results(
        self, results: List[Dict[str, Any]], task: str
    ) -> Dict[str, Any]:
        if len(results) == 1:
            return results[0]

        inputs = []
        for result in results:
            agent_name = result.get("agent_name", "Unknown")
            output = result.get("output", "")
            if isinstance(output, str):
                trimmed_output = output[:100] if len(output) > 100 else output
            else:
                trimmed_output = ""
            inputs.append(f"Agent: {agent_name}\nOutput: {trimmed_output}\n")

        combined_input = "\n".join(inputs)

        supervisor = None
        if "WorldSupervisor" in self.supervisors:
            supervisor = self.supervisors["WorldSupervisor"]
        elif "NarrativeSupervisor" in self.supervisors:
            supervisor = self.supervisors["NarrativeSupervisor"]
        else:
            supervisor = self.meta_agents.get("OrganizerAgent")

        if not supervisor:
            return {"combined_output": combined_input, "method": "simple_concatenation"}

        prompt = (
            f"Task: {task}\n\n"
            "Combine the following outputs:\n{combined_input}\n\nUnified Response:"
        )

        try:
            # Construct the full prompt including system instructions if available
            # Note: _get_llm_response in Agent class might need adjustment
            # if it doesn't inherently use agent.instructions as system prompt.
            # For now, assume _get_llm_response handles the full interaction.
            # A potential refinement could be passing system message explicitly if needed.

            combined_output = supervisor._get_llm_response(prompt)

            # Handle potential non-string responses from _get_llm_response if necessary
            if not isinstance(combined_output, str):
                logger.warning(
                    f"Unexpected response type from supervisor._get_llm_response: {type(combined_output)}. Converting to string."
                )
                combined_output = str(combined_output)

            return {
                "combined_output": combined_output,
                "method": f"combined_by_{supervisor.name}",
            }

        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return {"error": str(e), "method": "error_in_combination"}

    def _register_subtask(self, parent_task_id: str, subtask_description: str, assigned_agent_name: str) -> str:
        subtask_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        cursor = self.db_connection.cursor() # Local cursor
        try:
            logger.info(f"Swarm: Registering subtask {subtask_id} for parent {parent_task_id}. Assigned to: {assigned_agent_name}")
            cursor.execute(
                "INSERT INTO tasks (task_id, description, status, created_at, updated_at, parent_task_id, is_subtask, assigned_agent_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (subtask_id, subtask_description, "queued", now, now, parent_task_id, True, assigned_agent_name), # is_subtask = True
            )
            self.db_connection.commit()
            logger.info(f"Swarm: Subtask {subtask_id} registered successfully.")
            return subtask_id
        except sqlite3.Error as e:
            logger.error(f"Swarm: Database error in _register_subtask for parent {parent_task_id}, subtask {subtask_id}: {e}", exc_info=True)
            return "" # Indicates failure

    def run(self, task_description: str, task_id: str) -> Dict[str, Any]:
        """
        Main entry point for processing a task. 
        This will typically involve organizing the task for supervision (decomposition).
        """
        logger.info(f"Swarm.run called for task_id: {task_id}, description: {task_description[:100]}...")
        # For now, assume all tasks passed to Swarm.run are to be supervised and decomposed.
        # In the future, logic could be added here to differentiate simple vs. complex tasks.
        return self.organize_task_for_supervision(task=task_description, parent_task_id=task_id)

    def organize_task_for_supervision(self, task: str, parent_task_id: str) -> Dict[str, Any]:
        """
        Orchestrates task decomposition using an OrganizerAgent, registers subtasks,
        updates the parent task status to 'awaiting_subtasks', and returns information about the subtasks.
        """
        logger.info(
            f"[{parent_task_id}] Supervisor: Organizing task for supervision: {task[:100]}..."
        )
        organizer = self.meta_agents.get("OrganizerAgent")
        if not organizer:
            logger.error(f"[{parent_task_id}] Swarm.organize_task_for_supervision: OrganizerAgent not found. Parent task {parent_task_id} will be marked as failed.")
            cursor = self.db_connection.cursor() # Local cursor
            try:
                cursor.execute(
                    "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",
                    ("failed", "OrganizerAgent not found for decomposition", datetime.datetime.now().isoformat(), parent_task_id),
                )
                self.db_connection.commit()
            except sqlite3.Error as db_err:
                logger.error(f"Swarm: DB error in organize_task_for_supervision updating task {parent_task_id} to failed (Organizer not found): {db_err}", exc_info=True)
            return {"error": "OrganizerAgent not found"}

        # Construct a more detailed and prescriptive prompt for the OrganizerAgent
        worker_agent_names = [name for name, agent_obj in self.agents.items() if hasattr(agent_obj, 'role') and agent_obj.role == "worker"]
        supervisor_agent_names = [name for name, agent_obj in self.supervisors.items() if hasattr(agent_obj, 'role') and agent_obj.role == "supervisor"]
    
        # If no agents have roles defined, fall back to using all agents
        if not worker_agent_names and not supervisor_agent_names:
            worker_agent_names = list(self.agents.keys())
            supervisor_agent_names = list(self.supervisors.keys())
            
        assignable_agents = worker_agent_names + supervisor_agent_names
        
        if not assignable_agents:
            assignable_agents_str = "No specific worker/supervisor agents available. You may need to suggest generic roles or indicate if the task cannot be handled."
        else:
            assignable_agents_str = ", ".join(sorted(list(set(assignable_agents)))) # Sort and unique

        prompt = (
            f"You are an Organizer Agent. Your primary function is to decompose a given complex task into a sequence of smaller, actionable subtasks. \n\n"
            f"YOU MUST FORMAT YOUR OUTPUT AS A SINGLE JSON OBJECT with this exact structure:\n"
            f"{{\n"
            f"  \"subtasks\": [\n"
            f"    {{\"agent\": \"AgentName1\", \"subtask\": \"Description for subtask 1\"}},\n"
            f"    {{\"agent\": \"AgentName2\", \"subtask\": \"Description for subtask 2\"}},\n"
            f"    ...\n"
            f"  ]\n"
            f"}}\n\n"
            
            f"IMPORTANT RULES:\n"
            f"1. Your output MUST be ONLY the JSON object with no additional text\n"
            f"2. Each subtask object MUST have exactly two keys: 'agent' and 'subtask'\n"
            f"3. The 'agent' value MUST be one of these exact names: {assignable_agents_str}\n"
            f"4. If you think a new agent type is needed, assign that subtask to a supervisor agent\n\n"
            
            f"MAIN TASK TO DECOMPOSE:\n{task}\n\n"
            
            f"AVAILABLE AGENTS: {assignable_agents_str}\n\n"
            
            f"Remember: Output ONLY the JSON object with the subtasks array.\n"
            f"If the task is simple, you can create just one subtask assigned to the most appropriate agent."
        )

        try:
            logger.info(
                f"[{parent_task_id}] About to call OrganizerAgent ({organizer.llm_model_identifier}) _get_llm_response for task: {task[:50]}..."
            )
            workflow_plan_str = organizer._get_llm_response(prompt)
            if not workflow_plan_str:
                logger.error(f"[{parent_task_id}] OrganizerAgent returned an empty response. Falling back to single task execution.")
                raise ValueError("OrganizerAgent returned an empty plan")
                
            logger.info(
                f"[{parent_task_id}] OrganizerAgent call returned. Workflow plan received (first 200 chars): {str(workflow_plan_str)[:200]}"
            )

            # Handle potential non-string responses from _get_llm_response if necessary
            if not isinstance(workflow_plan_str, str):
                logger.warning(
                    f"[{parent_task_id}] Unexpected response type from organizer._get_llm_response: {type(workflow_plan_str)}. Converting to string."
                )
                workflow_plan_str = str(workflow_plan_str)

            # Enhanced JSON extraction with multiple patterns
            workflow_json_str = None
            
            # Try to find JSON in code blocks (```json ... ```)
            json_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', workflow_plan_str, re.DOTALL | re.IGNORECASE)
            if json_block_match:
                workflow_json_str = json_block_match.group(1).strip()
                logger.info(f"[{parent_task_id}] Extracted JSON from code block format")
        
            # If not found in code blocks, try to find a JSON object directly
            if not workflow_json_str:
                # Look for a complete JSON object with balanced braces
                json_obj_match = re.search(r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})', workflow_plan_str, re.DOTALL)
                if json_obj_match:
                    workflow_json_str = json_obj_match.group(1).strip()
                    logger.info(f"[{parent_task_id}] Extracted JSON using direct object pattern match")
            
            # If still not found, assume the entire response might be JSON (last resort)
            if not workflow_json_str:
                workflow_json_str = workflow_plan_str.strip()
                logger.info(f"[{parent_task_id}] No specific JSON pattern found, attempting to parse entire response")

            # Try to parse the extracted JSON string
            try:
                parsed_workflow = json.loads(workflow_json_str)
                logger.info(f"[{parent_task_id}] Successfully parsed JSON: {str(parsed_workflow)[:100]}...")
            except json.JSONDecodeError as e:
                logger.error(f"[{parent_task_id}] JSON parsing error: {e}. Workflow JSON: {workflow_json_str[:200]}...")
                
                # Attempt to fix common JSON issues
                fixed_json = False
                
                # Try adding missing quotes around keys
                try:
                    fixed_json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', workflow_json_str)
                    parsed_workflow = json.loads(fixed_json_str)
                    fixed_json = True
                    logger.info(f"[{parent_task_id}] Fixed JSON by adding quotes around keys")
                except:
                    pass
                
                # If still not fixed, fall back to a default structure
                if not fixed_json:
                    logger.warning(f"[{parent_task_id}] Could not fix JSON. Using fallback structure.")
                    # Fallback: Create a default workflow with the original task assigned to the first available agent
                    first_agent_name = next(iter(self.agents), "GeographyAgent")
                    parsed_workflow = {
                        "subtasks": [
                            {
                                "agent": first_agent_name,
                                "subtask": task
                            }
                        ]
                    }

            # Extract subtasks from the parsed workflow
            actual_steps = None
            if isinstance(parsed_workflow, dict):
                # Try common keys where the list of steps might be nested
                if "subtasks" in parsed_workflow and isinstance(parsed_workflow["subtasks"], list):
                    actual_steps = parsed_workflow["subtasks"]
                    logger.info(f"[{parent_task_id}] Found subtasks list with {len(actual_steps)} items")
                elif "steps" in parsed_workflow and isinstance(parsed_workflow["steps"], list):
                    actual_steps = parsed_workflow["steps"]
                    logger.info(f"[{parent_task_id}] Found steps list with {len(actual_steps)} items")
                elif "workflow" in parsed_workflow:
                    # If 'workflow' key exists, check if IT is the list or contains the list
                    if isinstance(parsed_workflow["workflow"], list):
                        actual_steps = parsed_workflow["workflow"]
                        logger.info(f"[{parent_task_id}] Found workflow list with {len(actual_steps)} items")
                    elif isinstance(parsed_workflow["workflow"], dict):
                        if "subtasks" in parsed_workflow["workflow"] and isinstance(
                            parsed_workflow["workflow"]["subtasks"], list
                        ):
                            actual_steps = parsed_workflow["workflow"]["subtasks"]
                            logger.info(f"[{parent_task_id}] Found nested subtasks list with {len(actual_steps)} items")
                        elif "steps" in parsed_workflow["workflow"] and isinstance(
                            parsed_workflow["workflow"]["steps"], list
                        ):
                            actual_steps = parsed_workflow["workflow"]["steps"]
                            logger.info(f"[{parent_task_id}] Found nested steps list with {len(actual_steps)} items")
            elif isinstance(parsed_workflow, list):
                # Handle case where the root JSON object IS the list of steps
                actual_steps = parsed_workflow
                logger.info(f"[{parent_task_id}] Found root list with {len(actual_steps)} items")

            if not actual_steps:
                logger.warning(
                    f"[{parent_task_id}] Could not extract a list of steps from the parsed workflow: {parsed_workflow}. Falling back to single step execution."
                )
                # Fallback: Treat the original task as a single step for the first available agent
                first_agent_name = next(
                    iter(self.agents), "GeographyAgent"
                )  # Default fallback agent
                actual_steps = [{"step": 1, "agent": first_agent_name, "subtask": task}]
                parsed_workflow = {
                    "subtasks": actual_steps
                }  # Ensure parsed_workflow variable holds the steps for later return

            # Register subtasks
            subtask_ids = []
            for step in actual_steps:
                # Ensure step has the required fields
                if not isinstance(step, dict) or "agent" not in step or "subtask" not in step:
                    if "step" in step and "agent" in step and "subtask" in step:
                        # Handle the case where we have {"step": 1, "agent": "...", "subtask": "..."}
                        agent_name = step["agent"]
                        subtask_description = step["subtask"]
                    else:
                        logger.warning(f"[{parent_task_id}] Invalid step format: {step}. Skipping.")
                        continue
                else:
                    agent_name = step["agent"]
                    subtask_description = step["subtask"]
                
                # Validate agent exists
                if not (agent_name in self.agents or agent_name in self.supervisors or agent_name in self.meta_agents):
                    logger.warning(f"[{parent_task_id}] Agent '{agent_name}' not found. Assigning to first available agent.")
                    agent_name = next(iter(self.agents), None)
                    if not agent_name:
                        logger.error(f"[{parent_task_id}] No agents available to assign subtask. Skipping.")
                        continue
                
                subtask_id = self._register_subtask(parent_task_id, subtask_description, agent_name)
                if subtask_id:
                    subtask_ids.append(subtask_id)
                    logger.info(f"[{parent_task_id}] Registered subtask {subtask_id} for agent '{agent_name}': {subtask_description[:50]}...")
                else:
                    logger.error(f"[{parent_task_id}] Failed to register subtask for step: {step}")

            if not subtask_ids:
                logger.error(f"[{parent_task_id}] No subtasks were successfully registered. Task cannot proceed.")
                raise ValueError("No subtasks registered from the plan")

            # IMPORTANT: Update parent task status to 'awaiting_subtasks'
            cursor = self.db_connection.cursor() # Local cursor for this update
            try:
                cursor.execute(
                    "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
                    ("awaiting_subtasks", datetime.datetime.now().isoformat(), parent_task_id),
                )
                self.db_connection.commit()
                logger.info(f"[{parent_task_id}] Swarm.organize_task_for_supervision: Parent task status updated to 'awaiting_subtasks'. {len(subtask_ids)} subtasks created.")
            except sqlite3.Error as db_err:
                 logger.error(f"Swarm: DB error in organize_task_for_supervision updating task {parent_task_id} to awaiting_subtasks: {db_err}", exc_info=True)
                 # This is tricky, subtasks are registered but parent failed to update.
                 # For now, log and proceed, supervisor might pick it up or it might need manual fix.
                 # Depending on desired atomicity, might need more complex rollback of subtask registration.
                 # Returning the subtask_ids is still useful.
                 return {"status": "error", "message": f"Failed to update parent task status, but subtasks registered: {db_err}", "subtask_ids": subtask_ids}

            return {
                "status": "decomposed",
                "message": f"Task decomposed into {len(subtask_ids)} subtasks",
                "parent_task_id": parent_task_id,
                "subtask_ids": subtask_ids,
                "workflow": parsed_workflow
            }
        except Exception as e:
            logger.error(
                f"[{parent_task_id}] Error organizing task for supervision: {e}", exc_info=True
            )
            # Update parent task to failed if decomposition fails critically
            cursor = self.db_connection.cursor() # Local cursor
            try:
                cursor.execute(
                    "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",
                    ("failed", f"Failed to decompose task: {str(e)}", datetime.datetime.now().isoformat(), parent_task_id),
                )
                self.db_connection.commit()
            except sqlite3.Error as db_err: # More specific exception
                logger.error(f"Swarm: DB error in organize_task_for_supervision marking task {parent_task_id} as failed after exception: {db_err}", exc_info=True)
            return {"status": "error", "message": f"Failed to organize task: {str(e)}"}