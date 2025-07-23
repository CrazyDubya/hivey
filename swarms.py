import datetime
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import openai
from dotenv import load_dotenv

from llm_clients import call_ollama_chat, call_xai_chat
from utils import client as openai_client
from utils import cosine_similarity, get_embedding

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger("SwarmMind")

# Configure OpenAI API (This global openai.api_key might be redundant if client from utils is used consistently)  # noqa: E501
# It's good practice to ensure it's set for any direct openai legacy calls if they exist elsewhere, but new calls should use the client.  # noqa: E501
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    logger.warning("OPENAI_API_KEY not found in environment variables.")

# Removed redundant logging.basicConfig - will be handled by utils.configure_logging()  # noqa: E501
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # noqa: E501
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
                f"KnowledgeBase creating NEW non-shared DB connection for path: {db_path}"  # noqa: E501
            )
            try:
                self.conn = sqlite3.connect(
                    DB_NAME
                )  # Tables are now initialized by utils.initialize_database()
            except sqlite3.Error as e:
                logger.error(
                    f"SQLite error creating connection in KB __init__ (path: {db_path}): {e}"  # noqa: E501
                )
                raise

        if not self.conn:
            raise Exception(
                "Failed to establish DB connection in KnowledgeBase"
            )

        self.cursor = self.conn.cursor()
        # Table creation logic moved to utils.initialize_database()
        # try:
        #     self.cursor.execute(
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
        #             status TEXT NOT NULL, -- e.g., queued, running, completed, failed  # noqa: E501
        #             result TEXT,          -- Store JSON or text result
        #             error_message TEXT,   -- Store error if status is 'failed'  # noqa: E501
        #             created_at TEXT NOT NULL,
        #             updated_at TEXT NOT NULL,
        #             parent_task_id TEXT,          -- Added for subtask relationship  # noqa: E501
        #             is_subtask BOOLEAN DEFAULT 0  -- Added to identify subtasks  # noqa: E501
        #         )
        #     """
        #     )
        #     # Add index for parent_task_id for faster querying
        #     self.cursor.execute(
        #         "CREATE INDEX IF NOT EXISTS idx_parent_task_id ON tasks(parent_task_id)"  # noqa: E501
        #     )
        #     self.conn.commit()

        # except sqlite3.Error as e:
        #     logger.error(f"SQLite error during KB table init/commit: {e}")
        #     if not self.shared_connection and hasattr(self, "conn") and self.conn:  # noqa: E501
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
        embedding_json = (
            json.dumps(content_embedding) if content_embedding else None
        )
        metadata_json_str = json.dumps(metadata) if metadata else None

        try:
            self.cursor.execute(
                """
                INSERT INTO memories (agent_name, memory_type, content_text, content_embedding, metadata_json, timestamp)  # noqa: E501
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
            logger.debug(f"Added memory '{content_text[:30]}...' to KB.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error during add_memory: {e}")
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

        try:
            self.cursor.execute(
                """
                INSERT INTO experiences (task, agent_name, content, confidence_score, feedback, timestamp, embedding)  # noqa: E501
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
            logger.info(f"Saved experience: {task} - {content[:50]}...")
        except sqlite3.Error as e:
            logger.error(f"SQLite error during save_experience: {e}")
            raise

    def query_experiences_by_similarity(
        self,
        task_embedding: List[float],
        agent_name_filter: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if not task_embedding:
            logger.warning(
                "Task embedding is empty, cannot perform semantic search for experiences."  # noqa: E501
            )
            return []

        query = "SELECT task, agent_name, content, confidence_score, feedback, embedding, timestamp FROM experiences"  # noqa: E501
        params = []
        if agent_name_filter:
            query += " WHERE agent_name = ?"
            params.append(agent_name_filter)

        try:
            self.cursor.execute(query, tuple(params))
            all_experiences = self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(
                f"SQLite error during query_experiences_by_similarity fetch: {e}"  # noqa: E501
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
                ):  # Handle case where embedding string is 'null' or empty list  # noqa: E501
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
                    f"Failed to decode embedding JSON for experience: {exp_content[:50]}..."  # noqa: E501
                )
            except (
                Exception
            ) as e:  # Catch other errors during similarity calculation
                logger.error(
                    f"Error calculating similarity for experience '{exp_content[:50]}...': {e}"  # noqa: E501
                )

        # Sort by similarity in descending order
        experiences_with_similarity.sort(
            key=lambda x: x["similarity"], reverse=True
        )

        # Return top N
        return experiences_with_similarity[:limit]

    def semantic_search(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Performs a semantic search for memories based on query embedding."""
        logger.info(
            f"Semantic search called with limit {limit}. Currently a stub."
        )
        # This is a placeholder. Actual implementation will query 'memories' table.  # noqa: E501
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
        self,
        query: Optional[str] = None,
        long_term: bool = False,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if long_term:
            if not query:
                logger.warning("Long-term recall attempted without a query.")
                # Decide what to return: empty list, or perhaps all memories up to limit?  # noqa: E501
                # For now, let's assume if no query, it implies no specific search, maybe return generic memories?  # noqa: E501
                # This behavior might need refinement based on desired functionality.  # noqa: E501
                # Returning empty for now if no query for long_term.
                return []
            query_embedding = get_embedding(query)
            if query_embedding is None:
                logger.error(
                    f"Could not generate embedding for query: {query}"
                )
                return []
            # semantic_search expects query_embedding and limit.
            # Agent name filtering would need to be part of semantic_search implementation itself if desired.  # noqa: E501
            return self.knowledge_base.semantic_search(
                query_embedding, limit=limit
            )
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
                similarity = cosine_similarity(
                    query_embedding, entry_embedding
                )
                scores.append(similarity)

            sorted_pairs = sorted(
                zip(scores, self.task_history),
                key=lambda x: x[0],
                reverse=True,
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
        self,
        task_description: str,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        model_params = model_params or {}
        system_prompt = (
            f"You are {self.name}, a specialized agent in the SwarmMind collective intelligence system.\n\n"  # noqa: E501
            f"Your instructions: {self.instructions}\n\n"
            f"Task: {task_description}\n\n"
            "Respond with clear, concise, and detailed information related to your specialization. "  # noqa: E501
            "Focus on producing high-quality output that will contribute to the collective task."  # noqa: E501
        )

        # Use specific types for OpenAI compatibility
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description},
        ]

        # Standardized variable for extracted text content
        extracted_text_content: str | None = None

        try:
            logger.info(
                f"Agent {self.name} using model: {self.llm_model_identifier} for task: {task_description[:100]}..."  # noqa: E501
            )

            if self.llm_model_identifier.startswith("ollama/"):
                model_name = self.llm_model_identifier.split("/", 1)[1]
                raw_response = call_ollama_chat(
                    model_name=model_name,
                    messages=messages,
                    options=model_params,
                )
                if (
                    raw_response
                    and raw_response.get("message")
                    and isinstance(raw_response["message"], dict)
                ):
                    extracted_text_content = raw_response["message"].get(
                        "content"
                    )
                elif raw_response and raw_response.get("error"):
                    logger.error(
                        f"Ollama API error for {self.name} ({model_name}): {str(raw_response.get('error')) if isinstance(raw_response, dict) else 'Unknown error'}"  # noqa: E501
                    )
                else:
                    logger.warning(
                        f"Unexpected Ollama response structure for {self.name} ({model_name}): {str(raw_response)}"  # noqa: E501
                    )

            elif self.llm_model_identifier.startswith("xai/"):
                model_name = self.llm_model_identifier.split("/", 1)[1]
                raw_response = call_xai_chat(
                    model_name=model_name, messages=messages
                )
                if (
                    raw_response
                    and raw_response.get("choices")
                    and raw_response["choices"]
                ):
                    message = raw_response["choices"][0].get("message")
                    if message and isinstance(message, dict):
                        extracted_text_content = message.get("content")
                else:
                    logger.warning(
                        f"Unexpected X.AI response structure for {self.name} ({model_name}): {str(raw_response)}"  # noqa: E501
                    )

            else:
                openai_model_name = self.llm_model_identifier
                if openai_model_name.startswith("openai/"):
                    openai_model_name = openai_model_name.split("/", 1)[1]

                completion = openai_client.chat.completions.create(
                    model=openai_model_name,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=model_params.get("temperature", 0.7),
                    max_tokens=model_params.get("max_tokens", 1024),
                )  # type: ignore[arg-type]
                if (
                    completion.choices
                    and completion.choices[0].message
                    and isinstance(completion.choices[0].message.content, str)
                ):
                    extracted_text_content = completion.choices[
                        0
                    ].message.content
                # else response_content remains None, addressing old L468 error.  # noqa: E501

            # Common post-processing logic based on extracted_text_content
            if isinstance(extracted_text_content, str):
                self.remember(
                    {"role": "assistant", "content": extracted_text_content}
                )
                logger.info(
                    f"Agent {self.name} received response: {extracted_text_content[:100]}..."  # noqa: E501
                )
                if isinstance(extracted_text_content, str):
                    trimmed_content = (
                        extracted_text_content[:100]
                        if len(extracted_text_content) > 100
                        else extracted_text_content
                    )
                else:
                    trimmed_content = ""
                extracted_text_content = trimmed_content
                if isinstance(extracted_text_content, str):
                    extracted_text_content = extracted_text_content.strip()
                # Ensure response_content is a string before returning
                return (
                    str(extracted_text_content)
                    if extracted_text_content is not None
                    else ""
                )
            else:
                logger.error(
                    f"Agent {self.name} received no content or non-string content from LLM ({self.llm_model_identifier})."  # noqa: E501
                )
                return "Error: No or invalid content received from LLM."

        except openai.APIError as e:
            logger.error(
                f"OpenAI API Error for agent {self.name} ({self.llm_model_identifier}): {str(e)}"  # noqa: E501
            )
            return "Error: OpenAI API Error - " + str(e)
        except Exception as e:
            logger.error(
                f"General error in _get_llm_response for agent {self.name} ({self.llm_model_identifier}): {str(e)}"  # noqa: E501
            )
            return "Error: An unexpected error occurred - " + str(e)

    def run(
        self,
        task_description: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self.task_history.append({"task": task_description})
        return self._get_llm_response(task_description)

    def _evaluate_output(self, task: str, output: str) -> Dict[str, Any]:
        judge = (
            self.swarm.meta_agents.get("JudgeAgent")
            if hasattr(self, "swarm")
            else None
        )
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
            "Provide a confidence score between 0.001 (poor) and 0.999 (excellent) "  # noqa: E501
            "and specific feedback including strengths and areas for improvement. "  # noqa: E501
            "Format your response clearly, ensuring the confidence score is easily parsable (e.g., 'Confidence Score: 0.85')."  # noqa: E501
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
                f"JudgeAgent ({judge_model_identifier}) evaluating output from {self.name} for task: {task[:50]}..."  # noqa: E501
            )

            if judge_model_identifier.startswith("ollama/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                evaluation_text = call_ollama_chat(
                    model_name=model_name,
                    messages=messages,
                    options={"temperature": 0.3},
                )
                if (
                    evaluation_text
                    and evaluation_text.get("message")
                    and isinstance(evaluation_text["message"], dict)
                ):
                    evaluation_text = evaluation_text["message"].get("content")
                elif evaluation_text and evaluation_text.get("error"):
                    logger.error(
                        f"Ollama API error for JudgeAgent ({model_name}): {str(evaluation_text.get('error')) if isinstance(evaluation_text, dict) else 'Unknown error'}"  # noqa: E501
                    )
                else:
                    logger.warning(
                        f"Unexpected Ollama response structure for JudgeAgent ({model_name}): {str(evaluation_text)}"  # noqa: E501
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
                if (
                    completion.choices
                    and completion.choices[0].message
                    and isinstance(completion.choices[0].message.content, str)
                ):
                    evaluation_text = completion.choices[0].message.content
                # else evaluation_text remains None, addressing old L587 error.

            if not evaluation_text:  # evaluation_text is str | None here
                logger.error(
                    f"JudgeAgent ({judge_model_identifier}) received no content."  # noqa: E501
                )
                raise ValueError(
                    "No content received from LLM for evaluation."
                )

            confidence_score = 0.75

            if isinstance(evaluation_text, str):
                score_match = re.search(
                    r"(?:confidence\s*score|score):?\s*(\d(?:\.\d+)?)",
                    evaluation_text,
                    re.IGNORECASE,
                )
            else:
                score_match = None
            if score_match:
                try:
                    confidence_score = float(score_match.group(1))
                    confidence_score = max(0.001, min(0.999, confidence_score))
                except ValueError:
                    logger.warning(
                        f"Could not parse confidence score from: {score_match.group(1) if score_match else 'N/A'}"  # noqa: E501
                    )
                    pass
            else:
                logger.warning(
                    f"No confidence score found in JudgeAgent output: {evaluation_text[:100]}..."  # noqa: E501
                    if isinstance(evaluation_text, str)
                    else "No confidence score found."
                )

            return {
                "confidence_score": confidence_score,
                "feedback": (
                    evaluation_text
                    if isinstance(evaluation_text, str)
                    else "Evaluation error occurred."
                ),
            }

        except Exception as e:
            logger.error(
                f"Error evaluating output with JudgeAgent ({judge_model_identifier}): {e}"  # noqa: E501
            )
            return {
                "confidence_score": 0.5,
                "feedback": f"Error in evaluation: {str(e)}",
            }


class Swarm:
    def __init__(self, db_connection: Optional[sqlite3.Connection]):
        """Initialize the swarm with a shared database connection and load agents"""  # noqa: E501
        self.db_connection = db_connection  # Store the shared connection
        self.agents: Dict[str, Agent] = {}
        self.supervisors: Dict[str, Agent] = {}
        self.meta_agents: Dict[str, Agent] = {}
        self.context_variables: Dict[str, Any] = {}
        self.global_memory: List[Dict[str, Any]] = []
        self._init_db()  # Ensures tables are created if they don't exist
        self.knowledge_base = KnowledgeBase(connection=self.db_connection)
        self._initialize_essential_agents()  # Load/create initial agents

    def _init_db(self):
        if not self.db_connection:
            logger.error(
                "Database connection is not initialized for Swarm._init_db"
            )
            raise ConnectionError("Database connection not initialized")

        self.cursor = (
            self.db_connection.cursor()
        )  # Ensure cursor is from self.db_connection

        # Ensure 'agents' table exists
        self.cursor.execute(
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
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL, -- e.g., queued, running, completed, failed  # noqa: E501
                result TEXT,          -- Store JSON or text result
                error_message TEXT,   -- Store error if status is 'failed'
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                parent_task_id TEXT,          -- Added for subtask relationship
                is_subtask BOOLEAN DEFAULT 0,  -- Added to identify subtasks
                assigned_agent_name TEXT      -- Added to specify which agent should run a subtask  # noqa: E501
            )
        """
        )

        # Schema migration for existing 'tasks' table
        self.cursor.execute("PRAGMA table_info(tasks)")
        columns = [info[1] for info in self.cursor.fetchall()]

        if "parent_task_id" not in columns:
            logger.info(
                "Migrating 'tasks' table: Adding column 'parent_task_id'"
            )
            self.cursor.execute(
                "ALTER TABLE tasks ADD COLUMN parent_task_id TEXT"
            )
        if "is_subtask" not in columns:
            logger.info("Migrating 'tasks' table: Adding column 'is_subtask'")
            self.cursor.execute(
                "ALTER TABLE tasks ADD COLUMN is_subtask BOOLEAN DEFAULT 0"
            )
        if "assigned_agent_name" not in columns:
            logger.info(
                "Migrating 'tasks' table: Adding column 'assigned_agent_name'"
            )
            self.cursor.execute(
                "ALTER TABLE tasks ADD COLUMN assigned_agent_name TEXT"
            )

        # Add index for parent_task_id for faster querying (now columns are guaranteed to exist)  # noqa: E501
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_parent_task_id ON tasks(parent_task_id)"  # noqa: E501
        )
        self.db_connection.commit()
        logger.info(
            "Swarm database initialized: 'agents' and 'tasks' tables ensured with new fields."  # noqa: E501
        )

    def _initialize_essential_agents(self):
        # Calls to self.add_agent will use self.knowledge_base.conn (shared) via Agent constructor  # noqa: E501
        # Meta-agents (highest tier)
        self.add_agent(
            name="OrganizerAgent",
            instructions=(
                "You are the Organizer Agent responsible for coordinating the swarm. "  # noqa: E501
                "You analyze tasks, delegate to appropriate agents, and ensure coherent outputs. "  # noqa: E501
                "You can propose new agents when needed based on task requirements."  # noqa: E501
            ),
            agent_type="meta",
        )

        self.add_agent(
            name="JudgeAgent",
            instructions=(
                "You evaluate the outputs of other agents, providing confidence scores and feedback. "  # noqa: E501
                "You ensure the quality, relevance, and coherence of content generated by the swarm. "  # noqa: E501
                "Score from 0.001 to 0.999 and provide specific feedback including strengths and areas for improvement. "  # noqa: E501
                "Format your response clearly, ensuring the confidence score is easily parsable (e.g., 'Confidence Score: 0.85')."  # noqa: E501
            ),
            agent_type="meta",
        )

        self.add_agent(
            name="InspiratorAgent",
            instructions=(
                "You are a highly creative and analytical AI. Your role is to identify gaps in the swarm's capabilities "  # noqa: E501
                "and propose new, specialized agents that would enhance the collective intelligence. When proposing, "  # noqa: E501
                "suggest a suitable name, detailed instructions, an agent_type (worker/supervisor). For the LLM model, "  # noqa: E501
                "you must choose between 'low' (for simpler tasks, maps to ollama/long-gemma) or 'high' (for complex tasks, maps to xai/grok-3-latest). "  # noqa: E501
                "Provide a clear rationale for your proposal, including your choice of 'low' or 'high' for the model."  # noqa: E501
            ),
            agent_type="meta",
            llm_model_identifier=DEFAULT_LLM_MODEL,
        )

        # Supervisor agents (middle tier)
        self.add_agent(
            name="WorldSupervisor",
            instructions=(
                "You supervise agents involved in world-building tasks. "
                "You coordinate their efforts and ensure consistency across geographical, cultural, historical, "  # noqa: E501
                "and technological aspects of created worlds."
            ),
            agent_type="supervisor",
        )

        self.add_agent(
            name="NarrativeSupervisor",
            instructions=(
                "You supervise agents involved in narrative creation. "
                "You ensure coherent storylines, character development, and plot progression. "  # noqa: E501
                "You coordinate between character, plot, and dialogue agents."
            ),
            agent_type="supervisor",
        )

        # Worker agents (base tier)
        self.add_agent(
            name="GeographyAgent",
            instructions=(
                "Create detailed geographical aspects of worlds including continents, climates, "  # noqa: E501
                "terrain features, natural resources, and ecosystems. Consider how geography "  # noqa: E501
                "influences other aspects of the world such as culture and politics."  # noqa: E501
            ),
            llm_model_identifier="ollama/long-gemma",
            agent_type="worker",
        )

        self.add_agent(
            name="CultureAgent",
            instructions=(
                "Develop rich cultural elements including customs, traditions, languages, "  # noqa: E501
                "arts, religions, social structures, and values. Create distinct cultural groups "  # noqa: E501
                "and explain how they interact with each other and their environment."  # noqa: E501
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker",
        )

        self.add_agent(
            name="HistoryAgent",
            instructions=(
                "Craft detailed historical timelines including major events, wars, discoveries, "  # noqa: E501
                "technological advancements, political shifts, and cultural developments. "  # noqa: E501
                "Create a sense of how the past shapes the present world state."  # noqa: E501
            ),
            llm_model_identifier="xai/grok-3-latest",
            agent_type="worker",
        )

        self.add_agent(
            name="CharacterAgent",
            instructions=(
                "Create compelling characters with distinct personalities, motivations, backgrounds, "  # noqa: E501
                "relationships, strengths, and flaws. Ensure characters feel authentic and reflect "  # noqa: E501
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

        self.knowledge_base.cursor.execute(
            """
            INSERT OR REPLACE INTO agents
            (name, instructions, tier, creation_time, task_count, success_rate, last_active)  # noqa: E501
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                name,
                instructions,
                agent_type,
                agent.creation_time,
                0,
                0.0,
                None,
            ),
        )
        if self.knowledge_base.conn:
            self.knowledge_base.conn.commit()
        logger.info(f"Added {agent_type} agent: {name}")
        return agent

    def run_agent(
        self,
        agent_name: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        agent = self._get_agent(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return {"error": f"Agent {agent_name} not found"}

        logger.info(f"Running agent: {agent_name} on task: {task[:50]}...")

        full_context = self.context_variables.copy()
        if context:
            full_context.update(context)

        relevant_experiences = self._get_relevant_experiences(
            task, agent_name, limit=3
        )
        experiences_text = self._format_experiences(relevant_experiences)

        relevant_memories = agent.recall(query=task, limit=3)
        memories_text = self._format_memories(relevant_memories)

        system_prompt = (
            f"You are {agent_name}, a specialized agent in the SwarmMind collective intelligence system.\n\n"  # noqa: E501
            f"Your instructions: {agent.instructions}\n\n"
            f"Task: {task}\n\n"
        )

        if experiences_text:
            system_prompt += (
                "Relevant past experiences:\n" + experiences_text + "\n\n"
            )

        if memories_text:
            system_prompt += (
                "Your relevant memories:\n" + memories_text + "\n\n"
            )

        if full_context:
            system_prompt += (
                "Context variables:\n"
                + json.dumps(full_context, indent=2)
                + "\n\n"
            )

        system_prompt += (
            "Respond with clear, concise, and detailed information related to your specialization. "  # noqa: E501
            "Focus on producing high-quality output that will contribute to the collective task."  # noqa: E501
        )

        try:
            response = agent.run(task)

            output: str = response

            agent.remember({"task": task, "content": output}, long_term=True)

            evaluation = self._evaluate_output(agent_name, output, task)

            # --- Update Agent Stats in DB ---
            try:
                # Fetch current stats
                self.knowledge_base.cursor.execute(
                    "SELECT task_count, success_rate FROM agents WHERE name = ?",  # noqa: E501
                    (agent_name,),
                )
                result = self.knowledge_base.cursor.fetchone()
                if result:
                    current_task_count, current_success_rate = result
                    current_task_count = current_task_count or 0  # Handle None
                    current_success_rate = (
                        current_success_rate or 0.0
                    )  # Handle None
                else:
                    logger.warning(
                        f"Could not fetch stats for agent {agent_name} to update."  # noqa: E501
                    )
                    current_task_count, current_success_rate = 0, 0.0

                # Calculate new stats
                new_task_count = current_task_count + 1
                evaluation_score = evaluation.get(
                    "confidence_score", 0.0
                )  # Default to 0 if missing

                # Weighted average calculation
                new_success_rate = (
                    (current_success_rate * current_task_count)
                    + evaluation_score
                ) / new_task_count

                # Update DB
                self.knowledge_base.cursor.execute(
                    "UPDATE agents SET task_count = ?, success_rate = ?, last_active = ? WHERE name = ?",  # noqa: E501
                    (
                        new_task_count,
                        new_success_rate,
                        datetime.datetime.now().isoformat(),
                        agent_name,
                    ),
                )
                if self.knowledge_base.conn:
                    self.knowledge_base.conn.commit()
                logger.debug(
                    f"Updated stats for {agent_name}: tasks={new_task_count}, success_rate={new_success_rate:.2f}"  # noqa: E501
                )

            except sqlite3.Error as db_err:
                logger.error(
                    f"Database error updating agent stats for {agent_name}: {db_err}"  # noqa: E501
                )
            except (
                Exception
            ) as calc_err:  # Catch potential calculation errors (e.g., division by zero if logic flawed)  # noqa: E501
                logger.error(
                    f"Error calculating agent stats for {agent_name}: {calc_err}"  # noqa: E501
                )
            # --- End Update Agent Stats ---

            self.knowledge_base.save_experience(
                task=task,
                agent_name=agent_name,
                content=output,
                confidence_score=evaluation["confidence_score"],
                feedback=evaluation["feedback"],
            )

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
            "Provide a confidence score between 0.001 (poor) and 0.999 (excellent) "  # noqa: E501
            "and specific feedback including strengths and areas for improvement. "  # noqa: E501
            "Format your response clearly, ensuring the confidence score is easily parsable (e.g., 'Confidence Score: 0.85')."  # noqa: E501
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
                f"JudgeAgent ({judge_model_identifier}) evaluating output from {agent_name} for task: {task[:50]}..."  # noqa: E501
            )

            if judge_model_identifier.startswith("ollama/"):
                model_name = judge_model_identifier.split("/", 1)[1]
                evaluation_text = call_ollama_chat(
                    model_name=model_name,
                    messages=messages,
                    options={"temperature": 0.3},
                )
                if (
                    evaluation_text
                    and evaluation_text.get("message")
                    and isinstance(evaluation_text["message"], dict)
                ):
                    evaluation_text = evaluation_text["message"].get("content")
                elif evaluation_text and evaluation_text.get("error"):
                    logger.error(
                        f"Ollama API error for JudgeAgent ({model_name}): {str(evaluation_text.get('error')) if isinstance(evaluation_text, dict) else 'Unknown error'}"  # noqa: E501
                    )
                else:
                    logger.warning(
                        f"Unexpected Ollama response structure for JudgeAgent ({model_name}): {str(evaluation_text)}"  # noqa: E501
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
                if (
                    completion.choices
                    and completion.choices[0].message
                    and isinstance(completion.choices[0].message.content, str)
                ):
                    evaluation_text = completion.choices[0].message.content
                # else evaluation_text remains None, addressing old L1041 error.  # noqa: E501

            if not evaluation_text:  # evaluation_text is str | None here
                logger.error(
                    f"JudgeAgent ({judge_model_identifier}) received no content."  # noqa: E501
                )
                raise ValueError(
                    "No content received from LLM for evaluation."
                )

            confidence_score = 0.75

            if isinstance(evaluation_text, str):
                score_match = re.search(
                    r"(?:confidence\s*score|score):?\s*(\d(?:\.\d+)?)",
                    evaluation_text,
                    re.IGNORECASE,
                )
            else:
                score_match = None
            if score_match:
                try:
                    confidence_score = float(score_match.group(1))
                    confidence_score = max(0.001, min(0.999, confidence_score))
                except ValueError:
                    logger.warning(
                        f"Could not parse confidence score from: {score_match.group(1) if score_match else 'N/A'}"  # noqa: E501
                    )
                    pass
            else:
                logger.warning(
                    f"No confidence score found in JudgeAgent output: {evaluation_text[:100]}..."  # noqa: E501
                    if isinstance(evaluation_text, str)
                    else "No confidence score found."
                )

            return {
                "confidence_score": confidence_score,
                "feedback": (
                    evaluation_text
                    if isinstance(evaluation_text, str)
                    else "Evaluation error occurred."
                ),
            }

        except Exception as e:
            logger.error(
                f"Error evaluating output with JudgeAgent ({judge_model_identifier}): {e}"  # noqa: E501
            )
            return {
                "confidence_score": 0.5,
                "feedback": f"Error in evaluation: {str(e)}",
            }

    def _get_relevant_experiences(
        self, task: str, agent_name: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        logger.debug(
            f"Getting relevant experiences for task: '{task[:50]}...' for agent: {agent_name}"  # noqa: E501
        )
        try:
            task_embedding = get_embedding(task)
            if not task_embedding:
                logger.warning(
                    f"Could not generate embedding for task: {task}"
                )
                return []

            relevant_experiences = (
                self.knowledge_base.query_experiences_by_similarity(
                    task_embedding=task_embedding,
                    agent_name_filter=agent_name,  # Filter by agent_name
                    limit=limit,
                )
            )
            logger.debug(
                f"Found {len(relevant_experiences)} relevant experiences."
            )
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
            if isinstance(exp.get("content"), str):
                formatted.append(
                    f"Agent: {exp['agent_name']}\nTask: {exp['task']}\nOutput: {exp['content'][:100]}..."  # noqa: E501
                )
            else:
                formatted.append(
                    f"Agent: {exp['agent_name']}\nTask: {exp['task']}\nOutput: {str(exp.get('content', ''))[:100]}..."  # noqa: E501
                )

        return "\n".join(formatted)

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""

        formatted = []
        for mem in memories:
            formatted.append(
                f"Task: {mem.get('task', 'Unknown')}\nContent: {mem.get('content', '')[:100]}...\nTimestamp: {mem.get('timestamp', 'Unknown')}\n"  # noqa: E501
            )

        return "\n".join(formatted)

    def propose_new_agent(self, task: str) -> Dict[str, Any]:
        inspirator = self.meta_agents.get("InspiratorAgent")
        if not inspirator:
            logger.error("InspirationAgent not found")
            return {"error": "InspirationAgent not found"}

        prompt = (
            f"Analyze this task and propose a new specialized agent that would enhance the swarm's capabilities:\n\n"  # noqa: E501
            f"Task: {task}\n\n"
            f"Current agents: {', '.join(list(self.agents.keys()) + list(self.supervisors.keys()))}\n\n"  # noqa: E501
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
            r"Instructions:\s*(.+?)(?=Agent Type:|LLM Model Identifier:|Rationale:|$)",  # noqa: E501
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
                        f"InspiratorAgent suggested LLM tier '{suggested_model_tier}', defaulting to 'high' (xai/grok-3-latest)."  # noqa: E501
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

    def execute_subtask(
        self, agent_name: str, task_description: str, subtask_id: str
    ) -> Dict[str, Any]:
        logger.info(
            f"[{subtask_id}] Executing subtask by agent '{agent_name}': {task_description[:50]}..."  # noqa: E501
        )
        agent = self._get_agent(agent_name)

        if not agent:
            logger.error(
                f"[{subtask_id}] Agent '{agent_name}' not found for subtask execution."  # noqa: E501
            )
            error_message = f"Agent '{agent_name}' not found."
            self.cursor.execute(
                "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
                (
                    "failed",
                    error_message,
                    datetime.datetime.now().isoformat(),
                    subtask_id,
                ),
            )
            self.db_connection.commit()
            return {"error": error_message, "subtask_id": subtask_id}

        try:
            # Assuming the core work of a subtask agent is to process the description (e.g., via LLM)  # noqa: E501
            # We'll use _get_llm_response for now, as it's a common method for LLM-based agents.  # noqa: E501
            # If agents have a more general 'execute' or 'run_task' method, that could be used.  # noqa: E501
            # For non-LLM agents, this would need to be adapted.

            # Construct a prompt that is suitable for a direct task, if necessary.  # noqa: E501
            # For now, let's assume task_description is already well-formed for the agent.  # noqa: E501
            # If the agent has specific prompting needs, this might be where it's adapted.  # noqa: E501
            # Example: prompt = f"You are {agent.name}. Your task is: {task_description}"  # noqa: E501
            # result_data = agent._get_llm_response(prompt)

            result_data = agent._get_llm_response(
                task_description
            )  # Direct call for now

            logger.info(
                f"[{subtask_id}] Subtask executed by '{agent_name}'. Result: {str(result_data)[:100]}..."  # noqa: E501
            )

            # Determine if the result_data itself indicates an error from the agent  # noqa: E501
            if isinstance(result_data, dict) and result_data.get("error"):
                error_from_agent = str(result_data.get("error"))
                logger.error(
                    f"[{subtask_id}] Agent '{agent_name}' reported an error: {error_from_agent}"  # noqa: E501
                )
                self.cursor.execute(
                    "UPDATE tasks SET status = ?, result = ?, error_message = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
                    (
                        "failed",
                        json.dumps(result_data),
                        error_from_agent,
                        datetime.datetime.now().isoformat(),
                        subtask_id,
                    ),
                )
                final_status = {
                    "error": error_from_agent,
                    "subtask_id": subtask_id,
                    "output": result_data,
                }
            else:
                self.cursor.execute(
                    "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
                    (
                        "completed",
                        json.dumps(result_data),
                        datetime.datetime.now().isoformat(),
                        subtask_id,
                    ),
                )
                final_status = {
                    "success": True,
                    "subtask_id": subtask_id,
                    "output": result_data,
                }

            self.db_connection.commit()
            return final_status

        except Exception as e:
            logger.error(
                f"[{subtask_id}] Error during subtask execution by '{agent_name}': {e}",  # noqa: E501
                exc_info=True,
            )
            error_message = (
                f"Execution error by agent '{agent_name}': {str(e)}"
            )
            self.cursor.execute(
                "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
                (
                    "failed",
                    error_message,
                    datetime.datetime.now().isoformat(),
                    subtask_id,
                ),
            )
            self.db_connection.commit()
            return {"error": error_message, "subtask_id": subtask_id}

    def organize_task(self, task: str) -> Dict[str, Any]:
        # This method is now a legacy method or can be deprecated/refactored.
        organizer = self.meta_agents.get("OrganizerAgent")
        if not organizer:
            logger.error("OrganizerAgent not found")
            return {"error": "OrganizerAgent not found"}

        prompt = (
            f"Analyze this task and organize a workflow to accomplish it effectively:\n\n"  # noqa: E501
            f"Task: {task}\n\n"
            f"Available agents: {', '.join(list(self.agents.keys()) + list(self.supervisors.keys()))}\n\n"  # noqa: E501
            "Break down the task into subtasks and assign them to appropriate agents. "  # noqa: E501
            "For each subtask, specify:\n"
            "1. The agent to handle it\n"
            "2. The specific subtask description\n"
            "3. The order of execution\n\n"
            "Provide your workflow plan in JSON format."
        )

        try:
            logger.info(
                f"[{self.__class__.__name__}] About to call OrganizerAgent ({organizer.llm_model_identifier}) _get_llm_response for task: {task[:50]}..."  # noqa: E501
            )
            workflow_plan = organizer._get_llm_response(prompt)
            logger.info(
                f"[{self.__class__.__name__}] OrganizerAgent call returned. Workflow plan received (first 100 chars): {str(workflow_plan)[:100]}"  # noqa: E501
            )

            # Handle potential non-string responses from _get_llm_response if necessary  # noqa: E501
            if not isinstance(workflow_plan, str):
                logger.warning(
                    f"Unexpected response type from organizer._get_llm_response: {type(workflow_plan)}. Converting to string."  # noqa: E501
                )
                workflow_plan = str(workflow_plan)

            json_match = re.search(
                r"```json\n(.*?)\n```", workflow_plan, re.DOTALL
            )
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
                    f"Could not parse workflow JSON: {workflow_json}. Falling back."  # noqa: E501
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

            # Initialize parsed_workflow to ensure it's always defined
            parsed_workflow = workflow

            actual_steps = None
            if isinstance(parsed_workflow, dict):
                # Try common keys where the list of steps might be nested
                if "subtasks" in parsed_workflow and isinstance(
                    parsed_workflow["subtasks"], list
                ):
                    actual_steps = parsed_workflow["subtasks"]
                elif "steps" in parsed_workflow and isinstance(
                    parsed_workflow["steps"], list
                ):
                    actual_steps = parsed_workflow["steps"]
                elif "workflow" in parsed_workflow:
                    # If 'workflow' key exists, check if IT is the list or contains the list  # noqa: E501
                    if isinstance(parsed_workflow["workflow"], list):
                        actual_steps = parsed_workflow["workflow"]
                    elif isinstance(parsed_workflow["workflow"], dict):
                        if "subtasks" in parsed_workflow[
                            "workflow"
                        ] and isinstance(
                            parsed_workflow["workflow"]["subtasks"], list
                        ):
                            actual_steps = parsed_workflow["workflow"][
                                "subtasks"
                            ]
                        elif "steps" in parsed_workflow[
                            "workflow"
                        ] and isinstance(
                            parsed_workflow["workflow"]["steps"], list
                        ):
                            actual_steps = parsed_workflow["workflow"]["steps"]
            elif isinstance(parsed_workflow, list):
                # Handle case where the root JSON object IS the list of steps
                actual_steps = parsed_workflow

            if not actual_steps:
                logger.warning(
                    f"Could not extract a list of steps from the parsed workflow: {parsed_workflow}. Falling back to single step execution."  # noqa: E501
                )
                # Fallback: Treat the original task as a single step for the first available agent  # noqa: E501
                first_agent_name = next(
                    iter(self.agents), "GeographyAgent"
                )  # Default fallback agent
                actual_steps = [
                    {"step": 1, "agent": first_agent_name, "subtask": task}
                ]
                parsed_workflow = {
                    "subtasks": actual_steps
                }  # Ensure parsed_workflow variable holds the steps for later return  # noqa: E501
            # --- End Improved Extraction ---

            # Now execute the extracted steps
            logger.info(
                f"Executing workflow with {len(actual_steps)} steps extracted."
            )
            results: List[Dict[str, Any]] = self._execute_workflow(
                actual_steps, task
            )
            # Combine results (using the corrected _combine_results method)
            combined_result = self._combine_results(results, task)
            # Return success case inside the try block
            return {
                "status": "success",
                "workflow": workflow,  # Return the workflow structure used (might be the fallback)  # noqa: E501
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
                    "No 'order' or 'step' key found in steps. Assuming order as given."  # noqa: E501
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
            return {
                "combined_output": combined_input,
                "method": "simple_concatenation",
            }

        prompt = (
            f"Task: {task}\n\n"
            "Combine the following outputs:\n{combined_input}\n\nUnified Response:"  # noqa: E501
        )

        try:
            # Construct the full prompt including system instructions if available  # noqa: E501
            # Note: _get_llm_response in Agent class might need adjustment
            # if it doesn't inherently use agent.instructions as system prompt.
            # For now, assume _get_llm_response handles the full interaction.
            # A potential refinement could be passing system message explicitly if needed.  # noqa: E501

            combined_output = supervisor._get_llm_response(prompt)

            # Handle potential non-string responses from _get_llm_response if necessary  # noqa: E501
            if not isinstance(combined_output, str):
                logger.warning(
                    f"Unexpected response type from supervisor._get_llm_response: {type(combined_output)}. Converting to string."  # noqa: E501
                )
                combined_output = str(combined_output)

            return {
                "combined_output": combined_output,
                "method": f"combined_by_{supervisor.name}",
            }

        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return {"error": str(e), "method": "error_in_combination"}

    def _register_subtask(
        self,
        parent_task_id: str,
        subtask_description: str,
        assigned_agent_name: str,
    ) -> str:
        subtask_id = str(uuid.uuid4())
        now = datetime.datetime.now().isoformat()
        try:
            logger.info(
                f"Registering subtask {subtask_id} for parent {parent_task_id}. Assigned to: {assigned_agent_name}"  # noqa: E501
            )
            self.cursor.execute(
                "INSERT INTO tasks "
                "(task_id, description, status, created_at, updated_at, "
                "parent_task_id, is_subtask, assigned_agent_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    subtask_id,
                    subtask_description,
                    "queued",
                    now,
                    now,
                    parent_task_id,
                    1,
                    assigned_agent_name,
                ),  # is_subtask = 1 (True)
            )
            self.db_connection.commit()
            logger.info(f"Subtask {subtask_id} registered successfully.")
            return subtask_id
        except sqlite3.Error as e:
            logger.error(f"Database error registering subtask: {e}")
            return ""

    def run(self, task_description: str, task_id: str) -> Dict[str, Any]:
        """
        Main entry point for processing a task.
        This will typically involve organizing the task for supervision (decomposition).  # noqa: E501
        """
        logger.info(
            f"Swarm.run called for task_id: {task_id}, description: {task_description[:100]}..."  # noqa: E501
        )
        # For now, assume all tasks passed to Swarm.run are to be supervised and decomposed.  # noqa: E501
        # In the future, logic could be added here to differentiate simple vs. complex tasks.  # noqa: E501
        return self.organize_task_for_supervision(
            task=task_description, parent_task_id=task_id
        )

    def organize_task_for_supervision(
        self, task: str, parent_task_id: str
    ) -> Dict[str, Any]:
        """
        Orchestrates task decomposition using an OrganizerAgent, registers subtasks,  # noqa: E501
        updates the parent task status to 'awaiting_subtasks', and returns information about the subtasks.  # noqa: E501
        """
        logger.info(
            f"[{parent_task_id}] Supervisor: Organizing task for supervision: {task[:100]}..."  # noqa: E501
        )
        organizer = self.meta_agents.get("OrganizerAgent")
        if not organizer:
            logger.error(
                f"[{parent_task_id}] OrganizerAgent not found. Parent task {parent_task_id} will be marked as failed."  # noqa: E501
            )
            self.cursor.execute(
                "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
                (
                    "failed",
                    "OrganizerAgent not found for decomposition",
                    datetime.datetime.now().isoformat(),
                    parent_task_id,
                ),
            )
            self.db_connection.commit()
            return {"error": "OrganizerAgent not found"}

        # Construct a more detailed and prescriptive prompt for the OrganizerAgent  # noqa: E501
        worker_agent_names = [
            name
            for name, agent_obj in self.agents.items()
            if hasattr(agent_obj, "role") and agent_obj.role == "worker"
        ]
        supervisor_agent_names = [
            name
            for name, agent_obj in self.supervisors.items()
            if hasattr(agent_obj, "role") and agent_obj.role == "supervisor"
        ]

        # If no agents have roles defined, fall back to using all agents
        if not worker_agent_names and not supervisor_agent_names:
            worker_agent_names = list(self.agents.keys())
            supervisor_agent_names = list(self.supervisors.keys())

        assignable_agents = worker_agent_names + supervisor_agent_names

        if not assignable_agents:
            assignable_agents_str = "No specific worker/supervisor agents available. You may need to suggest generic roles or indicate if the task cannot be handled."  # noqa: E501
        else:
            assignable_agents_str = ", ".join(
                sorted(list(set(assignable_agents)))
            )  # Sort and unique

        prompt = (
            f"You are an Organizer Agent. Your primary function is to decompose a given complex task into a sequence of smaller, actionable subtasks. \n\n"  # noqa: E501
            f"YOU MUST FORMAT YOUR OUTPUT AS A SINGLE JSON OBJECT with this exact structure:\n"  # noqa: E501
            f"{{\n"
            f'  "subtasks": [\n'
            f'    {{"agent": "AgentName1", "subtask": "Description for subtask 1"}},\n'  # noqa: E501
            f'    {{"agent": "AgentName2", "subtask": "Description for subtask 2"}},\n'  # noqa: E501
            f"    ...\n"
            f"  ]\n"
            f"}}\n\n"
            f"IMPORTANT RULES:\n"
            f"1. Your output MUST be ONLY the JSON object with no additional text\n"  # noqa: E501
            f"2. Each subtask object MUST have exactly two keys: 'agent' and 'subtask'\n"  # noqa: E501
            f"3. The 'agent' value MUST be one of these exact names: {assignable_agents_str}\n"  # noqa: E501
            f"4. If you think a new agent type is needed, assign that subtask to a supervisor agent\n\n"  # noqa: E501
            f"MAIN TASK TO DECOMPOSE:\n{task}\n\n"
            f"AVAILABLE AGENTS: {assignable_agents_str}\n\n"
            f"Remember: Output ONLY the JSON object with the subtasks array.\n"
            f"If the task is simple, you can create just one subtask assigned to the most appropriate agent."  # noqa: E501
        )

        try:
            logger.info(
                f"[{parent_task_id}] About to call OrganizerAgent ({organizer.llm_model_identifier}) _get_llm_response for task: {task[:50]}..."  # noqa: E501
            )
            workflow_plan_str = organizer._get_llm_response(prompt)
            if not workflow_plan_str:
                logger.error(
                    f"[{parent_task_id}] OrganizerAgent returned an empty response. Falling back to single task execution."  # noqa: E501
                )
                raise ValueError("OrganizerAgent returned an empty plan")

            logger.info(
                f"[{parent_task_id}] OrganizerAgent call returned. Workflow plan received (first 200 chars): {str(workflow_plan_str)[:200]}"  # noqa: E501
            )

            # Handle potential non-string responses from _get_llm_response if necessary  # noqa: E501
            if not isinstance(workflow_plan_str, str):
                logger.warning(
                    f"[{parent_task_id}] Unexpected response type from organizer._get_llm_response: {type(workflow_plan_str)}. Converting to string."  # noqa: E501
                )
                workflow_plan_str = str(workflow_plan_str)

            # Enhanced JSON extraction with multiple patterns
            workflow_json_str = None

            # Try to find JSON in code blocks (```json ... ```)
            json_block_match = re.search(
                r"```(?:json)?\s*\n(.*?)\n\s*```",
                workflow_plan_str,
                re.DOTALL | re.IGNORECASE,
            )
            if json_block_match:
                workflow_json_str = json_block_match.group(1).strip()
                logger.info(
                    f"[{parent_task_id}] Extracted JSON from code block format"
                )

            # If not found in code blocks, try to find a JSON object directly
            if not workflow_json_str:
                # Look for a complete JSON object with balanced braces
                json_obj_match = re.search(
                    r"(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})",
                    workflow_plan_str,
                    re.DOTALL,
                )
                if json_obj_match:
                    workflow_json_str = json_obj_match.group(1).strip()
                    logger.info(
                        f"[{parent_task_id}] Extracted JSON using direct object pattern match"  # noqa: E501
                    )

            # If still not found, assume the entire response might be JSON (last resort)  # noqa: E501
            if not workflow_json_str:
                workflow_json_str = workflow_plan_str.strip()
                logger.info(
                    f"[{parent_task_id}] No specific JSON pattern found, attempting to parse entire response"  # noqa: E501
                )

            # Try to parse the extracted JSON string
            parsed_workflow = None
            try:
                parsed_workflow = json.loads(workflow_json_str)
                logger.info(
                    f"[{parent_task_id}] Successfully parsed JSON: {str(parsed_workflow)[:100]}..."  # noqa: E501
                )
            except json.JSONDecodeError as e:
                logger.error(
                    f"[{parent_task_id}] JSON parsing error: {e}. Workflow JSON: {workflow_json_str[:200]}..."  # noqa: E501
                )

                # Attempt to fix common JSON issues
                fixed_json = False

                # Try adding missing quotes around keys
                try:
                    fixed_json_str = re.sub(
                        r"([{,])\s*([a-zA-Z0-9_]+)\s*:",
                        r'\1"\2":',
                        workflow_json_str,
                    )
                    parsed_workflow = json.loads(fixed_json_str)
                    fixed_json = True
                    logger.info(
                        f"[{parent_task_id}] Fixed JSON by adding quotes around keys"  # noqa: E501
                    )
                except (json.JSONDecodeError, re.error) as e:
                    logger.warning(
                        f"[{parent_task_id}] Failed to fix JSON: {e}"
                    )
                    pass

                # If still not fixed, fall back to a default structure
                if not fixed_json:
                    logger.warning(
                        f"[{parent_task_id}] Could not fix JSON. Using fallback structure."  # noqa: E501
                    )
                    # Fallback: Create a default workflow with the original task assigned to the first available agent  # noqa: E501
                    first_agent_name = next(
                        iter(self.agents), "GeographyAgent"
                    )
                    parsed_workflow = {
                        "subtasks": [
                            {"agent": first_agent_name, "subtask": task}
                        ]
                    }

            # Extract subtasks from the parsed workflow
            actual_steps = None
            if isinstance(parsed_workflow, dict):
                # Try common keys where the list of steps might be nested
                if "subtasks" in parsed_workflow and isinstance(
                    parsed_workflow["subtasks"], list
                ):
                    actual_steps = parsed_workflow["subtasks"]
                    logger.info(
                        f"[{parent_task_id}] Found subtasks list with {len(actual_steps)} items"  # noqa: E501
                    )
                elif "steps" in parsed_workflow and isinstance(
                    parsed_workflow["steps"], list
                ):
                    actual_steps = parsed_workflow["steps"]
                    logger.info(
                        f"[{parent_task_id}] Found steps list with {len(actual_steps)} items"  # noqa: E501
                    )
                elif "workflow" in parsed_workflow:
                    # If 'workflow' key exists, check if IT is the list or contains the list  # noqa: E501
                    if isinstance(parsed_workflow["workflow"], list):
                        actual_steps = parsed_workflow["workflow"]
                        logger.info(
                            f"[{parent_task_id}] Found workflow list with {len(actual_steps)} items"  # noqa: E501
                        )
                    elif isinstance(parsed_workflow["workflow"], dict):
                        if "subtasks" in parsed_workflow[
                            "workflow"
                        ] and isinstance(
                            parsed_workflow["workflow"]["subtasks"], list
                        ):
                            actual_steps = parsed_workflow["workflow"][
                                "subtasks"
                            ]
                            logger.info(
                                f"[{parent_task_id}] Found nested subtasks list with {len(actual_steps)} items"  # noqa: E501
                            )
                        elif "steps" in parsed_workflow[
                            "workflow"
                        ] and isinstance(
                            parsed_workflow["workflow"]["steps"], list
                        ):
                            actual_steps = parsed_workflow["workflow"]["steps"]
                            logger.info(
                                f"[{parent_task_id}] Found nested steps list with {len(actual_steps)} items"  # noqa: E501
                            )
            elif isinstance(parsed_workflow, list):
                # Handle case where the root JSON object IS the list of steps
                actual_steps = parsed_workflow
                logger.info(
                    f"[{parent_task_id}] Found root list with {len(actual_steps)} items"  # noqa: E501
                )

            if not actual_steps:
                logger.warning(
                    f"[{parent_task_id}] Could not extract a list of steps from the parsed workflow: {parsed_workflow}. Falling back to single step execution."  # noqa: E501
                )
                # Fallback: Treat the original task as a single step for the first available agent  # noqa: E501
                first_agent_name = next(
                    iter(self.agents), "GeographyAgent"
                )  # Default fallback agent
                actual_steps = [
                    {"step": 1, "agent": first_agent_name, "subtask": task}
                ]
                parsed_workflow = {
                    "subtasks": actual_steps
                }  # Ensure parsed_workflow variable holds the steps for later return  # noqa: E501

            # Register subtasks
            subtask_ids = []
            for step in actual_steps:
                # Ensure step has the required fields
                if (
                    not isinstance(step, dict)
                    or "agent" not in step
                    or "subtask" not in step
                ):
                    if (
                        "step" in step
                        and "agent" in step
                        and "subtask" in step
                    ):
                        # Handle the case where we have {"step": 1, "agent": "...", "subtask": "..."}  # noqa: E501
                        agent_name = step["agent"]
                        subtask_description = step["subtask"]
                    else:
                        logger.warning(
                            f"[{parent_task_id}] Invalid step format: {step}. Skipping."  # noqa: E501
                        )
                        continue
                else:
                    agent_name = step["agent"]
                    subtask_description = step["subtask"]

                # Validate agent exists
                if not (
                    agent_name in self.agents
                    or agent_name in self.supervisors
                    or agent_name in self.meta_agents
                ):
                    logger.warning(
                        f"[{parent_task_id}] Agent '{agent_name}' not found. Assigning to first available agent."  # noqa: E501
                    )
                    agent_name = next(iter(self.agents), None)
                    if not agent_name:
                        logger.error(
                            f"[{parent_task_id}] No agents available to assign subtask. Skipping."  # noqa: E501
                        )
                        continue

                subtask_id = self._register_subtask(
                    parent_task_id, subtask_description, agent_name
                )
                if subtask_id:
                    subtask_ids.append(subtask_id)
                    logger.info(
                        f"[{parent_task_id}] Registered subtask {subtask_id} for agent '{agent_name}': {subtask_description[:50]}..."  # noqa: E501
                    )
                else:
                    logger.error(
                        f"[{parent_task_id}] Failed to register subtask for step: {step}"  # noqa: E501
                    )

            if not subtask_ids:
                logger.error(
                    f"[{parent_task_id}] No subtasks were successfully registered. Task cannot proceed."  # noqa: E501
                )
                raise ValueError("No subtasks registered from the plan")

            # IMPORTANT: Update parent task status to 'awaiting_subtasks' instead of 'running'  # noqa: E501
            self.cursor.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
                (
                    "awaiting_subtasks",
                    datetime.datetime.now().isoformat(),
                    parent_task_id,
                ),
            )
            self.db_connection.commit()
            logger.info(
                f"[{parent_task_id}] Parent task status updated to 'awaiting_subtasks'. {len(subtask_ids)} subtasks created."  # noqa: E501
            )

            return {
                "status": "decomposed",
                "message": f"Task decomposed into {len(subtask_ids)} subtasks",
                "parent_task_id": parent_task_id,
                "subtask_ids": subtask_ids,
                "workflow": parsed_workflow,
            }
        except Exception as e:
            logger.error(
                f"[{parent_task_id}] Error organizing task for supervision: {e}",  # noqa: E501
                exc_info=True,
            )
            # Update parent task to failed if decomposition fails critically
            try:
                self.cursor.execute(
                    "UPDATE tasks SET status = ?, error_message = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
                    (
                        "failed",
                        f"Failed to decompose task: {str(e)}",
                        datetime.datetime.now().isoformat(),
                        parent_task_id,
                    ),
                )
                self.db_connection.commit()
            except Exception as db_err:
                logger.error(
                    f"[{parent_task_id}] Additionally, DB error while marking task as failed: {db_err}"  # noqa: E501
                )

            return {
                "status": "error",
                "message": f"Failed to organize task: {str(e)}",
            }
