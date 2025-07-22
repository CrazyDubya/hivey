import json
import logging
import os
import sqlite3
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, AsyncGenerator, Dict, List, Optional

import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from swarms import DB_NAME, Swarm

# Load environment variables
dotenv_path = find_dotenv(usecwd=True) or find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path, override=True)

# --- Configure Logging ---
LOG_FILE = "swarm_service.log"

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers (if any from previous runs or imports)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Standard formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# StreamHandler for stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# RotatingFileHandler for the log file
# Max 1MB per file, keep 3 backup files
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=1 * 1024 * 1024, backupCount=3
)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Configure specific loggers if needed, or they will inherit from root
logger = logging.getLogger(__name__)  # This will use the root logger's config

# --- Globals and Configuration ---
EXPECTED_API_KEY = os.getenv("SWARM_API_KEY")
if not EXPECTED_API_KEY:
    logger.warning(
        "SWARM_API_KEY not loaded from environment. Authentication will fail."
    )
else:
    logger.info(
        f"SWARM_API_KEY successfully loaded. Length: {len(EXPECTED_API_KEY)}"
    )

# --- Global Variables (Simpler for now, consider dependency injection later) ---  # noqa: E501
swarm_instance = None
main_db_connection = None

# --- API Key Configuration ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
) -> str:
    """Dependency function to validate the API key."""
    if not EXPECTED_API_KEY:
        # Log error if key wasn't configured on server side
        logger.error(
            "API key check failed: SWARM_API_KEY not configured on server."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on server",
        )

    if api_key_header == EXPECTED_API_KEY:
        return api_key_header
    else:
        logger.warning(f"Invalid API Key received: {api_key_header}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )


# --- Lifespan Management for Startup and Shutdown ---
SUPERVISOR_INTERVAL_SECONDS = 10  # Check every 10 seconds
shutdown_event = threading.Event()


def periodic_supervisor_check_loop():
    """Periodically calls the supervisor check functions."""
    logger.info("Supervisor thread started.")
    while not shutdown_event.is_set():
        try:
            # First process any pending subtasks
            logger.debug(
                f"Supervisor loop: Calling process_pending_subtasks() at {datetime.now()}."  # noqa: E501
            )
            process_pending_subtasks()

            # Then check for stalled subtasks
            logger.debug(
                f"Supervisor loop: Calling check_for_stalled_subtasks() at {datetime.now()}."  # noqa: E501
            )
            check_for_stalled_subtasks()

            # Then check if any parent tasks are complete
            logger.debug(
                f"Supervisor loop: Calling check_and_collate_supervised_tasks() at {datetime.now()}."  # noqa: E501
            )
            check_and_collate_supervised_tasks()
        except Exception as e:
            logger.error(
                f"Supervisor loop: Unhandled exception: {e}", exc_info=True
            )
        # Wait for the interval or until shutdown is signaled
        shutdown_event.wait(SUPERVISOR_INTERVAL_SECONDS)
    logger.info("Supervisor thread finished.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup sequence
    global swarm_instance, main_db_connection
    logger.info(
        "Lifespan: Initializing Swarm service... (API key should already be loaded)"  # noqa: E501
    )

    try:
        db_file_path = os.path.abspath(DB_NAME)
        logger.info(
            f"Lifespan: Attempting to connect to database: {db_file_path}"
        )
        # Use check_same_thread=False for SQLite with FastAPI/multi-threaded access  # noqa: E501
        main_db_connection = sqlite3.connect(DB_NAME, check_same_thread=False)
        logger.info(
            f"Lifespan: Database connection successful to {db_file_path}"
        )

        # Initialize Swarm with the shared connection
        swarm_instance = Swarm(db_connection=main_db_connection)
        logger.info("Lifespan: Swarm initialized successfully.")

        # Start supervisor thread
        shutdown_event.clear()  # Ensure it's not set from a previous run (if any weird state)  # noqa: E501
        supervisor_thread = threading.Thread(
            target=periodic_supervisor_check_loop, daemon=True
        )
        supervisor_thread.start()
        logger.info("Lifespan: Supervisor thread started.")

    except sqlite3.Error as e:
        logger.error(f"Lifespan: Database connection failed: {e}")
        raise

    except Exception as e:
        logger.error(f"Lifespan: Swarm initialization failed: {e}")
        raise

    yield  # Application runs here

    # Shutdown sequence
    logger.info("Lifespan: Initiating shutdown sequence...")
    shutdown_event.set()  # Signal supervisor thread to stop
    # supervisor_thread.join() # Optionally wait for thread to finish cleanly, might delay shutdown  # noqa: E501
    logger.info("Lifespan: Shutdown event set for supervisor thread.")

    if main_db_connection:
        logger.info("Lifespan: Closing database connection on shutdown.")
        main_db_connection.close()
        main_db_connection = None
    logger.info("Lifespan: Service shutdown complete.")


# Create the FastAPI app instance with the lifespan manager
app = FastAPI(title="SwarmMind Service", lifespan=lifespan)
print("-----> FastAPI app object CREATED <-----")


# Define /hello immediately to test if it registers on the app object Uvicorn uses  # noqa: E501
@app.get("/hello")
async def hello_world():
    print("-----> HELLO ENDPOINT (HANDLER) REACHED AND EXECUTING <-----")
    return {"message": "Hello, world!"}


print("-----> /hello route DEFINED ON APP <-----")


# --- Pydantic Models for API Data ---
class TaskRequest(BaseModel):
    task_description: str

    class Config:
        str_strip_whitespace = True
        str_min_length = 1
        str_max_length = 10000


class TaskResponse(BaseModel):
    task_id: str


# New response model for checking task status
class TaskStatusResponse(BaseModel):
    task_id: str
    description: Optional[str] = None  # Made optional for flexibility
    status: str
    result: Optional[Any] = None  # Can be any type, like dict or str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    parent_task_id: Optional[str] = None
    is_subtask: bool
    assigned_agent_name: Optional[str] = None


class TaskStatusOnlyResponse(BaseModel):
    task_id: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# New model for task summary including description
class TaskSummaryResponse(BaseModel):
    task_id: str
    description: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# --- Database Helper Functions for Tasks ---
def _get_db_cursor() -> Optional[sqlite3.Cursor]:
    """Helper to get a cursor from the global connection."""
    if not main_db_connection:
        logger.error("Database connection is not available.")
        raise HTTPException(
            status_code=503, detail="Database connection not available"
        )
    return main_db_connection.cursor()


def create_task_record(
    task_id: str,
    description: str,
    parent_task_id: Optional[str] = None,  # New parameter
    is_subtask: bool = False,  # New parameter
) -> None:
    """Adds a new task to the database with 'queued' status."""
    now = datetime.now().isoformat()
    db_cursor = _get_db_cursor()  # Use the helper to get a fresh cursor
    try:
        logger.info(
            f"Attempting to insert task {task_id} into database. Parent: {parent_task_id}, Is Subtask: {is_subtask}"  # noqa: E501
        )
        db_cursor.execute(
            "INSERT INTO tasks (task_id, description, status, created_at, updated_at, parent_task_id, is_subtask) VALUES (?, ?, ?, ?, ?, ?, ?)",  # noqa: E501
            (
                task_id,
                description,
                "queued",
                now,
                now,
                parent_task_id,
                is_subtask,
            ),
        )
        logger.info(
            f"Task {task_id} insert executed. Cursor rowcount: {db_cursor.rowcount if db_cursor else 'N/A'}"  # noqa: E501
        )
        main_db_connection.commit()
        logger.info(f"Database commit successful for task {task_id}.")
    except sqlite3.Error as e:
        logger.error(
            f"Database error creating task {task_id}: {e}", exc_info=True
        )
        # Optionally re-raise or handle, depending on desired error flow
        # For now, logging the error. The submit_task's try-except will catch higher-level issues.  # noqa: E501
    except Exception as e:  # Catch any other unexpected errors
        logger.error(
            f"Unexpected error in create_task_record for task {task_id}: {e}",
            exc_info=True,
        )


def update_task_status(
    task_id: str,
    status: str,
    result: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Updates the status and result/error of a task."""
    now = datetime.now().isoformat()
    cursor = _get_db_cursor()
    try:
        cursor.execute(
            "UPDATE tasks SET status = ?, result = ?, error_message = ?, updated_at = ? WHERE task_id = ?",  # noqa: E501
            (status, result, error_message, now, task_id),
        )
        if main_db_connection:
            main_db_connection.commit()
        logger.info(f"Task {task_id} status updated to {status}.")
    except sqlite3.Error as e:
        logger.error(f"Failed to update task status for {task_id}: {e}")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves task details from the database."""
    cursor = _get_db_cursor()
    try:
        cursor.execute(
            "SELECT task_id, description, status, result, error_message, "
            "created_at, updated_at, parent_task_id, is_subtask, "
            "assigned_agent_name FROM tasks WHERE task_id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    except sqlite3.Error as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        return None


def get_task_status_summary(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves only the status-related fields for a task from the database."""  # noqa: E501
    cursor = _get_db_cursor()
    if not cursor:
        logger.error(
            f"Failed to get DB cursor for get_task_status_summary(task_id={task_id})"  # noqa: E501
        )
        return None
    try:
        cursor.execute(
            "SELECT task_id, description, status, error_message, created_at, updated_at FROM tasks WHERE task_id = ?",  # noqa: E501
            (task_id,),
        )
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    except sqlite3.Error as e:
        logger.error(
            f"Database error in get_task_status_summary for {task_id}: {e}",
            exc_info=True,
        )
        return None


def get_all_tasks_summary() -> List[Dict[str, Any]]:
    """Retrieves status-related fields for all tasks from the database, ordered by creation date descending."""  # noqa: E501
    cursor = _get_db_cursor()
    if not cursor:
        logger.error("Failed to get DB cursor for get_all_tasks_summary")
        return []
    try:
        cursor.execute(
            "SELECT task_id, status, error_message, created_at, updated_at FROM tasks ORDER BY created_at DESC"  # noqa: E501
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    except sqlite3.Error as e:
        logger.error(
            f"Database error in get_all_tasks_summary: {e}", exc_info=True
        )
        return []


# --- Background Task Worker ---
def run_swarm_task(task_id: str, description: str) -> None:
    """The actual function that runs the swarm task in the background."""
    logger.info(
        f"[{task_id}] Background processing initiated for: {description[:50]}..."  # noqa: E501
    )
    # Initial status update to 'running' still makes sense.
    update_task_status(task_id, "running")
    logger.info(f"[{task_id}] Status updated to running.")

    if not swarm_instance:
        logger.error(f"[{task_id}] Swarm instance not available.")
        update_task_status(
            task_id, "failed", error_message="Swarm service not initialized"
        )
        return

    try:
        # Fetch full task details to check if it's a subtask
        task_details = get_task_status(task_id)
        if not task_details:
            logger.error(
                f"[{task_id}] Failed to retrieve task details for processing."
            )
            update_task_status(
                task_id,
                "failed",
                error_message="Could not retrieve task details.",
            )
            return

        current_task_description = task_details.get(
            "description", description
        )  # Use fetched desc if available
        is_subtask = task_details.get("is_subtask", False)
        assigned_agent_name = task_details.get("assigned_agent_name")

        if is_subtask:
            logger.info(
                f"[{task_id}] Is a subtask. Assigned to: {assigned_agent_name}. Description: {current_task_description[:50]}..."  # noqa: E501
            )
            if not assigned_agent_name:
                logger.error(
                    f"[{task_id}] Subtask has no assigned_agent_name."
                )
                update_task_status(
                    task_id,
                    "failed",
                    error_message="Subtask is missing an assigned agent.",
                )
                return

            # Call the new method for direct subtask execution
            # This method will be responsible for its own status updates upon completion/failure  # noqa: E501
            result_dict = swarm_instance.execute_subtask(
                agent_name=assigned_agent_name,
                task_description=current_task_description,
                subtask_id=task_id,
            )
            # execute_subtask should handle updating its own status internally.
            # For now, let's assume it returns a similar dict to organize_task/run for consistency if needed.  # noqa: E501
            # Log the result for debugging
            logger.info(
                f"Subtask execution result: {str(result_dict)[:100]}..."
            )
            # else:
            #    update_task_status(task_id, "completed", result=json.dumps(subtask_result_dict))  # noqa: E501
            logger.info(
                f"[{task_id}] Subtask execution handled by swarm_instance.execute_subtask."  # noqa: E501
            )

        else:  # Parent task or simple task (not a subtask)
            logger.info(
                f"[{task_id}] Is a parent or simple task. Description: {current_task_description[:50]}..."  # noqa: E501
            )
            # This will call Swarm.run(), which internally calls organize_task_for_supervision()  # noqa: E501
            # organize_task_for_supervision() will update the parent task's status to 'awaiting_subtasks' if it decomposes.  # noqa: E501
            task_output_dict = swarm_instance.run(
                current_task_description, task_id
            )

            # If 'subtask_ids' are returned, it means it was decomposed and parent task status is already 'awaiting_subtasks'.  # noqa: E501
            # In this case, run_swarm_task for THIS parent task is done for now.  # noqa: E501
            if task_output_dict and "subtask_ids" in task_output_dict:
                logger.info(
                    f"[{task_id}] Task decomposed into subtasks: {task_output_dict.get('subtask_ids')}. Parent status set to 'awaiting_subtasks' by Swarm.run."  # noqa: E501
                )
                # No further status update needed here for the parent task, as Swarm.run handled it.  # noqa: E501
            elif task_output_dict and "error" in task_output_dict:
                logger.error(
                    f"[{task_id}] Error returned by swarm_instance.run: {task_output_dict['error']}"  # noqa: E501
                )
                update_task_status(
                    task_id,
                    "failed",
                    result=json.dumps(
                        task_output_dict
                    ),  # Store the full output as result
                    error_message=str(task_output_dict["error"]),
                )
                logger.info(f"[{task_id}] Task status updated to failed.")
            elif (
                task_output_dict
            ):  # Assume it's a direct result for a simple, non-decomposed task
                logger.info(
                    f"[{task_id}] Task processed directly by Swarm.run (no decomposition). Output: {str(task_output_dict)[:100]}..."  # noqa: E501
                )
                update_task_status(
                    task_id,
                    "completed",
                    result=json.dumps(
                        task_output_dict
                    ),  # Store the full output as result
                )
                logger.info(f"[{task_id}] Task status updated to completed.")
            else:
                # This case should ideally not be reached if Swarm.run always returns a dict  # noqa: E501
                logger.warning(
                    f"[{task_id}] swarm_instance.run returned an unexpected or None value. Marking as failed."  # noqa: E501
                )
                update_task_status(
                    task_id,
                    "failed",
                    error_message="Swarm.run returned unexpected output.",
                )

    except Exception as e:
        logger.error(
            f"[{task_id}] Unexpected error in run_swarm_task: {e}",
            exc_info=True,
        )
        update_task_status(
            task_id,
            "failed",
            error_message=f"Internal error during task execution: {str(e)}",
        )
        logger.info(
            f"[{task_id}] Task status updated to failed due to unexpected error."  # noqa: E501
        )


# --- Supervisor Logic for Collating Subtasks ---
def process_pending_subtasks():
    """Process any subtasks that are in 'queued' status."""
    if not swarm_instance or not main_db_connection:
        logger.debug("Supervisor: Swarm instance or DB connection not ready.")
        return

    logger.debug("Supervisor: Checking for pending subtasks...")
    db_cursor = _get_db_cursor()

    try:
        # Find subtasks that are in 'queued' status
        db_cursor.execute(
            "SELECT task_id, description, assigned_agent_name FROM tasks WHERE status = 'queued' AND is_subtask = 1"  # noqa: E501
        )
        pending_subtasks = db_cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(
            f"Supervisor: DB error fetching pending subtasks: {e}",
            exc_info=True,
        )
        return

    if not pending_subtasks:
        logger.debug("Supervisor: No pending subtasks to process.")
        return

    logger.info(
        f"Supervisor: Found {len(pending_subtasks)} pending subtasks to process."  # noqa: E501
    )
    for subtask_row in pending_subtasks:
        subtask_id, subtask_description, assigned_agent_name = subtask_row
        logger.info(
            f"Supervisor: Processing pending subtask {subtask_id} assigned to {assigned_agent_name}."  # noqa: E501
        )

        try:
            # Update status to 'running' before execution
            update_task_status(subtask_id, "running")

            # Start a new thread to execute the subtask
            thread = threading.Thread(
                target=run_swarm_task,
                args=(subtask_id, subtask_description),
                daemon=True,
            )
            thread.start()
            logger.info(f"Supervisor: Started thread for subtask {subtask_id}")
        except Exception as e:
            logger.error(
                f"Supervisor: Error starting thread for subtask {subtask_id}: {e}",  # noqa: E501
                exc_info=True,
            )
            update_task_status(
                subtask_id,
                "failed",
                error_message=f"Failed to start execution: {str(e)}",
            )

    logger.debug("Supervisor: Finished processing pending subtasks.")


def check_for_stalled_subtasks():
    """Check for subtasks that have been in the running state for too long and mark them as failed."""  # noqa: E501
    if not main_db_connection:
        logger.debug("Supervisor: DB connection not ready.")
        return

    logger.debug("Supervisor: Checking for stalled subtasks...")
    db_cursor = _get_db_cursor()

    # Define the timeout threshold (30 minutes)
    timeout_minutes = 30
    timeout_threshold = datetime.now() - timedelta(minutes=timeout_minutes)
    timeout_iso = timeout_threshold.isoformat()

    try:
        # Find subtasks that have been running for too long
        db_cursor.execute(
            """SELECT task_id, description, assigned_agent_name, updated_at
               FROM tasks
               WHERE status = 'running'
               AND is_subtask = 1
               AND updated_at < ?""",
            (timeout_iso,),
        )
        stalled_subtasks = db_cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(
            f"Supervisor: DB error fetching stalled subtasks: {e}",
            exc_info=True,
        )
        return

    if not stalled_subtasks:
        logger.debug("Supervisor: No stalled subtasks found.")
        return

    logger.info(f"Supervisor: Found {len(stalled_subtasks)} stalled subtasks.")
    for subtask_row in stalled_subtasks:
        subtask_id, subtask_description, assigned_agent_name, updated_at = (
            subtask_row
        )
        logger.warning(
            f"Supervisor: Found stalled subtask {subtask_id} assigned to {assigned_agent_name}. Last updated: {updated_at}"  # noqa: E501
        )

        # Mark the subtask as failed
        error_message = (
            f"Subtask timed out after {timeout_minutes} minutes of inactivity."
        )
        update_task_status(subtask_id, "failed", error_message=error_message)
        logger.info(
            f"Supervisor: Marked stalled subtask {subtask_id} as failed due to timeout."  # noqa: E501
        )


def check_and_collate_supervised_tasks():
    if not swarm_instance or not main_db_connection:
        logger.debug("Supervisor: Swarm instance or DB connection not ready.")
        return

    logger.debug(
        "Supervisor: Checking for tasks awaiting subtask completion..."
    )
    db_cursor = _get_db_cursor()

    try:
        db_cursor.execute(
            "SELECT task_id, description FROM tasks WHERE status = 'awaiting_subtasks' AND is_subtask = 0"  # noqa: E501
        )
        parent_tasks_awaiting = db_cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(
            f"Supervisor: DB error fetching parent tasks: {e}", exc_info=True
        )
        return

    if not parent_tasks_awaiting:
        logger.debug(
            "Supervisor: No tasks currently awaiting subtask completion."
        )
        return

    for parent_task_row in parent_tasks_awaiting:
        parent_task_id = parent_task_row[0]
        parent_task_description = parent_task_row[1]
        logger.info(
            f"Supervisor: Checking parent task {parent_task_id} ('{parent_task_description[:50]}...')"  # noqa: E501
        )

        try:
            db_cursor.execute(
                "SELECT task_id, status, result, error_message FROM tasks WHERE parent_task_id = ? AND is_subtask = 1",  # noqa: E501
                (parent_task_id,),
            )
            subtasks = db_cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(
                f"Supervisor: DB error fetching subtasks for {parent_task_id}: {e}",  # noqa: E501
                exc_info=True,
            )
            continue  # Move to next parent task

        if not subtasks:
            logger.warning(
                f"Supervisor: Parent task {parent_task_id} is 'awaiting_subtasks' but has no subtasks found. Marking as error."  # noqa: E501
            )
            update_task_status(
                parent_task_id,
                "failed",
                error_message="Awaited subtasks but none found.",
            )
            continue

        all_subtasks_finished = True
        subtask_results_for_collation = []
        failed_subtask_errors = []

        for sub_row in subtasks:
            sub_task_id, sub_status, sub_result_json, sub_error_msg = sub_row
            if sub_status not in ("completed", "failed"):
                all_subtasks_finished = False
                logger.debug(
                    f"Supervisor: Parent {parent_task_id} still waiting for subtask {sub_task_id} (status: {sub_status})."  # noqa: E501
                )
                break  # Not all subtasks are done for this parent

            if sub_status == "completed":
                try:
                    # First, check if sub_result_json is already a dictionary or a string  # noqa: E501
                    if isinstance(sub_result_json, dict):
                        # If it's already a dict, use it directly
                        sub_result_data = sub_result_json
                    else:
                        # If it's a string, try to parse it as JSON
                        try:
                            sub_result_data = (
                                json.loads(sub_result_json)
                                if sub_result_json
                                else {}
                            )
                        except json.JSONDecodeError:
                            # If it's not valid JSON, just use the string directly  # noqa: E501
                            logger.warning(
                                f"Supervisor: Could not parse subtask result as JSON: {sub_result_json[:100]}..."  # noqa: E501
                            )
                            subtask_results_for_collation.append(
                                sub_result_json
                            )
                            continue

                    # If we have a dictionary, extract the output
                    if isinstance(sub_result_data, dict):
                        actual_agent_output = sub_result_data.get("output")
                        if actual_agent_output:
                            # Try to parse the output as JSON if it's a string
                            if isinstance(actual_agent_output, str):
                                try:
                                    parsed_agent_output = json.loads(
                                        actual_agent_output
                                    )
                                    subtask_results_for_collation.append(
                                        parsed_agent_output
                                    )
                                except json.JSONDecodeError:
                                    # If it's not valid JSON, use the string directly  # noqa: E501
                                    subtask_results_for_collation.append(
                                        actual_agent_output
                                    )
                            else:
                                # If it's already a dict or other type, use it directly  # noqa: E501
                                subtask_results_for_collation.append(
                                    actual_agent_output
                                )
                        else:
                            # No output field, use the whole result
                            subtask_results_for_collation.append(
                                sub_result_data
                            )
                    else:
                        # If it's not a dict, use it directly
                        subtask_results_for_collation.append(sub_result_data)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Supervisor: Error decoding result JSON for subtask {sub_task_id}: {e}. Original: {sub_result_json}"  # noqa: E501
                    )
                    subtask_results_for_collation.append(
                        {
                            "error": "Failed to parse subtask result",
                            "original_result": sub_result_json,
                        }
                    )
            elif sub_status == "failed":
                failed_subtask_errors.append(
                    f"Subtask {sub_task_id} failed: {sub_error_msg or 'Unknown error'}"  # noqa: E501
                )

        if all_subtasks_finished:
            logger.info(
                f"Supervisor: All subtasks finished for parent {parent_task_id}."  # noqa: E501
            )
            if failed_subtask_errors:
                parent_error_message = (
                    "One or more subtasks failed: "
                    + "; ".join(failed_subtask_errors)
                )
                logger.error(
                    f"Supervisor: Parent task {parent_task_id} failed due to subtask errors: {parent_error_message}"  # noqa: E501
                )
                update_task_status(
                    parent_task_id,
                    "failed",
                    error_message=parent_error_message,
                )
            else:
                try:
                    logger.info(
                        f"Supervisor: Collating {len(subtask_results_for_collation)} results for parent {parent_task_id}."  # noqa: E501
                    )
                    # We need to ensure swarm_instance._combine_results is thread-safe if called from a separate thread.  # noqa: E501
                    # The Swarm class uses a single DB connection/cursor which might be an issue.  # noqa: E501
                    # For now, assuming it's okay or this runs in a way that serializes calls.  # noqa: E501
                    combined_output = swarm_instance._combine_results(
                        subtask_results_for_collation, parent_task_description
                    )

                    final_result_json = json.dumps(combined_output)
                    update_task_status(
                        parent_task_id, "completed", result=final_result_json
                    )
                    logger.info(
                        f"Supervisor: Parent task {parent_task_id} completed successfully after collation."  # noqa: E501
                    )
                except Exception as e:
                    collation_error_msg = f"Error during result collation for {parent_task_id}: {e}"  # noqa: E501
                    logger.error(
                        f"Supervisor: {collation_error_msg}", exc_info=True
                    )
                    update_task_status(
                        parent_task_id,
                        "failed",
                        error_message=collation_error_msg,
                    )
        else:
            logger.debug(
                f"Supervisor: Parent {parent_task_id} still has pending subtasks."  # noqa: E501
            )

    logger.debug(
        "Supervisor: Finished checking tasks awaiting subtask completion."
    )


# --- Core API Endpoints ---
@app.post(
    "/tasks", response_model=TaskResponse, dependencies=[Security(get_api_key)]
)
async def submit_task(
    request: TaskRequest, background_tasks: BackgroundTasks
) -> TaskResponse:
    logger.info(
        f"submit_task entered. Request description (first 50 chars): {request.task_description[:50]}..."  # noqa: E501
    )
    task_id = str(uuid.uuid4())
    logger.info(
        f"Generated task_id: {task_id} for request: {request.task_description[:50]}..."  # noqa: E501
    )

    try:
        logger.info(f"Attempting to create task record for task_id: {task_id}")
        create_task_record(task_id, request.task_description)
        logger.info(
            f"Task record created for task_id: {task_id}. Adding to background tasks."  # noqa: E501
        )

        background_tasks.add_task(
            run_swarm_task, task_id, request.task_description
        )
        logger.info(f"Task {task_id} added to background_tasks.")

        logger.info(f"Returning TaskResponse for task_id: {task_id}")
        return TaskResponse(task_id=task_id)
    except Exception as e:
        logger.error(
            f"UNEXPECTED ERROR IN SUBMIT_TASK for task_id {task_id if 'task_id' in locals() else 'UNKNOWN'}: {e}",  # noqa: E501
            exc_info=True,
        )
        # Re-raise or return an error response if an exception occurs
        # Consider if you want to expose full error 'e' to client or a generic message  # noqa: E501
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during task submission: {str(e)}",
        )


@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    dependencies=[Security(get_api_key)],
)
async def get_task_details(task_id: str) -> TaskStatusResponse:
    """Retrieves the status and result of a specific task (requires API Key)."""  # noqa: E501
    logger.info(f"Received request for task status: {task_id}")
    task_data = get_task_status(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    # Convert the dictionary to the Pydantic model
    # Handle potential None for result/error_message if DB stores NULL
    return TaskStatusResponse(
        task_id=task_data["task_id"],
        description=task_data.get(
            "description"
        ),  # Use .get for optional field
        status=task_data["status"],
        result=task_data.get("result"),
        error_message=task_data.get("error_message"),
        created_at=task_data["created_at"],
        updated_at=task_data["updated_at"],
        parent_task_id=task_data.get("parent_task_id"),
        is_subtask=task_data["is_subtask"],
        assigned_agent_name=task_data.get("assigned_agent_name"),
    )


@app.get(
    "/tasks/all/summary",
    response_model=List[TaskStatusOnlyResponse],
    dependencies=[Security(get_api_key)],
)
async def get_all_tasks_list_summary() -> List[TaskStatusOnlyResponse]:
    tasks_data = get_all_tasks_summary()
    if not tasks_data:
        # Return empty list if no tasks, not an error, unless DB error specifically occurred and was logged  # noqa: E501
        return []
    return [TaskStatusOnlyResponse(**task) for task in tasks_data]


@app.get(
    "/tasks/{task_id}/summary",
    response_model=TaskSummaryResponse,  # Use new model
    dependencies=[Security(get_api_key)],
)
async def get_task_summary(
    task_id: str,
) -> TaskSummaryResponse:  # Update return type hint
    task_data = get_task_status_summary(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskSummaryResponse(**task_data)  # Map to new model


# Run the service using Uvicorn
if __name__ == "__main__":
    logger.info("Starting SwarmMind Service with Uvicorn...")
    uvicorn.run(
        "swarm_service:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
