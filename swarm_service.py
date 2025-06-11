print("!!!! EXECUTING SWARM_SERVICE.PY - LATEST VERSION MARKER !!!!")
import os
from dotenv import find_dotenv, load_dotenv
from typing import Dict, Optional, Any, AsyncGenerator, List
import json
import threading
import time

# --- Environment Debug Prints (Moved to very top) ---
print(f"DEBUG: Current Working Directory: {os.getcwd()}")
print(f"DEBUG: About to call find_dotenv(). Current __file__: {__file__}")
dotenv_path_debug = find_dotenv(usecwd=True) # Try forcing CWD search
print(f"DEBUG: find_dotenv(usecwd=True) result: {dotenv_path_debug}")
if not dotenv_path_debug:
    dotenv_path_debug = find_dotenv() # Default behavior
    print(f"DEBUG: find_dotenv() (default) result: {dotenv_path_debug}")

if dotenv_path_debug:
    print(f"DEBUG: Attempting to load .env file from: {dotenv_path_debug}")
    load_dotenv(dotenv_path_debug, override=True) # Override for good measure
    print(f"DEBUG: SWARM_API_KEY after load_dotenv: '{os.getenv('SWARM_API_KEY')}'")
    print(f"DEBUG: OPENAI_API_KEY after load_dotenv: '{os.getenv('OPENAI_API_KEY')}'")
else:
    print("DEBUG: .env file not found by find_dotenv().")
# --- End Environment Debug Prints ---

# --- Standard library and FastAPI imports AFTER dotenv loading ---
import sqlite3
import logging
from fastapi import FastAPI, Security, HTTPException, status, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
import uvicorn
import uuid
from pydantic import BaseModel, constr
from swarms import Swarm # DB_NAME will be imported from config
from config import DB_NAME # Import DB_NAME from config
from contextlib import asynccontextmanager
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta

# --- Configure Logging ---
LOG_FILE = "swarm_service.log"

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers (if any from previous runs or imports)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Standard formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# StreamHandler for stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# RotatingFileHandler for the log file
# Max 1MB per file, keep 3 backup files
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Configure specific loggers if needed, or they will inherit from root
logger = logging.getLogger(__name__) # This will use the root logger's config

# --- Globals and Configuration ---
EXPECTED_API_KEY = os.getenv("SWARM_API_KEY")
if not EXPECTED_API_KEY:
    logger.warning(
        "Top-level: SWARM_API_KEY not loaded from environment. Value is None. Auth will fail."
    )
else:
    logger.info(
        f"Top-level: SWARM_API_KEY successfully loaded. Length: {len(EXPECTED_API_KEY)}"
    )

# --- Global Variables (Simpler for now, consider dependency injection later) ---
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
        logger.error("API key check failed: SWARM_API_KEY not configured on server.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on server",
        )

    if api_key_header == EXPECTED_API_KEY:
        return api_key_header
    else:
        logger.warning("Invalid API Key received.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )


# --- Lifespan Management for Startup and Shutdown ---
SUPERVISOR_INTERVAL_SECONDS = 10 # Check every 10 seconds
shutdown_event = threading.Event()

def periodic_supervisor_check_loop():
    """Periodically calls the supervisor check functions."""
    logger.info("Supervisor thread started.")
    while not shutdown_event.is_set():
        try:
            # First process any pending subtasks
            logger.debug(f"Supervisor loop: Calling process_pending_subtasks() at {datetime.now()}.")
            process_pending_subtasks()
            
            # Then check for stalled subtasks
            logger.debug(f"Supervisor loop: Calling check_for_stalled_subtasks() at {datetime.now()}.")
            check_for_stalled_subtasks()
            
            # Then check if any parent tasks are complete
            logger.debug(f"Supervisor loop: Calling check_and_collate_supervised_tasks() at {datetime.now()}.")
            check_and_collate_supervised_tasks()
        except Exception as e:
            logger.error(f"Supervisor loop: Unhandled exception: {e}", exc_info=True)
        # Wait for the interval or until shutdown is signaled
        shutdown_event.wait(SUPERVISOR_INTERVAL_SECONDS) 
    logger.info("Supervisor thread finished.")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup sequence
    global swarm_instance, main_db_connection, shutdown_event
    logger.info(
        "Lifespan: Initializing Swarm service... (API key should already be loaded)"
    )

    try:
        db_file_path = os.path.abspath(DB_NAME)
        logger.info(f"Lifespan: Attempting to connect to database: {db_file_path}")
        # Use check_same_thread=False for SQLite with FastAPI/multi-threaded access
        main_db_connection = sqlite3.connect(DB_NAME, check_same_thread=False)
        logger.info(f"Lifespan: Database connection successful to {db_file_path}")

        # Initialize Swarm with the shared connection
        swarm_instance = Swarm(db_connection=main_db_connection)
        logger.info("Lifespan: Swarm initialized successfully.")

        # Start supervisor thread
        shutdown_event.clear() # Ensure it's not set from a previous run (if any weird state)
        supervisor_thread = threading.Thread(target=periodic_supervisor_check_loop, daemon=True)
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
    shutdown_event.set() # Signal supervisor thread to stop
    # supervisor_thread.join() # Optionally wait for thread to finish cleanly, might delay shutdown
    logger.info("Lifespan: Shutdown event set for supervisor thread.")

    if main_db_connection:
        logger.info("Lifespan: Closing database connection on shutdown.")
        main_db_connection.close()
        main_db_connection = None
    logger.info("Lifespan: Service shutdown complete.")


# Create the FastAPI app instance with the lifespan manager
app = FastAPI(title="SwarmMind Service", lifespan=lifespan)
print("-----> FastAPI app object CREATED <-----")

# Define /hello immediately to test if it registers on the app object Uvicorn uses
@app.get("/hello")
async def hello_world():
    print("-----> HELLO ENDPOINT (HANDLER) REACHED AND EXECUTING <-----")
    return {"message": "Hello, world!"}
print("-----> /hello route DEFINED ON APP <-----")

# --- Pydantic Models for API Data ---
class TaskRequest(BaseModel):
    task_description: constr(min_length=5, max_length=5000)


class TaskResponse(BaseModel):
    task_id: str


# New response model for checking task status
class TaskStatusResponse(BaseModel):
    task_id: str
    description: Optional[str] = None # Made optional for flexibility
    status: str
    result: Optional[Any] = None # Can be any type, like dict or str
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


class AgentDetail(BaseModel):
    name: str
    agent_type: str # meta, supervisor, worker
    llm_model_identifier: str
    # instructions: Optional[str] = None # Potentially too verbose for a list


# --- Database Helper Functions for Tasks ---
def _get_db_cursor() -> Optional[sqlite3.Cursor]:
    """Helper to get a cursor from the global connection."""
    if not main_db_connection:
        logger.error("Database connection is not available.")
        raise HTTPException(status_code=503, detail="Database connection not available")
    return main_db_connection.cursor()


def create_task_record(
    task_id: str,
    description: str,
    parent_task_id: Optional[str] = None,  # New parameter
    is_subtask: bool = False,             # New parameter
) -> None:
    """Adds a new task to the database with 'queued' status."""
    now = datetime.now().isoformat()
    db_cursor = _get_db_cursor()  # Use the helper to get a fresh cursor
    try:
        logger.info(f"Attempting to insert task {task_id} into database. Parent: {parent_task_id}, Is Subtask: {is_subtask}")
        db_cursor.execute(
            "INSERT INTO tasks (task_id, description, status, created_at, updated_at, parent_task_id, is_subtask) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, description, "queued", now, now, parent_task_id, is_subtask),
        )
        logger.info(f"Task {task_id} insert executed. Cursor rowcount: {db_cursor.rowcount if db_cursor else 'N/A'}")
        main_db_connection.commit()
        logger.info(f"Database commit successful for task {task_id}.")
    except sqlite3.Error as e:
        logger.error(f"Database error creating task {task_id}: {e}", exc_info=True)
        # Optionally re-raise or handle, depending on desired error flow
        # For now, logging the error. The submit_task's try-except will catch higher-level issues.
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error in create_task_record for task {task_id}: {e}", exc_info=True)


def update_task_status(
    task_id: str,
    status: str,
    result: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Updates the status and result/error of a task."""
    now = datetime.now().isoformat()

    current_status_val = "unknown"
    task_details_before_update = get_task_status(task_id) # Fetch current state
    if task_details_before_update:
        current_status_val = task_details_before_update.get("status", "unknown")

    cursor = _get_db_cursor()
    try:
        cursor.execute(
            "UPDATE tasks SET status = ?, result = ?, error_message = ?, updated_at = ? WHERE task_id = ?",
            (status, result, error_message, now, task_id),
        )
        if main_db_connection:
            main_db_connection.commit()
        # Log the transition including result/error presence
        log_message = f"Task {task_id} status transition: {current_status_val} -> {status}."
        if result is not None:
            # Avoid logging potentially large results; just indicate presence
            log_message += " Result updated."
        if error_message is not None:
            # Log only a snippet of the error message to avoid overly long logs
            log_message += f" Error message: \"{error_message[:100]}...\"" if len(error_message) > 100 else f" Error message: \"{error_message}\""
        logger.info(log_message)
    except sqlite3.Error as e:
        logger.error(f"Failed to update task status for {task_id} from {current_status_val} to {status}: {e}")


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves task details from the database."""
    cursor = _get_db_cursor()
    try:
        cursor.execute(
            "SELECT task_id, description, status, result, error_message, created_at, updated_at, parent_task_id, is_subtask, assigned_agent_name FROM tasks WHERE task_id = ?",
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
    """Retrieves only the status-related fields for a task from the database."""
    cursor = _get_db_cursor()
    if not cursor:
        logger.error(f"Failed to get DB cursor for get_task_status_summary(task_id={task_id})")
        return None
    try:
        cursor.execute(
            "SELECT task_id, description, status, error_message, created_at, updated_at FROM tasks WHERE task_id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    except sqlite3.Error as e:
        logger.error(f"Database error in get_task_status_summary for {task_id}: {e}", exc_info=True)
        return None

def get_all_tasks_summary() -> List[Dict[str, Any]]:
    """Retrieves status-related fields for all tasks from the database, ordered by creation date descending."""
    cursor = _get_db_cursor()
    if not cursor:
        logger.error("Failed to get DB cursor for get_all_tasks_summary")
        return []
    try:
        cursor.execute(
            "SELECT task_id, status, error_message, created_at, updated_at FROM tasks ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Database error in get_all_tasks_summary: {e}", exc_info=True)
        return []


# --- Background Task Worker ---
def run_swarm_task(task_id: str, description: str) -> None:
    """The actual function that runs the swarm task in the background."""
    logger.info(f"[{task_id}] Background processing initiated for: {description[:50]}...")

    # Fetch initial status to log transition
    task_details_initial = get_task_status(task_id)
    initial_status = task_details_initial.get("status", "unknown") if task_details_initial else "unknown"
    # Log the transition before calling update_task_status, which will log the actual update
    logger.info(f"[{task_id}] Preparing to transition task from {initial_status} -> running.")
    update_task_status(task_id, "running")
    # logger.info(f"[{task_id}] Status updated to running.") # This is redundant as update_task_status now logs the transition

    if not swarm_instance:
        logger.error(f"[{task_id}] Swarm instance not available.")
        update_task_status(task_id, "failed", error_message="Swarm service not initialized")
        return

    try:
        # Fetch full task details to check if it's a subtask
        task_details = get_task_status(task_id)
        if not task_details:
            logger.error(f"[{task_id}] Failed to retrieve task details for processing.")
            update_task_status(task_id, "failed", error_message="Could not retrieve task details.")
            return

        current_task_description = task_details.get("description", description) # Use fetched desc if available
        is_subtask = task_details.get("is_subtask", False)
        assigned_agent_name = task_details.get("assigned_agent_name")

        if is_subtask:
            logger.info(f"[{task_id}] Is a subtask. Assigned to: {assigned_agent_name}. Description: {current_task_description[:50]}...")
            if not assigned_agent_name:
                logger.error(f"[{task_id}] Subtask has no assigned_agent_name.")
                update_task_status(task_id, "failed", error_message="Subtask is missing an assigned agent.")
                return
            
            # Call the new method for direct subtask execution
            # This method will be responsible for its own status updates upon completion/failure
            subtask_result_dict = swarm_instance.execute_subtask(
                agent_name=assigned_agent_name, 
                task_description=current_task_description, 
                subtask_id=task_id
            )
            # execute_subtask should handle updating its own status internally.
            # For now, let's assume it returns a similar dict to organize_task/run for consistency if needed.
            # Example: if subtask_result_dict.get("error"):
            #    update_task_status(task_id, "failed", result=json.dumps(subtask_result_dict), error_message=subtask_result_dict["error"])
            # else:
            #    update_task_status(task_id, "completed", result=json.dumps(subtask_result_dict))
            logger.info(f"[{task_id}] Subtask execution handled by swarm_instance.execute_subtask.")

        else: # Parent task or simple task (not a subtask)
            logger.info(f"[{task_id}] Is a parent or simple task. Description: {current_task_description[:50]}...")
            # This will call Swarm.run(), which internally calls organize_task_for_supervision()
            # organize_task_for_supervision() will update the parent task's status to 'awaiting_subtasks' if it decomposes.
            task_output_dict = swarm_instance.run(current_task_description, task_id)

            # If 'subtask_ids' are returned, it means it was decomposed and parent task status is already 'awaiting_subtasks'.
            # In this case, run_swarm_task for THIS parent task is done for now.
            if task_output_dict and "subtask_ids" in task_output_dict:
                logger.info(f"[{task_id}] Task decomposed into subtasks: {task_output_dict.get('subtask_ids')}. Parent status set to 'awaiting_subtasks' by Swarm.run.")
                # No further status update needed here for the parent task, as Swarm.run handled it.
            elif task_output_dict and "error" in task_output_dict:
                logger.error(f"[{task_id}] Error returned by swarm_instance.run: {task_output_dict['error']}")
                update_task_status(
                    task_id,
                    "failed",
                    result=json.dumps(task_output_dict), # Store the full output as result
                    error_message=str(task_output_dict["error"])
                )
                logger.info(f"[{task_id}] Task status updated to failed.")
            elif task_output_dict: # Assume it's a direct result for a simple, non-decomposed task
                logger.info(f"[{task_id}] Task processed directly by Swarm.run (no decomposition). Output: {str(task_output_dict)[:100]}...")
                update_task_status(
                    task_id,
                    "completed",
                    result=json.dumps(task_output_dict) # Store the full output as result
                )
                logger.info(f"[{task_id}] Task status updated to completed.")
            else:
                # This case should ideally not be reached if Swarm.run always returns a dict
                logger.warning(f"[{task_id}] swarm_instance.run returned an unexpected or None value. Marking as failed.")
                update_task_status(task_id, "failed", error_message="Swarm.run returned unexpected output.")

    except Exception as e:
        logger.error(f"[{task_id}] Unexpected error in run_swarm_task: {e}", exc_info=True)
        update_task_status(
            task_id,
            "failed",
            error_message=f"Internal error during task execution: {str(e)}"
        )
        logger.info(
            f"[{task_id}] Task status updated to failed due to unexpected error."
        )


# --- Supervisor Logic for Collating Subtasks ---
def process_pending_subtasks():
    """Process any subtasks that are in 'queued' status."""
    global swarm_instance, main_db_connection
    if not swarm_instance or not main_db_connection:
        logger.debug("Supervisor: Swarm instance or DB connection not ready for process_pending_subtasks.")
        return

    logger.debug("Supervisor: Checking for pending subtasks (status 'queued')...")
    db_cursor = _get_db_cursor()

    try:
        # Find subtasks that are in 'queued' status
        db_cursor.execute(
            "SELECT task_id, description, assigned_agent_name FROM tasks WHERE status = 'queued' AND is_subtask = 1"
        )
        pending_subtasks = db_cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Supervisor: DB error fetching pending 'queued' subtasks: {e}", exc_info=True)
        return

    if not pending_subtasks:
        logger.debug("Supervisor: No pending subtasks in 'queued' status found.")
        return

    logger.info(f"Supervisor: Found {len(pending_subtasks)} pending subtasks in 'queued' status to process.")
    for subtask_row in pending_subtasks:
        subtask_id, subtask_description, assigned_agent_name = subtask_row
        logger.info(f"Supervisor: Initiating processing for pending subtask {subtask_id} ('{subtask_description[:30]}...') assigned to {assigned_agent_name}.")

        try:
            # update_task_status will log the transition from queued -> running
            update_task_status(subtask_id, "running")
            
            # Start a new thread to execute the subtask
            thread = threading.Thread(
                target=run_swarm_task,
                args=(subtask_id, subtask_description),
                daemon=True
            )
            thread.start()
            logger.info(f"Supervisor: Started thread for subtask {subtask_id}")
        except Exception as e:
            logger.error(f"Supervisor: Error starting thread for subtask {subtask_id}: {e}", exc_info=True)
            update_task_status(subtask_id, "failed", error_message=f"Failed to start execution: {str(e)}")

    logger.debug("Supervisor: Finished processing pending subtasks.")


def check_for_stalled_subtasks():
    """Check for subtasks that have been in the running state for too long and mark them as failed."""
    global main_db_connection
    if not main_db_connection:
        logger.debug("Supervisor: DB connection not ready for check_for_stalled_subtasks.")
        return
        
    logger.debug("Supervisor: Checking for stalled subtasks (status 'running' for too long)...")
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
            (timeout_iso,)
        )
        stalled_subtasks = db_cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Supervisor: DB error fetching stalled subtasks: {e}", exc_info=True)
        return
    
    if not stalled_subtasks:
        logger.debug(f"Supervisor: No subtasks found in 'running' state beyond the timeout threshold of {timeout_minutes} minutes.")
        return
    
    logger.info(f"Supervisor: Found {len(stalled_subtasks)} potentially stalled subtasks (running longer than {timeout_minutes} minutes).")
    for subtask_row in stalled_subtasks:
        subtask_id, subtask_description, assigned_agent_name, updated_at = subtask_row
        logger.warning(f"Supervisor: Stalled subtask {subtask_id} ('{subtask_description[:30]}...') assigned to {assigned_agent_name}. Last updated: {updated_at}. Marking as failed due to timeout.")
        
        # update_task_status will log the transition from running -> failed
        error_message = f"Subtask timed out after {timeout_minutes} minutes of inactivity."
        update_task_status(subtask_id, "failed", error_message=error_message)
        # logger.info(f"Supervisor: Marked stalled subtask {subtask_id} as failed due to timeout.") # Redundant, update_task_status logs


def check_and_collate_supervised_tasks():
    global swarm_instance, main_db_connection
    if not swarm_instance or not main_db_connection:
        logger.debug("Supervisor: Swarm instance or DB connection not ready for check_and_collate_supervised_tasks.")
        return

    logger.debug("Supervisor: Checking for parent tasks in 'awaiting_subtasks' status...")
    db_cursor = _get_db_cursor()

    try:
        db_cursor.execute("SELECT task_id, description FROM tasks WHERE status = 'awaiting_subtasks' AND is_subtask = 0")
        parent_tasks_awaiting = db_cursor.fetchall()
    except sqlite3.Error as e:
        logger.error(f"Supervisor: DB error fetching parent tasks in 'awaiting_subtasks': {e}", exc_info=True)
        return

    if not parent_tasks_awaiting:
        logger.debug("Supervisor: No parent tasks found in 'awaiting_subtasks' status.")
        return

    logger.info(f"Supervisor: Found {len(parent_tasks_awaiting)} parent tasks in 'awaiting_subtasks' status to check for collation.")
    for parent_task_row in parent_tasks_awaiting:
        parent_task_id = parent_task_row[0]
        parent_task_description = parent_task_row[1]
        logger.info(f"Supervisor: Checking subtasks for parent task {parent_task_id} ('{parent_task_description[:30]}...').")

        try:
            db_cursor.execute(
                "SELECT task_id, status, result, error_message FROM tasks WHERE parent_task_id = ? AND is_subtask = 1", 
                (parent_task_id,)
            )
            subtasks = db_cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Supervisor: DB error fetching subtasks for {parent_task_id}: {e}", exc_info=True)
            continue # Move to next parent task

        if not subtasks:
            logger.warning(f"Supervisor: Parent task {parent_task_id} is 'awaiting_subtasks' but has no subtasks found. Marking as error.")
            update_task_status(parent_task_id, "failed", error_message="Awaited subtasks but none found.")
            continue

        all_subtasks_finished = True
        subtask_results_for_collation = []
        failed_subtask_errors = []

        for sub_row in subtasks:
            sub_task_id, sub_status, sub_result_json, sub_error_msg = sub_row
            if sub_status not in ("completed", "failed"):
                all_subtasks_finished = False
                logger.debug(f"Supervisor: Parent {parent_task_id} still waiting for subtask {sub_task_id} (status: {sub_status}).")
                break # Not all subtasks are done for this parent
            
            if sub_status == "completed":
                try:
                    # First, check if sub_result_json is already a dictionary or a string
                    if isinstance(sub_result_json, dict):
                        # If it's already a dict, use it directly
                        sub_result_data = sub_result_json
                    else:
                        # If it's a string, try to parse it as JSON
                        try:
                            sub_result_data = json.loads(sub_result_json) if sub_result_json else {}
                        except json.JSONDecodeError:
                            # If it's not valid JSON, just use the string directly
                            logger.warning(f"Supervisor: Could not parse subtask result as JSON: {sub_result_json[:100]}...")
                            subtask_results_for_collation.append(sub_result_json)
                            continue
                    
                    # If we have a dictionary, extract the output
                    if isinstance(sub_result_data, dict):
                        actual_agent_output = sub_result_data.get("output")
                        if actual_agent_output:
                            # Try to parse the output as JSON if it's a string
                            if isinstance(actual_agent_output, str):
                                try:
                                    parsed_agent_output = json.loads(actual_agent_output)
                                    subtask_results_for_collation.append(parsed_agent_output)
                                except json.JSONDecodeError:
                                    # If it's not valid JSON, use the string directly
                                    subtask_results_for_collation.append(actual_agent_output)
                            else:
                                # If it's already a dict or other type, use it directly
                                subtask_results_for_collation.append(actual_agent_output)
                        else:
                            # No output field, use the whole result
                            subtask_results_for_collation.append(sub_result_data)
                    else:
                        # If it's not a dict, use it directly
                        subtask_results_for_collation.append(sub_result_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Supervisor: Error decoding result JSON for subtask {sub_task_id}: {e}. Original: {sub_result_json}")
                    subtask_results_for_collation.append({"error": "Failed to parse subtask result", "original_result": sub_result_json})
            elif sub_status == "failed":
                failed_subtask_errors.append(f"Subtask {sub_task_id} failed: {sub_error_msg or 'Unknown error'}")

        if all_subtasks_finished:
            logger.info(f"Supervisor: All subtasks finished for parent {parent_task_id}.")
            if failed_subtask_errors:
                parent_error_message = "One or more subtasks failed: " + "; ".join(failed_subtask_errors)
                logger.error(f"Supervisor: Parent task {parent_task_id} failed due to subtask errors: {parent_error_message}")
                update_task_status(parent_task_id, "failed", error_message=parent_error_message)
            else:
                try:
                    logger.info(f"Supervisor: Collating {len(subtask_results_for_collation)} results for parent {parent_task_id}.")
                    # We need to ensure swarm_instance._combine_results is thread-safe if called from a separate thread.
                    # The Swarm class uses a single DB connection/cursor which might be an issue.
                    # For now, assuming it's okay or this runs in a way that serializes calls.
                    combined_output = swarm_instance._combine_results(subtask_results_for_collation, parent_task_description)
                    
                    final_result_json = json.dumps(combined_output)
                    update_task_status(parent_task_id, "completed", result=final_result_json)
                    logger.info(f"Supervisor: Parent task {parent_task_id} completed successfully after collation.")
                except Exception as e:
                    collation_error_msg = f"Error during result collation for {parent_task_id}: {e}"
                    logger.error(f"Supervisor: {collation_error_msg}", exc_info=True)
                    update_task_status(parent_task_id, "failed", error_message=collation_error_msg)
        else:
            logger.debug(f"Supervisor: Parent {parent_task_id} still has pending subtasks.")

    logger.debug("Supervisor: Finished checking tasks awaiting subtask completion.")


# --- Core API Endpoints ---
@app.post("/tasks", response_model=TaskResponse, dependencies=[Security(get_api_key)])
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks) -> TaskResponse:
    logger.info("Endpoint submit_task called.")
    logger.info(f"submit_task entered. Request description (first 50 chars): {request.task_description[:50]}...")
    task_id = str(uuid.uuid4())
    logger.info(f"Generated task_id: {task_id} for request: {request.task_description[:50]}...")
    
    try:
        logger.info(f"Attempting to create task record for task_id: {task_id}")
        create_task_record(task_id, request.task_description)
        logger.info(f"Task record created for task_id: {task_id}. Adding to background tasks.")
        
        background_tasks.add_task(run_swarm_task, task_id, request.task_description)
        logger.info(f"Task {task_id} added to background_tasks.")
        
        logger.info(f"Returning TaskResponse for task_id: {task_id}")
        logger.info("Endpoint submit_task finished.")
        return TaskResponse(task_id=task_id)
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR IN SUBMIT_TASK for task_id {task_id if 'task_id' in locals() else 'UNKNOWN'}: {e}", exc_info=True)
        # Re-raise or return an error response if an exception occurs
        # Consider if you want to expose full error 'e' to client or a generic message
        logger.info("Endpoint submit_task finished with error.")
        raise HTTPException(status_code=500, detail=f"Internal server error during task submission: {str(e)}")


@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    dependencies=[Security(get_api_key)],
)
async def get_task_details(task_id: str) -> TaskStatusResponse:
    """Retrieves the status and result of a specific task (requires API Key)."""
    logger.info("Endpoint get_task_details called.")
    logger.info(f"Received request for task status: {task_id}")
    task_data = get_task_status(task_id)
    if not task_data:
        logger.info("Endpoint get_task_details finished with error.")
        raise HTTPException(status_code=404, detail="Task not found")

    # Convert the dictionary to the Pydantic model
    # Handle potential None for result/error_message if DB stores NULL
    response = TaskStatusResponse(
        task_id=task_data["task_id"],
        description=task_data.get("description"), # Use .get for optional field
        status=task_data["status"],
        result=task_data.get("result"),
        error_message=task_data.get("error_message"),
        created_at=task_data["created_at"],
        updated_at=task_data["updated_at"],
        parent_task_id=task_data.get("parent_task_id"),
        is_subtask=task_data["is_subtask"],
        assigned_agent_name=task_data.get("assigned_agent_name"),
    )
    logger.info("Endpoint get_task_details finished.")
    return response

@app.get(
    "/tasks/all/summary",
    response_model=List[TaskStatusOnlyResponse],
    dependencies=[Security(get_api_key)],
)
async def get_all_tasks_list_summary() -> List[TaskStatusOnlyResponse]:
    logger.info("Endpoint get_all_tasks_list_summary called.")
    tasks_data = get_all_tasks_summary()
    if not tasks_data:
        # Return empty list if no tasks, not an error, unless DB error specifically occurred and was logged
        logger.info("Endpoint get_all_tasks_list_summary finished (no tasks).")
        return []
    response = [TaskStatusOnlyResponse(**task) for task in tasks_data]
    logger.info("Endpoint get_all_tasks_list_summary finished.")
    return response

@app.get(
    "/tasks/{task_id}/summary",
    response_model=TaskSummaryResponse, # Use new model
    dependencies=[Security(get_api_key)],
)
async def get_task_summary(task_id: str) -> TaskSummaryResponse: # Update return type hint
    logger.info("Endpoint get_task_summary called.")
    task_data = get_task_status_summary(task_id)
    if not task_data:
        logger.info("Endpoint get_task_summary finished with error (task not found).")
        raise HTTPException(status_code=404, detail="Task not found")
    response = TaskSummaryResponse(**task_data) # Map to new model
    logger.info("Endpoint get_task_summary finished.")
    return response


@app.get("/agents", response_model=List[AgentDetail], dependencies=[Security(get_api_key)])
async def list_available_agents():
    """Lists all available agents in the swarm."""
    logger.info("Endpoint /agents called.")
    if not swarm_instance:
        logger.error("/agents: Swarm instance not available.")
        raise HTTPException(status_code=503, detail="Swarm service not initialized or unavailable.")

    agent_details_list = []
    try:
        # Helper to process a dictionary of agents
        def process_agent_dict(agents_dict: Dict[str, Any], agent_type: str):
            for agent_name, agent_obj in agents_dict.items():
                agent_details_list.append(
                    AgentDetail(
                        name=agent_obj.name,
                        agent_type=agent_obj.agent_type, # Assuming agent_obj has an agent_type attribute
                        llm_model_identifier=agent_obj.llm_model_identifier,
                    )
                )

        process_agent_dict(swarm_instance.agents, "worker")
        process_agent_dict(swarm_instance.supervisors, "supervisor")
        process_agent_dict(swarm_instance.meta_agents, "meta")

        logger.info(f"/agents: Successfully retrieved {len(agent_details_list)} agents.")
    except AttributeError as e:
        logger.error(f"/agents: AttributeError accessing agent attributes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving agent details.")
    except Exception as e:
        logger.error(f"/agents: Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

    logger.info("Endpoint /agents finished.")
    return agent_details_list

# Run the service using Uvicorn
if __name__ == "__main__":
    logger.info("Starting SwarmMind Service with Uvicorn...")
    uvicorn.run(
        "swarm_service:app", host="127.0.0.1", port=8000, reload=False, log_level="info"
    )
