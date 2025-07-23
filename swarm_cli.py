import ast
import json
import os

import requests
import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text

# Configuration
BASE_URL = os.getenv("SWARM_BASE_URL", "http://localhost:8001")
# Changed to match server's expected environment variable
API_KEY_ENV_VAR = "SWARM_API_KEY"

app = typer.Typer(help="CLI for interacting with the Swarm Service.")


def get_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        typer.secho(
            f"Error: {API_KEY_ENV_VAR} environment variable not set.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    return api_key


@app.command()
def submit(
    description: str = typer.Argument(
        ..., help="The description of the task to submit."
    )
):
    """Submits a new task to the Swarm service."""
    api_key = get_api_key()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    payload = {"task_description": description}

    try:
        response = requests.post(
            f"{BASE_URL}/tasks", headers=headers, json=payload
        )
        response.raise_for_status()  # Raises an exception for 4XX/5XX errors
        task_response = response.json()
        typer.secho("Task submitted successfully!", fg=typer.colors.GREEN)
        typer.echo(f"Task ID: {task_response.get('task_id')}")
    except requests.exceptions.HTTPError as e:
        typer.secho(f"Error submitting task: {e}", fg=typer.colors.RED)
        if e.response is not None:
            try:
                typer.echo(f"Server response: {e.response.json()}")
            except json.JSONDecodeError:
                typer.echo(f"Server response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        typer.secho(f"Request failed: {e}", fg=typer.colors.RED)


@app.command()
def list_tasks(ctx: typer.Context):
    """Lists summaries of all tasks."""
    api_key = get_api_key()
    headers = {"X-API-Key": api_key}

    try:
        response = requests.get(
            f"{BASE_URL}/tasks/all/summary", headers=headers
        )
        response.raise_for_status()
        tasks = response.json()
        if tasks:
            typer.secho("Current Tasks:", bold=True)
            # Using rich table via typer's print
            table = Table(
                title=None, show_header=True, header_style="bold magenta"
            )
            table.add_column("Task ID", style="cyan", no_wrap=True)
            table.add_column("Status", style="magenta")
            table.add_column("Created At", style="green")
            table.add_column("Updated At", style="green")
            table.add_column(
                "Error Message", style="red", overflow="fold", max_width=50
            )

            for task in tasks:
                error_msg = task.get("error_message", "")
                error_color = "red" if error_msg else ""
                table.add_row(
                    task["task_id"],
                    task["status"],
                    task["created_at"],
                    task["updated_at"],
                    (
                        f"[{error_color}]{error_msg}[/{error_color}]"
                        if error_msg
                        else "N/A"
                    ),
                )
            console = Console()
            console.print(table)
        else:
            typer.secho("No tasks found.", fg=typer.colors.YELLOW)
    except requests.exceptions.HTTPError as e:
        typer.secho(f"Error listing tasks: {e}", fg=typer.colors.RED)
        if e.response is not None:
            try:
                typer.echo(f"Server response: {e.response.json()}")
            except json.JSONDecodeError:
                typer.echo(f"Server response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        typer.secho(f"Request failed: {e}", fg=typer.colors.RED)


# Placeholder for future commands
@app.command()
def get_task(
    task_id: str = typer.Argument(..., help="The ID of the task to retrieve.")
):
    """Gets full details of a specific task."""
    api_key = get_api_key()
    headers = {"X-API-Key": api_key}
    try:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}", headers=headers)
        response.raise_for_status()  # Raises an exception for HTTP errors
        task_data = response.json()

        console = Console()
        if not task_data:
            console.print(
                f"No data found for task ID: {task_id}", style="yellow"
            )
            return

        # Prepare content for the panel
        content = Text()
        content.append(
            f"Task ID: {task_data.get('task_id', 'N/A')}\n", style="bold cyan"
        )
        content.append(
            f"Description: {task_data.get('description', 'N/A')}\n",
            style="green",
        )
        content.append(
            f"Status: {task_data.get('status', 'N/A')}\n", style="magenta"
        )
        content.append(
            f"Created At: {task_data.get('created_at', 'N/A')}\n", style="blue"
        )
        content.append(
            f"Updated At: {task_data.get('updated_at', 'N/A')}\n", style="blue"
        )

        error_message = task_data.get("error_message")
        if error_message:
            content.append(
                f"Error Message: {error_message}\n", style="bold red"
            )

        result = task_data.get("result")

        if result is not None:
            content.append(
                "Result (summary):\n", style="bold yellow"
            )  # Indicate this is just a header in the panel
            console.print(
                Panel(
                    content,
                    title=f"Task Information: {task_id}",
                    border_style="blue",
                    expand=False,
                )
            )

            parsed_result = None
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    try:
                        # Fallback to ast.literal_eval if json.loads fails (e.g. for single quotes)  # noqa: E501
                        parsed_result = ast.literal_eval(result)
                    except (ValueError, SyntaxError):
                        # If both fail, keep it as a string
                        parsed_result = result
            else:
                # If result is not a string (e.g., already a dict/list, or None), use it as is  # noqa: E501
                parsed_result = result

            if isinstance(parsed_result, (dict, list)):
                console.print("--- Detailed Result ---", style="bold green")
                console.print(Pretty(parsed_result))
                console.print(
                    "--- End of Detailed Result ---", style="bold green"
                )
            elif (
                parsed_result is not None
            ):  # Catch cases where result was a non-JSON string or other non-None, non-dict/list type  # noqa: E501
                console.print(
                    f"Result (plain text): {str(parsed_result)}", style="white"
                )
        else:
            console.print(
                Panel(
                    content,
                    title=f"Task Information: {task_id}",
                    border_style="blue",
                    expand=False,
                )
            )

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            typer.secho(
                f"Task with ID '{task_id}' not found.", fg=typer.colors.YELLOW
            )
        else:
            typer.secho(
                f"Error fetching task {task_id}: {e}. Response: {e.response.text}",  # noqa: E501
                fg=typer.colors.RED,
            )
    except requests.exceptions.RequestException as e:
        typer.secho(f"Request failed: {e}", fg=typer.colors.RED)


@app.command()
def get_summary(
    task_id: str = typer.Argument(
        ..., help="The ID of the task to get a summary for."
    )
):
    """Gets a summary of a specific task."""
    api_key = get_api_key()
    headers = {"X-API-Key": api_key}
    try:
        response = requests.get(
            f"{BASE_URL}/tasks/{task_id}/summary", headers=headers
        )
        response.raise_for_status()  # Raises an exception for HTTP errors
        summary_data = response.json()

        console = Console()
        if not summary_data:
            console.print(
                f"No summary data found for task ID: {task_id}", style="yellow"
            )
            return

        content = Text()
        content.append(
            f"Task ID: {summary_data.get('task_id', 'N/A')}\n",
            style="bold cyan",
        )
        content.append(
            f"Status: {summary_data.get('status', 'N/A')}\n", style="magenta"
        )
        content.append(
            f"Description: {summary_data.get('description', 'N/A')}\n",
            style="green",
        )

        console.print(
            Panel(
                content,
                title=f"Task Summary: {task_id}",
                border_style="blue",
                expand=False,
            )
        )

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            typer.secho(
                f"Task with ID '{task_id}' not found (for summary).",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                f"Error fetching task summary for {task_id}: {e}. Response: {e.response.text}",  # noqa: E501
                fg=typer.colors.RED,
            )
    except requests.exceptions.RequestException as e:
        typer.secho(f"Request failed: {e}", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
