"""
Data models for the Hivey swarm intelligence system.
Contains core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str
    content: str


@dataclass
class Result:
    """Represents the result of an agent's operation."""

    value: str = ""
    agent: Optional[Any] = None  # Forward reference to avoid circular import
    context_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Function:
    """Represents a function that can be called by an agent."""

    name: str
    func: Callable
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
