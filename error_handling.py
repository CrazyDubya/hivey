"""
Enhanced error handling utilities for the Hivey project.
Provides standardized error handling patterns and custom exceptions.
"""

import functools
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ErrorDetails:
    """Contains detailed information about an error."""

    error_type: str
    message: str
    details: Optional[str] = None
    traceback: Optional[str] = None
    context: Optional[dict] = None


class HiveyBaseException(Exception):
    """Base exception for all Hivey-specific errors."""

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        self.message = message
        self.details = details
        self.context = context or {}
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert exception to dictionary format."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "context": self.context,
        }


class ConfigurationError(HiveyBaseException):
    """Raised when there's a configuration-related error."""

    pass


class DatabaseError(HiveyBaseException):
    """Raised when there's a database-related error."""

    pass


class LLMError(HiveyBaseException):
    """Raised when there's an LLM-related error."""

    pass


class ValidationError(HiveyBaseException):
    """Raised when input validation fails."""

    pass


def safe_execute(
    operation: Callable,
    *args,
    logger: Optional[logging.Logger] = None,
    error_message: str = "Operation failed",
    return_on_error: Any = None,
    reraise: bool = False,
    **kwargs,
) -> Any:
    """
    Safely execute an operation with standardized error handling.

    Args:
        operation: The function/method to execute
        *args: Positional arguments for the operation
        logger: Logger instance to use for error logging
        error_message: Custom error message prefix
        return_on_error: Value to return if operation fails
        reraise: Whether to re-raise the exception after logging
        **kwargs: Keyword arguments for the operation

    Returns:
        Operation result or return_on_error value
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        return operation(*args, **kwargs)
    except Exception as e:
        error_details = ErrorDetails(
            error_type=type(e).__name__,
            message=str(e),
            details=error_message,
            traceback=traceback.format_exc(),
            context={"args": str(args), "kwargs": str(kwargs)},
        )

        logger.error(
            f"{error_message}: {error_details.error_type} - {error_details.message}",  # noqa: E501
            extra={"error_details": error_details},
        )

        if reraise:
            raise

        return return_on_error


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by on each retry
        exceptions: Tuple of exception types to catch and retry on
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            logger = logging.getLogger(func.__module__)

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"  # noqa: E501
                        )
                        raise

                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_attempts}: {e}. "  # noqa: E501
                        f"Retrying in {current_delay:.1f} seconds..."
                    )

                    import time

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

        return wrapper

    return decorator


def handle_database_errors(func: Callable) -> Callable:
    """Decorator to standardize database error handling."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import sqlite3

            if isinstance(e, sqlite3.Error):
                raise DatabaseError(
                    f"Database operation failed in {func.__name__}",
                    details=str(e),
                    context={"function": func.__name__, "args": str(args)},
                )
            else:
                raise HiveyBaseException(
                    f"Unexpected error in database operation {func.__name__}",
                    details=str(e),
                    context={"function": func.__name__, "args": str(args)},
                )

    return wrapper


def validate_input(
    validator: Callable[[Any], bool],
    error_message: str = "Input validation failed",
):
    """
    Decorator to validate function inputs.

    Args:
        validator: Function that takes the input and returns True if valid
        error_message: Error message to use if validation fails
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all arguments
            for arg in args:
                if not validator(arg):
                    raise ValidationError(
                        error_message,
                        details=f"Invalid argument: {arg}",
                        context={"function": func.__name__},
                    )

            # Validate keyword arguments
            for key, value in kwargs.items():
                if not validator(value):
                    raise ValidationError(
                        error_message,
                        details=f"Invalid keyword argument {key}: {value}",
                        context={"function": func.__name__},
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator
