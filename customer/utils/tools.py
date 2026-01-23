"""
Utility functions for the chat service.
"""
import time
import functools
from typing import Any, Callable
from datetime import datetime, timezone, timedelta
import uuid


def perf_counter_timer(func: Callable) -> Callable:
    """Performance counter decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(".4f")
        return result
    return wrapper


def get_cur_timezone_time() -> tuple[str, str]:
    """
    Get current time in Beijing timezone.

    Returns:
        tuple: (formatted_time, weekday)
    """
    # Beijing timezone (UTC+8)
    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz)

    # Format time
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Get weekday in Chinese
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]

    return time_str, weekday


def generate_msg_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


def limit_history_length(history: list, max_length: int = 20) -> list:
    """
    Limit the length of conversation history.

    Args:
        history: List of conversation messages
        max_length: Maximum number of messages to keep

    Returns:
        Limited history list
    """
    if len(history) > max_length * 2:  # Each conversation turn has 2 messages (user + assistant)
        return history[-(max_length * 2):]
    return history


def chunk_text(text: str, chunk_size: int = 50) -> list[str]:
    """
    Split text into chunks.

    Args:
        text: Text to split
        chunk_size: Size of each chunk

    Returns:
        List of text chunks
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def normalize_vector(vector: list[float]) -> list[float]:
    """
    Normalize a vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    import math
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]
