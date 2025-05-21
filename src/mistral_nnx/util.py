import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(desc: str) -> Generator[None, None, None] :
    """
    A context manager to time the execution of a code block.

    Args:
        description (str): A description for the code block being timed.
                           This will be included in the output message.
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{elapsed_time:4.4f}s elapsed for {desc}")