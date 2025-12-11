import os
import sys
from pathlib import Path

import pytest

from stem.logger import enable_writing_to_log_file


@pytest.fixture
def restore_output():
    """
    Pytest fixture to restore stdout/stderr after test

    Yields:
        - None
    """
    # Save original file descriptors
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())

    # Go to test
    yield

    # Restore original stdout/stderr
    os.dup2(original_stdout_fd, sys.stdout.fileno())
    os.dup2(original_stderr_fd, sys.stderr.fileno())
    os.close(original_stdout_fd)
    os.close(original_stderr_fd)


def test_enable_writing_to_log_file(restore_output: None, capsys: pytest.CaptureFixture[str]):
    """
    Test that enabling writing to a log file correctly redirects stdout and stderr
    to the specified log file.

    Args:
        - restore_output (None): pytest fixture to restore stdout/stderr after test
        - capsys (pytest.CaptureFixture[str]): pytest fixture to capture output
    """

    # disable capsys temporarily to allow redirection
    with capsys.disabled():
        log_file = Path("output_test_logger.log")

        enable_writing_to_log_file(log_file)

        # Anything printed after the redirect should go into the log
        print("hello stdout")
        print("hello stderr", file=sys.stderr)

    # Now read the log and verify content
    contents = log_file.read_text()
    assert "hello stdout" in contents
    assert "hello stderr" in contents
