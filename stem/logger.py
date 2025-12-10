import os
import sys

from typing import Union
from pathlib import Path


def enable_writing_to_log_file(log_path: Union[Path, str]):
    """
    Redirects all stdout/stderr to a log file

    Args:
        - log_path (Union[Path, str]): Path to the log file
    """

    # Open log file for writing
    logfile = open(log_path, "w")

    # Duplicate the low-level file descriptor so that *all* output goes to the log file
    os.dup2(logfile.fileno(), sys.stdout.fileno())
    os.dup2(logfile.fileno(), sys.stderr.fileno())
