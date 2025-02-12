# cereal/log.py
"""
Log Module for Cereal Package in FrogPilot

This module handles logging functionalities using Cap'n Proto for FrogPilot.
"""

import capnp
import os

# Path to the compiled log_capnp.py generated from log.capnp
LOG_CAPNP_PATH = os.path.join(os.path.dirname(__file__), 'log.capnp')

# Load the Cap'n Proto schema
log_capnp = capnp.load(LOG_CAPNP_PATH)

class LogMessage:
    """
    Represents a single log message.
    """
    @staticmethod
    def read_multiple(dat: bytes):
        """
        Read multiple log messages from bytes.
        
        Args:
            dat (bytes): Compressed or raw log data.
        
        Returns:
            list: List of LogMessage objects.
        """
        try:
            return log_capnp.LogMessage.read_multiple(dat)
        except capnp.KjException as e:
            raise Exception("Error reading log messages") from e

    @staticmethod
    def read(log_path: str, sort_by_time: bool = False):
        """
        Read log messages from a given log file path.
        
        Args:
            log_path (str): Path to the log file.
            sort_by_time (bool): Whether to sort the logs by time.
        
        Yields:
            LogMessage: Parsed log messages.
        """
        with open(log_path, 'rb') as f:
            dat = f.read()
            messages = LogMessage.read_multiple(dat)
            if sort_by_time:
                messages.sort(key=lambda msg: msg.logMonoTime)
            for msg in messages:
                yield msg