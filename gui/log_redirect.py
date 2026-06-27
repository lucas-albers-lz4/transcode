"""Redirect stdout/stderr to a queue for GUI log panels."""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from queue import Queue


class QueueStream(io.TextIOBase):
    """Write text lines into a queue for the UI thread to consume."""

    def __init__(self, queue: Queue, prefix: str = ""):
        self._queue = queue
        self._prefix = prefix
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._queue.put(("log", self._prefix + line))
        if "\r" in self._buffer:
            parts = self._buffer.split("\r")
            self._buffer = parts[-1]
            line = parts[-1].strip()
            if line:
                self._queue.put(("log", self._prefix + line))
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._queue.put(("log", self._prefix + self._buffer))
            self._buffer = ""


@contextmanager
def redirect_output(queue: Queue) -> Iterator[None]:
    """Capture stdout and stderr into queue log events."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = QueueStream(queue)
    sys.stderr = QueueStream(queue, prefix="[stderr] ")
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err
