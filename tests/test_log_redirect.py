"""Tests for stdout/stderr queue redirect."""

from queue import Empty, Queue

import pytest

from gui.log_redirect import QueueStream, redirect_output


def test_queue_stream_splits_lines():
    queue: Queue = Queue()
    stream = QueueStream(queue)
    stream.write("line one\nline two\n")
    stream.flush()

    assert queue.get_nowait() == ("log", "line one")
    assert queue.get_nowait() == ("log", "line two")


def test_queue_stream_carriage_return():
    queue: Queue = Queue()
    stream = QueueStream(queue)
    stream.write("Progress: 50%\rProgress: 100%")
    stream.flush()

    assert queue.get_nowait() == ("log", "Progress: 100%")


def test_redirect_output_captures_print():
    queue: Queue = Queue()
    with redirect_output(queue):
        print("hello from worker")

    assert queue.get_nowait() == ("log", "hello from worker")
    with pytest.raises(Empty):
        queue.get_nowait()
