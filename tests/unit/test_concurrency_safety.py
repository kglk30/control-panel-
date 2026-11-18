import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from control_panel.common.events import EventBus, Event, EventType, get_event_bus, reset_event_bus
from control_panel.worker.runtime import WorkerRuntime
from control_panel.common.models import WorkerEvent

def test_eventbus_emit_serializes_subscribers():
    """Verify EventBus.emit() uses a lock to prevent interleaved subscriber execution."""
    bus = EventBus()

    # We will track the active thread inside the handler
    # If the lock works, we should never see active_threads > 1
    active_threads = 0
    max_active_threads = 0
    lock_tracker = threading.Lock()

    def slow_handler(event: Event) -> None:
        nonlocal active_threads, max_active_threads
        with lock_tracker:
            active_threads += 1
            if active_threads > max_active_threads:
                max_active_threads = active_threads

        time.sleep(0.01)  # Simulate slow UI work

        with lock_tracker:
            active_threads -= 1

    bus.subscribe(EventType.WORKER_STARTED, slow_handler)

    # Launch concurrent emits
    def worker_thread():
        bus.emit_simple(EventType.WORKER_STARTED, task_id="t1", worker_id="w1")

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(worker_thread) for _ in range(10)]
        for f in futures:
            f.result()

    # If lock works properly, max_active_threads MUST be exactly 1
    assert max_active_threads == 1
    assert active_threads == 0


def test_workerruntime_progress_callback_serializes():
    """Verify WorkerRuntime._emit() serializes progress callbacks from concurrent workers."""
    runtime = WorkerRuntime()

    active_threads = 0
    max_active_threads = 0
    lock_tracker = threading.Lock()

    def slow_progress_callback(event: WorkerEvent) -> None:
        nonlocal active_threads, max_active_threads
        with lock_tracker:
            active_threads += 1
            if active_threads > max_active_threads:
                max_active_threads = active_threads

        time.sleep(0.01)  # Simulate UI progress update

        with lock_tracker:
            active_threads -= 1

    # Launch concurrent emit calls directly on the runtime
    def worker_thread():
        event = WorkerEvent(event_type="started", worker_id="w1", task_id="t1", message="test")
        runtime._emit(slow_progress_callback, event)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(worker_thread) for _ in range(10)]
        for f in futures:
            f.result()

    assert max_active_threads == 1
    assert active_threads == 0

def test_workerruntime_sequential_progress_behaves_normally():
    """Verify sequential calls to progress lock do not deadlock or fail."""
    runtime = WorkerRuntime()

    call_count = 0

    def simple_progress(event: WorkerEvent) -> None:
        nonlocal call_count
        call_count += 1

    event = WorkerEvent(event_type="started", worker_id="w1", task_id="t1", message="test")
    runtime._emit(simple_progress, event)
    runtime._emit(simple_progress, event)

    assert call_count == 2
