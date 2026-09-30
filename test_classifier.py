from control_panel.worker.runtime import WorkerRuntime
from control_panel.common.models import WorkerSpec, WorkerConstraints, TaskSpec
from control_panel.safety.models import SafetyContext
from control_panel.worker.models import Worker, WorkerState

runtime = WorkerRuntime()

spec = WorkerSpec(
    worker_id="w1",
    task_id="t1",
    role="file_classifier",
    tools_granted=[],
    permission_mode="read_only",
    input_data={
        "target_path": "/test/folder",
        "categories": {
            "Images": [".jpg", ".png"],
            "Documents": [".txt", ".md"]
        },
        "entries": [
            {"path": "/test/folder/pic.jpg", "name": "pic.jpg", "is_dir": False},
            {"path": "/test/folder/notes.txt", "name": "notes.txt", "is_dir": False},
            {"path": "/test/folder/unknown.xyz", "name": "unknown.xyz", "is_dir": False},
            {"path": "/test/folder/sub", "name": "sub", "is_dir": True}
        ]
    },
    constraints=WorkerConstraints(max_runtime_seconds=60, max_retries=1, allowed_paths=[])
)
ctx = SafetyContext(worker_id="w1", task_id="t1", permission_mode="read_only", allowed_paths=[])
worker = Worker(worker_id="w1", task_id="t1", spec=spec, state=WorkerState.READY, tools={}, safety_context=ctx, input_data=spec.input_data)

class DummyTrace:
    steps = []
trace = DummyTrace()

import time
out = runtime._run_worker_logic(worker, trace, 0, None, time.perf_counter(), 60.0)
import json
print(json.dumps(out["moves"], indent=2))
