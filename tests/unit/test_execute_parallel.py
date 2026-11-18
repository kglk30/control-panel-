import pytest
import time
from unittest.mock import MagicMock
from control_panel.worker.models import Worker, WorkerResult, WorkerSpec
from control_panel.common.models import ToolResult
from control_panel.worker.runtime import WorkerRuntime
from control_panel.orchestrator.depgraph import DependencyGraph, DepNode
from control_panel.safety.models import SafetyContext
from control_panel.tools.base import Tool

class DummyTool(Tool):
    def __init__(self, name, duration=0.1, fails=False):
        self._name = name
        self.duration = duration
        self.fails = fails

    @property
    def name(self) -> str: return self._name
    @property
    def category(self) -> str: return "test"
    @property
    def permissions(self) -> set[str]: return set()
    @property
    def parameters(self) -> dict: return {}
    def execute(self, params, ctx):
        time.sleep(self.duration)
        if self.fails:
            raise RuntimeError("dummy error")
        return ToolResult(success=True, output={"did_"+self._name: True})

def _make_worker(role, tools):
    spec = WorkerSpec(task_id="test_task", role=role, description="dummy", tools_granted=[t.name for t in tools])
    worker = Worker(
        task_id="test_task",
        spec=spec,
        tools={t.name: t for t in tools},
        safety_context=SafetyContext(worker_id="w", task_id="t", permission_mode="read_only", allowed_paths=[])
    )
    return worker

def test_execute_parallel_independent_workers_concurrently():
    runtime = WorkerRuntime()
    runtime._get_tool_sequence = lambda w: [(t, {}) for t in w.spec.tools_granted]
    w1 = _make_worker("A", [DummyTool("fs.list", duration=0.5)])
    w2 = _make_worker("B", [DummyTool("fs.list", duration=0.5)])

    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))

    t0 = time.perf_counter()
    results = runtime.execute_parallel([w1, w2], graph=graph, n_max=2)
    t1 = time.perf_counter()

    assert len(results) == 2
    assert t1 - t0 < 0.9, "Should run concurrently"
    assert {r.role for r in results} == {"A", "B"}
    assert all(r.success for r in results)

def test_dependent_workers_in_separate_groups():
    runtime = WorkerRuntime()
    runtime._get_tool_sequence = lambda w: [(t, {}) for t in w.spec.tools_granted]
    w1 = _make_worker("A", [DummyTool("t1", duration=0.2)])
    w2 = _make_worker("B", [DummyTool("t2", duration=0.2)])
    w3 = _make_worker("C", [DummyTool("t3", duration=0.2)])

    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))
    graph.add_node(DepNode(operation_id="C"))
    graph.add_edge("A", "C")
    graph.add_edge("B", "C")

    # Check that C receives output of A and B
    results = runtime.execute_parallel([w1, w2, w3], graph=graph, n_max=5)

    assert len(results) == 3
    c_worker = next(w for w in [w1, w2, w3] if w.spec.role == "C")
    assert c_worker.input_data.get("did_t1") is True
    assert c_worker.input_data.get("did_t2") is True

    roles = [r.role for r in results]
    assert set(roles[:2]) == {"A", "B"}
    assert roles[2] == "C"

def test_max_worker_limit_respected():
    runtime = WorkerRuntime()
    runtime._get_tool_sequence = lambda w: [(t, {}) for t in w.spec.tools_granted]
    workers = []
    graph = DependencyGraph()

    for i in range(5):
        role = f"W{i}"
        workers.append(_make_worker(role, [DummyTool("fs.list", duration=0.2)]))
        graph.add_node(DepNode(operation_id=role))

    t0 = time.perf_counter()
    # N=2 limit for 5 tasks of 0.2s should take approx 0.6s
    results = runtime.execute_parallel(workers, graph=graph, n_max=2)
    t1 = time.perf_counter()

    assert len(results) == 5
    assert t1 - t0 >= 0.5, "Worker limit not respected"

def test_worker_failure_isolates_and_stops_dependents():
    runtime = WorkerRuntime()
    runtime._get_tool_sequence = lambda w: [(t, {}) for t in w.spec.tools_granted]

    # Mock _run_worker_logic to force a failure for worker B
    original_run_logic = runtime._run_worker_logic
    def mock_run_logic(w, trace, step_index, on_progress, t0, max_runtime_s):
        if w.spec.role == "B":
            raise RuntimeError("dummy error")
        return original_run_logic(w, trace, step_index, on_progress, t0, max_runtime_s)
    runtime._run_worker_logic = mock_run_logic

    w1 = _make_worker("A", [DummyTool("t1", duration=0.1)])
    w2 = _make_worker("B", [DummyTool("fail", duration=0.1)])
    w3 = _make_worker("C", [DummyTool("t3", duration=0.1)])

    # Need to set max_retries to 0 so it fails immediately instead of retrying
    w2.spec.constraints.max_retries = 0

    graph = DependencyGraph()
    for role in ["A", "B", "C"]:
        graph.add_node(DepNode(operation_id=role))
    graph.add_edge("A", "C")
    graph.add_edge("B", "C")

    results = runtime.execute_parallel([w1, w2, w3], graph=graph, n_max=2)

    # C should not execute because B failed, but it returns a skipped result
    assert len(results) == 3
    roles = {r.role for r in results}
    assert roles == {"A", "B", "C"}

    # B should have success=False, A should have success=True
    a_res = next(r for r in results if r.role == "A")
    b_res = next(r for r in results if r.role == "B")
    c_res = next(r for r in results if r.role == "C")

    assert a_res.success is True
    assert b_res.success is False
    assert "dummy error" in b_res.error

    assert c_res.success is False
    assert c_res.error == "Prerequisite failed or skipped."
