import pytest
from unittest.mock import MagicMock
from control_panel.worker.runtime import WorkerRuntime
from control_panel.common.models import WorkerSpec, WorkerConstraints
from control_panel.worker.models import WorkerResult
from control_panel.orchestrator.depgraph import DependencyGraph, DepNode

class MockWorker:
    def __init__(self, role, result_success=True):
        self.spec = WorkerSpec(
            worker_id=f"id_{role}",
            task_id="task_1",
            role=role,
            tools_granted=[],
            permission_mode="read_only",
            input_data={},
            constraints=WorkerConstraints()
        )
        self.result_success = result_success
        self.input_data = {}

    @property
    def id(self):
        return self.spec.worker_id

def setup_dag_test():
    """
    DAG:
        A -> C -> E
        B -> D
    """
    graph = DependencyGraph()
    for op in ["A", "B", "C", "D", "E"]:
        graph.add_node(DepNode(operation_id=op))

    graph.add_edge("A", "C")
    graph.add_edge("C", "E")
    graph.add_edge("B", "D")

    return graph

def _run_test(graph, worker_configs):
    runtime = WorkerRuntime()
    workers = []

    for role, success in worker_configs.items():
        workers.append(MockWorker(role, result_success=success))

    # Mock self._execute_worker to just return based on MockWorker setup
    def fake_execute(worker, on_progress):
        return WorkerResult(
            task_id=worker.spec.task_id,
            worker_id=worker.spec.worker_id,
            role=worker.spec.role,
            success=worker.result_success,
            error=None if worker.result_success else "Simulated error"
        )

    runtime._execute_worker = MagicMock(side_effect=fake_execute)

    results = runtime.execute_parallel(workers, graph, n_max=4)
    result_by_role = {r.role: r for r in results}
    return result_by_role, runtime._execute_worker.call_count

def test_a_and_b_succeed():
    """1. A and B succeed → C and D execute."""
    graph = setup_dag_test()
    configs = {"A": True, "B": True, "C": True, "D": True, "E": True}

    res, call_count = _run_test(graph, configs)

    assert call_count == 5
    assert all(r.success for r in res.values())

def test_a_fails_b_succeeds():
    """2. A fails, B succeeds → C skipped, D executes."""
    graph = setup_dag_test()
    configs = {"A": False, "B": True, "C": True, "D": True, "E": True}

    res, call_count = _run_test(graph, configs)

    assert res["A"].success is False
    assert res["B"].success is True
    assert res["D"].success is True

    assert res["C"].success is False
    assert res["C"].error == "Prerequisite failed or skipped."

    assert res["E"].success is False
    assert res["E"].error == "Prerequisite failed or skipped."

def test_a_succeeds_b_fails():
    """3. A succeeds, B fails → D skipped, C executes."""
    graph = setup_dag_test()
    configs = {"A": True, "B": False, "C": True, "D": True, "E": True}

    res, call_count = _run_test(graph, configs)

    assert res["A"].success is True
    assert res["B"].success is False
    assert res["C"].success is True
    assert res["E"].success is True

    assert res["D"].success is False
    assert res["D"].error == "Prerequisite failed or skipped."

def test_a_and_b_fail():
    """4. A and B fail → C and D skipped."""
    graph = setup_dag_test()
    configs = {"A": False, "B": False, "C": True, "D": True, "E": True}

    res, call_count = _run_test(graph, configs)

    assert res["A"].success is False
    assert res["B"].success is False

    assert res["C"].success is False
    assert res["D"].success is False
    assert res["E"].success is False

def test_transitive_failure():
    """5. Transitive failure: A fails → C skipped → E skipped."""
    graph = setup_dag_test()
    configs = {"A": False, "B": True, "C": True, "D": True, "E": True}

    res, call_count = _run_test(graph, configs)

    assert res["A"].success is False
    assert res["C"].success is False
    assert res["E"].success is False

    # E's failure should specifically be transitive because C didn't run
    assert res["E"].error == "Prerequisite failed or skipped."
