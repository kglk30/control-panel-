import pytest
import threading
from unittest.mock import MagicMock, AsyncMock, patch
import time

from control_panel.orchestrator.core import Orchestrator
from control_panel.common.models import TaskSpec, ExecutionPlan, WorkerSpec, WorkerConstraints
from control_panel.templates.models import Template

@pytest.mark.asyncio
async def test_e2e_dag_parallel_execution():
    """Prove V0.2 complete DAG parallelism and data passing."""
    # 1. Setup Orchestrator (real runtime, real governor, real dep graph)
    orch = Orchestrator()
    orch._experience_store.find_by_category = AsyncMock(return_value=[])
    orch._experience_store.save = AsyncMock()

    # Mock analysis to trigger our mock template
    orch._analyzer.analyze = MagicMock(return_value=TaskSpec(
        task_id="e2e_dag_task",
        raw_input="do the parallel thing",
        objective="Run A and B parallel, then C",
        category="parallel.test",
        structured_requirements={}
    ))

    # 2. Provide the Mock Template
    template = Template(
        id="tpl_parallel",
        version=1,
        name="Parallel Template",
        category="parallel.test",
        description="Parallel DAG",
        workflow={
            "workers": [
                {"role": "worker_a", "depends_on": []},
                {"role": "worker_b", "depends_on": []},
                {"role": "worker_c", "depends_on": ["worker_a", "worker_b"]}
            ]
        },
        requirements={}
    )
    orch._template_store.find_by_category = AsyncMock(return_value=[template])

    # 3. Use synchronization primitives to prove overlap
    barrier = threading.Barrier(2, timeout=5.0)

    # Track execution timing and order
    exec_history = []
    lock = threading.Lock()

    # We will override the Planner to actually yield workers
    # instead of doing its default file_management logic
    original_plan = orch._planner.plan
    def fake_plan(task_spec, constraints, template, target_path, strategy):
        # We need to construct WorkerSpecs because the MVP planner
        # doesn't natively map arbitrary DAGs into generic plans yet
        # (It only has file_management or generic single worker)
        workers = [
            WorkerSpec(
                worker_id="w_a", task_id="e2e", role="worker_a", tools_granted=[],
                permission_mode="read_only", input_data={}, constraints=WorkerConstraints()
            ),
            WorkerSpec(
                worker_id="w_b", task_id="e2e", role="worker_b", tools_granted=[],
                permission_mode="read_only", input_data={}, constraints=WorkerConstraints()
            ),
            WorkerSpec(
                worker_id="w_c", task_id="e2e", role="worker_c", tools_granted=[],
                permission_mode="read_only", input_data={}, constraints=WorkerConstraints()
            )
        ]
        return ExecutionPlan(
            task_id="e2e",
            execution_mode=strategy.strategy_type if strategy else "parallel",
            workers=workers
        )
    orch._planner.plan = fake_plan

    # 4. Mock Worker Logic to pause and accumulate
    def fake_run_worker_logic(worker, trace, step_index, on_progress, t0, max_runtime_s):
        nonlocal exec_history
        role = worker.spec.role

        with lock:
            exec_history.append(f"{role}_start")

        if role in ["worker_a", "worker_b"]:
            # Prove overlap by waiting at barrier
            barrier.wait()
            # Return specific output
            return {f"output_{role}": f"data_from_{role}"}

        if role == "worker_c":
            # worker_c should receive inputs from A and B
            assert "output_worker_a" in worker.input_data, "Missing output from A"
            assert "output_worker_b" in worker.input_data, "Missing output from B"
            assert worker.input_data["output_worker_a"] == "data_from_worker_a"
            assert worker.input_data["output_worker_b"] == "data_from_worker_b"
            with lock:
                exec_history.append(f"{role}_done")
            return {"final": "success"}

    # We need to patch the internal execution to use our fake_run_worker_logic
    with patch.object(orch._worker_runtime, "_run_worker_logic", side_effect=fake_run_worker_logic):

        # We also need to spy on execute_parallel to check the DependencyGraph groups
        with patch.object(orch._worker_runtime, "execute_parallel", wraps=orch._worker_runtime.execute_parallel) as spy_exec:

            # Execute
            result = await orch.run_task("do the parallel thing")

            # Verify Orchestrator chose parallel
            spy_exec.assert_called_once()
            _, kwargs = spy_exec.call_args

            graph = kwargs["graph"]
            groups = graph.parallel_groups()

            # Verify Dependency Graph
            assert len(groups) == 2
            assert set(groups[0]) == {"worker_a", "worker_b"}
            assert groups[1] == ["worker_c"]

            # Verify concurrency limit was enforced
            assert kwargs["n_max"] <= orch._governor.get_resource_constraints().max_concurrent_workers

            # Verify execution overlap actually happened (barrier was passed)
            # If they didn't overlap, barrier.wait() would have raised BrokenBarrierError or TimeoutError

            # Verify execution ordering
            # C MUST start after A and B start
            assert exec_history[0] in ["worker_a_start", "worker_b_start"]
            assert exec_history[1] in ["worker_a_start", "worker_b_start"]
            assert exec_history[2] == "worker_c_start"
            assert exec_history[3] == "worker_c_done"

            assert result.success is True
