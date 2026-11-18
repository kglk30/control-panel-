from control_panel.common.logging import setup_logging
setup_logging(verbose=False)

from control_panel.inference.models import get_registry
from control_panel.resources.governor import ResourceGovernor
from control_panel.inference.selector import BackendSelectionEngine
from control_panel.common.models import InferenceRequest

reg = get_registry()
gov = ResourceGovernor()
snap = gov.snapshot()
constraints = gov.get_resource_constraints()

phi3 = reg.get("phi3_mini_int4_cpu")
print(f"Available RAM: {snap.ram_available_mb} MB")
print(f"Max model RAM: {constraints.max_model_ram_mb} MB")
print(f"Phi3 needs:    {phi3.requirements.estimated_ram_mb} MB")
print(f"Disk free:     {snap.disk_free_mb} MB")
print(f"On disk:       {phi3.is_available_on_disk()}")

sel = BackendSelectionEngine(reg)
req = InferenceRequest(prompt="test", task_type="analyze")
result = sel.select(req, constraints, snap.accelerators)
print(f"Selected: backend={result.backend} model={result.model_id} runtime={result.runtime_type}")
