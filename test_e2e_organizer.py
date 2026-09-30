import os
import shutil
import pathlib
import time
import asyncio
from control_panel.orchestrator.core import Orchestrator

async def main():
    base_dir = pathlib.Path("scratch/test_e2e_organize").resolve()
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    
    (base_dir / "pic.jpg").write_text("fake image")
    (base_dir / "notes.txt").write_text("fake text")
    
    orch = Orchestrator()
    await orch.initialize()
    
    result = await orch.run_task(
        raw_input="organize this folder",
        target_path=str(base_dir)
    )
    
    print("Orchestrator success:", result.success)
    if not result.success:
        print("Error:", result.error)
        
    for w in result.worker_results:
        print(f"Worker {w.role}: {w.success}, trace steps: {len(w.trace.steps) if w.trace else 0}")
    
    print("Items processed:", result.items_processed)
    for p in base_dir.rglob("*"):
        print("  Found:", p.relative_to(base_dir))

if __name__ == "__main__":
    asyncio.run(main())
