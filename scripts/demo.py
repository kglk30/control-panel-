#!/usr/bin/env python3
"""
Control Panel — Live Demo Script
Snapdragon AI Lab Build & Present Challenge 2026

This script runs a complete end-to-end demo showing:
1. System resource detection (CPU/NPU/GPU probe)
2. Backend selection (which inference engine is used)
3. File organization task with dry-run
4. Experience learning (memory after first run)
5. Second run using cached experience (faster)

Usage:
    $env:PYTHONUTF8="1"
    uv run python scripts/demo.py
    uv run python scripts/demo.py --path "C:\\Users\\me\\Downloads" --real
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Make demo runnable from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def run_demo(target_path: str, dry_run: bool = True) -> None:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.rule import Rule

    console = Console()

    from control_panel.common.logging import setup_logging
    setup_logging(verbose=False)

    console.print()
    console.print(Panel(
        "[bold cyan]Control Panel[/bold cyan] [dim]— AI-Native Task Engine[/dim]\n"
        "[dim]Snapdragon AI Lab Build & Present Challenge 2026[/dim]",
        style="cyan bold",
        expand=False,
    ))
    console.print()

    # ── 1. Resource Probe ────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1: Hardware Detection[/bold]"))
    console.print("[dim]Probing Snapdragon X Plus: CPU / Hexagon NPU / Adreno GPU...[/dim]")

    from control_panel.resources.governor import ResourceGovernor
    gov = ResourceGovernor()
    snap = gov.snapshot()

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Detail")

    table.add_row("Oryon CPU", "[green]✓ READY[/green]",
                  f"{snap.cpu_count} cores @ {snap.cpu_percent:.0f}% load")
    table.add_row("RAM", "[green]✓ READY[/green]",
                  f"{snap.ram_available_mb:,} MB free of {snap.ram_total_mb:,} MB")
    table.add_row("Disk", "[green]✓ READY[/green]",
                  f"{snap.disk_free_mb:,} MB free")
    table.add_row("Power", "[yellow]⚡ BATTERY[/yellow]" if snap.power_source == "battery" else "[green]⚡ PLUGGED[/green]",
                  f"{snap.battery_percent:.0f}%" if snap.battery_percent else "plugged in")

    for name, acc in snap.accelerators.items():
        status = "[green]✓ READY[/green]" if acc.available else "[red]✗ UNAVAILABLE[/red]"
        reason = acc.failure_reason[:50] if acc.failure_reason else "Available for inference"
        table.add_row(f"{name.upper()} Accelerator", status, reason)

    console.print(table)
    console.print()
    time.sleep(0.5)

    # ── 2. Backend Selection ──────────────────────────────────────────────
    console.print(Rule("[bold]Step 2: Inference Backend Selection[/bold]"))

    from control_panel.inference.runtime import InferenceRuntime
    runtime = InferenceRuntime(governor=gov)
    backends = runtime.available_backends()

    table2 = Table(show_header=True)
    table2.add_column("Priority")
    table2.add_column("Backend")
    table2.add_column("Accelerator")
    table2.add_column("Status")

    priority_map = {
        "snapdragon": ("1", "Hexagon NPU / Adreno GPU"),
        "local_cpu": ("2", "Oryon CPU"),
        "api": ("3", "Remote"),
        "rule_based": ("4", "In-process"),
    }

    for b in backends:
        pri, accel = priority_map.get(b.name, ("?", "?"))
        status = "[green]✓ ACTIVE[/green]" if b.available else "[dim]—[/dim]"
        table2.add_row(pri, b.name, accel, status)

    console.print(table2)
    console.print()

    active = next((b for b in backends if b.available), None)
    if active:
        console.print(f"[bold green]→ Using:[/bold green] [cyan]{active.name}[/cyan] ({active.device or 'none'})")
    console.print()
    time.sleep(0.5)

    # ── 3. Live Task ──────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 3: Execute — File Organization Task[/bold]"))
    mode_label = "[yellow]DRY RUN[/yellow]" if dry_run else "[bold red]LIVE[/bold red]"
    console.print(f"Task: [italic]\"Organize files by type\"[/italic]  Mode: {mode_label}")
    console.print(f"Path: [dim]{target_path}[/dim]")
    console.print()

    # Initialize persistence
    from control_panel.persistence.database import get_database, close_database
    db_path = Path(__file__).parent.parent / "data" / "demo.db"
    db_path.parent.mkdir(exist_ok=True)
    await get_database(db_path)

    from control_panel.orchestrator.core import Orchestrator
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    orch = Orchestrator(dry_run=dry_run)
    await orch.initialize()

    states_seen = []

    def on_progress(event):
        pass  # We'll show state transitions instead

    t0 = time.perf_counter()

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                  TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Initializing...", total=None)

        def on_prog(event):
            progress.update(task, description=f"[blue]{event.message}")

        result = await orch.run_task(
            raw_input="Organize files by type into category folders",
            target_path=target_path,
            on_progress=on_prog,
        )

    total_ms = int((time.perf_counter() - t0) * 1000)
    console.print()

    # ── 4. Results ────────────────────────────────────────────────────────
    if result.success:
        console.print(Panel(
            f"[bold green]✓ COMPLETED[/bold green]\n\n"
            f"[dim]Task ID:[/dim]  {result.task_id}\n"
            f"[dim]Backend:[/dim]  {result.backend_used} ({result.device_used})\n"
            f"[dim]Runtime:[/dim]  {result.total_runtime_ms}ms wall-clock\n"
            f"[dim]Workers:[/dim]  {len(result.worker_results)} ephemeral workers\n"
            f"[dim]Scanned:[/dim]  {result.items_processed} items\n"
            f"[dim]Mode:[/dim]     {'DRY RUN (no files moved)' if dry_run else 'LIVE (files organized)'}",
            title="[bold]Result[/bold]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]✗ FAILED[/bold red]: {result.error}",
            border_style="red",
        ))

    console.print()

    # ── 5. Worker trace ───────────────────────────────────────────────────
    console.print(Rule("[bold]Step 4: Worker Execution Trace[/bold]"))
    for wr in result.worker_results:
        icon = "✓" if wr.success else "✗"
        color = "green" if wr.success else "red"
        steps = len(wr.trace.steps) if wr.trace else 0
        console.print(f"  [{color}]{icon}[/{color}] [bold]{wr.role}[/bold] — {steps} tool calls — {wr.runtime_ms}ms")
        if wr.trace:
            for step in wr.trace.steps[:5]:
                sc = "green" if step.status == "success" else "red"
                console.print(
                    f"      [{sc}]→[/{sc}] [dim]{step.tool_name}[/dim] "
                    f"[dim]({step.duration_ms}ms)[/dim]"
                )
            if len(wr.trace.steps) > 5:
                console.print(f"      [dim]... +{len(wr.trace.steps)-5} more steps[/dim]")

    console.print()

    # ── 6. Experience memory ──────────────────────────────────────────────
    console.print(Rule("[bold]Step 5: Experience Memory (Learning)[/bold]"))
    from control_panel.experience.store import ExperienceStore
    from control_panel.persistence.database import get_database
    db = await get_database(db_path)
    store = ExperienceStore(db)
    history = await store.list_recent(limit=5)

    console.print(f"[dim]Experiences stored:[/dim] {len(history)}")
    if history:
        last = history[0]
        console.print(
            f"[dim]Last task:[/dim] {last.task.raw_input[:60]} → "
            f"{'[green]✓[/green]' if last.outcome.success else '[red]✗[/red]'} "
            f"{last.execution.total_runtime_ms}ms via {last.backend_used}"
        )
        console.print("[dim]The next run will retrieve this experience and plan faster.[/dim]")

    console.print()
    console.print(Panel(
        "[bold cyan]Demo complete.[/bold cyan]\n\n"
        "Control Panel is a Windows ARM64-native AI task engine that:\n"
        "  • Runs entirely locally on Snapdragon X Plus\n"
        "  • Routes inference to CPU / NPU / GPU based on availability\n"
        "  • Executes file operations safely through ephemeral workers\n"
        "  • Learns from every task to get faster over time\n"
        "  • Zero cloud dependency — works offline",
        style="cyan",
    ))

    await close_database()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Control Panel live demo")
    parser.add_argument("--path", default=None,
                        help="Target path to organize (default: temp demo dir)")
    parser.add_argument("--real", action="store_true",
                        help="Run in LIVE mode (actually moves files, not dry-run)")
    args = parser.parse_args()

    if args.path:
        target = args.path
    else:
        # Create a demo directory with sample files
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp(prefix="cp_demo_"))
        for name in ["photo.jpg", "photo2.png", "doc.pdf", "report.docx",
                     "song.mp3", "video.mp4", "code.py", "notes.txt",
                     "archive.zip", "data.csv", "presentation.pptx"]:
            (tmp / name).write_bytes(f"demo file: {name}".encode())
        print(f"Demo directory created: {tmp}")
        target = str(tmp)

    asyncio.run(run_demo(target, dry_run=not args.real))


if __name__ == "__main__":
    main()
