"""Control Panel CLI - Typer commands.

Commands:
  run <task>   Execute a natural-language task
  history      Show past task executions
  backends     Show inference backend availability
  status       Show system resource status
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from control_panel.cli.display import Display
from control_panel.common.config import get_app_dirs, load_config, set_config
from control_panel.common.logging import setup_logging

__all__ = ["app"]

app = typer.Typer(
    name="control-panel",
    help="AI-native control plane for Windows ARM64 PCs.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def _early_bootstrap(ctx: typer.Context) -> None:
    """Bootstrap logging before any command runs."""
    if ctx.invoked_subcommand is not None:
        # Setup with defaults - each command may override via --verbose
        dirs = get_app_dirs()
        dirs.ensure_all()
        setup_logging(log_dir=dirs.logs_dir, verbose=False)


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Natural-language task description")],
    path: Annotated[Optional[str], typer.Option("--path", "-p", help="Target filesystem path")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate without mutating filesystem")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full execution trace")] = False,
    backend: Annotated[Optional[str], typer.Option("--backend", "-b", help="Force backend: snapdragon|local_cpu|api|rule_based")] = None,
    config_file: Annotated[Optional[str], typer.Option("--config", help="Path to config TOML file")] = None,
) -> None:
    """Execute a natural-language task."""
    _bootstrap(verbose=verbose, config_file=config_file)
    display = Display(verbose=verbose)
    display.print_header()

    if path:
        p = Path(path)
        if not p.exists():
            err_console.print(f"[red]Error:[/red] Path does not exist: {path}")
            raise typer.Exit(code=2)
    else:
        # Default to current directory
        path = str(Path.cwd())

    console.print(f"[dim]Task:[/dim] {task}")
    console.print(f"[dim]Path:[/dim] {path}")
    if dry_run:
        console.print("[yellow]Dry run mode - no files will be modified.[/yellow]")
    console.print()

    # Initialize and run
    result = asyncio.run(_run_task(task, path, dry_run=dry_run, verbose=verbose, force_backend=backend, display=display))

    # Display result
    display.print_result(result)

    # Exit codes: 0=success, 1=partial failure, 2=total failure
    if result.success:
        raise typer.Exit(code=0)
    elif result.items_processed > 0:
        raise typer.Exit(code=1)
    else:
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# history command
# ---------------------------------------------------------------------------

@app.command()
def history(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of records to show")] = 20,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Show past task execution history."""
    _bootstrap(verbose=verbose)
    display = Display(verbose=verbose)
    display.print_header()

    experiences = asyncio.run(_get_history(limit))
    display.print_history(experiences)


# ---------------------------------------------------------------------------
# backends command
# ---------------------------------------------------------------------------

@app.command()
def backends(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Show inference backend availability."""
    _bootstrap(verbose=verbose)
    display = Display(verbose=verbose)
    display.print_header()

    from control_panel.inference.runtime import InferenceRuntime
    runtime = InferenceRuntime()
    available = runtime.available_backends()
    display.print_backends(available)


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

@app.command()
def status(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Show current system resource status."""
    _bootstrap(verbose=verbose)
    from rich.table import Table
    from control_panel.resources.governor import ResourceGovernor

    governor = ResourceGovernor()
    snap = governor.snapshot()

    table = Table(title="System Status", show_header=True)
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("CPU Usage", f"{snap.cpu_percent:.1f}%")
    table.add_row("CPU Cores", str(snap.cpu_count))
    table.add_row("RAM Total", f"{snap.ram_total_mb:,} MB")
    table.add_row("RAM Available", f"{snap.ram_available_mb:,} MB")
    table.add_row("Disk Free", f"{snap.disk_free_mb:,} MB")
    table.add_row("Power", snap.power_source or "unknown")
    if snap.battery_percent is not None:
        table.add_row("Battery", f"{snap.battery_percent:.0f}%")

    for name, acc in snap.accelerators.items():
        status_str = "[green]OK[/green]" if acc.available else "[red]FAIL[/red]"
        reason = f" ({acc.failure_reason[:50]})" if acc.failure_reason else ""
        table.add_row(f"Accelerator:{name.upper()}", f"{status_str}{reason}")

    console.print(table)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bootstrap(verbose: bool = False, config_file: Optional[str] = None) -> None:
    """Initialize logging, dirs, and config."""
    dirs = get_app_dirs()
    dirs.ensure_all()
    setup_logging(log_dir=dirs.logs_dir, verbose=verbose)

    if config_file:
        cfg = load_config(Path(config_file))
        set_config(cfg)


async def _run_task(
    task: str,
    path: str,
    dry_run: bool,
    verbose: bool,
    force_backend: Optional[str],
    display: Display,
) -> "OrchestratorResult":
    from control_panel.orchestrator.core import Orchestrator
    from control_panel.persistence.database import close_database

    orch = Orchestrator(dry_run=dry_run)
    try:
        await orch.initialize()
        with display.make_progress() as progress:
            prog_task = progress.add_task("Initializing...", total=None)
            callback = display.progress_handler(progress, prog_task)
            return await orch.run_task(
                raw_input=task,
                target_path=path,
                on_progress=callback,
            )
    finally:
        await close_database()


async def _get_history(limit: int) -> list:
    from control_panel.experience.store import ExperienceStore
    from control_panel.persistence.database import close_database, get_database

    try:
        db = await get_database()
        return await ExperienceStore(db).list_recent(limit=limit)
    finally:
        await close_database()
