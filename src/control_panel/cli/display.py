"""Rich terminal display helpers for Control Panel CLI."""

from __future__ import annotations

from typing import Optional

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from control_panel.common.models import WorkerEvent
from control_panel.orchestrator.core import OrchestratorResult
from control_panel.worker.models import WorkerResult

__all__ = ["Display"]

console = Console()


class Display:
    """Rich-based terminal display for the Control Panel CLI."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def print_header(self) -> None:
        """Print the Control Panel banner."""
        console.print(Panel(
            "[bold cyan]Control Panel[/bold cyan] [dim]v0.1 - AI-native task engine[/dim]",
            style="cyan",
            expand=False,
        ))

    def print_result(self, result: OrchestratorResult) -> None:
        """Print the final task result summary."""
        if result.success:
            status = "[bold green]COMPLETED[/bold green]"
        else:
            status = "[bold red]FAILED[/bold red]"

        console.print()
        console.print(Panel(
            f"{status}\n"
            f"[dim]Task ID:[/dim] {result.task_id}\n"
            f"[dim]Backend:[/dim] {result.backend_used} ({result.device_used})\n"
            f"[dim]Runtime:[/dim] {result.total_runtime_ms}ms\n"
            f"[dim]Workers:[/dim] {len(result.worker_results)}\n"
            f"[dim]Items:[/dim] {result.items_processed}",
            title="Result",
            border_style="green" if result.success else "red",
        ))

        if result.error:
            console.print(f"[red]Error:[/red] {result.error}")

        if self._verbose:
            self._print_worker_traces(result.worker_results)

    def _print_worker_traces(self, results: list[WorkerResult]) -> None:
        """Print detailed worker traces in verbose mode."""
        for r in results:
            status = "[green]OK[/green]" if r.success else "[red]FAIL[/red]"
            console.print(f"\n{status} [bold]{r.role}[/bold] ({r.runtime_ms}ms)")
            if r.trace and r.trace.steps:
                for step in r.trace.steps:
                    step_status = "OK" if step.status == "success" else "FAIL"
                    console.print(
                        f"    {step_status} [dim]{step.tool_name}[/dim] "
                        f"[dim]{step.duration_ms}ms[/dim]"
                    )
                    if self._verbose and step.tool_output_summary:
                        console.print(f"       -> {step.tool_output_summary[:120]}")

    def print_backends(self, backends: list) -> None:
        """Print backend availability table."""
        table = Table(title="Inference Backends", show_header=True)
        table.add_column("Backend", style="bold")
        table.add_column("Available")
        table.add_column("Device")
        table.add_column("Note")

        for b in backends:
            status = "[green]OK[/green]" if b.available else "[red]FAIL[/red]"
            table.add_row(
                b.name,
                status,
                b.device or "-",
                b.reason or "",
            )
        console.print(table)

    def print_history(self, experiences: list) -> None:
        """Print experience history table."""
        if not experiences:
            console.print("[dim]No history yet.[/dim]")
            return

        table = Table(title="Task History", show_header=True)
        table.add_column("Date", style="dim")
        table.add_column("Category")
        table.add_column("Success")
        table.add_column("Backend")
        table.add_column("Runtime")
        table.add_column("Input", max_width=40)

        for e in experiences:
            status = "[green]OK[/green]" if e.outcome.success else "[red]FAIL[/red]"
            table.add_row(
                e.created_at.strftime("%Y-%m-%d %H:%M") if e.created_at else "-",
                e.task.category,
                status,
                e.backend_used,
                f"{e.execution.total_runtime_ms}ms",
                e.task.raw_input[:40],
            )

        console.print(table)

    def make_progress(self) -> Progress:
        """Create a Rich progress bar for task execution."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )

    def progress_handler(self, progress: Progress, task_id: object) -> callable:
        """Return a WorkerEvent callback that updates the progress display."""
        def on_progress(event: WorkerEvent) -> None:
            progress.update(task_id, description=f"[blue]{event.message}")

        return on_progress
