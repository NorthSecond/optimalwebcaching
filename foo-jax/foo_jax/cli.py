"""
CLI for FOO-JAX: GPU-accelerated Flow Offline Optimal caching.

Usage:
    python -m foo_jax solve <trace> <cache_size> <output> [options]
    python -m foo_jax info <trace>

Examples:
    # Solve FOO on a trace
    python -m foo_jax solve trace.zst 128974848 output.txt

    # With options
    python -m foo_jax solve trace.zst 1000000 output.txt --max-iters 20000 --tol 1e-3

    # Get trace info
    python -m foo_jax info trace.zst
"""

import time
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .trace_parser import parse_trace_fast, TraceData
from .topology import build_topology, Topology
from .solver import solve, SolverConfig, SolverResult
from .output import process_result, print_summary, write_output, FOOResult
from .pairwise import compute_features, extract_pairs, write_pairs_csv

app = typer.Typer(
    name="foo-jax",
    help="GPU-accelerated FOO (Flow Offline Optimal) caching algorithm"
)
console = Console()


@app.command(name="solve")
def solve_cmd(
    trace_path: str = typer.Argument(..., help="Path to trace file (.dat or .zst)"),
    cache_size: int = typer.Argument(..., help="Cache size in bytes"),
    output_path: str = typer.Argument(..., help="Output file for decision variables"),
    max_iters: int = typer.Option(10000, "--max-iters", "-i", help="Maximum iterations"),
    tol: float = typer.Option(1e-4, "--tol", "-t", help="Convergence tolerance"),
    step_size: float = typer.Option(0.1, "--step", "-s", help="PDHG step size"),
    max_requests: Optional[int] = typer.Option(None, "--max-requests", "-n", help="Limit trace size"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Solve FOO optimal caching problem.

    Computes optimal cache decisions using the r2HPDHG algorithm.
    """
    console.print(f"[bold blue]FOO-JAX Solver[/bold blue]")
    console.print(f"  Trace: {trace_path}")
    console.print(f"  Cache: {cache_size:,} bytes ({cache_size/1024/1024:.1f} MiB)")
    console.print()

    total_start = time.time()

    # Phase 1: Parse trace
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Parsing trace...", total=None)

        trace = parse_trace_fast(trace_path, max_requests)

        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Parsed {trace.n_requests:,} requests, {trace.n_unique_objects:,} unique objects")

    # Phase 2: Build topology
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Building topology...", total=None)

        topo = build_topology(trace, cache_size)

        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Built graph: {topo.n_nodes:,} nodes, {topo.n_inner + topo.n_outer:,} arcs")

    # Phase 3: Solve
    console.print()
    console.print("[cyan]Solving with r2HPDHG...[/cyan]")

    config = SolverConfig(
        max_iters=max_iters,
        tol=tol,
        step_size=step_size,
        verbose=verbose,
        check_interval=max(100, max_iters // 100)
    )

    def progress_callback(msg):
        if verbose:
            console.print(f"    {msg}")

    solve_start = time.time()
    solver_result = solve(topo, config, progress_callback)
    solve_time = time.time() - solve_start

    status = "[green]Converged[/green]" if solver_result.converged else "[yellow]Max iterations[/yellow]"
    console.print(f"  {status} in {solver_result.iterations:,} iterations ({solve_time:.2f}s)")

    # Phase 4: Process results
    result = process_result(trace, topo, solver_result)

    total_time = time.time() - total_start

    # Print summary
    console.print()
    print_summary(result, cache_size, total_time)

    # Write output
    write_output(output_path, trace, result.dvars)
    console.print()
    console.print(f"[green]Output written to:[/green] {output_path}")


@app.command()
def info(
    trace_path: str = typer.Argument(..., help="Path to trace file"),
    max_requests: Optional[int] = typer.Option(None, "--max-requests", "-n", help="Limit trace size")
):
    """
    Show information about a trace file.
    """
    console.print(f"[bold blue]Trace Info[/bold blue]")
    console.print(f"  File: {trace_path}")
    console.print()

    trace = parse_trace_fast(trace_path, max_requests)

    console.print(f"  Requests: {trace.n_requests:,}")
    console.print(f"  Unique objects: {trace.n_unique_objects:,}")
    console.print(f"  Reuse ratio: {(trace.n_requests - trace.n_unique_objects) / trace.n_requests * 100:.1f}%")
    console.print()

    # Size statistics
    sizes = trace.obj_sizes
    console.print(f"  Object sizes:")
    console.print(f"    Min: {sizes.min():,} bytes")
    console.print(f"    Max: {sizes.max():,} bytes")
    console.print(f"    Mean: {sizes.mean():.0f} bytes")
    console.print(f"    Total: {sizes.sum():,} bytes ({sizes.sum()/1024/1024/1024:.2f} GiB)")


@app.command()
def benchmark(
    trace_path: str = typer.Argument(..., help="Path to trace file"),
    cache_size: int = typer.Argument(..., help="Cache size in bytes"),
    max_requests: int = typer.Option(10000, "--max-requests", "-n", help="Trace size to test"),
    max_iters: int = typer.Option(5000, "--max-iters", "-i", help="Maximum iterations"),
):
    """
    Run a benchmark on a trace.
    """
    console.print(f"[bold blue]FOO-JAX Benchmark[/bold blue]")
    console.print(f"  Trace: {trace_path}")
    console.print(f"  Cache: {cache_size:,} bytes")
    console.print(f"  Max requests: {max_requests:,}")
    console.print()

    # Parse
    parse_start = time.time()
    trace = parse_trace_fast(trace_path, max_requests)
    parse_time = time.time() - parse_start

    # Build topology
    topo_start = time.time()
    topo = build_topology(trace, cache_size)
    topo_time = time.time() - topo_start

    # Solve
    config = SolverConfig(max_iters=max_iters, tol=1e-4)
    solve_start = time.time()
    solver_result = solve(topo, config)
    solve_time = time.time() - solve_start

    # Results
    result = process_result(trace, topo, solver_result)

    console.print(f"[green]Benchmark Results[/green]")
    console.print(f"  Parse time: {parse_time*1000:.1f}ms")
    console.print(f"  Topology time: {topo_time*1000:.1f}ms")
    console.print(f"  Solve time: {solve_time*1000:.1f}ms")
    console.print(f"  Time/iteration: {solve_time/solver_result.iterations*1000:.3f}ms")
    console.print(f"  Converged: {solver_result.converged}")
    console.print(f"  OHR: {result.ohr:.4f}")


@app.command()
def export_pairs(
    trace_path: str = typer.Argument(..., help="Path to trace file (.dat or .zst)"),
    cache_size: int = typer.Argument(..., help="Cache size in bytes"),
    output_path: str = typer.Argument(..., help="Output CSV file for pairwise data"),
    max_pairs: int = typer.Option(10000, "--max-pairs", "-p", help="Maximum pairs to extract"),
    threshold_hi: float = typer.Option(0.7, "--threshold-hi", help="dvar threshold for HIT (keep)"),
    threshold_lo: float = typer.Option(0.3, "--threshold-lo", help="dvar threshold for MISS (evict)"),
    max_iters: int = typer.Option(10000, "--max-iters", "-i", help="Maximum solver iterations"),
    tol: float = typer.Option(1e-4, "--tol", "-t", help="Convergence tolerance"),
    step_size: float = typer.Option(0.1, "--step", "-s", help="PDHG step size"),
    max_requests: Optional[int] = typer.Option(None, "--max-requests", "-n", help="Limit trace size"),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Export pairwise (hi, lo) dataset from FOO solution.

    Extracts pairs where hi objects are kept (high dvar) and lo objects are evicted (low dvar).
    Useful for training learning-to-cache models.
    """
    console.print(f"[bold blue]FOO-JAX Pairwise Export[/bold blue]")
    console.print(f"  Trace: {trace_path}")
    console.print(f"  Cache: {cache_size:,} bytes ({cache_size/1024/1024:.1f} MiB)")
    console.print(f"  Max pairs: {max_pairs:,}")
    console.print()

    total_start = time.time()

    # Phase 1: Parse trace
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Parsing trace...", total=None)

        trace = parse_trace_fast(trace_path, max_requests)

        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Parsed {trace.n_requests:,} requests, {trace.n_unique_objects:,} unique objects")

    # Phase 2: Build topology
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Building topology...", total=None)

        topo = build_topology(trace, cache_size)

        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Built graph: {topo.n_nodes:,} nodes, {topo.n_inner + topo.n_outer:,} arcs")

    # Phase 3: Solve
    console.print()
    console.print("[cyan]Solving with r2HPDHG...[/cyan]")

    config = SolverConfig(
        max_iters=max_iters,
        tol=tol,
        step_size=step_size,
        verbose=verbose,
        check_interval=max(100, max_iters // 100)
    )

    def progress_callback(msg):
        if verbose:
            console.print(f"    {msg}")

    solve_start = time.time()
    solver_result = solve(topo, config, progress_callback)
    solve_time = time.time() - solve_start

    status = "[green]Converged[/green]" if solver_result.converged else "[yellow]Max iterations[/yellow]"
    console.print(f"  {status} in {solver_result.iterations:,} iterations ({solve_time:.2f}s)")

    # Phase 4: Process results
    result = process_result(trace, topo, solver_result)

    console.print()
    console.print(f"  OHR: {result.ohr:.6f} ({result.ohr*100:.2f}%)")

    # Phase 5: Extract pairs
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Extracting pairwise data...", total=None)

        pairs = extract_pairs(
            trace=trace,
            result=result,
            topo=topo,
            max_pairs=max_pairs,
            threshold_hi=threshold_hi,
            threshold_lo=threshold_lo,
            seed=seed
        )

        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Extracted {len(pairs):,} pairs")

    if len(pairs) == 0:
        console.print("[yellow]Warning: No pairs found. Try adjusting thresholds.[/yellow]")
        return

    # Phase 6: Write output
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Writing CSV...", total=None)

        write_pairs_csv(output_path, pairs)

        progress.update(task, completed=True)

    total_time = time.time() - total_start

    console.print()
    console.print(f"[green]Pairwise data written to:[/green] {output_path}")
    console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
