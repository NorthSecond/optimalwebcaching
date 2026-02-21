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
import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .trace_parser import parse_trace, TraceData
from .topology import build_topology, Topology
from .solver import solve, SolverConfig, SolverResult
from .output import process_result, print_summary, write_output, FOOResult

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

        # Use streaming parser so --max-requests on huge .zst traces does not
        # require full-file decompression.
        trace = parse_trace(trace_path, max_requests=max_requests)

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

    trace = parse_trace(trace_path, max_requests=max_requests)

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
    trace = parse_trace(trace_path, max_requests=max_requests)
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
    dvar_file: Optional[str] = typer.Option(None, "--dvar-file", "-d", help="Load dvars from C++ FOO output (skip solver)"),
    max_pairs_per_point: int = typer.Option(100, "--max-pairs-per-point", "-p", help="Max pairs per eviction event"),
    max_iters: int = typer.Option(10000, "--max-iters", "-i", help="Maximum solver iterations"),
    tol: float = typer.Option(1e-4, "--tol", "-t", help="Convergence tolerance"),
    step_size: float = typer.Option(0.1, "--step", help="PDHG step size"),
    max_requests: Optional[int] = typer.Option(None, "--max-requests", "-n", help="Limit trace size"),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
    sampling_strategy: str = typer.Option(
        "stratified",
        "--sampling-strategy", "-s",
        help="Sampling strategy: random (baseline), similar (difficult cases), stratified (mixed, recommended)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Export pairwise (hi, lo) dataset from FOO solution.

    Uses libCacheSim for accurate cache simulation. Samples pairs at actual
    eviction events (true decision points) where cache is full.
    Extracts pairs where hi objects are kept (dvar=1) and lo objects are evicted (dvar=0).

    Sampling strategies:
    - random: Random sampling (baseline, may create trivially separable pairs)
    - similar: Focus on pairs with similar features (difficult cases)
    - stratified: Mix of easy (30%) and difficult (70%) cases (recommended)

    For optimal accuracy, use --dvar-file to load dvars from C++ FOO output:
        ./OHRgoal/FOO/foo trace.dat cache_size 4 dvars.txt
        python -m foo_jax export-pairs trace.dat cache_size output.csv --dvar-file dvars.txt
    """
    try:
        from .pairwise_libcachesim import (
            export_pairwise_libcachesim,
            load_dvars_from_cpp_foo,
        )
    except ImportError as exc:
        console.print(
            "[red]Missing optional dependency 'libcachesim'.[/red] "
            "Install it before using export-pairs."
        )
        raise typer.Exit(1) from exc

    # Validate sampling strategy
    valid_strategies = ["random", "similar", "stratified"]
    if sampling_strategy not in valid_strategies:
        console.print(f"[red]Error: Invalid sampling strategy '{sampling_strategy}'[/red]")
        console.print(f"Valid options: {', '.join(valid_strategies)}")
        raise typer.Exit(1)

    console.print(f"[bold blue]FOO-JAX Pairwise Export[/bold blue]")
    console.print(f"  Trace: {trace_path}")
    console.print(f"  Cache: {cache_size:,} bytes ({cache_size/1024/1024:.1f} MiB)")
    if dvar_file:
        console.print(f"  [cyan]Using external dvars:[/cyan] {dvar_file}")
    console.print(f"  Max pairs/eviction: {max_pairs_per_point:,}")
    console.print(f"  Sampling strategy: {sampling_strategy}")
    console.print()

    total_start = time.time()

    # Phase 1: Parse trace
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Parsing trace...", total=None)

        trace = parse_trace(trace_path, max_requests=max_requests)

        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Parsed {trace.n_requests:,} requests, {trace.n_unique_objects:,} unique objects")

    if dvar_file:
        # Load external dvars from C++ FOO output
        console.print()
        console.print("[cyan]Loading dvars from C++ FOO output...[/cyan]")
        dvars = load_dvars_from_cpp_foo(dvar_file, trace)

        # Compute OHR from dvars
        dvars_rounded = np.round(dvars).astype(np.int32)
        n_dvar_1 = np.sum(dvars_rounded == 1)
        # OHR = sum(dvars) / n_requests (exact for integer dvars)
        ohr_estimate = float(np.sum(dvars)) / trace.n_requests
        console.print(f"  [green]✓[/green] Loaded {len(dvars):,} dvars")
        console.print(f"  hi (dvar=1): {n_dvar_1:,} ({100*n_dvar_1/len(dvars):.1f}%)")

        # Create minimal FOOResult for pairwise generator
        result = FOOResult(
            dvars=dvars,
            n_requests=trace.n_requests,
            n_unique_objects=trace.n_unique_objects,
            float_hits=float(np.sum(dvars)),
            integer_hits=int(np.sum(dvars > 0.99)),
            ohr=ohr_estimate,
            primal_obj=0.0,
            iterations=0,
            converged=True,
            x_outer=np.zeros(0)
        )
    else:
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

    # Phase 5: Generate and export pairwise data
    console.print()
    console.print("[cyan]Generating pairwise data (libCacheSim simulation)...[/cyan]")

    n_pairs = export_pairwise_libcachesim(
        trace=trace,
        foo_result=result,
        output_path=output_path,
        cache_size=cache_size,
        max_pairs_per_point=max_pairs_per_point,
        seed=seed,
        sampling_strategy=sampling_strategy
    )

    total_time = time.time() - total_start

    console.print()
    if n_pairs > 0:
        console.print(f"[green]Pairwise data written to:[/green] {output_path}")
        console.print(f"  Pairs generated: {n_pairs:,}")
    else:
        console.print("[yellow]Warning: No pairs generated.[/yellow]")
    console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
