# benchmark.py
# ---------------------------------------------------------------
#  Benchmark algorithms on a directory of instances.
#
#  Supports:
#    - benchmark-level multiprocessing:
#        multiple instance files evaluated concurrently
#    - solver-level multiprocessing:
#        ALNS / Memetic can use multiple islands per instance
#        through solver_parallel_jobs
#
#  User-facing behavior:
#    - ALNS with solver_parallel_jobs = 1:
#        ordinary single-island ALNS
#    - ALNS with solver_parallel_jobs > 1:
#        parallel multi-start/island ALNS
#    - Memetic with solver_parallel_jobs = 1:
#        ordinary single-island Memetic
#    - Memetic with solver_parallel_jobs > 1:
#        parallel island Memetic
#
#  Saves:
#    - graphs/runtime_comparison.png
#    - graphs/distance_comparison.png
#    - graphs/evs_comparison.png
#    - graphs/runtime_comparison_details.csv
#    - graphs/runtime_comparison_summary.csv
# ---------------------------------------------------------------

from __future__ import annotations

import csv
import multiprocessing as mp
import queue
import re

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.parser import Parser
from core.solver_runner import solve_with_optional_timeout


DEFAULT_SOLVER_NAMES = [
    "ClarkeWright",
    "ALNS",
    "Memetic",
]

# These public solver names are internally parallel-capable.
PARALLEL_SOLVER_NAMES = {
    "ALNS",
    "Memetic",
}


# ===============================================================
#  CSV helpers
# ===============================================================

def _write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


# ===============================================================
#  Instance balancing helpers
# ===============================================================

def _instance_size_key(path: Path) -> str:
    """
    Extract input-size key from paths such as:

        data/Customer_5/foo.txt
        data/Customer_10/bar.txt

    Used only for distributing similar-size instances across
    benchmark workers.
    """
    for part in reversed(path.parts):
        match = re.search(r"Customer[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return path.parent.name


def _sort_size_key(x: str) -> Any:
    if str(x).isdigit():
        return int(x)

    return str(x)


def _balanced_chunks_by_input_size(
    instance_files: List[Path],
    num_workers: int,
) -> List[List[Path]]:
    """
    Split instance files across workers while giving each worker
    approximately identical input-size composition.

    Example with 300 files for each of:
      Customer_5, Customer_10, Customer_15, Customer_50, Customer_100

    and 10 workers:

      each worker receives about:
        30 Customer_5
        30 Customer_10
        30 Customer_15
        30 Customer_50
        30 Customer_100
    """
    by_size: Dict[str, List[Path]] = defaultdict(list)

    for path in instance_files:
        by_size[_instance_size_key(path)].append(path)

    chunks: List[List[Path]] = [[] for _ in range(num_workers)]

    for size_key in sorted(by_size.keys(), key=_sort_size_key):
        files = sorted(by_size[size_key])

        for i, path in enumerate(files):
            worker_idx = i % num_workers
            chunks[worker_idx].append(path)

    chunks = [chunk for chunk in chunks if chunk]

    for chunk in chunks:
        chunk.sort(
            key=lambda p: (
                _sort_size_key(_instance_size_key(p)),
                str(p),
            )
        )

    return chunks


def _chunk_size_counts(chunk: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)

    for path in chunk:
        counts[_instance_size_key(path)] += 1

    return dict(
        sorted(
            counts.items(),
            key=lambda kv: _sort_size_key(kv[0]),
        )
    )


def _effective_solver_parallel_jobs(
    solver_name: str,
    solver_parallel_jobs: int,
) -> int:
    """
    ALNS and Memetic are inherently parallel-capable.

    If solver_parallel_jobs == 1:
        they behave like their ordinary single-island versions.

    If solver_parallel_jobs > 1:
        they launch multiple independent islands and return the best.

    Other solvers run single-core per instance.
    """
    if solver_name in PARALLEL_SOLVER_NAMES:
        return max(1, int(solver_parallel_jobs))

    return 1


# ===============================================================
#  Benchmark execution on subset
# ===============================================================

def _benchmark_instance_subset(
    instance_files: List[Path],
    solver_names: List[str],
    timeout_sec: Optional[float],
    solver_parallel_jobs: int = 1,
    worker_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run benchmark on a subset of instance files.

    Used by:
      - sequential benchmark mode
      - each benchmark multiprocessing worker
    """
    detail_rows: List[Dict[str, Any]] = []

    prefix = ""
    if worker_id is not None:
        prefix = f"[worker {worker_id}] "

    for instance_path in instance_files:
        try:
            data = Parser(instance_path).data
            n_customers = len(data["customers"])

            print(
                f"{prefix}Instance: {instance_path.name} | "
                f"customers={n_customers}",
                flush=True,
            )

        except Exception as err:
            print(
                f"{prefix}Instance: {instance_path.name} | "
                f"PARSE ERROR: {err}",
                flush=True,
            )

            for solver_name in solver_names:
                effective_jobs = _effective_solver_parallel_jobs(
                    solver_name,
                    solver_parallel_jobs,
                )

                detail_rows.append({
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": "",
                    "solver": solver_name,
                    "solver_parallel_jobs": effective_jobs,
                    "status": "parse_error",
                    "time_sec": "",
                    "solution_found": "",
                    "evs": "",
                    "distance": "",
                    "error": str(err),
                })

            continue

        for solver_name in solver_names:
            effective_jobs = _effective_solver_parallel_jobs(
                solver_name,
                solver_parallel_jobs,
            )

            print(
                f"{prefix}  - {solver_name:<13} "
                f"jobs={effective_jobs:<3} ... ",
                end="",
                flush=True,
            )

            res = solve_with_optional_timeout(
                solver_name=solver_name,
                data=data,
                timeout_sec=timeout_sec,
                solver_parallel_jobs=effective_jobs,
            )

            if res["status"] == "ok":
                solved = res["result"]
                solution_found = solved.get("solution") is not None

                row = {
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": n_customers,
                    "solver": solver_name,
                    "solver_parallel_jobs": effective_jobs,
                    "status": "ok",
                    "time_sec": res["time_sec"],
                    "solution_found": solution_found,
                    "evs": solved.get("evs", ""),
                    "distance": solved.get("distance", ""),
                    "error": "",
                }

                if solution_found:
                    print(
                        f"{res['time_sec']:.3f} s | "
                        f"dist={row['distance']} | "
                        f"EVs={row['evs']}",
                        flush=True,
                    )
                else:
                    print(
                        f"{res['time_sec']:.3f} s | no solution",
                        flush=True,
                    )

            elif res["status"] == "timeout":
                row = {
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": n_customers,
                    "solver": solver_name,
                    "solver_parallel_jobs": effective_jobs,
                    "status": "timeout",
                    "time_sec": res.get("time_sec", ""),
                    "solution_found": "",
                    "evs": "",
                    "distance": "",
                    "error": "",
                }

                print("TIMEOUT", flush=True)

            else:
                row = {
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": n_customers,
                    "solver": solver_name,
                    "solver_parallel_jobs": effective_jobs,
                    "status": res.get("status", "error"),
                    "time_sec": res.get("time_sec", ""),
                    "solution_found": "",
                    "evs": "",
                    "distance": "",
                    "error": res.get("error", ""),
                }

                print(
                    f"ERROR: {row['error']}",
                    flush=True,
                )

            detail_rows.append(row)

    if worker_id is not None:
        print(
            f"{prefix}Finished {len(instance_files)} instances.",
            flush=True,
        )

    return detail_rows


# ===============================================================
#  Benchmark multiprocessing workers
# ===============================================================

def _benchmark_worker(
    worker_id: int,
    instance_files: List[Path],
    solver_names: List[str],
    timeout_sec: Optional[float],
    solver_parallel_jobs: int,
    result_queue: mp.Queue,
) -> None:
    """
    Top-level benchmark worker.

    Returns CSV-friendly result rows only.
    """
    try:
        rows = _benchmark_instance_subset(
            instance_files=instance_files,
            solver_names=solver_names,
            timeout_sec=timeout_sec,
            solver_parallel_jobs=solver_parallel_jobs,
            worker_id=worker_id,
        )

        result_queue.put({
            "worker_id": worker_id,
            "status": "ok",
            "rows": rows,
            "error": "",
        })

    except Exception as err:
        result_queue.put({
            "worker_id": worker_id,
            "status": "error",
            "rows": [],
            "error": repr(err),
        })


def _run_parallel_benchmark(
    instance_files: List[Path],
    solver_names: List[str],
    timeout_sec: Optional[float],
    max_workers: int,
    solver_parallel_jobs: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run benchmark using multiple benchmark workers.

    Each benchmark worker gets a balanced subset of instance sizes.
    """
    num_workers = min(max(1, int(max_workers)), len(instance_files))
    chunks = _balanced_chunks_by_input_size(instance_files, num_workers)

    print(f"\nParallel benchmark workers : {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        counts = _chunk_size_counts(chunk)
        counts_text = ", ".join(
            f"{size}: {count}" for size, count in counts.items()
        )

        print(
            f"  Worker {i:<3}: {len(chunk):<5} instances "
            f"({counts_text})"
        )

    result_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for worker_id, chunk in enumerate(chunks, start=1):
        proc = mp.Process(
            target=_benchmark_worker,
            args=(
                worker_id,
                chunk,
                solver_names,
                timeout_sec,
                solver_parallel_jobs,
                result_queue,
            ),
        )

        proc.start()
        processes.append(proc)

    messages: Dict[int, Dict[str, Any]] = {}

    # Read while processes are running so queue pipes do not fill up.
    while len(messages) < len(processes):
        try:
            msg = result_queue.get(timeout=0.5)
            messages[msg["worker_id"]] = msg

        except queue.Empty:
            if all(not proc.is_alive() for proc in processes):
                break

    for proc in processes:
        proc.join()

    # Drain messages that arrived just before join.
    while len(messages) < len(processes):
        try:
            msg = result_queue.get_nowait()
            messages[msg["worker_id"]] = msg
        except queue.Empty:
            break

    worker_errors: List[str] = []

    for i, proc in enumerate(processes, start=1):
        if i not in messages:
            worker_errors.append(
                f"Worker {i} exited without returning results. "
                f"Exit code: {proc.exitcode}"
            )
            continue

        msg = messages[i]

        if msg["status"] != "ok":
            worker_errors.append(
                f"Worker {i} failed: {msg.get('error', 'unknown error')}"
            )

    if worker_errors:
        raise RuntimeError(
            "One or more benchmark workers failed:\n"
            + "\n".join(worker_errors)
        )

    detail_rows: List[Dict[str, Any]] = []

    for worker_id in sorted(messages):
        detail_rows.extend(messages[worker_id]["rows"])

    return detail_rows


# ===============================================================
#  Summary generation
# ===============================================================

def _build_summary(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_time: Dict[tuple, List[float]] = defaultdict(list)
    grouped_dist: Dict[tuple, List[float]] = defaultdict(list)
    grouped_evs: Dict[tuple, List[float]] = defaultdict(list)

    for row in detail_rows:
        if row["status"] != "ok":
            continue

        solver = row["solver"]
        customers = int(row["customers"])
        solver_parallel_jobs = int(row.get("solver_parallel_jobs", 1) or 1)

        key = (
            solver,
            solver_parallel_jobs,
            customers,
        )

        if _is_number(row["time_sec"]):
            grouped_time[key].append(float(row["time_sec"]))

        # Distance and EV count are only meaningful if a solution exists.
        if row.get("solution_found") and _is_number(row["distance"]):
            grouped_dist[key].append(float(row["distance"]))

        if row.get("solution_found") and _is_number(row["evs"]):
            grouped_evs[key].append(float(row["evs"]))

    all_keys = sorted(
        set(grouped_time) | set(grouped_dist) | set(grouped_evs),
        key=lambda x: (
            x[2],  # customers
            x[0],  # solver
            x[1],  # solver_parallel_jobs
        ),
    )

    summary_rows: List[Dict[str, Any]] = []

    for solver, solver_parallel_jobs, customers in all_keys:
        key = (
            solver,
            solver_parallel_jobs,
            customers,
        )

        time_vals = grouped_time.get(key, [])
        dist_vals = grouped_dist.get(key, [])
        ev_vals = grouped_evs.get(key, [])

        summary_rows.append({
            "solver": solver,
            "solver_parallel_jobs": solver_parallel_jobs,
            "customers": customers,

            "time_runs": len(time_vals),
            "avg_time_sec": mean(time_vals) if time_vals else "",
            "min_time_sec": min(time_vals) if time_vals else "",
            "max_time_sec": max(time_vals) if time_vals else "",

            "distance_runs": len(dist_vals),
            "avg_distance": mean(dist_vals) if dist_vals else "",
            "min_distance": min(dist_vals) if dist_vals else "",
            "max_distance": max(dist_vals) if dist_vals else "",

            "ev_runs": len(ev_vals),
            "avg_evs": mean(ev_vals) if ev_vals else "",
            "min_evs": min(ev_vals) if ev_vals else "",
            "max_evs": max(ev_vals) if ev_vals else "",
        })

    return summary_rows


# ===============================================================
#  Plotting
# ===============================================================

def _plot_metric(
    summary_rows: List[Dict[str, Any]],
    y_key: str,
    ylabel: str,
    title: str,
    plot_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    by_solver: Dict[str, List[tuple]] = defaultdict(list)

    for row in summary_rows:
        y = row.get(y_key, "")

        if y == "":
            continue

        jobs = int(row.get("solver_parallel_jobs", 1) or 1)

        if row["solver"] in PARALLEL_SOLVER_NAMES:
            label = f"{row['solver']} ({jobs} cores)"
        else:
            label = row["solver"]

        by_solver[label].append(
            (
                int(row["customers"]),
                float(y),
            )
        )

    if not by_solver:
        plt.text(
            0.5,
            0.5,
            f"No data available for {ylabel}",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return

    for solver_label, pts in sorted(by_solver.items()):
        pts.sort()

        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=solver_label,
        )

    plt.xlabel("Number of customers")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


# ===============================================================
#  Public API
# ===============================================================

def benchmark_algorithms(
    data_root: Path,
    graphs_dir: Path,
    solver_names: Optional[List[str]] = None,
    timeout_sec: Optional[float] = None,
    max_workers: int = 1,
    solver_parallel_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Benchmark algorithms on all .txt instances under data_root.

    Parameters
    ----------
    data_root:
        Root directory containing instance files.

    graphs_dir:
        Output directory for CSV files and plots.

    solver_names:
        Solvers to run. Recommended for HPC comparison:

            ["ClarkeWright", "ALNS", "Memetic"]

        ALNS and Memetic are parallel-capable. With solver_parallel_jobs=1,
        they run as single-island/sequential versions. With
        solver_parallel_jobs>1, they run as parallel portfolios.

    timeout_sec:
        Wall-clock timeout per solver-instance run.
        For your HPC setup, use 10.0.

    max_workers:
        Number of benchmark workers, i.e. how many instances are
        evaluated concurrently.

    solver_parallel_jobs:
        Number of islands/processes used inside ALNS and Memetic for each
        individual instance.
    """
    data_root = Path(data_root)
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if solver_names is None:
        solver_names = list(DEFAULT_SOLVER_NAMES)

    instance_files = sorted(data_root.rglob("*.txt"))

    if not instance_files:
        raise FileNotFoundError(
            f"No .txt instance files found under: {data_root}"
        )

    max_workers = max(1, int(max_workers))
    solver_parallel_jobs = max(1, int(solver_parallel_jobs))

    print(f"\nBenchmarking instances under: {data_root}")
    print(f"Saving outputs to          : {graphs_dir}")
    print(f"Algorithms                 : {', '.join(solver_names)}")
    print(f"Total instances            : {len(instance_files)}")
    print(f"Benchmark workers          : {max_workers}")
    print(f"Solver parallel jobs       : {solver_parallel_jobs}")

    if timeout_sec is None:
        print("Per-run timeout            : none")
    else:
        print(f"Per-run timeout            : {timeout_sec:.1f} s")

    if max_workers <= 1:
        print("Execution mode             : sequential benchmark")

        detail_rows = _benchmark_instance_subset(
            instance_files=instance_files,
            solver_names=solver_names,
            timeout_sec=timeout_sec,
            solver_parallel_jobs=solver_parallel_jobs,
            worker_id=None,
        )

    else:
        print("Execution mode             : multiprocessing benchmark")

        detail_rows = _run_parallel_benchmark(
            instance_files=instance_files,
            solver_names=solver_names,
            timeout_sec=timeout_sec,
            max_workers=max_workers,
            solver_parallel_jobs=solver_parallel_jobs,
        )

    # Deterministic final output ordering.
    detail_rows.sort(
        key=lambda row: (
            str(row["instance_path"]),
            str(row["solver"]),
            int(row.get("solver_parallel_jobs", 1) or 1),
        )
    )

    summary_rows = _build_summary(detail_rows)

    detail_csv = graphs_dir / "runtime_comparison_details.csv"
    summary_csv = graphs_dir / "runtime_comparison_summary.csv"
    runtime_plot_path = graphs_dir / "runtime_comparison.png"
    distance_plot_path = graphs_dir / "distance_comparison.png"
    evs_plot_path = graphs_dir / "evs_comparison.png"

    _write_csv(
        detail_csv,
        detail_rows,
        fieldnames=[
            "instance",
            "instance_path",
            "customers",
            "solver",
            "solver_parallel_jobs",
            "status",
            "time_sec",
            "solution_found",
            "evs",
            "distance",
            "error",
        ],
    )

    _write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "solver",
            "solver_parallel_jobs",
            "customers",

            "time_runs",
            "avg_time_sec",
            "min_time_sec",
            "max_time_sec",

            "distance_runs",
            "avg_distance",
            "min_distance",
            "max_distance",

            "ev_runs",
            "avg_evs",
            "min_evs",
            "max_evs",
        ],
    )

    _plot_metric(
        summary_rows,
        y_key="avg_time_sec",
        ylabel="Average solve time (s)",
        title="2E-EVRP runtime comparison",
        plot_path=runtime_plot_path,
    )

    _plot_metric(
        summary_rows,
        y_key="avg_distance",
        ylabel="Average total distance",
        title="2E-EVRP distance comparison",
        plot_path=distance_plot_path,
    )

    _plot_metric(
        summary_rows,
        y_key="avg_evs",
        ylabel="Average number of EVs used",
        title="2E-EVRP EV usage comparison",
        plot_path=evs_plot_path,
    )

    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "runtime_plot_path": runtime_plot_path,
        "distance_plot_path": distance_plot_path,
        "evs_plot_path": evs_plot_path,
        "detail_rows": detail_rows,
        "summary_rows": summary_rows,
    }
