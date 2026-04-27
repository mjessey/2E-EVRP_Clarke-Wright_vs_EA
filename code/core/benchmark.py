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
#  Also produces plots split by instance type:
#    - Clustered: filename starts with "C"
#    - Random:    filename starts with "R"
#    - Mixed:     filename starts with "RC"
#
#  Important:
#    Check "RC" before "R", otherwise mixed instances would be
#    classified as random.
#
#  For each metric:
#    1. Overall algorithm comparison
#    2. Clustered-only algorithm comparison
#    3. Random-only algorithm comparison
#    4. Mixed-only algorithm comparison
#    5. One per-solver plot comparing that solver across types
#
#  With n algorithms and 3 metrics:
#      total plots = 3 * (4 + n)
#
#  Saves legacy plots:
#    - graphs/runtime_comparison.png
#    - graphs/distance_comparison.png
#    - graphs/evs_comparison.png
#
#  Saves additional plots:
#    - graphs/runtime_comparison_clustered.png
#    - graphs/runtime_comparison_random.png
#    - graphs/runtime_comparison_mixed.png
#    - graphs/runtime_by_type_<solver>.png
#    - and equivalent distance/EV plots
#
#  Saves:
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

INSTANCE_TYPES = [
    "Clustered",
    "Random",
    "Mixed",
]

INSTANCE_TYPE_ALL = "All"


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
#  Instance type helpers
# ===============================================================

def _instance_type_from_name(path: Path) -> str:
    """
    Determine distribution type from filename.

    Rules:
      - starts with "RC" -> Mixed
      - starts with "R"  -> Random
      - starts with "C"  -> Clustered

    "RC" must be checked before "R".
    """
    name = path.name.upper()

    if name.startswith("RC"):
        return "Mixed"

    if name.startswith("R"):
        return "Random"

    if name.startswith("C"):
        return "Clustered"

    return "Unknown"


def _safe_filename_part(s: str) -> str:
    """
    Make a solver/type label safe for filenames.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


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


def _sort_size_key(x: str) -> tuple:
    """
    Sort numeric size keys numerically and non-numeric keys lexically.
    """
    if str(x).isdigit():
        return (0, int(x))

    return (1, str(x))


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
        instance_type = _instance_type_from_name(instance_path)

        try:
            data = Parser(instance_path).data
            n_customers = len(data["customers"])

            print(
                f"{prefix}Instance: {instance_path.name} | "
                f"customers={n_customers} | type={instance_type}",
                flush=True,
            )

        except Exception as err:
            print(
                f"{prefix}Instance: {instance_path.name} | "
                f"type={instance_type} | PARSE ERROR: {err}",
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
                    "instance_type": instance_type,
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
                    "instance_type": instance_type,
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
                    "instance_type": instance_type,
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
                    "instance_type": instance_type,
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

        type_counts: Dict[str, int] = defaultdict(int)
        for path in chunk:
            type_counts[_instance_type_from_name(path)] += 1

        type_counts_text = ", ".join(
            f"{typ}: {count}"
            for typ, count in sorted(type_counts.items())
        )

        print(
            f"  Worker {i:<3}: {len(chunk):<5} instances "
            f"({counts_text}) | types=({type_counts_text})"
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

def _build_summary_for_type(
    detail_rows: List[Dict[str, Any]],
    instance_type_filter: Optional[str],
    summary_instance_type: str,
) -> List[Dict[str, Any]]:
    """
    Build summary rows for either:
      - all instance types combined
      - one specific instance type

    summary_instance_type is written into the CSV, e.g.:
      - "All"
      - "Clustered"
      - "Random"
      - "Mixed"
    """
    grouped_time: Dict[tuple, List[float]] = defaultdict(list)
    grouped_dist: Dict[tuple, List[float]] = defaultdict(list)
    grouped_evs: Dict[tuple, List[float]] = defaultdict(list)

    for row in detail_rows:
        if row["status"] != "ok":
            continue

        row_type = row.get("instance_type", "Unknown")

        if instance_type_filter is not None and row_type != instance_type_filter:
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
            "instance_type": summary_instance_type,
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


def _build_summary(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build summary for:
      - all instance types combined
      - Clustered only
      - Random only
      - Mixed only
    """
    summary_rows: List[Dict[str, Any]] = []

    summary_rows.extend(
        _build_summary_for_type(
            detail_rows=detail_rows,
            instance_type_filter=None,
            summary_instance_type=INSTANCE_TYPE_ALL,
        )
    )

    for instance_type in INSTANCE_TYPES:
        summary_rows.extend(
            _build_summary_for_type(
                detail_rows=detail_rows,
                instance_type_filter=instance_type,
                summary_instance_type=instance_type,
            )
        )

    summary_rows.sort(
        key=lambda row: (
            str(row["instance_type"]),
            int(row["customers"]),
            str(row["solver"]),
            int(row.get("solver_parallel_jobs", 1) or 1),
        )
    )

    return summary_rows


# ===============================================================
#  Plotting
# ===============================================================

def _solver_label(solver: str, jobs: int) -> str:
    if solver in PARALLEL_SOLVER_NAMES:
        return f"{solver} ({jobs} cores)"

    return solver


def _plot_no_data(plot_path: Path, message: str) -> None:
    plt.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def _plot_metric_algorithm_comparison(
    summary_rows: List[Dict[str, Any]],
    y_key: str,
    ylabel: str,
    title: str,
    plot_path: Path,
    instance_type: str,
) -> None:
    """
    Plot algorithm comparison for one instance type.

    instance_type can be:
      - "All"
      - "Clustered"
      - "Random"
      - "Mixed"
    """
    plt.figure(figsize=(9, 6))

    by_solver: Dict[str, List[tuple]] = defaultdict(list)

    for row in summary_rows:
        if row.get("instance_type") != instance_type:
            continue

        y = row.get(y_key, "")

        if y == "":
            continue

        jobs = int(row.get("solver_parallel_jobs", 1) or 1)
        label = _solver_label(row["solver"], jobs)

        by_solver[label].append(
            (
                int(row["customers"]),
                float(y),
            )
        )

    if not by_solver:
        _plot_no_data(
            plot_path,
            f"No data available for {ylabel}\nInstance type: {instance_type}",
        )
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


def _plot_metric_solver_by_type(
    summary_rows: List[Dict[str, Any]],
    y_key: str,
    ylabel: str,
    title: str,
    plot_path: Path,
    solver_name: str,
    solver_parallel_jobs: int,
) -> None:
    """
    Plot one solver against itself across distribution types.

    Lines:
      - Clustered
      - Random
      - Mixed
    """
    plt.figure(figsize=(9, 6))

    by_type: Dict[str, List[tuple]] = defaultdict(list)

    for row in summary_rows:
        if row.get("instance_type") not in INSTANCE_TYPES:
            continue

        if row.get("solver") != solver_name:
            continue

        row_jobs = int(row.get("solver_parallel_jobs", 1) or 1)

        if row_jobs != solver_parallel_jobs:
            continue

        y = row.get(y_key, "")

        if y == "":
            continue

        by_type[row["instance_type"]].append(
            (
                int(row["customers"]),
                float(y),
            )
        )

    if not by_type:
        _plot_no_data(
            plot_path,
            f"No data available for {ylabel}\nSolver: {solver_name}",
        )
        return

    # Keep a stable, meaningful order.
    for instance_type in INSTANCE_TYPES:
        pts = by_type.get(instance_type, [])

        if not pts:
            continue

        pts.sort()

        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=instance_type,
        )

    plt.xlabel("Number of customers")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Instance type")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def _make_all_plots(
    summary_rows: List[Dict[str, Any]],
    solver_names: List[str],
    graphs_dir: Path,
) -> Dict[str, Path]:
    """
    Create all metric plots.

    For each metric:
      1. Overall comparison across all instance types
      2. Clustered comparison
      3. Random comparison
      4. Mixed comparison
      5. Per-solver type comparison
    """
    plot_paths: Dict[str, Path] = {}

    metric_specs = [
        {
            "metric_name": "runtime",
            "y_key": "avg_time_sec",
            "ylabel": "Average solve time (s)",
            "title_base": "2E-EVRP runtime comparison",
            "legacy_path": graphs_dir / "runtime_comparison.png",
            "type_prefix": "runtime_comparison",
            "self_prefix": "runtime_by_type",
        },
        {
            "metric_name": "distance",
            "y_key": "avg_distance",
            "ylabel": "Average total distance",
            "title_base": "2E-EVRP distance comparison",
            "legacy_path": graphs_dir / "distance_comparison.png",
            "type_prefix": "distance_comparison",
            "self_prefix": "distance_by_type",
        },
        {
            "metric_name": "evs",
            "y_key": "avg_evs",
            "ylabel": "Average number of EVs used",
            "title_base": "2E-EVRP EV usage comparison",
            "legacy_path": graphs_dir / "evs_comparison.png",
            "type_prefix": "evs_comparison",
            "self_prefix": "evs_by_type",
        },
    ]

    # Determine actual solver/job combinations present in the summary.
    solver_job_pairs = sorted({
        (
            row["solver"],
            int(row.get("solver_parallel_jobs", 1) or 1),
        )
        for row in summary_rows
        if row.get("instance_type") == INSTANCE_TYPE_ALL
    })

    # Preserve selected solver order where possible.
    def solver_pair_sort_key(pair: tuple) -> tuple:
        solver, jobs = pair

        try:
            idx = solver_names.index(solver)
        except ValueError:
            idx = 10**9

        return (idx, solver, jobs)

    solver_job_pairs.sort(key=solver_pair_sort_key)

    for spec in metric_specs:
        metric_name = spec["metric_name"]
        y_key = spec["y_key"]
        ylabel = spec["ylabel"]
        title_base = spec["title_base"]

        # 1. Overall comparison, legacy filename.
        legacy_path = spec["legacy_path"]
        plot_paths[f"{metric_name}_all"] = legacy_path

        _plot_metric_algorithm_comparison(
            summary_rows=summary_rows,
            y_key=y_key,
            ylabel=ylabel,
            title=title_base,
            plot_path=legacy_path,
            instance_type=INSTANCE_TYPE_ALL,
        )

        # 2-4. Type-specific algorithm comparisons.
        for instance_type in INSTANCE_TYPES:
            type_slug = instance_type.lower()
            type_path = graphs_dir / f"{spec['type_prefix']}_{type_slug}.png"
            plot_paths[f"{metric_name}_{type_slug}"] = type_path

            _plot_metric_algorithm_comparison(
                summary_rows=summary_rows,
                y_key=y_key,
                ylabel=ylabel,
                title=f"{title_base} — {instance_type}",
                plot_path=type_path,
                instance_type=instance_type,
            )

        # 5. Per-solver self comparison across instance types.
        for solver_name, jobs in solver_job_pairs:
            solver_slug = _safe_filename_part(
                _solver_label(solver_name, jobs)
            )

            self_path = graphs_dir / f"{spec['self_prefix']}_{solver_slug}.png"
            plot_paths[f"{metric_name}_by_type_{solver_slug}"] = self_path

            _plot_metric_solver_by_type(
                summary_rows=summary_rows,
                y_key=y_key,
                ylabel=ylabel,
                title=f"{title_base} by input type — {_solver_label(solver_name, jobs)}",
                plot_path=self_path,
                solver_name=solver_name,
                solver_parallel_jobs=jobs,
            )

    return plot_paths


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

    type_counts: Dict[str, int] = defaultdict(int)
    for path in instance_files:
        type_counts[_instance_type_from_name(path)] += 1

    print(f"\nBenchmarking instances under: {data_root}")
    print(f"Saving outputs to          : {graphs_dir}")
    print(f"Algorithms                 : {', '.join(solver_names)}")
    print(f"Total instances            : {len(instance_files)}")
    print(f"Benchmark workers          : {max_workers}")
    print(f"Solver parallel jobs       : {solver_parallel_jobs}")

    print("Instance types             : " + ", ".join(
        f"{typ}={count}"
        for typ, count in sorted(type_counts.items())
    ))

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

    _write_csv(
        detail_csv,
        detail_rows,
        fieldnames=[
            "instance",
            "instance_path",
            "customers",
            "instance_type",
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
            "instance_type",
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

    plot_paths = _make_all_plots(
        summary_rows=summary_rows,
        solver_names=solver_names,
        graphs_dir=graphs_dir,
    )

    print("\nGenerated plots:")
    for key, path in sorted(plot_paths.items()):
        print(f"  {key:<35}: {path}")

    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,

        # Legacy return keys expected by main.py.
        "runtime_plot_path": plot_paths["runtime_all"],
        "distance_plot_path": plot_paths["distance_all"],
        "evs_plot_path": plot_paths["evs_all"],

        # New complete plot dictionary.
        "plot_paths": plot_paths,

        "detail_rows": detail_rows,
        "summary_rows": summary_rows,
    }
