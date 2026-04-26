# benchmark.py
# ---------------------------------------------------------------
#  Benchmark algorithms on a directory of instances.
#  Saves:
#    - graphs/runtime_comparison.png
#    - graphs/distance_comparison.png
#    - graphs/evs_comparison.png
#    - graphs/runtime_comparison_details.csv
#    - graphs/runtime_comparison_summary.csv
# ---------------------------------------------------------------

from __future__ import annotations

import csv
import queue
import re
import multiprocessing as mp

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.parser import Parser
from core.solver_runner import solve_with_optional_timeout


DEFAULT_SOLVER_NAMES = ["ClarkeWright", "ALNS"]


# ===============================================================
# CSV helpers
# ===============================================================

def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


# ===============================================================
# Balanced multiprocessing helpers
# ===============================================================

def _instance_size_key(path: Path) -> str:
    """
    Extracts a size key from paths such as:

        data/Customer_5/foo.txt
        data/Customer_10/bar.txt

    The returned key is used only for balancing work between processes.
    """
    for part in reversed(path.parts):
        match = re.search(r"Customer[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    # Fallback if the directory does not follow Customer_N naming.
    return path.parent.name


def _balanced_chunks_by_input_size(
    instance_files: List[Path],
    num_workers: int,
) -> List[List[Path]]:
    """
    Split files between workers while preserving roughly identical input-size
    composition in every worker.

    Example with 5 input sizes and 300 files per size:

        Worker 1 gets roughly:
            5-customer files   : 300 / num_workers
            10-customer files  : 300 / num_workers
            15-customer files  : 300 / num_workers
            50-customer files  : 300 / num_workers
            100-customer files : 300 / num_workers

    This prevents one worker from receiving mostly large instances while
    another receives mostly small instances.
    """
    by_size: Dict[str, List[Path]] = defaultdict(list)

    for path in instance_files:
        by_size[_instance_size_key(path)].append(path)

    chunks: List[List[Path]] = [[] for _ in range(num_workers)]

    for size_key in sorted(by_size.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        files = sorted(by_size[size_key])

        for i, path in enumerate(files):
            worker_idx = i % num_workers
            chunks[worker_idx].append(path)

    # Remove empty chunks if num_workers > number of files.
    chunks = [chunk for chunk in chunks if chunk]

    # Keep output deterministic.
    for chunk in chunks:
        chunk.sort(
            key=lambda p: (
                int(_instance_size_key(p)) if _instance_size_key(p).isdigit() else 10**9,
                str(p),
            )
        )

    return chunks


def _chunk_size_counts(chunk: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for path in chunk:
        counts[_instance_size_key(path)] += 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 10**9))


# ===============================================================
# Benchmark execution
# ===============================================================

def _benchmark_instance_subset(
    instance_files: List[Path],
    solver_names: List[str],
    timeout_sec: Optional[float],
    worker_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run benchmark on a subset of instance files.

    This function is used both by:
      - sequential benchmarking
      - each multiprocessing worker
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
                f"{prefix}Instance: {instance_path.name} | customers={n_customers}",
                flush=True,
            )

        except Exception as err:
            print(
                f"{prefix}Instance: {instance_path.name} | PARSE ERROR: {err}",
                flush=True,
            )

            for solver_name in solver_names:
                detail_rows.append({
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": "",
                    "solver": solver_name,
                    "status": "parse_error",
                    "time_sec": "",
                    "solution_found": "",
                    "evs": "",
                    "distance": "",
                    "error": str(err),
                })

            continue

        for solver_name in solver_names:
            res = solve_with_optional_timeout(
                solver_name=solver_name,
                data=data,
                timeout_sec=timeout_sec,
            )

            if res["status"] == "ok":
                solved = res["result"]
                solution_found = solved.get("solution") is not None

                row = {
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": n_customers,
                    "solver": solver_name,
                    "status": "ok",
                    "time_sec": res["time_sec"],
                    "solution_found": solution_found,
                    "evs": solved.get("evs", ""),
                    "distance": solved.get("distance", ""),
                    "error": "",
                }

                if solution_found:
                    print(
                        f"{prefix}{instance_path.name} | "
                        f"{solver_name:<13} | "
                        f"{res['time_sec']:.3f} s | "
                        f"dist={row['distance']} | EVs={row['evs']}",
                        flush=True,
                    )
                else:
                    print(
                        f"{prefix}{instance_path.name} | "
                        f"{solver_name:<13} | "
                        f"{res['time_sec']:.3f} s | no solution",
                        flush=True,
                    )

            elif res["status"] == "timeout":
                row = {
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": n_customers,
                    "solver": solver_name,
                    "status": "timeout",
                    "time_sec": res.get("time_sec", ""),
                    "solution_found": "",
                    "evs": "",
                    "distance": "",
                    "error": "",
                }

                print(
                    f"{prefix}{instance_path.name} | "
                    f"{solver_name:<13} | TIMEOUT",
                    flush=True,
                )

            else:
                row = {
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": n_customers,
                    "solver": solver_name,
                    "status": res.get("status", "error"),
                    "time_sec": res.get("time_sec", ""),
                    "solution_found": "",
                    "evs": "",
                    "distance": "",
                    "error": res.get("error", ""),
                }

                print(
                    f"{prefix}{instance_path.name} | "
                    f"{solver_name:<13} | ERROR: {row['error']}",
                    flush=True,
                )

            detail_rows.append(row)

    if worker_id is not None:
        print(
            f"{prefix}Finished {len(instance_files)} instances.",
            flush=True,
        )

    return detail_rows


def _benchmark_worker(
    worker_id: int,
    instance_files: List[Path],
    solver_names: List[str],
    timeout_sec: Optional[float],
    result_queue: mp.Queue,
) -> None:
    """
    Top-level multiprocessing worker.

    It returns only CSV-friendly result rows, not Solution objects.
    """
    try:
        rows = _benchmark_instance_subset(
            instance_files=instance_files,
            solver_names=solver_names,
            timeout_sec=timeout_sec,
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
) -> List[Dict[str, Any]]:
    num_workers = min(max_workers, len(instance_files))
    chunks = _balanced_chunks_by_input_size(instance_files, num_workers)

    print(f"\nParallel workers           : {len(chunks)}")

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
                result_queue,
            ),
        )

        proc.start()
        processes.append(proc)

    messages: Dict[int, Dict[str, Any]] = {}

    # Read results while processes are running. This avoids the queue pipe
    # filling up and blocking a worker at process exit.
    while len(messages) < len(processes):
        try:
            msg = result_queue.get(timeout=0.5)
            messages[msg["worker_id"]] = msg

        except queue.Empty:
            if all(not proc.is_alive() for proc in processes):
                break

    for proc in processes:
        proc.join()

    # Drain any messages that arrived just before join.
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
# Summary and plotting
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
        key = (solver, customers)

        if _is_number(row["time_sec"]):
            grouped_time[key].append(float(row["time_sec"]))

        # Only meaningful if a feasible solution was found.
        if row.get("solution_found") and _is_number(row["distance"]):
            grouped_dist[key].append(float(row["distance"]))

        if row.get("solution_found") and _is_number(row["evs"]):
            grouped_evs[key].append(float(row["evs"]))

    all_keys = sorted(
        set(grouped_time) | set(grouped_dist) | set(grouped_evs),
        key=lambda x: (x[1], x[0]),
    )

    summary_rows = []

    for solver, customers in all_keys:
        time_vals = grouped_time.get((solver, customers), [])
        dist_vals = grouped_dist.get((solver, customers), [])
        ev_vals = grouped_evs.get((solver, customers), [])

        summary_rows.append({
            "solver": solver,
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

        by_solver[row["solver"]].append(
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

    for solver, pts in sorted(by_solver.items()):
        pts.sort()

        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]

        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=solver,
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
# Public API
# ===============================================================

def benchmark_algorithms(
    data_root: Path,
    graphs_dir: Path,
    solver_names: Optional[List[str]] = None,
    timeout_sec: Optional[float] = None,
    max_workers: int = 1,
) -> Dict[str, Any]:
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

    print(f"\nBenchmarking instances under: {data_root}")
    print(f"Saving outputs to          : {graphs_dir}")
    print(f"Algorithms                 : {', '.join(solver_names)}")
    print(f"Total instances            : {len(instance_files)}")

    if timeout_sec is None:
        print("Per-run timeout            : none")
    else:
        print(f"Per-run timeout            : {timeout_sec:.1f} s")

    if max_workers <= 1:
        print("Execution mode             : sequential")

        detail_rows = _benchmark_instance_subset(
            instance_files=instance_files,
            solver_names=solver_names,
            timeout_sec=timeout_sec,
            worker_id=None,
        )

    else:
        print("Execution mode             : multiprocessing")

        detail_rows = _run_parallel_benchmark(
            instance_files=instance_files,
            solver_names=solver_names,
            timeout_sec=timeout_sec,
            max_workers=max_workers,
        )

    # Keep final output deterministic.
    detail_rows.sort(
        key=lambda row: (
            str(row["instance_path"]),
            str(row["solver"]),
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
