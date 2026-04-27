# scaling_benchmark.py
# ---------------------------------------------------------------
#  Scaling benchmark for ALNS and Memetic.
#
#  Purpose:
#    Test how solution quality changes with:
#      1. More cores/islands
#      2. More solve time
#
#  Experiments:
#
#    A. ALNS core scaling
#       - customer size fixed at 100
#       - solve time fixed at 10 seconds
#       - core counts: 1, 2, 4, 8
#       - ALNS ignores MAX_ITERATIONS in this scaling benchmark
#         and stops by time.
#       - plots:
#           average distance vs core count
#           average EVs vs core count
#
#    B. Memetic core scaling
#       - customer size fixed at 100
#       - solve time fixed at 10 seconds
#       - core counts: 1, 2, 4, 8
#       - plots:
#           average distance vs core count
#           average EVs vs core count
#
#    C. ALNS time scaling
#       - customer size fixed at 100
#       - cores fixed at 1
#       - solve times: 5, 10, 15 seconds
#       - ALNS ignores MAX_ITERATIONS in this scaling benchmark
#         and stops by time.
#       - plots:
#           average distance vs solve time
#           average EVs vs solve time
#
#    D. Memetic time scaling
#       - customer size fixed at 100
#       - cores fixed at 1
#       - solve times: 5, 10, 15 seconds
#       - plots:
#           average distance vs solve time
#           average EVs vs solve time
#
#  Each plot contains four lines:
#      All
#      Clustered
#      Random
#      Mixed
#
#  Output:
#    graphs/scaling_benchmark_details.csv
#    graphs/scaling_benchmark_summary.csv
#
#    graphs/scaling_ALNS_distance_vs_cores.png
#    graphs/scaling_ALNS_evs_vs_cores.png
#    graphs/scaling_ALNS_distance_vs_time.png
#    graphs/scaling_ALNS_evs_vs_time.png
#
#    graphs/scaling_Memetic_distance_vs_cores.png
#    graphs/scaling_Memetic_evs_vs_cores.png
#    graphs/scaling_Memetic_distance_vs_time.png
#    graphs/scaling_Memetic_evs_vs_time.png
# ---------------------------------------------------------------

from __future__ import annotations

import csv
import multiprocessing as mp
import queue
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.parser import Parser
from core.solver_runner import solve_with_optional_timeout


CORE_COUNTS = [1, 2, 4, 8]
SOLVE_TIMES_SEC = [4, 7, 10]

CUSTOMER_SIZE = 50
FIXED_CORE_SCALING_TIME_SEC = 7.0

INSTANCE_TYPES = ["Clustered", "Random", "Mixed"]
INSTANCE_TYPE_ALL = "All"


# ===============================================================
# Helpers
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


def _instance_type_from_name(path: Path) -> str:
    """
    Determine distribution type from filename.

    Important:
      "RC" must be checked before "R".

    Rules:
      - RC* -> Mixed
      - R*  -> Random
      - C*  -> Clustered
    """
    name = path.name.upper()

    if name.startswith("RC"):
        return "Mixed"

    if name.startswith("R"):
        return "Random"

    if name.startswith("C"):
        return "Clustered"

    return "Unknown"


def _find_customer_100_instances(data_root: Path) -> List[Path]:
    """
    Prefer data_root/Customer_100 if it exists.
    Otherwise, search all .txt files and keep those whose path suggests
    Customer_100.
    """
    data_root = Path(data_root)
    customer_100_dir = data_root / "Customer_100"

    if customer_100_dir.exists() and customer_100_dir.is_dir():
        files = sorted(customer_100_dir.rglob("*.txt"))
    else:
        files = sorted(
            p
            for p in data_root.rglob("*.txt")
            if "Customer_100" in str(p)
        )

    if not files:
        raise FileNotFoundError(
            f"No Customer_100 .txt files found under: {data_root}"
        )

    return files


def _balanced_chunks_by_type(
    instance_files: List[Path],
    num_workers: int,
) -> List[List[Path]]:
    """
    Distribute Clustered/Random/Mixed instances evenly across workers.
    """
    by_type: Dict[str, List[Path]] = defaultdict(list)

    for path in instance_files:
        by_type[_instance_type_from_name(path)].append(path)

    chunks: List[List[Path]] = [[] for _ in range(num_workers)]

    for typ in sorted(by_type):
        files = sorted(by_type[typ])

        for i, path in enumerate(files):
            chunks[i % num_workers].append(path)

    chunks = [chunk for chunk in chunks if chunk]

    for chunk in chunks:
        chunk.sort(key=lambda p: (_instance_type_from_name(p), str(p)))

    return chunks


def _chunk_type_counts(chunk: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)

    for path in chunk:
        counts[_instance_type_from_name(path)] += 1

    return dict(sorted(counts.items()))


def _alns_scaling_options(solver_name: str) -> Optional[Dict[str, Any]]:
    """
    In the scaling benchmark only, ALNS ignores MAX_ITERATIONS and
    stops by time. Normal benchmarks do not use this option.
    """
    if solver_name == "ALNS":
        return {
            "unlimited_iterations": True,
        }

    return None


# ===============================================================
# Worker execution
# ===============================================================

def _run_subset_for_scaling_point(
    instance_files: List[Path],
    experiment: str,
    solver_name: str,
    x_value: float,
    x_label: str,
    solver_parallel_jobs: int,
    timeout_sec: float,
    worker_id: Optional[int] = None,
    solver_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run one experiment point on a subset of Customer_100 instances.
    """
    rows: List[Dict[str, Any]] = []

    prefix = ""
    if worker_id is not None:
        prefix = f"[worker {worker_id}] "

    for instance_path in instance_files:
        instance_type = _instance_type_from_name(instance_path)

        try:
            data = Parser(instance_path).data
            n_customers = len(data["customers"])
        except Exception as err:
            rows.append({
                "experiment": experiment,
                "solver": solver_name,
                "instance": instance_path.name,
                "instance_path": str(instance_path),
                "customers": "",
                "instance_type": instance_type,
                "x_label": x_label,
                "x_value": x_value,
                "solver_parallel_jobs": solver_parallel_jobs,
                "timeout_sec": timeout_sec,
                "status": "parse_error",
                "time_sec": "",
                "solution_found": "",
                "evs": "",
                "distance": "",
                "error": str(err),
            })
            continue

        options_text = ""
        if solver_options:
            options_text = f" options={solver_options}"

        print(
            f"{prefix}{experiment} | {solver_name} | "
            f"{x_label}={x_value:g} | jobs={solver_parallel_jobs} | "
            f"timeout={timeout_sec:g}s{options_text} | "
            f"{instance_path.name} ({instance_type}) ... ",
            end="",
            flush=True,
        )

        res = solve_with_optional_timeout(
            solver_name=solver_name,
            data=data,
            timeout_sec=timeout_sec,
            solver_parallel_jobs=solver_parallel_jobs,
            solver_options=solver_options,
        )

        if res["status"] == "ok":
            solved = res["result"]
            solution_found = solved.get("solution") is not None

            row = {
                "experiment": experiment,
                "solver": solver_name,
                "instance": instance_path.name,
                "instance_path": str(instance_path),
                "customers": n_customers,
                "instance_type": instance_type,
                "x_label": x_label,
                "x_value": x_value,
                "solver_parallel_jobs": solver_parallel_jobs,
                "timeout_sec": timeout_sec,
                "status": "ok",
                "time_sec": res.get("time_sec", ""),
                "solution_found": solution_found,
                "evs": solved.get("evs", ""),
                "distance": solved.get("distance", ""),
                "error": "",
            }

            if solution_found:
                print(
                    f"{res.get('time_sec', 0):.3f}s | "
                    f"dist={row['distance']} | EVs={row['evs']}",
                    flush=True,
                )
            else:
                print(
                    f"{res.get('time_sec', 0):.3f}s | no solution",
                    flush=True,
                )

        elif res["status"] == "timeout":
            row = {
                "experiment": experiment,
                "solver": solver_name,
                "instance": instance_path.name,
                "instance_path": str(instance_path),
                "customers": n_customers,
                "instance_type": instance_type,
                "x_label": x_label,
                "x_value": x_value,
                "solver_parallel_jobs": solver_parallel_jobs,
                "timeout_sec": timeout_sec,
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
                "experiment": experiment,
                "solver": solver_name,
                "instance": instance_path.name,
                "instance_path": str(instance_path),
                "customers": n_customers,
                "instance_type": instance_type,
                "x_label": x_label,
                "x_value": x_value,
                "solver_parallel_jobs": solver_parallel_jobs,
                "timeout_sec": timeout_sec,
                "status": res.get("status", "error"),
                "time_sec": res.get("time_sec", ""),
                "solution_found": "",
                "evs": "",
                "distance": "",
                "error": res.get("error", ""),
            }

            print(f"ERROR: {row['error']}", flush=True)

        rows.append(row)

    return rows


def _scaling_worker(
    worker_id: int,
    instance_files: List[Path],
    experiment: str,
    solver_name: str,
    x_value: float,
    x_label: str,
    solver_parallel_jobs: int,
    timeout_sec: float,
    solver_options: Optional[Dict[str, Any]],
    result_queue: mp.Queue,
) -> None:
    try:
        rows = _run_subset_for_scaling_point(
            instance_files=instance_files,
            experiment=experiment,
            solver_name=solver_name,
            x_value=x_value,
            x_label=x_label,
            solver_parallel_jobs=solver_parallel_jobs,
            timeout_sec=timeout_sec,
            worker_id=worker_id,
            solver_options=solver_options,
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


def _run_scaling_point_parallel(
    instance_files: List[Path],
    experiment: str,
    solver_name: str,
    x_value: float,
    x_label: str,
    solver_parallel_jobs: int,
    timeout_sec: float,
    available_cpus: int,
    solver_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run one scaling point.

    If testing solver_parallel_jobs=4 on a 24-core machine, use:

        benchmark_workers = 24 // 4 = 6

    That means 6 instances are evaluated concurrently, and each solver
    instance may use 4 islands.
    """
    available_cpus = max(1, int(available_cpus))
    solver_parallel_jobs = max(1, int(solver_parallel_jobs))

    benchmark_workers = max(1, available_cpus // solver_parallel_jobs)
    benchmark_workers = min(benchmark_workers, len(instance_files))

    chunks = _balanced_chunks_by_type(instance_files, benchmark_workers)

    print("\n-------------------------------------------------")
    print(f"Experiment              : {experiment}")
    print(f"Solver                  : {solver_name}")
    print(f"{x_label:<23}: {x_value:g}")
    print(f"Timeout                 : {timeout_sec:.1f} s")
    print(f"Solver parallel jobs    : {solver_parallel_jobs}")
    print(f"Solver options          : {solver_options or {}}")
    print(f"Available CPUs          : {available_cpus}")
    print(f"Benchmark workers       : {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        counts = _chunk_type_counts(chunk)
        counts_text = ", ".join(f"{k}: {v}" for k, v in counts.items())
        print(f"  Worker {i:<3}: {len(chunk):<5} instances ({counts_text})")

    print("-------------------------------------------------")

    result_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    for worker_id, chunk in enumerate(chunks, start=1):
        proc = mp.Process(
            target=_scaling_worker,
            args=(
                worker_id,
                chunk,
                experiment,
                solver_name,
                x_value,
                x_label,
                solver_parallel_jobs,
                timeout_sec,
                solver_options,
                result_queue,
            ),
        )

        proc.start()
        processes.append(proc)

    messages: Dict[int, Dict[str, Any]] = {}

    while len(messages) < len(processes):
        try:
            msg = result_queue.get(timeout=0.5)
            messages[msg["worker_id"]] = msg
        except queue.Empty:
            if all(not proc.is_alive() for proc in processes):
                break

    for proc in processes:
        proc.join()

    while len(messages) < len(processes):
        try:
            msg = result_queue.get_nowait()
            messages[msg["worker_id"]] = msg
        except queue.Empty:
            break

    errors: List[str] = []

    for i, proc in enumerate(processes, start=1):
        if i not in messages:
            errors.append(
                f"Worker {i} exited without returning results. "
                f"Exit code: {proc.exitcode}"
            )
            continue

        if messages[i]["status"] != "ok":
            errors.append(
                f"Worker {i} failed: {messages[i].get('error', 'unknown error')}"
            )

    if errors:
        raise RuntimeError(
            "One or more scaling benchmark workers failed:\n"
            + "\n".join(errors)
        )

    rows: List[Dict[str, Any]] = []

    for worker_id in sorted(messages):
        rows.extend(messages[worker_id]["rows"])

    return rows


# ===============================================================
# Summary
# ===============================================================

def _summarize_type(
    rows: List[Dict[str, Any]],
    experiment: str,
    solver_name: str,
    x_label: str,
    x_value: float,
    instance_type: str,
) -> Dict[str, Any]:
    if instance_type == INSTANCE_TYPE_ALL:
        subset = [
            r for r in rows
            if r["experiment"] == experiment
            and r["solver"] == solver_name
            and r["x_label"] == x_label
            and float(r["x_value"]) == float(x_value)
        ]
    else:
        subset = [
            r for r in rows
            if r["experiment"] == experiment
            and r["solver"] == solver_name
            and r["x_label"] == x_label
            and float(r["x_value"]) == float(x_value)
            and r.get("instance_type") == instance_type
        ]

    ok_rows = [r for r in subset if r.get("status") == "ok"]

    sol_rows = [
        r for r in ok_rows
        if r.get("solution_found")
        and _is_number(r.get("distance"))
        and _is_number(r.get("evs"))
    ]

    dist_vals = [float(r["distance"]) for r in sol_rows]
    ev_vals = [float(r["evs"]) for r in sol_rows]
    time_vals = [
        float(r["time_sec"])
        for r in ok_rows
        if _is_number(r.get("time_sec"))
    ]

    solver_parallel_jobs = ""
    timeout_sec = ""

    if subset:
        solver_parallel_jobs = subset[0].get("solver_parallel_jobs", "")
        timeout_sec = subset[0].get("timeout_sec", "")

    return {
        "experiment": experiment,
        "solver": solver_name,
        "instance_type": instance_type,
        "customers": CUSTOMER_SIZE,
        "x_label": x_label,
        "x_value": x_value,
        "solver_parallel_jobs": solver_parallel_jobs,
        "timeout_sec": timeout_sec,

        "total_runs": len(subset),
        "ok_runs": len(ok_rows),
        "solution_runs": len(sol_rows),

        "avg_time_sec": mean(time_vals) if time_vals else "",
        "avg_distance": mean(dist_vals) if dist_vals else "",
        "avg_evs": mean(ev_vals) if ev_vals else "",

        "min_distance": min(dist_vals) if dist_vals else "",
        "max_distance": max(dist_vals) if dist_vals else "",

        "min_evs": min(ev_vals) if ev_vals else "",
        "max_evs": max(ev_vals) if ev_vals else "",
    }


def _build_scaling_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({
        (
            r["experiment"],
            r["solver"],
            r["x_label"],
            float(r["x_value"]),
        )
        for r in rows
    })

    summary: List[Dict[str, Any]] = []

    for experiment, solver_name, x_label, x_value in keys:
        for instance_type in [INSTANCE_TYPE_ALL] + INSTANCE_TYPES:
            summary.append(
                _summarize_type(
                    rows=rows,
                    experiment=experiment,
                    solver_name=solver_name,
                    x_label=x_label,
                    x_value=x_value,
                    instance_type=instance_type,
                )
            )

    summary.sort(
        key=lambda r: (
            r["experiment"],
            r["solver"],
            r["instance_type"],
            float(r["x_value"]),
        )
    )

    return summary


# ===============================================================
# Plotting
# ===============================================================

def _plot_scaling_metric(
    summary_rows: List[Dict[str, Any]],
    experiment: str,
    solver_name: str,
    x_label: str,
    y_key: str,
    ylabel: str,
    title: str,
    plot_path: Path,
) -> None:
    """
    Plot one scaling curve.

    Lines:
      - All
      - Clustered
      - Random
      - Mixed
    """
    plt.figure(figsize=(9, 6))

    by_type: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    for row in summary_rows:
        if row["experiment"] != experiment:
            continue

        if row["solver"] != solver_name:
            continue

        if row["x_label"] != x_label:
            continue

        y = row.get(y_key, "")

        if y == "":
            continue

        by_type[row["instance_type"]].append(
            (
                float(row["x_value"]),
                float(y),
            )
        )

    if not by_type:
        plt.text(
            0.5,
            0.5,
            f"No data available for {title}",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return

    line_order = [INSTANCE_TYPE_ALL] + INSTANCE_TYPES

    for instance_type in line_order:
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

    if x_label == "core_count":
        plt.xlabel("Number of cores / islands")
        plt.xticks(CORE_COUNTS)
    else:
        plt.xlabel("Solve time limit (s)")
        plt.xticks(SOLVE_TIMES_SEC)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Input type")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def _make_scaling_plots(
    summary_rows: List[Dict[str, Any]],
    graphs_dir: Path,
) -> Dict[str, Path]:
    plot_paths: Dict[str, Path] = {}

    specs = [
        {
            "key": "alns_distance_vs_cores",
            "experiment": "core_scaling",
            "solver": "ALNS",
            "x_label": "core_count",
            "y_key": "avg_distance",
            "ylabel": "Average distance",
            "title": "ALNS average distance vs core count",
            "path": graphs_dir / "scaling_ALNS_distance_vs_cores.png",
        },
        {
            "key": "alns_evs_vs_cores",
            "experiment": "core_scaling",
            "solver": "ALNS",
            "x_label": "core_count",
            "y_key": "avg_evs",
            "ylabel": "Average EVs used",
            "title": "ALNS average EVs vs core count",
            "path": graphs_dir / "scaling_ALNS_evs_vs_cores.png",
        },
        {
            "key": "alns_distance_vs_time",
            "experiment": "time_scaling",
            "solver": "ALNS",
            "x_label": "solve_time_sec",
            "y_key": "avg_distance",
            "ylabel": "Average distance",
            "title": "ALNS average distance vs solve time",
            "path": graphs_dir / "scaling_ALNS_distance_vs_time.png",
        },
        {
            "key": "alns_evs_vs_time",
            "experiment": "time_scaling",
            "solver": "ALNS",
            "x_label": "solve_time_sec",
            "y_key": "avg_evs",
            "ylabel": "Average EVs used",
            "title": "ALNS average EVs vs solve time",
            "path": graphs_dir / "scaling_ALNS_evs_vs_time.png",
        },
        {
            "key": "memetic_distance_vs_cores",
            "experiment": "core_scaling",
            "solver": "Memetic",
            "x_label": "core_count",
            "y_key": "avg_distance",
            "ylabel": "Average distance",
            "title": "Memetic average distance vs core count",
            "path": graphs_dir / "scaling_Memetic_distance_vs_cores.png",
        },
        {
            "key": "memetic_evs_vs_cores",
            "experiment": "core_scaling",
            "solver": "Memetic",
            "x_label": "core_count",
            "y_key": "avg_evs",
            "ylabel": "Average EVs used",
            "title": "Memetic average EVs vs core count",
            "path": graphs_dir / "scaling_Memetic_evs_vs_cores.png",
        },
        {
            "key": "memetic_distance_vs_time",
            "experiment": "time_scaling",
            "solver": "Memetic",
            "x_label": "solve_time_sec",
            "y_key": "avg_distance",
            "ylabel": "Average distance",
            "title": "Memetic average distance vs solve time",
            "path": graphs_dir / "scaling_Memetic_distance_vs_time.png",
        },
        {
            "key": "memetic_evs_vs_time",
            "experiment": "time_scaling",
            "solver": "Memetic",
            "x_label": "solve_time_sec",
            "y_key": "avg_evs",
            "ylabel": "Average EVs used",
            "title": "Memetic average EVs vs solve time",
            "path": graphs_dir / "scaling_Memetic_evs_vs_time.png",
        },
    ]

    for spec in specs:
        _plot_scaling_metric(
            summary_rows=summary_rows,
            experiment=spec["experiment"],
            solver_name=spec["solver"],
            x_label=spec["x_label"],
            y_key=spec["y_key"],
            ylabel=spec["ylabel"],
            title=spec["title"],
            plot_path=spec["path"],
        )

        plot_paths[spec["key"]] = spec["path"]

    return plot_paths


# ===============================================================
# Public API
# ===============================================================

def benchmark_scaling_experiments(
    data_root: Path,
    graphs_dir: Path,
    available_cpus: int,
    core_counts: Optional[List[int]] = None,
    solve_times_sec: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run ALNS/Memetic scaling benchmarks on Customer_100 instances.

    Parameters
    ----------
    data_root:
        Root data directory.

    graphs_dir:
        Output directory.

    available_cpus:
        Number of CPUs available for this benchmark. This controls how
        many instances are evaluated concurrently.

    core_counts:
        Core/island counts to test for ALNS and Memetic. Default:
            [1, 2, 4, 8]

    solve_times_sec:
        Solve times to test for ALNS and Memetic with one core. Default:
            [5, 10, 15]
    """
    data_root = Path(data_root)
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    available_cpus = max(1, int(available_cpus))

    if core_counts is None:
        core_counts = list(CORE_COUNTS)

    if solve_times_sec is None:
        solve_times_sec = list(SOLVE_TIMES_SEC)

    # Do not test more solver cores than are available.
    core_counts = [
        int(c)
        for c in core_counts
        if 1 <= int(c) <= available_cpus
    ]

    solve_times_sec = [
        int(t)
        for t in solve_times_sec
        if int(t) > 0
    ]

    if not core_counts:
        raise ValueError(
            f"No valid core counts for available_cpus={available_cpus}"
        )

    if not solve_times_sec:
        raise ValueError("No valid solve times were provided.")

    instance_files = _find_customer_100_instances(data_root)

    type_counts: Dict[str, int] = defaultdict(int)

    for path in instance_files:
        type_counts[_instance_type_from_name(path)] += 1

    print("\n=================================================")
    print("Scaling benchmark")
    print("=================================================")
    print(f"Data root              : {data_root}")
    print(f"Graphs dir             : {graphs_dir}")
    print(f"Customer size          : {CUSTOMER_SIZE}")
    print(f"Instances              : {len(instance_files)}")
    print(f"Available CPUs         : {available_cpus}")
    print(f"Core counts            : {core_counts}")
    print(f"Solve times            : {solve_times_sec}")
    print("Instance types         : " + ", ".join(
        f"{typ}={count}" for typ, count in sorted(type_counts.items())
    ))
    print("ALNS scaling mode      : unlimited iterations, time-limited")
    print("=================================================")

    detail_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Core scaling for ALNS and Memetic
    # ------------------------------------------------------------
    for solver_name in ["ALNS", "Memetic"]:
        for core_count in core_counts:
            solver_options = _alns_scaling_options(solver_name)

            rows = _run_scaling_point_parallel(
                instance_files=instance_files,
                experiment="core_scaling",
                solver_name=solver_name,
                x_value=float(core_count),
                x_label="core_count",
                solver_parallel_jobs=int(core_count),
                timeout_sec=FIXED_CORE_SCALING_TIME_SEC,
                available_cpus=available_cpus,
                solver_options=solver_options,
            )

            detail_rows.extend(rows)

    # ------------------------------------------------------------
    # Time scaling for ALNS and Memetic, single core
    # ------------------------------------------------------------
    for solver_name in ["ALNS", "Memetic"]:
        for solve_time_sec in solve_times_sec:
            solver_options = _alns_scaling_options(solver_name)

            rows = _run_scaling_point_parallel(
                instance_files=instance_files,
                experiment="time_scaling",
                solver_name=solver_name,
                x_value=float(solve_time_sec),
                x_label="solve_time_sec",
                solver_parallel_jobs=1,
                timeout_sec=float(solve_time_sec),
                available_cpus=available_cpus,
                solver_options=solver_options,
            )

            detail_rows.extend(rows)

    detail_rows.sort(
        key=lambda r: (
            r["experiment"],
            r["solver"],
            str(r["x_label"]),
            float(r["x_value"]),
            str(r["instance_path"]),
        )
    )

    summary_rows = _build_scaling_summary(detail_rows)

    detail_csv = graphs_dir / "scaling_benchmark_details.csv"
    summary_csv = graphs_dir / "scaling_benchmark_summary.csv"

    _write_csv(
        detail_csv,
        detail_rows,
        fieldnames=[
            "experiment",
            "solver",
            "instance",
            "instance_path",
            "customers",
            "instance_type",
            "x_label",
            "x_value",
            "solver_parallel_jobs",
            "timeout_sec",
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
            "experiment",
            "solver",
            "instance_type",
            "customers",
            "x_label",
            "x_value",
            "solver_parallel_jobs",
            "timeout_sec",

            "total_runs",
            "ok_runs",
            "solution_runs",

            "avg_time_sec",
            "avg_distance",
            "avg_evs",

            "min_distance",
            "max_distance",

            "min_evs",
            "max_evs",
        ],
    )

    plot_paths = _make_scaling_plots(
        summary_rows=summary_rows,
        graphs_dir=graphs_dir,
    )

    print("\nScaling benchmark complete.")
    print(f"Detail CSV : {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    print("\nGenerated scaling plots:")

    for key, path in sorted(plot_paths.items()):
        print(f"  {key:<30}: {path}")

    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "plot_paths": plot_paths,
        "detail_rows": detail_rows,
        "summary_rows": summary_rows,
    }
