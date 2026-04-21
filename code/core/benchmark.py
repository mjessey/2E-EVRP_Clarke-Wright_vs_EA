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
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.parser import Parser
from core.solver_runner import solve_with_optional_timeout


DEFAULT_SOLVER_NAMES = ["ClarkeWright", "ALNS"]


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


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

        # only meaningful if a feasible solution was found
        if row.get("solution_found") and _is_number(row["distance"]):
            grouped_dist[key].append(float(row["distance"]))

        if row.get("solution_found") and _is_number(row["evs"]):
            grouped_evs[key].append(float(row["evs"]))

    all_keys = sorted(
        set(grouped_time) | set(grouped_dist) | set(grouped_evs),
        key=lambda x: (x[1], x[0])
    )

    summary_rows = []
    for solver, customers in all_keys:
        time_vals = grouped_time.get((solver, customers), [])
        dist_vals = grouped_dist.get((solver, customers), [])
        ev_vals   = grouped_evs.get((solver, customers), [])

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
        by_solver[row["solver"]].append((int(row["customers"]), float(y)))

    if not by_solver:
        plt.text(0.5, 0.5, f"No data available for {ylabel}", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return

    for solver, pts in sorted(by_solver.items()):
        pts.sort()
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.plot(xs, ys, marker="o", linewidth=2, label=solver)

    plt.xlabel("Number of customers")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def benchmark_algorithms(
    data_root: Path,
    graphs_dir: Path,
    solver_names: Optional[List[str]] = None,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    data_root = Path(data_root)
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if solver_names is None:
        solver_names = list(DEFAULT_SOLVER_NAMES)

    instance_files = sorted(data_root.rglob("*.txt"))
    if not instance_files:
        raise FileNotFoundError(f"No .txt instance files found under: {data_root}")

    detail_rows: List[Dict[str, Any]] = []

    print(f"\nBenchmarking instances under: {data_root}")
    print(f"Saving outputs to          : {graphs_dir}")
    print(f"Algorithms                 : {', '.join(solver_names)}")
    if timeout_sec is None:
        print("Per-run timeout            : none")
    else:
        print(f"Per-run timeout            : {timeout_sec:.1f} s")

    for instance_path in instance_files:
        print(f"\nInstance: {instance_path.name}")

        try:
            data = Parser(instance_path).data
            n_customers = len(data["customers"])
            print(f"  Customers: {n_customers}")
        except Exception as err:
            print(f"  Parse failed: {err}")
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
            print(f"  - {solver_name:<13} ... ", end="", flush=True)
            res = solve_with_optional_timeout(solver_name, data, timeout_sec)

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
                    print(f"{res['time_sec']:.3f} s | dist={row['distance']} | EVs={row['evs']}")
                else:
                    print(f"{res['time_sec']:.3f} s | no solution")

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
                print("TIMEOUT")

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
                print("ERROR")

            detail_rows.append(row)

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
