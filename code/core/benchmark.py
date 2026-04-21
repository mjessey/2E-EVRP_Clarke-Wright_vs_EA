# benchmark.py
# ---------------------------------------------------------------
#  Benchmark both algorithms on a directory of instances.
#  Saves:
#    - graphs/runtime_comparison.png
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


def _build_summary(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[float]] = defaultdict(list)

    for row in detail_rows:
        if row["status"] == "ok":
            grouped[(row["solver"], int(row["customers"]))].append(float(row["time_sec"]))

    summary_rows = []
    for (solver, customers), times in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        summary_rows.append({
            "solver": solver,
            "customers": customers,
            "runs": len(times),
            "avg_time_sec": mean(times),
            "min_time_sec": min(times),
            "max_time_sec": max(times),
        })

    return summary_rows


def _plot_summary(summary_rows: List[Dict[str, Any]], plot_path: Path) -> None:
    plt.figure(figsize=(9, 6))

    if not summary_rows:
        plt.text(0.5, 0.5, "No successful benchmark runs", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return

    by_solver: Dict[str, List[tuple]] = defaultdict(list)
    for row in summary_rows:
        by_solver[row["solver"]].append((int(row["customers"]), float(row["avg_time_sec"])))

    for solver, pts in sorted(by_solver.items()):
        pts.sort()
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.plot(xs, ys, marker="o", linewidth=2, label=solver)

    plt.xlabel("Number of customers")
    plt.ylabel("Average solve time (s)")
    plt.title("2E-EVRP runtime comparison")
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
                row = {
                    "instance": instance_path.name,
                    "instance_path": str(instance_path),
                    "customers": n_customers,
                    "solver": solver_name,
                    "status": "ok",
                    "time_sec": res["time_sec"],
                    "solution_found": solved.get("solution") is not None,
                    "evs": solved.get("evs", ""),
                    "distance": solved.get("distance", ""),
                    "error": "",
                }
                print(f"{res['time_sec']:.3f} s")

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
    plot_path = graphs_dir / "runtime_comparison.png"

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
            "runs",
            "avg_time_sec",
            "min_time_sec",
            "max_time_sec",
        ],
    )

    _plot_summary(summary_rows, plot_path)

    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "plot_path": plot_path,
        "detail_rows": detail_rows,
        "summary_rows": summary_rows,
    }
