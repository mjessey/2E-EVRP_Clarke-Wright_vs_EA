# main.py
# -------------------------------------------------------------------
#  Command-line launcher
#   1) solve one instance with GUI + image export
#   2) benchmark algorithms on many instances + runtime plot
# -------------------------------------------------------------------

import sys
from pathlib import Path
from typing import Optional

from core.parser import Parser
from core.evaluator import Evaluator

from core.benchmark import benchmark_algorithms
from core.solver_runner import solve_with_optional_timeout

from gui.gui import GUI


# -------------------------------------------------------------------
def ask_for_mode() -> str:
    while True:
        p = input(
            "Choose mode:\n"
            "1: Solve one instance\n"
            "2: Benchmark Clarke-Wright vs ALNS\n> "
        ).strip()
        if p in {"1", "2"}:
            return p
        print(f"'{p}' is not a valid option - try again.\n")


def ask_for_path() -> Path:
    while True:
        p = input("Enter path to 2E-EVRP instance (or 'q' to quit): ").strip()
        if p.lower() in {"q", "quit", "exit"}:
            sys.exit(0)
        path = Path(p)
        if path.is_file():
            return path
        print(f"'{p}' is not a valid file – try again.\n")


def ask_for_solver() -> str:
    while True:
        p = input("Enter which algorithm to use:\n1: brute-force\n2: clarke-wright\n3: Adaptive Large Neighborhood Search:\n4: memetic: ")
        if p in {"1", "2", "3", "4"}:
            return p
        print(f"'{p}' is not a valid option - try again.\n")


def ask_for_directory(default_dir: Path, label: str) -> Path:
    while True:
        p = input(f"{label} [default: {default_dir}]\n> ").strip()
        if not p:
            return default_dir
        path = Path(p)
        if path.exists() and path.is_dir():
            return path
        print(f"'{p}' is not a valid directory - try again.\n")


def ask_for_timeout(prompt: str) -> Optional[float]:
    while True:
        p = input(prompt).strip()
        if p == "":
            return None
        try:
            val = float(p)
            if val > 0:
                return val
        except ValueError:
            pass
        print("Please enter a positive number or just press Enter.\n")


# -------------------------------------------------------------------
def run_single_instance(instance_path: Path, graphs_dir: Path) -> None:
    # parse ---------------------------------------------------------
    try:
        data = Parser(instance_path).data
    except Exception as err:
        print("Parsing failed:", err)
        sys.exit(1)

    # choose solver -------------------------------------------------
    solver_num = ask_for_solver()
    if solver_num == "1":
        solver_name = "BruteForce"
    elif solver_num == "2":
        solver_name = "ClarkeWright"
    elif solver_num == "3":
        solver_name = "ALNS"
    elif solver_num == "4":
        solver_name = "Memetic"


    timeout_sec = ask_for_timeout(
        "Solver timeout in seconds for this instance "
        "(press Enter for no timeout)\n> "
    )

    # solve ---------------------------------------------------------
    wrapped = solve_with_optional_timeout(solver_name, data, timeout_sec)

    if wrapped["status"] == "timeout":
        print(f"Solver timed out after {timeout_sec:.1f} seconds.")
        return

    if wrapped["status"] == "error":
        print("Solver failed:", wrapped.get("error", "unknown error"))
        if wrapped.get("traceback"):
            print(wrapped["traceback"])
        return

    result = wrapped["result"]
    best_sol = result["solution"]

    if best_sol is None:
        print("No feasible solution found.")
        return

    ev = Evaluator(data, check_sat_inventory=False)
    res = ev.evaluate(best_sol)

    print("-------------------------------------------------")
    print(f"Algorithm         : {solver_name}")
    print(f"Solve time        : {wrapped['time_sec']:.3f} s")
    print(f"#EVs used         : {result['evs']}")
    print(f"Total distance    : {result['distance']:.2f}")
    print(f"Feasible          : {res['feasible']}")
    print(f"Violations        : {res['violations']}")

    if "routes" in result:
        print(f"Routes enumerated : {result['routes']}")
    if "milp_vars" in result:
        print(f"MILP variables    : {result['milp_vars']}")
    if "milp_time" in result and result["milp_time"] is not None:
        print(f"MILP solve time   : {result['milp_time']:.3f} s")
    print("-------------------------------------------------")

    # visualise + save image into graphs/ --------------------------
    GUI(
        data,
        solution=best_sol,
        algorithm_name=solver_name,
        instance_name=instance_path.stem,
        save_dir=graphs_dir,
        save_on_start=True,
    ).run()


def run_benchmark(project_root: Path, graphs_dir: Path) -> None:
    data_dir = ask_for_directory(project_root / "data", "Enter benchmark data directory")
    timeout_sec = ask_for_timeout(
        "Per-run timeout in seconds for benchmarking "
        "(press Enter for no timeout)\n> "
    )

    result = benchmark_algorithms(
        data_root=data_dir,
        graphs_dir=graphs_dir,
        solver_names=["ClarkeWright", "ALNS"],
        timeout_sec=timeout_sec,
    )

    print("\n-------------------------------------------------")
    print(f"Detailed results : {result['detail_csv']}")
    print(f"Summary results  : {result['summary_csv']}")
    print(f"Runtime plot     : {result['runtime_plot_path']}")
    print(f"Distance plot    : {result['distance_plot_path']}")
    print(f"EV usage plot    : {result['evs_plot_path']}")
    print("-------------------------------------------------")


# -------------------------------------------------------------------
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    graphs_dir = project_root / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # CLI shortcuts ------------------------------------------------
    if len(sys.argv) >= 2 and sys.argv[1] == "--benchmark":
        run_benchmark(project_root, graphs_dir)
        return

    if len(sys.argv) >= 2:
        instance_path = Path(sys.argv[1])
        if instance_path.is_file():
            run_single_instance(instance_path, graphs_dir)
            return

    # interactive mode --------------------------------------------
    mode = ask_for_mode()
    if mode == "1":
        instance_path = ask_for_path()
        run_single_instance(instance_path, graphs_dir)
    else:
        run_benchmark(project_root, graphs_dir)


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
