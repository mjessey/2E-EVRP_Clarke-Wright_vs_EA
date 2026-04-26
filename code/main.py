# main.py
# -------------------------------------------------------------------
#  Command-line launcher
#   1) solve one instance with GUI + image export
#   2) benchmark algorithms on many instances + runtime plot
#
#  HPC/parallel benchmark support:
#   - benchmark_workers: number of instances evaluated concurrently
#   - solver_parallel_jobs: number of per-instance islands/processes
#     used by ParallelALNS and ParallelMemetic
# -------------------------------------------------------------------

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from core.parser import Parser
from core.evaluator import Evaluator

from core.benchmark import benchmark_algorithms
from core.solver_runner import solve_with_optional_timeout

from gui.gui import GUI


# -------------------------------------------------------------------
def available_cpu_count() -> int:
    """
    Prefer scheduler-provided CPU counts if available.
    Falls back to os.cpu_count().

    Common HPC scheduler variables:
      - SLURM_CPUS_PER_TASK
      - SLURM_CPUS_ON_NODE
      - PBS_NP
      - NSLOTS
    """
    for var in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "PBS_NP",
        "NSLOTS",
    ):
        val = os.environ.get(var)
        if val:
            try:
                n = int(val)
                if n > 0:
                    return n
            except ValueError:
                pass

    return os.cpu_count() or 1


def ask_for_mode() -> str:
    while True:
        p = input(
            "Choose mode:\n"
            "1: Solve one instance\n"
            "2: Benchmark algorithms\n> "
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
        p = input(
            "Enter which algorithm to use:\n"
            "1: brute-force\n"
            "2: clarke-wright\n"
            "3: Adaptive Large Neighborhood Search\n"
            "4: Memetic\n"
            "5: Parallel ALNS\n"
            "6: Parallel Memetic\n> "
        ).strip()

        if p in {"1", "2", "3", "4", "5", "6"}:
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


def ask_for_int(
    prompt: str,
    default: int,
    min_val: int,
    max_val: int,
) -> int:
    while True:
        p = input(f"{prompt} [default: {default}]\n> ").strip()

        if not p:
            return default

        try:
            val = int(p)
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass

        print(f"Please enter an integer between {min_val} and {max_val}.\n")


def ask_for_benchmark_parallelism() -> Tuple[int, int]:
    """
    Ask how to allocate CPUs between:
      1. benchmark-level parallelism:
           number of instances evaluated concurrently
      2. solver-level parallelism:
           number of islands/processes per ParallelALNS/ParallelMemetic run

    Returns
    -------
    benchmark_workers, solver_parallel_jobs

    Example
    -------
    If available CPUs = 64 and user chooses:

        solver_parallel_jobs = 16
        benchmark_workers = 4

    then the benchmark runs approximately:

        4 instances at the same time,
        each ParallelALNS/ParallelMemetic instance using 16 islands.
    """
    available = available_cpu_count()

    print(f"\nCPUs available according to system/scheduler: {available}")

    default_solver_jobs = min(16, available)

    solver_parallel_jobs = ask_for_int(
        prompt=(
            "CPUs/islands per solver instance for "
            "ParallelALNS/ParallelMemetic"
        ),
        default=default_solver_jobs,
        min_val=1,
        max_val=available,
    )

    max_benchmark_workers = max(1, available // solver_parallel_jobs)

    benchmark_workers = ask_for_int(
        prompt=(
            "Number of benchmark workers "
            "(instances evaluated concurrently)"
        ),
        default=max_benchmark_workers,
        min_val=1,
        max_val=max_benchmark_workers,
    )

    print("\nParallel benchmark configuration:")
    print(f"  CPUs available        : {available}")
    print(f"  CPUs per instance     : {solver_parallel_jobs}")
    print(f"  Benchmark workers     : {benchmark_workers}")
    print(f"  Max active solver jobs : ~{benchmark_workers * solver_parallel_jobs}")
    print()

    return benchmark_workers, solver_parallel_jobs


def solver_num_to_name(solver_num: str) -> str:
    if solver_num == "1":
        return "BruteForce"
    if solver_num == "2":
        return "ClarkeWright"
    if solver_num == "3":
        return "ALNS"
    if solver_num == "4":
        return "Memetic"
    if solver_num == "5":
        return "ParallelALNS"
    if solver_num == "6":
        return "ParallelMemetic"

    raise ValueError(f"Unknown solver selection: {solver_num}")


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
    solver_name = solver_num_to_name(solver_num)

    timeout_sec = ask_for_timeout(
        "Solver timeout in seconds for this instance "
        "(press Enter for no timeout)\n> "
    )

    solver_parallel_jobs = 1

    if solver_name in {"ParallelALNS", "ParallelMemetic"}:
        available = available_cpu_count()

        print(f"\nCPUs available according to system/scheduler: {available}")

        solver_parallel_jobs = ask_for_int(
            prompt="CPUs/islands to use for this solver instance",
            default=min(16, available),
            min_val=1,
            max_val=available,
        )

    # solve ---------------------------------------------------------
    wrapped = solve_with_optional_timeout(
        solver_name=solver_name,
        data=data,
        timeout_sec=timeout_sec,
        solver_parallel_jobs=solver_parallel_jobs,
    )

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

    if solver_name in {"ParallelALNS", "ParallelMemetic"}:
        print(f"Parallel islands  : {solver_parallel_jobs}")

        if "parallel_workers" in result:
            print(f"Workers used      : {result['parallel_workers']}")

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

    if "island_summaries" in result:
        successful = sum(
            1
            for row in result["island_summaries"]
            if row.get("status") == "ok" and row.get("solution_found")
        )
        total = len(result["island_summaries"])
        print(f"Successful islands: {successful}/{total}")

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
    data_dir = ask_for_directory(
        project_root / "data",
        "Enter benchmark data directory",
    )

    timeout_sec = ask_for_timeout(
        "Per-run wall-clock timeout in seconds "
        "(recommended for HPC comparison: 10; press Enter for no timeout)\n> "
    )

    benchmark_workers, solver_parallel_jobs = ask_for_benchmark_parallelism()

    result = benchmark_algorithms(
        data_root=data_dir,
        graphs_dir=graphs_dir,

        # Clarke-Wright is mostly deterministic and generally should be
        # parallelized across instances rather than within one instance.
        #
        # ParallelALNS and ParallelMemetic use solver_parallel_jobs
        # independent islands per instance.
        solver_names=[
            "ClarkeWright",
            "ParallelALNS",
            "ParallelMemetic",
        ],

        timeout_sec=timeout_sec,
        max_workers=benchmark_workers,
        solver_parallel_jobs=solver_parallel_jobs,
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
    import multiprocessing as mp

    mp.freeze_support()
    main()
