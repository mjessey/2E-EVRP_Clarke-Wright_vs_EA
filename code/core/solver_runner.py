# solver_runner.py
# ---------------------------------------------------------------
#  Shared solver execution utilities
#    - runs a solver once
#    - optionally enforces a timeout using multiprocessing
#    - passes timeout cooperatively to solvers that support it
#    - passes solver_parallel_jobs to parallel portfolio solvers
#    - safely serialises/deserialises Solution objects so they
#      can cross process boundaries
# ---------------------------------------------------------------

from __future__ import annotations

import inspect
import multiprocessing as mp
import time
import traceback
from typing import Dict, Any, Optional

from core.evaluator import Solution

from solvers.brute_force import BruteForce
from solvers.clarke_wright import ClarkeWright
from solvers.neighborhood_search import ALNS
from solvers.memetic import MemeticAlgorithm
from solvers.parallel_portfolio import ParallelALNS, ParallelMemetic


SOLVER_FACTORIES = {
    "BruteForce": BruteForce,
    "ClarkeWright": ClarkeWright,
    "ALNS": ALNS,
    "Memetic": MemeticAlgorithm,

    # Parallel portfolio / island-model variants.
    "ParallelALNS": ParallelALNS,
    "ParallelMemetic": ParallelMemetic,
}


# Small grace period so cooperative solvers can stop and return
# their best-so-far solution before being force-killed.
HARD_TIMEOUT_GRACE_SEC = 1.0


# ===============================================================
#  Solution/result serialisation
# ===============================================================

def _serialise_solution(sol: Optional[Solution]) -> Optional[Dict[str, Any]]:
    if sol is None:
        return None

    return {
        "lv_routes": sol.lv_routes,
        "ev_routes": sol.ev_routes,
    }


def _deserialise_solution(payload: Optional[Dict[str, Any]]) -> Optional[Solution]:
    if payload is None:
        return None

    return Solution(
        lv_routes=payload["lv_routes"],
        ev_routes=payload["ev_routes"],
    )


def _serialise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a solver result into a multiprocessing-safe payload.

    Expected result shape:

        {
            "solution": Solution | None,
            "evs": int,
            "distance": float,
            ...
        }

    Extra metadata fields are preserved.
    """
    out = dict(result)

    if "solution" in out:
        out["solution"] = _serialise_solution(out["solution"])

    return out


def _deserialise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a multiprocessing-safe payload back into a normal solver result.
    """
    out = dict(result)

    if "solution" in out:
        out["solution"] = _deserialise_solution(out["solution"])

    return out


# ===============================================================
#  Solver invocation helpers
# ===============================================================

def _solver_accepts_time_limit(solver: Any) -> bool:
    """
    Return True if solver.solve supports:

        solve(data, time_limit_sec=...)
    """
    try:
        sig = inspect.signature(solver.solve)
        return "time_limit_sec" in sig.parameters
    except Exception:
        return False


def _build_solver(
    solver_name: str,
    solver_parallel_jobs: int = 1,
) -> Any:
    """
    Instantiate a solver.

    Parallel portfolio solvers accept:

        ParallelALNS(n_jobs=...)
        ParallelMemetic(n_jobs=...)

    Ordinary solvers ignore solver_parallel_jobs.
    """
    if solver_name not in SOLVER_FACTORIES:
        raise ValueError(
            f"Unknown solver '{solver_name}'. "
            f"Valid options: {list(SOLVER_FACTORIES)}"
        )

    factory = SOLVER_FACTORIES[solver_name]

    # If the solver constructor accepts n_jobs, pass solver_parallel_jobs.
    try:
        sig = inspect.signature(factory)

        if "n_jobs" in sig.parameters:
            return factory(n_jobs=solver_parallel_jobs)

    except Exception:
        pass

    return factory()


def solve_once(
    solver_name: str,
    data: Dict[str, Any],
    timeout_sec: Optional[float] = None,
    solver_parallel_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Run one solver once.

    If the solver supports cooperative time limits, pass timeout_sec into it
    so it can stop gracefully and return its best-so-far solution.

    Parameters
    ----------
    solver_name:
        Name in SOLVER_FACTORIES.

    data:
        Parsed instance dictionary.

    timeout_sec:
        Optional cooperative time limit.

    solver_parallel_jobs:
        Number of internal jobs/islands for parallel portfolio solvers.
        Ignored by ordinary solvers.
    """
    try:
        solver = _build_solver(
            solver_name=solver_name,
            solver_parallel_jobs=solver_parallel_jobs,
        )

        # Optional cooperative timeout API:
        #
        #   solver.set_time_limit(timeout_sec)
        #
        # This is supported only if implemented by a solver.
        if hasattr(solver, "set_time_limit"):
            try:
                solver.set_time_limit(timeout_sec)
            except Exception:
                pass

        t0 = time.perf_counter()

        # Optional solve signature:
        #
        #   solve(data, time_limit_sec=...)
        #
        # ParallelPortfolioSolver and the modified MemeticAlgorithm support it.
        if _solver_accepts_time_limit(solver):
            result = solver.solve(data, time_limit_sec=timeout_sec)
        else:
            result = solver.solve(data)

        elapsed = time.perf_counter() - t0

        return {
            "status": "ok",
            "time_sec": elapsed,
            "result": _serialise_result(result),
        }

    except Exception as err:
        return {
            "status": "error",
            "error": str(err),
            "traceback": traceback.format_exc(),
        }


def _solve_worker(
    solver_name: str,
    data: Dict[str, Any],
    timeout_sec: Optional[float],
    solver_parallel_jobs: int,
    queue: mp.Queue,
) -> None:
    """
    Worker used when a hard timeout is requested.
    """
    queue.put(
        solve_once(
            solver_name=solver_name,
            data=data,
            timeout_sec=timeout_sec,
            solver_parallel_jobs=solver_parallel_jobs,
        )
    )


# ===============================================================
#  Public timeout wrapper
# ===============================================================

def solve_with_optional_timeout(
    solver_name: str,
    data: Dict[str, Any],
    timeout_sec: Optional[float],
    solver_parallel_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Run solver with optional timeout.

    If timeout_sec is None:
        Run directly in the current process.

    Otherwise:
        - Run the solver in a child process.
        - Pass timeout_sec cooperatively into the solver.
        - Allow a small grace period for graceful return.
        - If still running after timeout + grace, terminate process.

    For ParallelALNS and ParallelMemetic:
        solver_parallel_jobs controls how many independent islands are
        launched inside the solver process.
    """
    solver_parallel_jobs = max(1, int(solver_parallel_jobs))

    # No hard timeout: direct call in current process.
    if timeout_sec is None:
        res = solve_once(
            solver_name=solver_name,
            data=data,
            timeout_sec=None,
            solver_parallel_jobs=solver_parallel_jobs,
        )

        if res["status"] == "ok":
            res["result"] = _deserialise_result(res["result"])

        return res

    # Hard timeout mode: run solver in child process.
    queue: mp.Queue = mp.Queue()

    proc = mp.Process(
        target=_solve_worker,
        args=(
            solver_name,
            data,
            timeout_sec,
            solver_parallel_jobs,
            queue,
        ),
    )

    proc.start()

    hard_timeout = float(timeout_sec) + HARD_TIMEOUT_GRACE_SEC
    proc.join(hard_timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()

        return {
            "status": "timeout",
            "time_sec": timeout_sec,
            "result": None,
        }

    if queue.empty():
        return {
            "status": "error",
            "error": "Worker exited without returning a result.",
            "traceback": "",
        }

    res = queue.get()

    if res["status"] == "ok":
        res["result"] = _deserialise_result(res["result"])

    return res
