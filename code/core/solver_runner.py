# solver_runner.py
# ---------------------------------------------------------------
#  Shared solver execution utilities
#    - runs a solver once
#    - optionally enforces a timeout using multiprocessing
#    - passes timeout cooperatively to solvers that support it
#    - safely serialises/deserialises Solution objects so they
#      can cross process boundaries
# ---------------------------------------------------------------

from __future__ import annotations

import inspect
import time
import traceback
import multiprocessing as mp
from typing import Dict, Any, Optional

from core.evaluator import Solution
from solvers.brute_force import BruteForce
from solvers.clarke_wright import ClarkeWright
from solvers.neighborhood_search import ALNS
from solvers.memetic import MemeticAlgorithm


SOLVER_FACTORIES = {
    "BruteForce": BruteForce,
    "ClarkeWright": ClarkeWright,
    "ALNS": ALNS,
    "Memetic": MemeticAlgorithm,
}

# small grace period so cooperative solvers can stop and return
# their best-so-far solution before being force-killed
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
    out = dict(result)
    if "solution" in out:
        out["solution"] = _serialise_solution(out["solution"])
    return out


def _deserialise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(result)
    if "solution" in out:
        out["solution"] = _deserialise_solution(out["solution"])
    return out


# ===============================================================
#  Solver invocation helpers
# ===============================================================
def _solver_accepts_time_limit(solver: Any) -> bool:
    try:
        sig = inspect.signature(solver.solve)
        return "time_limit_sec" in sig.parameters
    except Exception:
        return False


def _build_solver(solver_name: str) -> Any:
    if solver_name not in SOLVER_FACTORIES:
        raise ValueError(
            f"Unknown solver '{solver_name}'. "
            f"Valid options: {list(SOLVER_FACTORIES)}"
        )
    return SOLVER_FACTORIES[solver_name]()


def solve_once(
    solver_name: str,
    data: Dict[str, Any],
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run one solver once.

    If the solver supports cooperative time limits, pass timeout_sec into it
    so it can stop gracefully and return its best-so-far solution.
    """
    try:
        solver = _build_solver(solver_name)

        # Optional cooperative timeout API:
        #   solver.set_time_limit(timeout_sec)
        if hasattr(solver, "set_time_limit"):
            try:
                solver.set_time_limit(timeout_sec)
            except Exception:
                pass

        t0 = time.perf_counter()

        # Optional solve signature:
        #   solve(data, time_limit_sec=...)
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
    queue: mp.Queue,
) -> None:
    queue.put(solve_once(solver_name, data, timeout_sec))


# ===============================================================
#  Public timeout wrapper
# ===============================================================
def solve_with_optional_timeout(
    solver_name: str,
    data: Dict[str, Any],
    timeout_sec: Optional[float],
) -> Dict[str, Any]:
    """
    Run solver with optional timeout.

    - If timeout_sec is None: run directly in current process.
    - Otherwise:
        * pass timeout_sec cooperatively into the solver
        * allow a small grace period for graceful return
        * if still running after that, terminate process
    """
    if timeout_sec is None:
        res = solve_once(solver_name, data, timeout_sec=None)
        if res["status"] == "ok":
            res["result"] = _deserialise_result(res["result"])
        return res

    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_solve_worker,
        args=(solver_name, data, timeout_sec, queue),
    )
    proc.start()

    hard_timeout = timeout_sec + HARD_TIMEOUT_GRACE_SEC
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
