# solver_runner.py
# ---------------------------------------------------------------
#  Shared solver execution utilities
#    - runs a solver once
#    - optionally enforces a timeout using multiprocessing
#    - safely serialises/deserialises Solution objects so they
#      can cross process boundaries
# ---------------------------------------------------------------

from __future__ import annotations

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


def solve_once(solver_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if solver_name not in SOLVER_FACTORIES:
            raise ValueError(
                f"Unknown solver '{solver_name}'. "
                f"Valid options: {list(SOLVER_FACTORIES)}"
            )

        solver = SOLVER_FACTORIES[solver_name]()
        t0 = time.perf_counter()
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


def _solve_worker(solver_name: str, data: Dict[str, Any], queue: mp.Queue) -> None:
    queue.put(solve_once(solver_name, data))


def solve_with_optional_timeout(
    solver_name: str,
    data: Dict[str, Any],
    timeout_sec: Optional[float],
) -> Dict[str, Any]:
    if timeout_sec is None:
        res = solve_once(solver_name, data)
        if res["status"] == "ok":
            res["result"] = _deserialise_result(res["result"])
        return res

    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_solve_worker, args=(solver_name, data, queue))
    proc.start()
    proc.join(timeout_sec)

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
