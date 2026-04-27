# parallel_portfolio.py
# ---------------------------------------------------------------
# Parallel portfolio / island-model wrapper for stochastic solvers.
#
# For one instance:
#   - launch n_jobs independent solver runs with different seeds
#   - each run receives approximately the same wall-clock limit
#   - collect all returned solutions
#   - return the best one
#
# This is intended for HPC-style parallelization:
#   wall-clock budget fixed, more CPUs -> more independent searches.
#
# Additional solver_options can be passed through to the underlying
# solver. This is used by the scaling benchmark, for example:
#
#   solver_options = {"unlimited_iterations": True}
#
# for ALNS time-controlled scaling experiments.
# ---------------------------------------------------------------

from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from typing import Any, Dict, Optional, List, Tuple

from solvers.neighborhood_search import ALNS
from solvers.memetic import MemeticAlgorithm


PORTFOLIO_SOLVERS = {
    "ALNS": ALNS,
    "Memetic": MemeticAlgorithm,
}


def _accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        return False


def _call_solver(
    base_solver_name: str,
    instance: Dict[str, Any],
    time_limit_sec: Optional[float],
    seed: Optional[int],
    solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call one underlying solver instance.

    Supports optional solver APIs:
      solve(instance, time_limit_sec=...)
      solve(instance, seed=...)
      solve(instance, unlimited_iterations=...)
      solver.set_time_limit(...)

    Extra options are passed through only if the underlying solver.solve()
    accepts them.
    """
    solver_cls = PORTFOLIO_SOLVERS[base_solver_name]
    solver = solver_cls()

    if seed is not None:
        random.seed(seed)

    if hasattr(solver, "set_time_limit"):
        try:
            solver.set_time_limit(time_limit_sec)
        except Exception:
            pass

    solve_fn = solver.solve
    kwargs: Dict[str, Any] = {}

    if _accepts_kwarg(solve_fn, "time_limit_sec"):
        kwargs["time_limit_sec"] = time_limit_sec

    if _accepts_kwarg(solve_fn, "seed"):
        kwargs["seed"] = seed

    if solver_options:
        for key, value in solver_options.items():
            if _accepts_kwarg(solve_fn, key):
                kwargs[key] = value

    if kwargs:
        return solve_fn(instance, **kwargs)

    return solve_fn(instance)


def _portfolio_worker(
    worker_id: int,
    base_solver_name: str,
    instance: Dict[str, Any],
    time_limit_sec: Optional[float],
    seed: int,
    solver_options: Optional[Dict[str, Any]],
    result_queue: mp.Queue,
) -> None:
    """
    Worker for one independent solver island.
    """
    t0 = time.perf_counter()

    try:
        result = _call_solver(
            base_solver_name=base_solver_name,
            instance=instance,
            time_limit_sec=time_limit_sec,
            seed=seed,
            solver_options=solver_options,
        )

        elapsed = time.perf_counter() - t0

        result_queue.put({
            "worker_id": worker_id,
            "status": "ok",
            "time_sec": elapsed,
            "result": result,
            "error": "",
            "traceback": "",
        })

    except Exception as err:
        elapsed = time.perf_counter() - t0

        result_queue.put({
            "worker_id": worker_id,
            "status": "error",
            "time_sec": elapsed,
            "result": None,
            "error": str(err),
            "traceback": traceback.format_exc(),
        })


class ParallelPortfolioSolver:
    """
    Base class for independent multi-start / island-model solvers.

    Subclasses set:

        BASE_SOLVER_NAME = "ALNS"

    or:

        BASE_SOLVER_NAME = "Memetic"
    """

    BASE_SOLVER_NAME: str = ""

    def __init__(
        self,
        n_jobs: int = 1,
        safety_margin_sec: float = 0.25,
    ) -> None:
        self.n_jobs = max(1, int(n_jobs))
        self.safety_margin_sec = max(0.0, float(safety_margin_sec))

    def solve(
        self,
        instance: Dict[str, Any],
        time_limit_sec: Optional[float] = None,
        seed: Optional[int] = None,
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the parallel portfolio solver.

        Parameters
        ----------
        instance:
            Parsed 2E-EVRP instance.

        time_limit_sec:
            Wall-clock time limit for the whole portfolio.

        seed:
            Optional seed for reproducibility.

        solver_options:
            Extra options passed through to the underlying solver.
            Example:

                {"unlimited_iterations": True}

            for ALNS scaling experiments.
        """
        available = os.cpu_count() or 1
        n_jobs = max(1, min(self.n_jobs, available))

        # --------------------------------------------------------
        # Single-core mode
        # --------------------------------------------------------
        # Even for a single island, subtract a small safety margin so
        # cooperative anytime solvers can return their best solution
        # before the outer hard timeout kills the process.
        if n_jobs <= 1:
            single_time_limit = time_limit_sec

            if single_time_limit is not None:
                single_time_limit = max(
                    0.1,
                    float(single_time_limit) - self.safety_margin_sec,
                )

            return _call_solver(
                base_solver_name=self.BASE_SOLVER_NAME,
                instance=instance,
                time_limit_sec=single_time_limit,
                seed=seed,
                solver_options=solver_options,
            )

        # --------------------------------------------------------
        # Multi-core / multi-island mode
        # --------------------------------------------------------
        if time_limit_sec is not None:
            island_time_limit = max(
                0.1,
                float(time_limit_sec) - self.safety_margin_sec,
            )
            hard_deadline = time.time() + float(time_limit_sec)
        else:
            island_time_limit = None
            hard_deadline = None

        rng = random.Random(seed)

        result_queue: mp.Queue = mp.Queue()
        processes: List[mp.Process] = []

        for worker_id in range(1, n_jobs + 1):
            worker_seed = rng.randint(0, 2**31 - 1)

            proc = mp.Process(
                target=_portfolio_worker,
                args=(
                    worker_id,
                    self.BASE_SOLVER_NAME,
                    instance,
                    island_time_limit,
                    worker_seed,
                    solver_options,
                    result_queue,
                ),
            )

            proc.start()
            processes.append(proc)

        messages: Dict[int, Dict[str, Any]] = {}

        while len(messages) < len(processes):
            if hard_deadline is not None and time.time() >= hard_deadline:
                break

            try:
                msg = result_queue.get(timeout=0.05)
                messages[msg["worker_id"]] = msg

            except queue.Empty:
                if all(not proc.is_alive() for proc in processes):
                    break

        # Terminate stragglers once the global wall-clock budget is reached.
        for proc in processes:
            if proc.is_alive():
                proc.terminate()

        for proc in processes:
            proc.join()

        # Drain late queue messages.
        while len(messages) < len(processes):
            try:
                msg = result_queue.get_nowait()
                messages[msg["worker_id"]] = msg
            except queue.Empty:
                break

        ok_results: List[Dict[str, Any]] = []
        island_summaries: List[Dict[str, Any]] = []

        for worker_id in range(1, n_jobs + 1):
            msg = messages.get(worker_id)

            if msg is None:
                island_summaries.append({
                    "worker_id": worker_id,
                    "status": "no_result",
                    "time_sec": "",
                    "evs": "",
                    "distance": "",
                    "solution_found": False,
                    "error": (
                        "Worker exited or was terminated without "
                        "returning a result."
                    ),
                })
                continue

            if msg["status"] == "ok" and msg["result"] is not None:
                result = msg["result"]
                solution_found = result.get("solution") is not None

                island_summaries.append({
                    "worker_id": worker_id,
                    "status": "ok",
                    "time_sec": msg.get("time_sec", ""),
                    "evs": result.get("evs", ""),
                    "distance": result.get("distance", ""),
                    "solution_found": solution_found,
                    "error": "",
                })

                if solution_found:
                    ok_results.append(result)

            else:
                island_summaries.append({
                    "worker_id": worker_id,
                    "status": msg.get("status", "error"),
                    "time_sec": msg.get("time_sec", ""),
                    "evs": "",
                    "distance": "",
                    "solution_found": False,
                    "error": msg.get("error", ""),
                })

        if not ok_results:
            return {
                "solution": None,
                "evs": 0,
                "distance": float("inf"),
                "parallel_workers": n_jobs,
                "base_solver": self.BASE_SOLVER_NAME,
                "island_summaries": island_summaries,
                "solver_options": solver_options or {},
            }

        def result_key(r: Dict[str, Any]) -> Tuple[float, int]:
            return (
                float(r.get("distance", float("inf"))),
                int(r.get("evs", 10**9)),
            )

        best = min(ok_results, key=result_key)

        out = dict(best)
        out["parallel_workers"] = n_jobs
        out["base_solver"] = self.BASE_SOLVER_NAME
        out["island_summaries"] = island_summaries
        out["solver_options"] = solver_options or {}

        return out


class ParallelALNS(ParallelPortfolioSolver):
    BASE_SOLVER_NAME = "ALNS"


class ParallelMemetic(ParallelPortfolioSolver):
    BASE_SOLVER_NAME = "Memetic"
