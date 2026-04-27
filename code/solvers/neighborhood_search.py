from __future__ import annotations

import math
import random
import copy
import time
from typing import Dict, Any, List, Optional, Tuple

from core.evaluator import Evaluator, Solution
from solvers.clarke_wright import ClarkeWright


class ALNS:
    """
    Adaptive Large Neighborhood Search for 2E-EVRP.

    solve(instance_dict, time_limit_sec=None, seed=None, unlimited_iterations=False) ->
        { 'solution' : Solution | None,
          'evs'      : int,
          'distance' : float }

    Notes
    -----
    Normal benchmark behavior:
        unlimited_iterations=False
        The solver stops after MAX_ITERATIONS or after the optional time limit.

    Scaling benchmark behavior:
        unlimited_iterations=True
        The solver ignores MAX_ITERATIONS and stops by time limit.
        This requires time_limit_sec to be provided.
    """

    MAX_ITERATIONS = 500
    SEGMENT_SIZE = 50
    MIN_REMOVAL = 2
    MAX_REMOVAL = 5

    START_TEMP = 100.0
    COOL_RATE = 0.997

    SCORE_BEST = 10
    SCORE_BETTER = 5
    SCORE_ACCEPTED = 2
    SCORE_REJECTED = 0

    REACTION = 0.5

    def __init__(self, time_limit_sec: Optional[float] = None) -> None:
        self.time_limit_sec = time_limit_sec
        self._deadline: Optional[float] = None

    def set_time_limit(self, time_limit_sec: Optional[float]) -> None:
        self.time_limit_sec = time_limit_sec

    def _start_timer(self, override: Optional[float] = None) -> None:
        limit = self.time_limit_sec if override is None else override

        if limit is None:
            self._deadline = None
        else:
            self._deadline = time.perf_counter() + float(limit)

    def _time_exceeded(self) -> bool:
        return (
            self._deadline is not None
            and time.perf_counter() >= self._deadline
        )

    def solve(
        self,
        instance: Dict[str, Any],
        time_limit_sec: Optional[float] = None,
        seed: Optional[int] = None,
        unlimited_iterations: bool = False,
    ) -> Dict[str, Any]:
        """
        Run ALNS.

        Parameters
        ----------
        instance:
            Parsed 2E-EVRP instance.

        time_limit_sec:
            Optional wall-clock time limit.

        seed:
            Optional random seed.

        unlimited_iterations:
            If False, stop after self.MAX_ITERATIONS.
            If True, ignore self.MAX_ITERATIONS and stop only by time limit.

            This is intended for the scaling benchmark only.
        """
        if seed is not None:
            random.seed(seed)

        if unlimited_iterations and time_limit_sec is None:
            raise ValueError(
                "ALNS unlimited_iterations=True requires time_limit_sec. "
                "Otherwise the solver may run forever."
            )

        self._start_timer(time_limit_sec)

        self.data = instance
        self.ev = Evaluator(instance, check_sat_inventory=False)
        self.params = instance["params"]
        self.EV_CAP = self.params["C"]
        self.LV_CAP = self.params["L"]
        self.BAT_CAP = self.params["Q"]
        self.SPEED = self.params["v"]
        self.INV_G = self.params["g"]

        coords = {
            n: (
                instance["nodes"][n]["x"],
                instance["nodes"][n]["y"],
            )
            for n in instance["nodes"]
        }

        self.dist: Dict[Tuple[str, str], float] = {
            (i, j): math.hypot(
                coords[i][0] - coords[j][0],
                coords[i][1] - coords[j][1],
            )
            for i in coords
            for j in coords
        }

        cw_result = ClarkeWright().solve(instance)
        current = cw_result["solution"]

        if current is None:
            return {
                "solution": None,
                "evs": 0,
                "distance": float("inf"),
                "iterations": 0,
                "unlimited_iterations": unlimited_iterations,
            }

        current_cost = self._cost(current)
        best = copy.deepcopy(current)
        best_cost = current_cost

        self.destroy_ops = [
            self._random_removal,
            self._worst_removal,
            self._route_removal,
            self._time_window_removal,
            self._cluster_removal,
        ]

        self.repair_ops = [
            self._greedy_insertion,
            self._regret_insertion,
        ]

        n_d = len(self.destroy_ops)
        n_r = len(self.repair_ops)

        d_weights = [1.0] * n_d
        r_weights = [1.0] * n_r
        d_scores = [0.0] * n_d
        r_scores = [0.0] * n_r
        d_counts = [0] * n_d
        r_counts = [0] * n_r

        temp = self.START_TEMP
        iteration = 0

        while True:
            if self._time_exceeded():
                break

            if not unlimited_iterations and iteration >= self.MAX_ITERATIONS:
                break

            iteration += 1

            d_idx = self._weighted_choice(d_weights)
            r_idx = self._weighted_choice(r_weights)

            removed, destroyed = self.destroy_ops[d_idx](
                copy.deepcopy(current)
            )

            if self._time_exceeded():
                break

            if not removed:
                continue

            candidate = self.repair_ops[r_idx](destroyed, removed)

            if self._time_exceeded():
                break

            if candidate is None:
                continue

            candidate_cost = self._cost(candidate)

            score = self.SCORE_REJECTED
            delta = candidate_cost - current_cost

            if candidate_cost < best_cost:
                best = copy.deepcopy(candidate)
                best_cost = candidate_cost
                score = self.SCORE_BEST

            if delta < 0:
                current = candidate
                current_cost = candidate_cost

                if score != self.SCORE_BEST:
                    score = self.SCORE_BETTER

            elif random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = candidate
                current_cost = candidate_cost
                score = self.SCORE_ACCEPTED

            d_scores[d_idx] += score
            r_scores[r_idx] += score
            d_counts[d_idx] += 1
            r_counts[r_idx] += 1

            temp *= self.COOL_RATE

            if iteration % self.SEGMENT_SIZE == 0:
                d_weights, r_weights = self._update_weights(
                    d_weights,
                    r_weights,
                    d_scores,
                    r_scores,
                    d_counts,
                    r_counts,
                )

                d_scores = [0.0] * n_d
                r_scores = [0.0] * n_r
                d_counts = [0] * n_d
                r_counts = [0] * n_r

        res = self.ev.evaluate(best)

        return {
            "solution": best,
            "evs": self.ev._count_evs(best),
            "distance": res["cost"],
            "iterations": iteration,
            "unlimited_iterations": unlimited_iterations,
        }

    # ============================================================
    # Destroy operators
    # ============================================================

    def _random_removal(self, sol: Solution) -> Tuple[List[str], Solution]:
        k = self._removal_count(sol)
        all_custs = self._all_customers_in_solution(sol)

        if len(all_custs) < k:
            return [], sol

        removed = random.sample(all_custs, k)

        return removed, self._remove_customers(sol, removed)

    def _worst_removal(self, sol: Solution) -> Tuple[List[str], Solution]:
        contributions = []

        for sat, routes in sol.ev_routes.items():
            for route in routes:
                for i, node in enumerate(route):
                    if self.data["nodes"][node]["Type"] != "c":
                        continue

                    prev = route[i - 1]
                    nxt = route[i + 1]

                    saving = (
                        self.dist[prev, node]
                        + self.dist[node, nxt]
                        - self.dist[prev, nxt]
                    )

                    contributions.append((saving, node))

        contributions.sort(reverse=True)

        k = self._removal_count(sol)
        removed = [node for _, node in contributions[:k]]

        return removed, self._remove_customers(sol, removed)

    def _route_removal(self, sol: Solution) -> Tuple[List[str], Solution]:
        all_routes = [
            (sat, route)
            for sat, routes in sol.ev_routes.items()
            for route in routes
        ]

        if not all_routes:
            return [], sol

        sat, route = random.choice(all_routes)

        removed = [
            n
            for n in route
            if self.data["nodes"][n]["Type"] == "c"
        ]

        new_ev = {}

        for s, rlist in sol.ev_routes.items():
            new_routes = [
                r
                for r in rlist
                if r is not route
            ]

            if new_routes:
                new_ev[s] = new_routes

        return removed, Solution(
            lv_routes=sol.lv_routes,
            ev_routes=new_ev,
        )

    def _time_window_removal(self, sol: Solution) -> Tuple[List[str], Solution]:
        windows = []

        for node_id in self._all_customers_in_solution(sol):
            node = self.data["nodes"][node_id]
            tw = node["DueDate"] - node["ReadyTime"]
            windows.append((tw, node_id))

        windows.sort()

        k = self._removal_count(sol)
        removed = [node for _, node in windows[:k]]

        return removed, self._remove_customers(sol, removed)

    def _cluster_removal(self, sol: Solution) -> Tuple[List[str], Solution]:
        all_custs = self._all_customers_in_solution(sol)

        if not all_custs:
            return [], sol

        seed = random.choice(all_custs)
        sx = self.data["nodes"][seed]["x"]
        sy = self.data["nodes"][seed]["y"]

        by_dist = sorted(
            all_custs,
            key=lambda c: math.hypot(
                self.data["nodes"][c]["x"] - sx,
                self.data["nodes"][c]["y"] - sy,
            ),
        )

        k = self._removal_count(sol)
        removed = by_dist[:k]

        return removed, self._remove_customers(sol, removed)

    # ============================================================
    # Repair operators
    # ============================================================

    def _greedy_insertion(
        self,
        sol: Solution,
        removed: List[str],
    ) -> Optional[Solution]:
        current = copy.deepcopy(sol)

        def min_insertion_cost(cust: str) -> float:
            best = self._best_insertion(current, cust)
            return best[0] if best else float("inf")

        for cust in sorted(removed, key=min_insertion_cost, reverse=True):
            if self._time_exceeded():
                return None

            best = self._best_insertion(current, cust)

            if best is None:
                sat = self._nearest_satellite(cust)
                new_route = [sat, cust, sat]

                if not self._route_feasible(sat, new_route):
                    return None

                current.ev_routes.setdefault(sat, []).append(new_route)

            else:
                _, sat, r_idx, pos = best
                route = current.ev_routes[sat][r_idx]
                current.ev_routes[sat][r_idx] = (
                    route[:pos] + [cust] + route[pos:]
                )

        return current if self._solution_feasible(current) else None

    def _regret_insertion(
        self,
        sol: Solution,
        removed: List[str],
    ) -> Optional[Solution]:
        current = copy.deepcopy(sol)
        remaining = list(removed)

        while remaining:
            if self._time_exceeded():
                return None

            regrets = []

            for cust in remaining:
                positions = self._all_insertions(current, cust)

                if len(positions) == 0:
                    regret = float("inf")
                elif len(positions) == 1:
                    regret = positions[0][0]
                else:
                    regret = positions[1][0] - positions[0][0]

                regrets.append((regret, cust))

            regrets.sort(reverse=True)

            _, chosen = regrets[0]
            remaining.remove(chosen)

            best = self._best_insertion(current, chosen)

            if best is None:
                sat = self._nearest_satellite(chosen)
                new_route = [sat, chosen, sat]

                if not self._route_feasible(sat, new_route):
                    return None

                current.ev_routes.setdefault(sat, []).append(new_route)

            else:
                _, sat, r_idx, pos = best
                route = current.ev_routes[sat][r_idx]
                current.ev_routes[sat][r_idx] = (
                    route[:pos] + [chosen] + route[pos:]
                )

        return current if self._solution_feasible(current) else None

    # ============================================================
    # Insertion helpers
    # ============================================================

    def _best_insertion(
        self,
        sol: Solution,
        cust: str,
    ) -> Optional[Tuple[float, str, int, int]]:
        best_cost = float("inf")
        best_pos = None

        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    if self._time_exceeded():
                        return best_pos

                    candidate = route[:pos] + [cust] + route[pos:]

                    load = sum(
                        self.data["nodes"][n]["DeliveryDemand"]
                        for n in candidate
                        if self.data["nodes"][n]["Type"] == "c"
                    )

                    if load > self.EV_CAP:
                        continue

                    if not self._route_feasible(sat, candidate):
                        continue

                    extra = (
                        self.dist[route[pos - 1], cust]
                        + self.dist[cust, route[pos]]
                        - self.dist[route[pos - 1], route[pos]]
                    )

                    if extra < best_cost:
                        best_cost = extra
                        best_pos = (
                            extra,
                            sat,
                            r_idx,
                            pos,
                        )

        return best_pos

    def _all_insertions(
        self,
        sol: Solution,
        cust: str,
    ) -> List[Tuple[float, str, int, int]]:
        results = []

        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
                    if self._time_exceeded():
                        results.sort()
                        return results

                    candidate = route[:pos] + [cust] + route[pos:]

                    load = sum(
                        self.data["nodes"][n]["DeliveryDemand"]
                        for n in candidate
                        if self.data["nodes"][n]["Type"] == "c"
                    )

                    if load > self.EV_CAP:
                        continue

                    if not self._route_feasible(sat, candidate):
                        continue

                    extra = (
                        self.dist[route[pos - 1], cust]
                        + self.dist[cust, route[pos]]
                        - self.dist[route[pos - 1], route[pos]]
                    )

                    results.append(
                        (
                            extra,
                            sat,
                            r_idx,
                            pos,
                        )
                    )

        results.sort()
        return results

    # ============================================================
    # Feasibility helpers
    # ============================================================

    def _route_feasible(
        self,
        sat: str,
        route: List[str],
    ) -> bool:
        """
        Check battery and time windows for a single EV route.
        """
        time_ = 0.0
        soc = self.BAT_CAP
        prev = route[0]

        for nxt in route[1:]:
            d = self.dist[prev, nxt]
            time_ += d / self.SPEED
            soc -= d

            if soc < -1e-9:
                return False

            ntype = self.data["nodes"][nxt]["Type"]

            if ntype == "f":
                time_ += (self.BAT_CAP - soc) * self.INV_G
                soc = self.BAT_CAP

            elif ntype == "c":
                node = self.data["nodes"][nxt]

                if time_ < node["ReadyTime"]:
                    time_ = node["ReadyTime"]

                if time_ > node["DueDate"]:
                    return False

                time_ += node["ServiceTime"]

            prev = nxt

        return True

    def _solution_feasible(self, sol: Solution) -> bool:
        return self.ev.evaluate(sol)["feasible"]

    # ============================================================
    # Operator weight update
    # ============================================================

    def _update_weights(
        self,
        d_weights: List[float],
        r_weights: List[float],
        d_scores: List[float],
        r_scores: List[float],
        d_counts: List[int],
        r_counts: List[int],
    ) -> Tuple[List[float], List[float]]:
        def update(
            weights: List[float],
            scores: List[float],
            counts: List[int],
        ) -> List[float]:
            new = []

            for w, s, c in zip(weights, scores, counts):
                if c > 0:
                    new.append(
                        (1 - self.REACTION) * w
                        + self.REACTION * (s / c)
                    )
                else:
                    new.append(w)

            mn = min(new)

            if mn < 0.1:
                new = [
                    max(v - mn + 0.1, 0.1)
                    for v in new
                ]

            return new

        return (
            update(d_weights, d_scores, d_counts),
            update(r_weights, r_scores, r_counts),
        )

    # ============================================================
    # Cost and utility helpers
    # ============================================================

    def _cost(self, sol: Solution) -> float:
        res = self.ev.evaluate(sol)
        return res["cost_with_M"]

    def _removal_count(self, sol: Solution) -> int:
        n = len(self._all_customers_in_solution(sol))

        if n <= 0:
            return 0

        return random.randint(
            min(self.MIN_REMOVAL, n),
            min(self.MAX_REMOVAL, n),
        )

    def _all_customers_in_solution(self, sol: Solution) -> List[str]:
        return [
            n
            for routes in sol.ev_routes.values()
            for route in routes
            for n in route
            if self.data["nodes"][n]["Type"] == "c"
        ]

    def _remove_customers(
        self,
        sol: Solution,
        to_remove: List[str],
    ) -> Solution:
        to_remove_set = set(to_remove)
        new_ev = {}

        for sat, routes in sol.ev_routes.items():
            new_routes = []

            for route in routes:
                new_route = [
                    n
                    for n in route
                    if n not in to_remove_set
                ]

                if any(
                    self.data["nodes"][n]["Type"] == "c"
                    for n in new_route
                ):
                    new_routes.append(new_route)

            if new_routes:
                new_ev[sat] = new_routes

        return Solution(
            lv_routes=sol.lv_routes,
            ev_routes=new_ev,
        )

    def _nearest_satellite(self, cust: str) -> str:
        cx = self.data["nodes"][cust]["x"]
        cy = self.data["nodes"][cust]["y"]

        return min(
            self.data["satellites"],
            key=lambda s: math.hypot(
                cx - self.data["nodes"][s]["x"],
                cy - self.data["nodes"][s]["y"],
            ),
        )

    @staticmethod
    def _weighted_choice(weights: List[float]) -> int:
        total = sum(weights)

        if total <= 0:
            return random.randrange(len(weights))

        r = random.random() * total
        cum = 0.0

        for i, w in enumerate(weights):
            cum += w

            if r <= cum:
                return i

        return len(weights) - 1
