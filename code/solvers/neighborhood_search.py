# ---------------------------------------------------------------
#  Adaptive Large Neighborhood Search for 2E-EVRP
#
#  Pipeline
#  --------
#  1. Build initial solution via Clarke-Wright
#  2. ALNS main loop:
#       a. select destroy operator (weighted random)
#       b. select repair operator  (weighted random)
#       c. destroy current solution
#       d. repair destroyed solution
#       e. accept/reject via simulated annealing
#       f. update operator weights
#  3. Return best feasible solution found
#
#  Destroy operators
#  -----------------
#  • random_removal      — remove k random customers
#  • worst_removal       — remove k customers with highest route cost contribution
#  • route_removal       — remove all customers from a random EV route
#  • time_window_removal — remove k customers with tightest time windows
#  • cluster_removal     — remove k geographically clustered customers
#
#  Repair operators
#  ----------------
#  • greedy_insertion    — insert each removed customer at cheapest feasible position
#  • regret_insertion    — insert customer with highest regret (best vs 2nd best) first
# ---------------------------------------------------------------

from __future__ import annotations

import math
import random
import copy
from typing import Dict, Any, List, Optional, Tuple

from core.evaluator import Evaluator, Solution
from solvers.clarke_wright import ClarkeWright


# ================================================================
#  Public entry class
# ================================================================
class ALNS:
    """
    solve(instance_dict) ->
        { 'solution' : Solution | None,
          'evs'      : int,
          'distance' : float }
    """

    # ---- tunable parameters ------------------------------------
    MAX_ITERATIONS   = 500       # main loop iterations
    SEGMENT_SIZE     = 50        # iterations between weight updates
    MIN_REMOVAL      = 2         # minimum customers removed per destroy
    MAX_REMOVAL      = 5         # maximum customers removed per destroy

    # simulated annealing
    START_TEMP       = 100.0     # initial temperature
    COOL_RATE        = 0.997     # temperature multiplier per iteration

    # weight update scores
    SCORE_BEST       = 10        # candidate is new global best
    SCORE_BETTER     = 5         # candidate improves on current
    SCORE_ACCEPTED   = 2         # candidate accepted but not better
    SCORE_REJECTED   = 0         # candidate rejected

    # weight update reaction factor (0=ignore history, 1=only history)
    REACTION         = 0.5

    # ------------------------------------------------------------
    def solve(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        self.data    = instance
        self.ev      = Evaluator(instance, check_sat_inventory=False)
        self.params  = instance["params"]
        self.EV_CAP  = self.params["C"]
        self.LV_CAP  = self.params["L"]
        self.BAT_CAP = self.params["Q"]
        self.SPEED   = self.params["v"]
        self.INV_G   = self.params["g"]

        # distance cache
        coords = {n: (instance["nodes"][n]["x"], instance["nodes"][n]["y"])
                  for n in instance["nodes"]}
        self.dist: Dict[Tuple[str, str], float] = {
            (i, j): math.hypot(coords[i][0] - coords[j][0],
                               coords[i][1] - coords[j][1])
            for i in coords for j in coords
        }

        # ---- build initial solution via Clarke-Wright -----------
        #print("  Building initial solution via Clarke-Wright...")
        cw_result  = ClarkeWright().solve(instance)
        current    = cw_result["solution"]

        if current is None:
            return {"solution": None, "evs": 0, "distance": float("inf")}

        current_cost = self._cost(current)
        best         = copy.deepcopy(current)
        best_cost    = current_cost

        #print(f"  Initial cost: {current_cost:.2f}")

        # ---- operator registries --------------------------------
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

        # weights and score accumulators
        d_weights = [1.0] * n_d
        r_weights = [1.0] * n_r
        d_scores  = [0.0] * n_d
        r_scores  = [0.0] * n_r
        d_counts  = [0]   * n_d
        r_counts  = [0]   * n_r

        temp = self.START_TEMP

        # ---- main loop ------------------------------------------
        for iteration in range(1, self.MAX_ITERATIONS + 1):

            # select operators
            d_idx = self._weighted_choice(d_weights)
            r_idx = self._weighted_choice(r_weights)

            # destroy + repair
            removed, destroyed = self.destroy_ops[d_idx](copy.deepcopy(current))
            if not removed:
                continue
            candidate = self.repair_ops[r_idx](destroyed, removed)
            if candidate is None:
                continue

            candidate_cost = self._cost(candidate)

            # score this iteration
            score = self.SCORE_REJECTED
            delta = candidate_cost - current_cost

            if candidate_cost < best_cost:
                best      = copy.deepcopy(candidate)
                best_cost = candidate_cost
                score     = self.SCORE_BEST

            if delta < 0:
                current      = candidate
                current_cost = candidate_cost
                if score != self.SCORE_BEST:
                    score = self.SCORE_BETTER
            elif random.random() < math.exp(-delta / max(temp, 1e-10)):
                current      = candidate
                current_cost = candidate_cost
                score        = self.SCORE_ACCEPTED

            d_scores[d_idx] += score
            r_scores[r_idx] += score
            d_counts[d_idx] += 1
            r_counts[r_idx] += 1

            # cool temperature
            temp *= self.COOL_RATE

            # update weights every segment
            if iteration % self.SEGMENT_SIZE == 0:
                d_weights, r_weights = self._update_weights(
                    d_weights, r_weights,
                    d_scores,  r_scores,
                    d_counts,  r_counts,
                )
                d_scores = [0.0] * n_d
                r_scores = [0.0] * n_r
                d_counts = [0]   * n_d
                r_counts = [0]   * n_r

                #print(f"  Iter {iteration:>4} | temp={temp:7.2f} | "
                      #f"best={best_cost:.2f} | current={current_cost:.2f}")

        # ---- final evaluation -----------------------------------
        res = self.ev.evaluate(best)
        return {
            "solution": best,
            "evs":      self.ev._count_evs(best),
            "distance": res["cost"],
        }

    # ================================================================
    #  DESTROY OPERATORS
    #  Each returns (removed_customers: List[str], partial_solution: Solution)
    # ================================================================

    def _random_removal(
        self, sol: Solution
    ) -> Tuple[List[str], Solution]:
        """Remove k randomly selected customers."""
        k        = self._removal_count(sol)
        all_custs = self._all_customers_in_solution(sol)
        if len(all_custs) < k:
            return [], sol
        removed = random.sample(all_custs, k)
        return removed, self._remove_customers(sol, removed)

    # ----------------------------------------------------------------
    def _worst_removal(
        self, sol: Solution
    ) -> Tuple[List[str], Solution]:
        """Remove k customers that contribute most to route cost."""
        contributions = []
        for sat, routes in sol.ev_routes.items():
            for r in routes:
                for i, node in enumerate(r):
                    if self.data["nodes"][node]["Type"] != "c":
                        continue
                    # cost saved if this customer were removed
                    prev = r[i - 1]
                    nxt  = r[i + 1]
                    saving = (self.dist[prev, node]
                              + self.dist[node, nxt]
                              - self.dist[prev, nxt])
                    contributions.append((saving, node))

        contributions.sort(reverse=True)
        k       = self._removal_count(sol)
        removed = [node for _, node in contributions[:k]]
        return removed, self._remove_customers(sol, removed)

    # ----------------------------------------------------------------
    def _route_removal(
        self, sol: Solution
    ) -> Tuple[List[str], Solution]:
        """Remove all customers from a randomly selected EV route."""
        all_routes = [
            (sat, r)
            for sat, routes in sol.ev_routes.items()
            for r in routes
        ]
        if not all_routes:
            return [], sol

        sat, route   = random.choice(all_routes)
        removed      = [n for n in route if self.data["nodes"][n]["Type"] == "c"]
        new_ev       = {
            s: [r for r in rlist if r is not route]
            for s, rlist in sol.ev_routes.items()
        }
        return removed, Solution(lv_routes=sol.lv_routes, ev_routes=new_ev)

    # ----------------------------------------------------------------
    def _time_window_removal(
        self, sol: Solution
    ) -> Tuple[List[str], Solution]:
        """Remove k customers with the tightest (smallest) time windows."""
        windows = []
        for node_id in self._all_customers_in_solution(sol):
            n  = self.data["nodes"][node_id]
            tw = n["DueDate"] - n["ReadyTime"]
            windows.append((tw, node_id))

        windows.sort()   # tightest first
        k       = self._removal_count(sol)
        removed = [node for _, node in windows[:k]]
        return removed, self._remove_customers(sol, removed)

    # ----------------------------------------------------------------
    def _cluster_removal(
        self, sol: Solution
    ) -> Tuple[List[str], Solution]:
        """Remove k geographically clustered customers (random seed)."""
        all_custs = self._all_customers_in_solution(sol)
        if not all_custs:
            return [], sol

        seed = random.choice(all_custs)
        sx   = self.data["nodes"][seed]["x"]
        sy   = self.data["nodes"][seed]["y"]

        by_dist = sorted(
            all_custs,
            key=lambda c: math.hypot(
                self.data["nodes"][c]["x"] - sx,
                self.data["nodes"][c]["y"] - sy,
            )
        )
        k       = self._removal_count(sol)
        removed = by_dist[:k]
        return removed, self._remove_customers(sol, removed)

    # ================================================================
    #  REPAIR OPERATORS
    #  Each takes (partial_solution, removed_customers) and returns
    #  a complete Solution or None if repair failed.
    # ================================================================

    def _greedy_insertion(
        self, sol: Solution, removed: List[str]
    ) -> Optional[Solution]:
        """
        Insert each removed customer at the cheapest feasible position
        across all existing EV routes (and new routes if needed).
        Customers are sorted by insertion cost descending (hardest first).
        """
        current = copy.deepcopy(sol)

        # sort by how expensive the cheapest insertion is — hardest first
        def min_insertion_cost(c):
            best = self._best_insertion(current, c)
            return best[0] if best else float("inf")

        for cust in sorted(removed, key=min_insertion_cost, reverse=True):
            best = self._best_insertion(current, cust)
            if best is None:
                # open a new spoke route at the nearest satellite
                sat = self._nearest_satellite(cust)
                new_route = [sat, cust, sat]
                if not self._route_feasible(sat, new_route):
                    return None
                current.ev_routes.setdefault(sat, []).append(new_route)
            else:
                _, sat, r_idx, pos = best
                route = current.ev_routes[sat][r_idx]
                current.ev_routes[sat][r_idx] = route[:pos] + [cust] + route[pos:]

        return current if self._solution_feasible(current) else None

    # ----------------------------------------------------------------
    def _regret_insertion(
        self, sol: Solution, removed: List[str]
    ) -> Optional[Solution]:
        """
        Regret-2 insertion: repeatedly insert the customer whose
        difference between best and 2nd-best insertion cost is largest.
        This prioritises customers with few good alternatives.
        """
        current   = copy.deepcopy(sol)
        remaining = list(removed)

        while remaining:
            regrets = []
            for cust in remaining:
                positions = self._all_insertions(current, cust)
                if len(positions) == 0:
                    regret = float("inf")   # must open new route — urgent
                elif len(positions) == 1:
                    regret = positions[0][0]
                else:
                    regret = positions[1][0] - positions[0][0]
                regrets.append((regret, cust))

            # pick customer with highest regret
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
                current.ev_routes[sat][r_idx] = route[:pos] + [chosen] + route[pos:]

        return current if self._solution_feasible(current) else None

    # ================================================================
    #  INSERTION HELPERS
    # ================================================================

    def _best_insertion(
        self, sol: Solution, cust: str
    ) -> Optional[Tuple[float, str, int, int]]:
        """
        Find the cheapest feasible insertion of cust into any existing route.
        Returns (cost, sat, route_index, position) or None.
        """
        best_cost = float("inf")
        best_pos  = None

        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
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
                    extra = (self.dist[route[pos-1], cust]
                             + self.dist[cust, route[pos]]
                             - self.dist[route[pos-1], route[pos]])
                    if extra < best_cost:
                        best_cost = extra
                        best_pos  = (extra, sat, r_idx, pos)

        return best_pos

    def _all_insertions(
        self, sol: Solution, cust: str
    ) -> List[Tuple[float, str, int, int]]:
        """Return all feasible insertions sorted by cost ascending."""
        results = []
        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                for pos in range(1, len(route)):
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
                    extra = (self.dist[route[pos-1], cust]
                             + self.dist[cust, route[pos]]
                             - self.dist[route[pos-1], route[pos]])
                    results.append((extra, sat, r_idx, pos))
        results.sort()
        return results

    # ================================================================
    #  FEASIBILITY CHECKS
    # ================================================================

    def _route_feasible(self, sat: str, route: List[str]) -> bool:
        """Check battery and time windows for a single EV route."""
        time = 0.0
        soc  = self.BAT_CAP
        prev = route[0]

        for nxt in route[1:]:
            d     = self.dist[prev, nxt]
            time += d / self.SPEED
            soc  -= d

            if soc < -1e-9:
                return False

            ntype = self.data["nodes"][nxt]["Type"]
            if ntype == "f":
                time += (self.BAT_CAP - soc) * self.INV_G
                soc   = self.BAT_CAP
            elif ntype == "c":
                node = self.data["nodes"][nxt]
                if time < node["ReadyTime"]:
                    time = node["ReadyTime"]
                if time > node["DueDate"]:
                    return False
                time += node["ServiceTime"]
            prev = nxt
        return True

    def _solution_feasible(self, sol: Solution) -> bool:
        return self.ev.evaluate(sol)["feasible"]

    # ================================================================
    #  WEIGHT UPDATE
    # ================================================================

    def _update_weights(
        self,
        d_weights, r_weights,
        d_scores,  r_scores,
        d_counts,  r_counts,
    ):
        def update(weights, scores, counts):
            new = []
            for w, s, c in zip(weights, scores, counts):
                if c > 0:
                    new.append(
                        (1 - self.REACTION) * w
                        + self.REACTION * (s / c)
                    )
                else:
                    new.append(w)
            # normalise so min weight stays above 0.1
            mn = min(new)
            if mn < 0.1:
                new = [max(v - mn + 0.1, 0.1) for v in new]
            return new

        return (
            update(d_weights, d_scores, d_counts),
            update(r_weights, r_scores, r_counts),
        )

    # ================================================================
    #  UTILITIES
    # ================================================================

    def _cost(self, sol: Solution) -> float:
        """Lexicographic cost: M * #EVs + distance."""
        res = self.ev.evaluate(sol)
        return res["cost_with_M"]

    def _removal_count(self, sol: Solution) -> int:
        n = len(self._all_customers_in_solution(sol))
        return random.randint(
            min(self.MIN_REMOVAL, n),
            min(self.MAX_REMOVAL, n),
        )

    def _all_customers_in_solution(self, sol: Solution) -> List[str]:
        return [
            n
            for routes in sol.ev_routes.values()
            for r in routes
            for n in r
            if self.data["nodes"][n]["Type"] == "c"
        ]

    def _remove_customers(
        self, sol: Solution, to_remove: List[str]
    ) -> Solution:
        new_ev = {}
        for sat, routes in sol.ev_routes.items():
            new_routes = []
            for r in routes:
                new_r = [n for n in r if n not in to_remove]
                # only keep route if it still has customers
                if any(self.data["nodes"][n]["Type"] == "c" for n in new_r):
                    new_routes.append(new_r)
            if new_routes:
                new_ev[sat] = new_routes
        return Solution(lv_routes=sol.lv_routes, ev_routes=new_ev)

    def _nearest_satellite(self, cust: str) -> str:
        cx = self.data["nodes"][cust]["x"]
        cy = self.data["nodes"][cust]["y"]
        return min(
            self.data["satellites"],
            key=lambda s: math.hypot(
                cx - self.data["nodes"][s]["x"],
                cy - self.data["nodes"][s]["y"],
            )
        )

    @staticmethod
    def _weighted_choice(weights: List[float]) -> int:
        total = sum(weights)
        r     = random.random() * total
        cum   = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1