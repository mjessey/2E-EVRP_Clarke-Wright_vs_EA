# ---------------------------------------------------------------
#  Stage 4: Memetic Algorithm — Main Loop + LCS Population Update
#
#  Based on Algorithm 1 and Section 4.5 of:
#  "A Memetic Algorithm for the Green Vehicle Routing Problem"
#  Peng et al. (2019)
#
#  Full pipeline
#  -------------
#  1. Generate initial population via k-Pseudo Greedy
#  2. Apply adaptive local search to each individual
#  3. Main loop until stopping criterion:
#       a. Randomly select two parents from population
#       b. Generate offspring via backbone crossover
#       c. Improve offspring with adaptive local search
#       d. Update global best if improved
#       e. LCS-based population update
#  4. Return best solution found
#
#  Anytime behavior
#  ----------------
#  This implementation supports:
#
#      solve(instance, time_limit_sec=..., seed=...)
#
#  and is designed to behave as an anytime solver:
#
#      - it keeps track of the best solution found so far
#      - it checks a deadline before expensive phases
#      - it gives local search only the remaining available time
#      - it skips optional route-elimination/extra-improvement steps
#        when too close to the deadline
#
#  This is important when used through solver_runner.py, where an
#  outer hard timeout may terminate the process if the solver does
#  not return in time.
# ---------------------------------------------------------------

from __future__ import annotations

import math
import time
import copy
import random
from typing import Dict, Any, List, Optional, Tuple

from core.evaluator import Evaluator, Solution
from solvers.memetic_helpers.k_pseudo_greedy import KPseudoGreedy
from solvers.memetic_helpers.adaptive_local_search import AdaptiveLocalSearch
from solvers.memetic_helpers.backbone_crossover import BackboneCrossover


class MemeticAlgorithm:
    """
    Memetic Algorithm for 2E-EVRP based on Peng et al. (2019).

    solve(instance_dict, time_limit_sec=None, seed=None) ->
        { 'solution' : Solution | None,
          'evs'      : int,
          'distance' : float }
    """

    # ---- parameters from paper Table 1 -------------------------
    POP_SIZE = 6        # np: population size
    DELTA = 0.7         # δ: quality vs diversity balance
    K_GREEDY = 2        # k: k-Pseudo Greedy candidates
    MAX_TIME = 10.0     # seconds: default stopping criterion

    # ---- anytime safety parameters -----------------------------
    # If less than this much time remains, skip optional expensive
    # improvement/cleanup steps and focus on returning best-so-far.
    RETURN_SAFETY_SEC = 0.25

    # Minimum time slice given to local search.
    MIN_LOCAL_SEARCH_SEC = 0.05

    # ------------------------------------------------------------
    def solve(
        self,
        instance: Dict[str, Any],
        time_limit_sec: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the Memetic Algorithm.

        Parameters
        ----------
        instance:
            Parsed 2E-EVRP instance dictionary.

        time_limit_sec:
            Optional wall-clock time limit. If None, self.MAX_TIME is used.

        seed:
            Optional random seed. Used by ParallelMemetic so each island
            explores a different region of the search space.
        """
        if seed is not None:
            random.seed(seed)

        max_time = (
            float(time_limit_sec)
            if time_limit_sec is not None
            else self.MAX_TIME
        )

        t_start = time.time()
        deadline = t_start + max_time

        self.data = instance
        self.ev = Evaluator(instance, check_sat_inventory=False)

        self.EV_CAP = instance["params"]["C"]
        self.BAT_CAP = instance["params"]["Q"]
        self.SPEED = instance["params"]["v"]
        self.INV_G = instance["params"]["g"]

        coords = {
            n: (
                instance["nodes"][n]["x"],
                instance["nodes"][n]["y"],
            )
            for n in instance["nodes"]
        }

        self.dist = {
            (i, j): math.hypot(
                coords[i][0] - coords[j][0],
                coords[i][1] - coords[j][1],
            )
            for i in coords
            for j in coords
        }

        # initialise component solvers
        greedy = KPseudoGreedy(instance, k=self.K_GREEDY)
        local = AdaptiveLocalSearch(instance)
        crossover = BackboneCrossover(instance)

        # --------------------------------------------------------
        # 1. initial population
        # --------------------------------------------------------
        print("  [MA] Generating initial population...")

        # For short anytime runs, use a smaller population so the
        # algorithm can reach the evolutionary loop instead of spending
        # the entire budget on initial population improvement.
        pop_size = self.POP_SIZE

        if max_time <= 5.0:
            pop_size = min(self.POP_SIZE, 5)
        elif max_time <= 15.0:
            pop_size = min(self.POP_SIZE, 8)

        population = greedy.generate_population(pop_size)

        if not population:
            return {
                "solution": None,
                "evs": 0,
                "distance": float("inf"),
            }

        # If construction itself consumed the whole budget, return the
        # best constructed solution.
        if self._time_nearly_exhausted(deadline):
            best = self._best_solution_from_population(population)
            return self._format_result(best)

        # --------------------------------------------------------
        # 2. improve each initial individual
        # --------------------------------------------------------
        print("  [MA] Running local search on initial population...")

        for idx, s in enumerate(population):
            remaining = self._remaining_time(deadline)

            if remaining <= self.RETURN_SAFETY_SEC:
                break

            remaining_individuals = max(1, len(population) - idx)

            # Reserve some time for at least a few generations.
            ls_limit = min(
                2.0,
                max(
                    self.MIN_LOCAL_SEARCH_SEC,
                    remaining / (remaining_individuals + 3),
                ),
            )

            improved = local.run(s, time_limit=ls_limit)

            # Route elimination is optional and can be expensive.
            # Skip it if close to the deadline.
            improved = self._maybe_eliminate_route(
                improved,
                deadline=deadline,
            )

            population[idx] = improved
            cost = self._cost(improved)

            print(
                f"  [MA] Individual {idx + 1}/{len(population)} "
                f"cost: {cost:.2f} | LS={ls_limit:.2f}s"
            )

        costs = [self._cost(s) for s in population]
        best_idx = min(range(len(population)), key=lambda i: costs[i])
        best = copy.deepcopy(population[best_idx])
        best_cost = costs[best_idx]

        print(f"  [MA] Initial best cost: {best_cost:.2f}")
        print(f"  [MA] Population costs: {[round(c, 2) for c in costs]}")

        # --------------------------------------------------------
        # 3. main evolutionary loop
        # --------------------------------------------------------
        generation = 0

        while not self._time_nearly_exhausted(deadline):
            generation += 1

            if len(population) < 2:
                break

            i, j = random.sample(range(len(population)), 2)
            pa, pb = population[i], population[j]

            # Crossover can be moderately expensive, so check the
            # deadline before starting it.
            if self._time_nearly_exhausted(deadline):
                break

            offspring = crossover.crossover(pa, pb)

            if offspring is None:
                continue

            parent_cost = min(self._cost(pa), self._cost(pb))
            offspring_cost_before_ls = self._cost(offspring)

            remaining = self._remaining_time(deadline)

            if remaining <= self.RETURN_SAFETY_SEC:
                break

            offspring_ls_limit = min(
                5.0,
                max(
                    self.MIN_LOCAL_SEARCH_SEC,
                    remaining * 0.5,
                ),
            )

            offspring = local.run(
                offspring,
                time_limit=offspring_ls_limit,
                quick=True,
            )

            offspring = self._maybe_eliminate_route(
                offspring,
                deadline=deadline,
            )

            offspring_cost = self._cost(offspring)

            # Optional extra stripped-parent attempt.
            # This can be useful but should be skipped near deadline.
            if (
                generation % 5 == 0
                and offspring_cost >= best_cost
                and not self._time_nearly_exhausted(deadline)
            ):
                pa_stripped = self._strip_smallest_route(pa)

                if pa_stripped is not None and not self._time_nearly_exhausted(deadline):
                    stripped_offspring = crossover.crossover(pa_stripped, pb)

                    if stripped_offspring is not None:
                        remaining = self._remaining_time(deadline)

                        if remaining > self.RETURN_SAFETY_SEC:
                            stripped_ls_limit = min(
                                5.0,
                                max(
                                    self.MIN_LOCAL_SEARCH_SEC,
                                    remaining * 0.5,
                                ),
                            )

                            stripped_offspring = local.run(
                                stripped_offspring,
                                time_limit=stripped_ls_limit,
                            )

                            stripped_offspring = self._maybe_eliminate_route(
                                stripped_offspring,
                                deadline=deadline,
                            )

                            stripped_cost = self._cost(stripped_offspring)

                            if stripped_cost < offspring_cost:
                                offspring = stripped_offspring
                                offspring_cost = stripped_cost

                        print(
                            f"  [MA] Gen {generation:>4} | "
                            f"parents={parent_cost:.2f} | "
                            f"pre-LS={offspring_cost_before_ls:.2f} | "
                            f"post-LS={offspring_cost:.2f}"
                        )

            # update global best
            if offspring_cost < best_cost:
                best = copy.deepcopy(offspring)
                best_cost = offspring_cost

                print(
                    f"  [MA] Gen {generation:>4} | "
                    f"New best: {best_cost:.2f} | "
                    f"t={time.time() - t_start:.1f}s"
                )

            # LCS update is useful but can cost O(p^2 n^2).
            # Population is small, but still skip it if too close
            # to the deadline. In that case, best-so-far is already safe.
            if self._time_nearly_exhausted(deadline):
                break

            population, costs = self._lcs_update(
                population,
                costs,
                offspring,
                offspring_cost,
            )

        print(
            f"  [MA] Finished — {generation} generations | "
            f"best cost: {best_cost:.2f}"
        )

        return self._format_result(best)

    # ============================================================
    #  Anytime helpers
    # ============================================================

    def _remaining_time(self, deadline: float) -> float:
        return deadline - time.time()

    def _time_nearly_exhausted(self, deadline: float) -> bool:
        return self._remaining_time(deadline) <= self.RETURN_SAFETY_SEC

    def _maybe_eliminate_route(
        self,
        sol: Solution,
        deadline: float,
    ) -> Solution:
        """
        Try route elimination only if there is enough time left.

        _try_eliminate_route can be expensive because it attempts
        greedy reinsertion of removed customers. For anytime behavior,
        skip it near the deadline so the solver can return best-so-far.
        """
        if self._time_nearly_exhausted(deadline):
            return sol

        eliminated = self._try_eliminate_route(sol)

        if eliminated is not None:
            return eliminated

        return sol

    def _best_solution_from_population(
        self,
        population: List[Solution],
    ) -> Solution:
        costs = [self._cost(s) for s in population]
        best_idx = min(range(len(population)), key=lambda i: costs[i])
        return copy.deepcopy(population[best_idx])

    def _format_result(self, sol: Optional[Solution]) -> Dict[str, Any]:
        if sol is None:
            return {
                "solution": None,
                "evs": 0,
                "distance": float("inf"),
            }

        res = self.ev.evaluate(sol)

        return {
            "solution": sol,
            "evs": self.ev._count_evs(sol),
            "distance": res["cost"],
        }

    # ============================================================
    #  LCS-based population updating
    # ============================================================

    def _lcs_update(
        self,
        population: List[Solution],
        costs: List[float],
        offspring: Solution,
        offspring_cost: float,
    ) -> Tuple[List[Solution], List[float]]:
        """
        Insert offspring into population if its goodness score
        exceeds that of the worst individual.
        Returns updated (population, costs).
        """
        n = len(population)
        all_sols = population + [offspring]
        all_costs = costs + [offspring_cost]

        seqs = [self._flatten(s) for s in all_sols]

        distances = []

        for i in range(len(all_sols)):
            min_dist = math.inf

            for j in range(len(all_sols)):
                if i == j:
                    continue

                lcs_len = self._lcs_length(seqs[i], seqs[j])

                # Symmetric LCS distance.
                dist_ij = len(seqs[i]) + len(seqs[j]) - 2 * lcs_len

                if dist_ij < min_dist:
                    min_dist = dist_ij

            distances.append(min_dist if min_dist != math.inf else 0.0)

        f_max = max(all_costs)
        f_min = min(all_costs)

        d_max = max(distances)
        d_min = min(distances)

        def goodness(idx: int) -> float:
            cost_spread = max(all_costs) - min(all_costs)

            # If solutions are close in cost, encourage diversity more.
            delta = 0.3 if cost_spread < 1000 else self.DELTA

            quality = (f_max - all_costs[idx]) / (f_max - f_min + 1)
            diversity = (distances[idx] - d_min) / (d_max - d_min + 1)

            return delta * quality + (1 - delta) * diversity

        scores = [goodness(i) for i in range(len(all_sols))]

        offspring_idx = len(all_sols) - 1
        offspring_gs = scores[offspring_idx]

        worst_idx = min(range(n), key=lambda i: scores[i])
        worst_gs = scores[worst_idx]

        if offspring_gs > worst_gs:
            population[worst_idx] = offspring
            costs[worst_idx] = offspring_cost

        return population, costs

    # ============================================================
    #  LCS helpers
    # ============================================================

    def _flatten(self, sol: Solution) -> List[str]:
        """
        Flatten a solution to a single sequence of customer and
        station node IDs across all EV routes.

        Satellite endpoints are excluded.
        """
        seq: List[str] = []

        for routes in sol.ev_routes.values():
            for r in routes:
                for node_id in r[1:-1]:
                    node_type = self.data["nodes"][node_id]["Type"]

                    if node_type in ("c", "f"):
                        seq.append(node_id)

        return seq

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """
        Compute the length of the longest common subsequence
        between sequences a and b.

        O(|a| * |b|) time and O(|b|) space.
        """
        m, n = len(a), len(b)

        if m == 0 or n == 0:
            return 0

        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])

            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    # ============================================================
    #  Route elimination helpers
    # ============================================================

    def _strip_smallest_route(self, sol: Solution) -> Optional[Solution]:
        min_custs = float("inf")
        min_sat = None
        min_ridx = None

        for sat, routes in sol.ev_routes.items():
            for ridx, route in enumerate(routes):
                n_custs = sum(
                    1
                    for node_id in route
                    if self.data["nodes"][node_id]["Type"] == "c"
                )

                if n_custs < min_custs:
                    min_custs = n_custs
                    min_sat = sat
                    min_ridx = ridx

        if min_sat is None or min_ridx is None:
            return None

        new_ev = copy.deepcopy(sol.ev_routes)
        new_ev[min_sat].pop(min_ridx)

        if not new_ev[min_sat]:
            del new_ev[min_sat]

        return Solution(
            lv_routes=sol.lv_routes,
            ev_routes=new_ev,
        )

    def _try_eliminate_route(self, sol: Solution) -> Optional[Solution]:
        """
        Find the route with fewest customers, remove all its customers,
        and greedily reinsert them into other routes.

        Returns improved/repaired solution if successful, None otherwise.

        Note:
        This function is intentionally not called near the deadline by
        _maybe_eliminate_route(), because it may be expensive.
        """
        min_custs = float("inf")
        min_sat = None
        min_ridx = None

        for sat, routes in sol.ev_routes.items():
            for ridx, route in enumerate(routes):
                n_custs = sum(
                    1
                    for node_id in route
                    if self.data["nodes"][node_id]["Type"] == "c"
                )

                if 0 < n_custs < min_custs:
                    min_custs = n_custs
                    min_sat = sat
                    min_ridx = ridx

        if min_sat is None or min_ridx is None:
            return None

        removed = [
            node_id
            for node_id in sol.ev_routes[min_sat][min_ridx]
            if self.data["nodes"][node_id]["Type"] == "c"
        ]

        new_ev = copy.deepcopy(sol.ev_routes)
        new_ev[min_sat].pop(min_ridx)

        if not new_ev[min_sat]:
            del new_ev[min_sat]

        partial = Solution(
            lv_routes=sol.lv_routes,
            ev_routes=new_ev,
        )

        for cust in removed:
            best_cost = float("inf")
            best_sat = None
            best_ridx = None
            best_pos = None

            for sat, routes in partial.ev_routes.items():
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        candidate = route[:pos] + [cust] + route[pos:]

                        load = sum(
                            self.data["nodes"][node_id]["DeliveryDemand"]
                            for node_id in candidate
                            if self.data["nodes"][node_id]["Type"] == "c"
                        )

                        if load > self.EV_CAP:
                            continue

                        if not self._route_feasible_check(candidate, sat):
                            continue

                        cost = sum(
                            self.dist[a, b]
                            for a, b in zip(candidate, candidate[1:])
                        )

                        if cost < best_cost:
                            best_cost = cost
                            best_sat = sat
                            best_ridx = r_idx
                            best_pos = pos

            if best_sat is None or best_ridx is None or best_pos is None:
                return None

            route = partial.ev_routes[best_sat][best_ridx]
            partial.ev_routes[best_sat][best_ridx] = (
                route[:best_pos] + [cust] + route[best_pos:]
            )

        return partial

    def _route_feasible_check(self, route: List[str], sat: str) -> bool:
        """
        Check battery and customer time windows for one EV route.
        """
        time_now = 0.0
        soc = self.BAT_CAP
        prev = route[0]

        for nxt in route[1:]:
            d = self.dist[prev, nxt]
            time_now += d / self.SPEED
            soc -= d

            if soc < -1e-9:
                return False

            node_type = self.data["nodes"][nxt]["Type"]

            if node_type == "f":
                time_now += (self.BAT_CAP - soc) * self.INV_G
                soc = self.BAT_CAP

            elif node_type == "c":
                node = self.data["nodes"][nxt]

                if time_now < node["ReadyTime"]:
                    time_now = node["ReadyTime"]

                if time_now > node["DueDate"]:
                    return False

                time_now += node["ServiceTime"]

            prev = nxt

        return True

    # ============================================================
    #  Cost helper
    # ============================================================

    def _cost(self, sol: Solution) -> float:
        return self.ev.evaluate(sol)["cost_with_M"]
