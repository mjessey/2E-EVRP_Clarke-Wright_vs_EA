# ---------------------------------------------------------------
#  Stage 4: Memetic Algorithm — Main Loop + LCS Population Update
#
#  Based on Algorithm 1 and Section 4.5 of:
#  "A Memetic Algorithm for the Green Vehicle Routing Problem"
#  Peng et al. (2019)
#
#  Full pipeline
#  -------------
#  1. Generate initial population via k-Pseudo Greedy (Stage 1)
#  2. Apply adaptive local search to each individual (Stage 2)
#  3. Main loop until stopping criterion:
#       a. Randomly select two parents from population
#       b. Generate offspring via backbone crossover (Stage 3)
#       c. Improve offspring with adaptive local search (Stage 2)
#       d. Update global best if improved
#       e. LCS-based population update (Section 4.5)
#  4. Return best solution found
#
#  LCS Population Updating (Section 4.5)
#  --------------------------------------
#  Decides whether to insert the offspring into the population
#  by computing a "goodness score" that balances:
#    - Solution quality  (objective value)
#    - Population diversity (LCS distance to nearest neighbour)
#
#  GS(Si) = δ * (fmax - f(Si)) / (fmax - fmin + 1)
#          + (1-δ) * (Dist(Si) - Distmin) / (Distmax - Distmin + 1)
#
#  Insert offspring if GS(offspring) > GS(worst individual).
#  This keeps the population both elite and diverse.
#
#  LCS distance between two solutions
#  -----------------------------------
#  Flatten each solution to a sequence of all customer+station
#  node IDs across all routes. The LCS distance is:
#    Dist(Si, Sj) = 2*(n+s) - LCS_length(Si, Sj)
#  where n+s = total customers + stations.
#  Lower LCS distance = more similar solutions.
#  The distance of a solution to the population = min over all
#  other individuals.
#
#  Parameters from paper Table 1
#  ------------------------------
#  np    = 15     population size
#  delta = 0.7    weight balancing quality vs diversity
#  k     = 3      k-Pseudo Greedy parameter
#  max computing time = stopping criterion
# ---------------------------------------------------------------

from __future__ import annotations

import math
import time
import copy
import random
from typing import Dict, Any, List, Optional, Tuple

from core.evaluator import Evaluator, Solution
from solvers.memetic.k_pseudo_greedy       import KPseudoGreedy
from solvers.memetic.adaptive_local_search import AdaptiveLocalSearch
from solvers.memetic.backbone_crossover    import BackboneCrossover


class MemeticAlgorithm:
    """
    Memetic Algorithm for 2E-EVRP based on Peng et al. (2019).

    solve(instance_dict) ->
        { 'solution' : Solution | None,
          'evs'      : int,
          'distance' : float }
    """

    # ---- parameters from paper Table 1 -------------------------
    POP_SIZE   = 15       # np:    population size
    DELTA      = 0.7      # δ:     quality vs diversity balance
    K_GREEDY   = 3        # k:     k-Pseudo Greedy candidates
    MAX_TIME   = 180.0     # seconds: stopping criterion

    # ------------------------------------------------------------
    def solve(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        self.data    = instance
        self.ev      = Evaluator(instance, check_sat_inventory=False)

        self.EV_CAP  = instance["params"]["C"]
        self.BAT_CAP = instance["params"]["Q"]
        self.SPEED   = instance["params"]["v"]
        self.INV_G   = instance["params"]["g"]
        coords = {n: (instance["nodes"][n]["x"], instance["nodes"][n]["y"])
                for n in instance["nodes"]}
        self.dist = {
            (i, j): math.hypot(coords[i][0] - coords[j][0],
                            coords[i][1] - coords[j][1])
            for i in coords for j in coords
        }

        # initialise component solvers
        greedy    = KPseudoGreedy(instance, k=self.K_GREEDY)
        local     = AdaptiveLocalSearch(instance)
        crossover = BackboneCrossover(instance)

        t_start = time.time()

        # ---- 1. initial population ------------------------------
        print("  [MA] Generating initial population...")
        population = greedy.generate_population(self.POP_SIZE)

        if not population:
            return {"solution": None, "evs": 0,
                    "distance": float("inf")}

        # ---- 2. improve each individual -------------------------
        print("  [MA] Running local search on initial population...")
        for idx, s in enumerate(population):
            improved   = local.run(s, time_limit=2.0)
            eliminated = self._try_eliminate_route(improved)
            if eliminated is not None:
                improved = eliminated
            population[idx] = improved
            cost = self._cost(improved)
            print(f"  [MA] Individual {idx+1}/15 cost: {cost:.2f}")

        costs = [self._cost(s) for s in population]
        best_idx  = min(range(len(population)), key=lambda i: costs[i])
        best      = copy.deepcopy(population[best_idx])
        best_cost = costs[best_idx]

        print(f"  [MA] Initial best cost: {best_cost:.2f}")
        print(f"  [MA] Population costs: {[round(c, 2) for c in costs]}")

        # ---- 3. main loop ---------------------------------------
        generation = 0
        while time.time() - t_start < self.MAX_TIME:
            generation += 1

            # select two distinct parents randomly
            if len(population) < 2:
                break
            i, j = random.sample(range(len(population)), 2)
            pa, pb = population[i], population[j]

            # crossover
            offspring = crossover.crossover(pa, pb)
            if offspring is None:
                continue

            parent_cost              = min(self._cost(pa), self._cost(pb))
            offspring_cost_before_ls = self._cost(offspring)
            offspring      = local.run(offspring, time_limit=5.0, quick=True)
            eliminated     = self._try_eliminate_route(offspring)
            if eliminated is not None:
                offspring  = eliminated
            offspring_cost = self._cost(offspring)
            if generation % 5 == 0 and offspring_cost >= best_cost:
                pa_stripped = self._strip_smallest_route(pa)
                if pa_stripped is not None:
                    stripped_offspring = crossover.crossover(pa_stripped, pb)
                    if stripped_offspring is not None:
                        stripped_offspring = local.run(
                            stripped_offspring, time_limit=5.0
                        )
                        stripped_cost = self._cost(stripped_offspring)
                        if stripped_cost < offspring_cost:
                            offspring      = stripped_offspring
                            offspring_cost = stripped_cost

                        print(f"  [MA] Gen {generation:>4} | "
                            f"parents={parent_cost:.2f} | "
                            f"pre-LS={offspring_cost_before_ls:.2f} | "
                            f"post-LS={offspring_cost:.2f}")

            # update global best
            if offspring_cost < best_cost:
                best      = copy.deepcopy(offspring)
                best_cost = offspring_cost
                print(f"  [MA] Gen {generation:>4} | "
                      f"New best: {best_cost:.2f} | "
                      f"t={time.time()-t_start:.1f}s")

            # LCS population update
            population, costs = self._lcs_update(
                population, costs, offspring, offspring_cost
            )

        print(f"  [MA] Finished — {generation} generations | "
              f"best cost: {best_cost:.2f}")

        # ---- 4. final evaluation --------------------------------
        res = self.ev.evaluate(best)
        return {
            "solution": best,
            "evs":      self.ev._count_evs(best),
            "distance": res["cost"],
        }

    # ============================================================
    #  LCS-based population updating  (Section 4.5)
    # ============================================================

    def _lcs_update(
        self,
        population:     List[Solution],
        costs:          List[float],
        offspring:      Solution,
        offspring_cost: float,
    ) -> Tuple[List[Solution], List[float]]:
        """
        Insert offspring into population if its goodness score
        exceeds that of the worst individual.
        Returns updated (population, costs).
        """
        n   = len(population)
        all_sols = population + [offspring]
        all_costs = costs + [offspring_cost]

        # ---- compute LCS distances ------------------------------
        # distance of each solution to its nearest neighbour
        seqs = [self._flatten(s) for s in all_sols]
        n_s  = len(seqs[0])   # 2*(customers + stations)

        distances = []
        for i in range(len(all_sols)):
            min_dist = math.inf
            for j in range(len(all_sols)):
                if i == j:
                    continue
                lcs_len  = self._lcs_length(seqs[i], seqs[j])
                dist_ij  = n_s - lcs_len
                if dist_ij < min_dist:
                    min_dist = dist_ij
            distances.append(min_dist if min_dist != math.inf else 0.0)

        # ---- compute goodness scores ----------------------------
        f_vals   = all_costs
        f_max    = max(f_vals)
        f_min    = min(f_vals)
        d_vals   = distances
        d_max    = max(d_vals)
        d_min    = min(d_vals)

        def goodness(idx: int) -> float:
            cost_spread = max(all_costs) - min(all_costs)
            delta       = 0.3 if cost_spread < 1000 else self.DELTA
            quality     = (f_max - all_costs[idx]) / (f_max - f_min + 1)
            diversity   = (distances[idx] - d_min) / (d_max - d_min + 1)
            return delta * quality + (1 - delta) * diversity

        scores = [goodness(i) for i in range(len(all_sols))]

        # offspring is the last element
        offspring_idx = len(all_sols) - 1
        offspring_gs  = scores[offspring_idx]

        # worst individual in current population (excluding offspring)
        worst_idx = min(range(n), key=lambda i: scores[i])
        worst_gs  = scores[worst_idx]

        if offspring_gs > worst_gs:
            # replace worst with offspring
            population[worst_idx] = offspring
            costs[worst_idx]      = offspring_cost

        return population, costs

    # ============================================================
    #  LCS helpers
    # ============================================================

    def _flatten(self, sol: Solution) -> List[str]:
        """
        Flatten a solution to a single sequence of customer and
        station node IDs across all EV routes.
        Satellite endpoints are excluded — only the inner nodes
        matter for LCS similarity comparison.
        """
        seq = []
        for routes in sol.ev_routes.values():
            for r in routes:
                for n in r[1:-1]:   # skip sat endpoints
                    t = self.data["nodes"][n]["Type"]
                    if t in ("c", "f"):
                        seq.append(n)
        return seq

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """
        Compute the length of the longest common subsequence
        between sequences a and b using standard DP.
        O(|a| * |b|) time and space.
        """
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0

        # space-optimised: only keep two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]
    def _strip_smallest_route(self, sol: Solution) -> Optional[Solution]:
        min_custs = float("inf")
        min_sat   = None
        min_ridx  = None
        for sat, routes in sol.ev_routes.items():
            for ridx, r in enumerate(routes):
                n = sum(1 for n in r
                        if self.data["nodes"][n]["Type"] == "c")
                if n < min_custs:
                    min_custs = n
                    min_sat   = sat
                    min_ridx  = ridx
        if min_sat is None:
            return None
        new_ev = copy.deepcopy(sol.ev_routes)
        new_ev[min_sat].pop(min_ridx)
        return Solution(lv_routes=sol.lv_routes, ev_routes=new_ev)
    
    def _try_eliminate_route(self, sol: Solution) -> Optional[Solution]:
        """
        Find the route with fewest customers, remove all its customers,
        and greedily reinsert them into other routes.
        Returns improved solution if successful, None otherwise.
        """
        # find smallest route
        min_custs = float("inf")
        min_sat   = None
        min_ridx  = None
        for sat, routes in sol.ev_routes.items():
            for ridx, r in enumerate(routes):
                n = sum(1 for n in r
                        if self.data["nodes"][n]["Type"] == "c")
                if 0 < n < min_custs:
                    min_custs = n
                    min_sat   = sat
                    min_ridx  = ridx

        if min_sat is None:
            return None

        # remove the route
        removed = [
            n for n in sol.ev_routes[min_sat][min_ridx]
            if self.data["nodes"][n]["Type"] == "c"
        ]
        new_ev = copy.deepcopy(sol.ev_routes)
        new_ev[min_sat].pop(min_ridx)
        if not new_ev[min_sat]:
            del new_ev[min_sat]
        partial = Solution(lv_routes=sol.lv_routes, ev_routes=new_ev)

        # try to reinsert all removed customers
        for cust in removed:
            best_cost = float("inf")
            best_sat  = best_ridx = best_pos = None
            for sat, routes in partial.ev_routes.items():
                for r_idx, r in enumerate(routes):
                    for pos in range(1, len(r)):
                        candidate = r[:pos] + [cust] + r[pos:]
                        load = sum(
                            self.data["nodes"][n]["DeliveryDemand"]
                            for n in candidate
                            if self.data["nodes"][n]["Type"] == "c"
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
                            best_sat  = sat
                            best_ridx = r_idx
                            best_pos  = pos

            if best_sat is None:
                return None  # couldn't reinsert — elimination failed
            r = partial.ev_routes[best_sat][best_ridx]
            partial.ev_routes[best_sat][best_ridx] = (
                r[:best_pos] + [cust] + r[best_pos:]
            )

        return partial


    def _route_feasible_check(self, route: List[str], sat: str) -> bool:
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

    # ============================================================
    #  Cost helper
    # ============================================================

    def _cost(self, sol: Solution) -> float:
        return self.ev.evaluate(sol)["cost_with_M"]