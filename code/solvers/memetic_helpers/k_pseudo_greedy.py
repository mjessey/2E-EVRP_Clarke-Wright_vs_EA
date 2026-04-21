# ---------------------------------------------------------------
#  Stage 1: k-Pseudo Greedy Initial Population Generator
#
#  Based on Felipe et al. (2014) as described in:
#  "A Memetic Algorithm for the Green Vehicle Routing Problem"
#  Peng et al. (2019)
#
#  How it works
#  ------------
#  Builds one solution by repeatedly extending routes until all
#  customers are served:
#
#  1. Start a new route from a satellite
#  2. From the current node, find the k closest unvisited
#     customers that are reachable (battery + time window)
#  3. Pick one at random from those k candidates
#  4. Check if we can return to satellite after visiting them
#     (directly, or via a charging stop)
#  5. If yes, add them to the route and continue
#  6. If no feasible customer found, close the current route
#     and start a new one
#  7. Repeat until all customers are served
#
#  k=1 reduces to nearest-neighbour (deterministic, one solution)
#  k>1 introduces randomness, producing different solutions
#     each run — this is what gives the population diversity
#
#  Adapted for 2E-EVRP
#  --------------------
#  The paper solves single-echelon GVRP (depot -> customers).
#  We adapt it to the second echelon of 2E-EVRP:
#     - Routes start and end at satellites, not the depot
#     - Customers are pre-assigned to their nearest satellite
#       (same as Clarke-Wright)
#     - LV routes are built with Clarke-Wright after EV routes
#       are fixed
# ---------------------------------------------------------------

from __future__ import annotations

import math
import random
import copy
from typing import Dict, Any, List, Optional, Tuple

from core.evaluator import Evaluator, Solution
from solvers.clarke_wright import ClarkeWright


class KPseudoGreedy:
    """
    Generates a population of feasible solutions using the
    k-Pseudo Greedy construction heuristic.

    Parameters
    ----------
    instance : dict
        Parsed 2E-EVRP instance
    k : int
        Number of candidate customers to consider at each step.
        k=1 -> nearest neighbour (deterministic)
        k=3 -> default from paper parameter tuning
    """

    def __init__(self, instance: Dict[str, Any], k: int = 3) -> None:
        self.data    = instance
        self.k       = k
        self.params  = instance["params"]
        self.EV_CAP  = self.params["C"]
        self.BAT_CAP = self.params["Q"]
        self.SPEED   = self.params["v"]
        self.INV_G   = self.params["g"]

        self.satellites = instance["satellites"]
        self.stations   = instance["stations"]
        self.customers  = instance["customers"]

        # distance cache
        coords = {n: (instance["nodes"][n]["x"], instance["nodes"][n]["y"])
                  for n in instance["nodes"]}
        self.dist: Dict[Tuple[str, str], float] = {
            (i, j): math.hypot(coords[i][0] - coords[j][0],
                               coords[i][1] - coords[j][1])
            for i in coords for j in coords
        }

        # assign customers to nearest satellite
        self.sat_customers = self._assign_customers()

    # ----------------------------------------------------------------
    def generate(self) -> Optional[Solution]:
        """
        Build and return one feasible solution using k-Pseudo Greedy.
        Returns None if construction fails (should be rare).
        """
        ev_routes: Dict[str, List[List[str]]] = {}

        for sat in self.satellites:
            custs = list(self.sat_customers.get(sat, []))
            if not custs:
                continue
            routes = self._build_ev_routes(sat, custs)
            if routes:
                ev_routes[sat] = routes

        # build LV routes via Clarke-Wright on the resulting demand
        lv_routes = self._build_lv_routes(ev_routes)

        sol = Solution(lv_routes=lv_routes, ev_routes=ev_routes)
        return sol

    def generate_population(self, size: int) -> List[Solution]:
        """
        Generate a population of `size` solutions.
        Uses k=1 for the first solution (best deterministic seed),
        then k>1 for the rest to introduce diversity.
        """
        population = []

        # first solution: nearest-neighbour (k=1) as a quality anchor
        original_k  = self.k
        self.k      = 1
        sol         = self.generate()
        if sol is not None:
            population.append(sol)
        self.k = original_k

        # remaining solutions: random k-greedy
        attempts = 0
        max_attempts = size * 10   # avoid infinite loop on hard instances
        while len(population) < size and attempts < max_attempts:
            sol = self.generate()
            if sol is not None:
                population.append(sol)
            attempts += 1

        print(f"  Generated {len(population)}/{size} initial solutions "
              f"({attempts} attempts)")
        return population

    # ================================================================
    #  EV route construction
    # ================================================================

    def _build_ev_routes(
        self, sat: str, customers: List[str]
    ) -> List[List[str]]:
        """
        Build EV routes for one satellite using k-Pseudo Greedy.
        """
        unvisited = set(customers)
        routes    = []

        while unvisited:
            route   = [sat]
            load    = 0.0
            soc     = self.BAT_CAP
            time    = 0.0
            current = sat

            while True:
                # find k closest reachable unvisited customers
                candidates = self._find_candidates(
                    current, unvisited, route, load, soc, time, sat
                )

                if not candidates:
                    # no feasible customer reachable — close route
                    break

                # pick one at random from the k candidates
                chosen = random.choice(candidates)

                # travel to chosen customer
                d      = self.dist[current, chosen]
                time  += d / self.SPEED
                soc   -= d

                # apply waiting if we arrive before ready time
                node   = self.data["nodes"][chosen]
                if time < node["ReadyTime"]:
                    time = node["ReadyTime"]

                time += node["ServiceTime"]
                load += node["DeliveryDemand"]

                route.append(chosen)
                unvisited.remove(chosen)
                current = chosen

                # check if we should insert a charging stop on the
                # way back to satellite
                if not self._can_reach_sat(current, soc, sat):
                    f = self._best_charging_stop(current, soc, sat)
                    if f is None:
                        # stranded — close route here
                        break
                    # insert charging stop
                    d_to_f  = self.dist[current, f]
                    soc    -= d_to_f
                    time   += d_to_f / self.SPEED
                    time   += (self.BAT_CAP - soc) * self.INV_G
                    soc     = self.BAT_CAP
                    route.append(f)
                    current = f

            # close route back to satellite
            if len(route) > 1:
                # insert charging stop before return if needed
                if not self._can_reach_sat(current, soc, sat):
                    f = self._best_charging_stop(current, soc, sat)
                    if f is not None:
                        route.append(f)
                route.append(sat)
                # only keep if route has at least one customer
                if any(self.data["nodes"][n]["Type"] == "c" for n in route):
                    routes.append(route)

            # safety: if no progress was made, remove a random
            # unvisited customer to avoid infinite loop
            if len(route) == 2 and route[0] == route[1] == sat:
                if unvisited:
                    unvisited.pop()

        return routes

    # ----------------------------------------------------------------
    def _find_candidates(
        self,
        current:   str,
        unvisited: set,
        route:     List[str],
        load:      float,
        soc:       float,
        time:      float,
        sat:       str,
    ) -> List[str]:
        """
        Return up to k unvisited customers that are:
          (a) reachable from current with remaining battery
          (b) within capacity
          (c) time window not already violated
          (d) can still return to satellite after visiting them
              (directly or via a charging stop)
        Sorted by distance ascending, return k closest.
        """
        reachable = []

        for c in unvisited:
            node = self.data["nodes"][c]

            # capacity check
            if load + node["DeliveryDemand"] > self.EV_CAP:
                continue

            d_to_c = self.dist[current, c]

            # battery to reach c
            if soc - d_to_c < 0:
                continue

            # time to reach c
            arr = time + d_to_c / self.SPEED
            if arr > node["DueDate"]:
                continue

            # after visiting c, can we return to sat?
            soc_after  = soc - d_to_c
            time_after = max(arr, node["ReadyTime"]) + node["ServiceTime"]

            if not self._can_reach_sat_after(c, soc_after, time_after, sat):
                continue

            reachable.append((d_to_c, c))

        reachable.sort()
        return [c for _, c in reachable[:self.k]]

    # ----------------------------------------------------------------
    def _can_reach_sat(self, node: str, soc: float, sat: str) -> bool:
        """Can we reach sat directly from node with current soc?"""
        return soc >= self.dist[node, sat]

    def _can_reach_sat_after(
        self, node: str, soc: float, time: float, sat: str
    ) -> bool:
        """
        After visiting node with given soc and time, can we return
        to sat — either directly or via one charging stop?
        """
        # direct return
        if self._can_reach_sat(node, soc, sat):
            return True

        # via a charging stop
        for f in self.stations:
            d_to_f = self.dist[node, f]
            d_f_sat = self.dist[f, sat]
            if soc >= d_to_f and self.BAT_CAP >= d_f_sat:
                return True

        return False

    def _best_charging_stop(
        self, current: str, soc: float, sat: str
    ) -> Optional[str]:
        """
        Find the charging station that minimises detour cost while
        allowing us to reach sat afterwards.
        """
        best_cost = math.inf
        best_f    = None

        for f in self.stations:
            d_to_f  = self.dist[current, f]
            d_f_sat = self.dist[f, sat]

            if soc < d_to_f:
                continue
            if self.BAT_CAP < d_f_sat:
                continue

            detour = d_to_f + d_f_sat - self.dist[current, sat]
            if detour < best_cost:
                best_cost = detour
                best_f    = f

        return best_f

    # ================================================================
    #  LV route construction (Clarke-Wright)
    # ================================================================

    def _build_lv_routes(
        self, ev_routes: Dict[str, List[List[str]]]
    ) -> List[List[str]]:
        """
        Build LV routes using Clarke-Wright savings on the actual
        demand per satellite.
        """
        depot = self.data["depots"][0]

        sat_demand = {
            sat: sum(
                self.data["nodes"][c]["DeliveryDemand"]
                for route in routes
                for c in route
                if self.data["nodes"][c]["Type"] == "c"
            )
            for sat, routes in ev_routes.items()
        }

        active = [s for s in sat_demand if sat_demand[s] > 0]
        if not active:
            return []

        lv_routes = [[depot, s, depot] for s in active]

        from itertools import combinations
        savings = sorted([
            (self.dist[depot, si] + self.dist[depot, sj]
             - self.dist[si, sj], si, sj)
            for si, sj in combinations(active, 2)
        ], reverse=True)

        for _, si, sj in savings:
            ri = next((r for r in lv_routes if si in r), None)
            rj = next((r for r in lv_routes if sj in r), None)
            if ri is None or rj is None or ri is rj:
                continue
            if ri[-2] == si and rj[1] == sj:
                candidate = ri[:-1] + rj[1:]
                if sum(sat_demand.get(n, 0) for n in candidate) <= self.data["params"]["L"]:
                    lv_routes.remove(ri)
                    lv_routes.remove(rj)
                    lv_routes.append(candidate)

        return lv_routes

    # ================================================================
    #  Utilities
    # ================================================================

    def _assign_customers(self) -> Dict[str, List[str]]:
        """Assign each customer to its nearest satellite."""
        sat_coords = {
            s: (self.data["nodes"][s]["x"], self.data["nodes"][s]["y"])
            for s in self.satellites
        }
        assignment: Dict[str, List[str]] = {s: [] for s in self.satellites}
        for c in self.customers:
            cx = self.data["nodes"][c]["x"]
            cy = self.data["nodes"][c]["y"]
            best = min(
                sat_coords,
                key=lambda s: math.hypot(
                    cx - sat_coords[s][0],
                    cy - sat_coords[s][1]
                )
            )
            assignment[best].append(c)
        return assignment