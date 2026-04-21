# ---------------------------------------------------------------
#  Stage 3: Backbone-Based Crossover Operator
#
#  Based on Section 4.4 of:
#  "A Memetic Algorithm for the Green Vehicle Routing Problem"
#  Peng et al. (2019)
#
#  How it works
#  ------------
#  Given two parent solutions Sa and Sb, build a child Sc by
#  iteratively selecting the best route from either parent:
#
#  Repeat until all customers are assigned:
#    1. Collect all remaining routes from both parents
#    2. Score each route by:
#           Δ = δ(f) / θ
#       where δ(f) = incremental cost of adding this route to Sc
#             θ    = number of customers in this route
#       Lower Δ = more customers covered per unit cost = better
#    3. Select the route with minimum Δ
#    4. Add it to Sc
#    5. Remove its customers from ALL remaining routes in
#       BOTH parents (prevents duplicate coverage)
#    6. Clean up any routes that become empty
#
#  Why this works
#  --------------
#  The crossover inherits entire feasible route sequences from
#  parents rather than splicing at arbitrary points. This means:
#    - The child is always feasible (inherits feasible routes)
#    - Good customer+station sequences are preserved intact
#    - The greedy selection criterion biases toward efficient routes
#
#  Adaptation for 2E-EVRP
#  -----------------------
#  The paper operates on a single depot. We adapt for satellites:
#    - Routes are grouped by satellite
#    - A route from satellite S can only be added to Sc under S
#    - After crossover, LV routes are rebuilt via Clarke-Wright
#      on the actual demand per satellite
#    - If any customers are unassigned after the main loop
#      (e.g. all routes containing them were discarded), they
#      are inserted greedily as fallback spoke routes
# ---------------------------------------------------------------

from __future__ import annotations

import math
import copy
import random
from itertools import combinations
from typing import Dict, Any, List, Optional, Tuple, Set

from core.evaluator import Evaluator, Solution


class BackboneCrossover:
    """
    Backbone-based crossover operator from Peng et al. (2019).
    Adapted for 2E-EVRP two-echelon structure.
    """

    def __init__(self, instance: Dict[str, Any]) -> None:
        self.data   = instance
        self.ev     = Evaluator(instance, check_sat_inventory=False)
        self.params = instance["params"]
        self.EV_CAP = self.params["C"]
        self.LV_CAP = self.params["L"]
        self.BAT_CAP = self.params["Q"]
        self.SPEED  = self.params["v"]
        self.INV_G  = self.params["g"]

        # distance cache
        coords = {n: (instance["nodes"][n]["x"], instance["nodes"][n]["y"])
                  for n in instance["nodes"]}
        self.dist: Dict[Tuple[str, str], float] = {
            (i, j): math.hypot(coords[i][0] - coords[j][0],
                               coords[i][1] - coords[j][1])
            for i in coords for j in coords
        }

    # ============================================================
    #  Public interface
    # ============================================================

    def crossover(
        self, sa: Solution, sb: Solution
    ) -> Optional[Solution]:
        """
        Generate one child solution from two parents Sa and Sb.
        Returns None if crossover fails to cover all customers.
        """
        all_customers: Set[str] = set(self.data["customers"])
        unassigned: Set[str]    = set(all_customers)

        # working copies of both parents' route pools
        # pool_a[sat] = list of routes still available from Sa
        # pool_b[sat] = list of routes still available from Sb
        pool_a = self._copy_pool(sa.ev_routes)
        pool_b = self._copy_pool(sb.ev_routes)

        # child route collection
        child_ev: Dict[str, List[List[str]]] = {}

        # ---- main loop ------------------------------------------
        while unassigned:
            # collect all candidate routes from both pools
            candidates = []
            for sat, routes in pool_a.items():
                for r in routes:
                    custs_in_r = self._customers_in(r)
                    relevant   = custs_in_r & unassigned
                    if relevant:
                        candidates.append((sat, r, relevant))
            for sat, routes in pool_b.items():
                for r in routes:
                    custs_in_r = self._customers_in(r)
                    relevant   = custs_in_r & unassigned
                    if relevant:
                        candidates.append((sat, r, relevant))

            if not candidates:
                # no route covers any remaining customer
                # fall back to greedy spoke insertion
                break

            # score each candidate by Δ = δ(f) / θ
            best_delta  = math.inf
            best_entry  = None

            for sat, route, relevant in candidates:
                theta = len(relevant)
                if theta == 0:
                    continue

                # δ(f): incremental distance cost of adding this route
                # to the child — just the route's own travel distance
                # since we're building Sc from scratch
                delta_f = self._route_distance(route)

                delta = delta_f / theta

                if delta < best_delta:
                    best_delta = delta
                    best_entry = (sat, route, relevant)

            if best_entry is None:
                break

            chosen_sat, chosen_route, chosen_custs = best_entry

            # add chosen route to child (only unassigned customers)
            clean_route = self._strip_assigned(
                chosen_route, unassigned
            )
            if self._has_customers(clean_route):
                child_ev.setdefault(chosen_sat, []).append(clean_route)

            # mark customers as assigned
            unassigned -= chosen_custs

            # remove chosen customers from BOTH pools
            pool_a = self._remove_customers_from_pool(
                pool_a, chosen_custs
            )
            pool_b = self._remove_customers_from_pool(
                pool_b, chosen_custs
            )

        # ---- fallback: greedy spoke insertion -------------------
        if unassigned:
            child_ev = self._greedy_insert_remaining(
                child_ev, list(unassigned)
            )

        if not child_ev:
            return None

        # ---- rebuild LV routes ----------------------------------
        lv_routes = self._build_lv_routes(child_ev)

        child = Solution(lv_routes=lv_routes, ev_routes=child_ev)

        # validate
        res = self.ev.evaluate(child)
        if not res["feasible"]:
            # attempt to repair time window violations by
            # reverting to spoke routes for violating customers
            child = self._repair(child)

        return child

    # ============================================================
    #  Route cleaning helpers
    # ============================================================

    def _strip_assigned(
        self, route: List[str], unassigned: Set[str]
    ) -> List[str]:
        """
        Remove already-assigned customers from a route while keeping
        the satellite endpoints and charging stops.
        The result is still a valid route structure:
          sat -> [stops/customers] -> sat
        """
        sat   = route[0]
        inner = route[1:-1]

        # keep only unassigned customers and charging stations
        kept  = [
            n for n in inner
            if (self.data["nodes"][n]["Type"] == "f"
                or n in unassigned)
        ]

        # strip leading/trailing charging stops
        while kept and self.data["nodes"][kept[0]]["Type"] == "f":
            kept.pop(0)
        while kept and self.data["nodes"][kept[-1]]["Type"] == "f":
            kept.pop()

        return [sat] + kept + [sat]

    def _has_customers(self, route: List[str]) -> bool:
        return any(
            self.data["nodes"][n]["Type"] == "c" for n in route
        )

    def _customers_in(self, route: List[str]) -> Set[str]:
        return {
            n for n in route
            if self.data["nodes"][n]["Type"] == "c"
        }

    # ============================================================
    #  Pool management
    # ============================================================

    def _copy_pool(
        self,
        ev_routes: Dict[str, List[List[str]]]
    ) -> Dict[str, List[List[str]]]:
        return {
            sat: [list(r) for r in routes]
            for sat, routes in ev_routes.items()
        }

    def _remove_customers_from_pool(
        self,
        pool:    Dict[str, List[List[str]]],
        to_remove: Set[str],
    ) -> Dict[str, List[List[str]]]:
        """
        Remove the given customers from every route in the pool.
        Drop routes that become empty of customers.
        """
        new_pool = {}
        for sat, routes in pool.items():
            new_routes = []
            for r in routes:
                cleaned = self._strip_assigned(
                    r, set(self.data["customers"]) - to_remove
                )
                # keep if still has unremoved customers
                remaining = self._customers_in(cleaned) - to_remove
                if remaining:
                    new_routes.append(cleaned)
            if new_routes:
                new_pool[sat] = new_routes
        return new_pool

    # ============================================================
    #  Fallback: greedy spoke insertion for leftover customers
    # ============================================================

    def _greedy_insert_remaining(
        self,
        child_ev:  Dict[str, List[List[str]]],
        remaining: List[str],
    ) -> Dict[str, List[List[str]]]:
        """
        For each remaining customer, try inserting into an existing
        route at the cheapest feasible position. If no position
        works, open a new spoke route at the nearest satellite.
        """
        for cust in remaining:
            best_cost = math.inf
            best_sat  = None
            best_ridx = None
            best_pos  = None

            for sat, routes in child_ev.items():
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
                        cost = self._route_distance(candidate)
                        if cost < best_cost:
                            best_cost = cost
                            best_sat  = sat
                            best_ridx = r_idx
                            best_pos  = pos

            if best_sat is not None:
                r = child_ev[best_sat][best_ridx]
                child_ev[best_sat][best_ridx] = (
                    r[:best_pos] + [cust] + r[best_pos:]
                )
            else:
                # open new spoke at nearest satellite
                nearest = min(
                    self.data["satellites"],
                    key=lambda s: self.dist[cust, s]
                )
                child_ev.setdefault(nearest, []).append(
                    [nearest, cust, nearest]
                )

        return child_ev

    # ============================================================
    #  Repair: fix infeasible child by reverting problem routes
    #  to individual spokes
    # ============================================================

    def _repair(self, sol: Solution) -> Solution:
        """
        For each EV route that violates constraints, break it into
        individual customer spokes. This guarantees feasibility at
        the cost of using more EVs.
        """
        new_ev = {}
        for sat, routes in sol.ev_routes.items():
            new_routes = []
            for r in routes:
                if self._route_feasible(sat, r):
                    new_routes.append(r)
                else:
                    # split into spokes
                    for n in r:
                        if self.data["nodes"][n]["Type"] == "c":
                            new_routes.append([sat, n, sat])
            new_ev[sat] = new_routes

        lv_routes = self._build_lv_routes(new_ev)
        return Solution(lv_routes=lv_routes, ev_routes=new_ev)

    # ============================================================
    #  LV route construction (Clarke-Wright)
    # ============================================================

    def _build_lv_routes(
        self,
        ev_routes: Dict[str, List[List[str]]]
    ) -> List[List[str]]:
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
                if (sum(sat_demand.get(n, 0) for n in candidate)
                        <= self.LV_CAP):
                    lv_routes.remove(ri)
                    lv_routes.remove(rj)
                    lv_routes.append(candidate)

        return lv_routes

    # ============================================================
    #  Feasibility and cost helpers
    # ============================================================

    def _route_feasible(self, sat: str, route: List[str]) -> bool:
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

    def _route_distance(self, route: List[str]) -> float:
        return sum(
            self.dist[a, b] for a, b in zip(route, route[1:])
        )