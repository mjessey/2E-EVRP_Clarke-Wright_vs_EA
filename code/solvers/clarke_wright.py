# ---------------------------------------------------------------
#  Clarke-Wright Savings Heuristic for 2E-EVRP
#
#  Strategy
#  --------
#  Both echelons are constructed independently using the same
#  savings principle:
#
#  SECOND ECHELON (EVs, per satellite)
#    1. Start with one "spoke" route per customer:
#           sat → customer_i → sat
#    2. Compute savings:
#           s(i,j) = d(sat,i) + d(sat,j) - d(i,j)
#       A merge of  …i → sat  and  sat → j…  into  …i → j…
#       saves that much distance.
#    3. Sort savings descending; greedily merge whenever:
#           (a) neither customer is an interior node,
#           (b) the two routes are different,
#           (c) merged load ≤ EV capacity C,
#           (d) battery feasibility holds (with optional charging
#               stop insertion).
#
#  FIRST ECHELON (LVs)
#    Same savings logic over satellites, with load = total EV
#    demand routed through each satellite, capacity = L.
#    (Simple version: one LV per depot–sat–depot spoke, then
#     merge satellites onto shared LV tours.)
#
#  Charging-stop insertion
#    After every EV merge, if the resulting route violates battery
#    capacity, the algorithm attempts to insert the nearest charging
#    station into the cheapest feasible position.  If no position
#    works the merge is rejected.
# ---------------------------------------------------------------

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Any, List, Optional, Tuple

from core.evaluator import Evaluator, Solution


# ================================================================
#  Public entry class
# ================================================================
class ClarkeWright:
    """
    solve(instance_dict) ->
        { 'solution' : Solution | None,
          'evs'      : int,
          'distance' : float }
    """

    def solve(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        cw = _CWSolver(instance)
        sol = cw.build()

        if sol is None:
            return {"solution": None, "evs": 0, "distance": float("inf")}

        ev = Evaluator(instance, check_sat_inventory=False)
        res = ev.evaluate(sol)

        return {
            "solution": sol,
            "evs":      ev._count_evs(sol),
            "distance": res["cost"],
        }


# ================================================================
#  Internal solver
# ================================================================
class _CWSolver:

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data      = data
        self.params    = data["params"]
        self.EV_CAP    = self.params["C"]
        self.LV_CAP    = self.params["L"]
        self.BAT_CAP   = self.params["Q"]
        self.SPEED     = self.params["v"]
        self.INV_G     = self.params["g"]

        self.depots     = data["depots"]
        self.satellites = data["satellites"]
        self.stations   = data["stations"]
        self.customers  = data["customers"]

        # Euclidean distance cache
        coords = {n: (data["nodes"][n]["x"], data["nodes"][n]["y"])
                  for n in data["nodes"]}
        self.dist: Dict[Tuple[str, str], float] = {
            (i, j): math.hypot(coords[i][0] - coords[j][0],
                               coords[i][1] - coords[j][1])
            for i in coords for j in coords
        }

    # ----------------------------------------------------------------
    def build(self) -> Optional[Solution]:
        ev_routes: Dict[str, List[List[str]]] = {}

        # ---- assign customers to their nearest satellite ------------
        sat_customers = self._assign_customers_to_satellites()

        # ---- second echelon: EV routes per satellite ---------------
        for sat in self.satellites:
            custs = sat_customers.get(sat, [])
            if not custs:
                continue
            routes = self._clarke_wright_ev(sat, custs)
            if routes:
                ev_routes[sat] = routes

        # ---- first echelon: LV routes ------------------------------
        # demand each satellite needs delivered
        sat_demand = {
            sat: sum(
                self.data["nodes"][c]["DeliveryDemand"]
                for route in ev_routes.get(sat, [])
                for c in route
                if self.data["nodes"][c]["Type"] == "c"
            )
            for sat in self.satellites
        }
        lv_routes = self._clarke_wright_lv(sat_demand)

        return Solution(lv_routes=lv_routes, ev_routes=ev_routes)

    # ==============================================================
    #  SECOND ECHELON  –  EV routes
    # ==============================================================
    def _clarke_wright_ev(
        self, sat: str, customers: List[str]
    ) -> List[List[str]]:
        """
        Returns a list of EV routes, each starting and ending at sat.
        """
        # -- initialise: one spoke per customer ---------------------
        # Each route is stored as a plain list  [sat, ..., sat]
        routes: List[List[str]] = [[sat, c, sat] for c in customers]

        # endpoint index: customer -> which route it's in, and
        # whether it's at the HEAD (index 1) or TAIL (index -2).
        # We only allow merging at the free endpoints.

        def route_load(r: List[str]) -> float:
            return sum(
                self.data["nodes"][n]["DeliveryDemand"]
                for n in r
                if self.data["nodes"][n]["Type"] == "c"
            )

        # -- compute all savings ------------------------------------
        savings = []
        for ci, cj in combinations(customers, 2):
            s = (self.dist[sat, ci] + self.dist[sat, cj]
                 - self.dist[ci, cj])
            savings.append((s, ci, cj))
        savings.sort(reverse=True)

        # -- greedy merge -------------------------------------------
        for _, ci, cj in savings:
            # find which routes contain ci and cj
            ri = self._find_route(routes, ci)
            rj = self._find_route(routes, cj)

            if ri is None or rj is None:
                continue
            if ri is rj:
                continue     # already in the same route

            # ci must be a *tail* endpoint, cj a *head* endpoint
            # Try both orientations
            merged = None
            for r_tail, c_tail, r_head, c_head in [
                (ri, ci, rj, cj),
                (rj, cj, ri, ci),
            ]:
                if (r_tail[-2] == c_tail          # c_tail is last customer
                        and r_head[1] == c_head):  # c_head is first customer
                    candidate = r_tail[:-1] + r_head[1:]
                    load = route_load(candidate)
                    if load > self.EV_CAP:
                        continue
                    candidate = self._ensure_battery(sat, candidate)
                    if candidate is not None:
                        merged = (candidate, r_tail, r_head)
                        break

            if merged is None:
                continue

            new_route, old_tail, old_head = merged
            routes.remove(old_tail)
            routes.remove(old_head)
            routes.append(new_route)

        # -- final battery pass (spokes may still be infeasible) ----
        final = []
        for r in routes:
            fixed = self._ensure_battery(sat, r)
            if fixed is not None:
                final.append(fixed)
            else:
                # fall back: split into individual spokes
                for c in r[1:-1]:
                    if self.data["nodes"][c]["Type"] == "c":
                        final.append([sat, c, sat])
        return final

    # ==============================================================
    #  FIRST ECHELON  –  LV routes
    # ==============================================================
    def _clarke_wright_lv(
        self, sat_demand: Dict[str, float]
    ) -> List[List[str]]:
        """
        Returns LV routes; each route is  [depot, sat1, sat2, …, depot].
        """
        depot = self.depots[0]

        # filter only satellites that actually have demand
        active_sats = [s for s in self.satellites if sat_demand.get(s, 0) > 0]

        if not active_sats:
            return []

        # initialise: one spoke per satellite
        routes: List[List[str]] = [[depot, s, depot] for s in active_sats]

        def route_demand(r: List[str]) -> float:
            return sum(sat_demand.get(n, 0) for n in r)

        # compute savings
        savings = []
        for si, sj in combinations(active_sats, 2):
            s = (self.dist[depot, si] + self.dist[depot, sj]
                 - self.dist[si, sj])
            savings.append((s, si, sj))
        savings.sort(reverse=True)

        # greedy merge
        for _, si, sj in savings:
            ri = self._find_route(routes, si)
            rj = self._find_route(routes, sj)

            if ri is None or rj is None:
                continue
            if ri is rj:
                continue

            for r_tail, s_tail, r_head, s_head in [
                (ri, si, rj, sj),
                (rj, sj, ri, si),
            ]:
                if r_tail[-2] == s_tail and r_head[1] == s_head:
                    candidate = r_tail[:-1] + r_head[1:]
                    if route_demand(candidate) <= self.LV_CAP:
                        routes.remove(r_tail)
                        routes.remove(r_head)
                        routes.append(candidate)
                        break

        return routes

    # ==============================================================
    #  Battery helpers
    # ==============================================================
    def _simulate_battery(self, route: List[str]) -> bool:
        """Return True if the route is battery-feasible as-is."""
        soc = self.BAT_CAP
        for a, b in zip(route, route[1:]):
            soc -= self.dist[a, b]
            if soc < -1e-9:
                return False
            if self.data["nodes"][b]["Type"] == "f":
                soc = self.BAT_CAP
        return True

    def _ensure_battery(
        self, sat: str, route: List[str]
    ) -> Optional[List[str]]:
        """
        If route is already battery-feasible, return it unchanged.
        Otherwise try inserting the cheapest charging station into
        every position until feasibility is achieved (greedy, one
        station at a time, up to len(route) attempts).

        Returns the (possibly augmented) route, or None if no
        insertion sequence achieves feasibility.
        """
        if not self.stations:
            return route if self._simulate_battery(route) else None

        r = list(route)
        for _ in range(len(route)):          # at most this many stations needed
            if self._simulate_battery(r):
                return r
            # find the best (station, position) to insert
            best_cost = math.inf
            best_r    = None
            for f in self.stations:
                for pos in range(1, len(r)):
                    # don't insert consecutively
                    if r[pos - 1] == f or r[pos] == f:
                        continue
                    new_r = r[:pos] + [f] + r[pos:]
                    if self._simulate_battery_partial(new_r, pos):
                        extra = (self.dist[r[pos-1], f]
                                 + self.dist[f, r[pos]]
                                 - self.dist[r[pos-1], r[pos]])
                        if extra < best_cost:
                            best_cost = extra
                            best_r    = new_r
            if best_r is None:
                return None
            r = best_r
        return r if self._simulate_battery(r) else None

    def _simulate_battery_partial(
        self, route: List[str], up_to: int
    ) -> bool:
        """
        Quick check: does SoC stay ≥ 0 from the start up to index up_to?
        Used during insertion to prune obviously bad candidates early.
        """
        soc = self.BAT_CAP
        for a, b in zip(route[:up_to], route[1:up_to+1]):
            soc -= self.dist[a, b]
            if soc < -1e-9:
                return False
            if self.data["nodes"][b]["Type"] == "f":
                soc = self.BAT_CAP
        return True

    # ==============================================================
    #  Utilities
    # ==============================================================
    def _assign_customers_to_satellites(self) -> Dict[str, List[str]]:
        """Assign each customer to its geographically nearest satellite."""
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
                key=lambda s: math.hypot(cx - sat_coords[s][0],
                                         cy - sat_coords[s][1]),
            )
            assignment[best].append(c)
        return assignment

    @staticmethod
    def _find_route(
        routes: List[List[str]], node: str
    ) -> Optional[List[str]]:
        """Return the route list that contains node (by identity search)."""
        for r in routes:
            if node in r:
                return r
        return None