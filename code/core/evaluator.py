# --------------------------------------------------------------------
#  Common cost / feasibility engine for Two-Echelon EVRP instances.
#  Designed to be solver-agnostic: every algorithm feeds a Solution
#  object to `Evaluator.evaluate()` and receives a score plus a list
#  of violated constraints (if any).
# --------------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


# --------------------------------------------------------------------
#  1)  SIMPLE SOLUTION CONTAINER
# --------------------------------------------------------------------
@dataclass
class Solution:
    """
    Minimalistic representation of a 2E-EVRP solution.

    lv_routes : list of LV tours (each is a list of node-IDs)
    ev_routes : dict  satellite-id -> list of EV tours
                 every EV tour again is a list of node-IDs
                 (must start/end with that satellite)
    """
    lv_routes: List[List[str]] = field(default_factory=list)
    ev_routes: Dict[str, List[List[str]]] = field(default_factory=dict)

    # helper ─────────────────────────────────────────────────────────
    def all_routes(self) -> List[List[str]]:
        """Iterator over *all* individual routes (both echelons)."""
        for r in self.lv_routes:
            yield r
        for rlist in self.ev_routes.values():
            for r in rlist:
                yield r


# --------------------------------------------------------------------
#  2)  EVALUATOR
# --------------------------------------------------------------------
class Evaluator:
    """
    Deterministic cost / feasibility engine for the 2-Echelon EVRP.
    It enforces
        • LV capacity
        • EV capacity
        • EV battery capacity + recharging at 'f' nodes
        • customer time windows
        • (optionally) satellite inventory balance   –  *see note below*
    and returns the total travel distance as objective value.

    NOTE
    ----
    Stock synchronisation at satellites is included in outline form
    but can be de-activated via constructor flag if it is not needed
    for an early prototype.
    """

    # ───────────────────────────────────────────────────────────────
    def __init__(self, instance: Dict[str, Any], *,
                 check_sat_inventory: bool = True) -> None:
        self.data = instance
        self.check_sat_inventory = check_sat_inventory

        p = instance["params"]
        self.LV_CAP = p["L"]
        self.EV_CAP = p["C"]
        self.BAT_CAP = p["Q"]          #   energy / kilometre is r == 1
        self.INV_RECHARGE = p["g"]     #   time per energy unit
        self.SPEED = p["v"]            #   km  per time unit

        # pre-compute distance / time / energy matrices ──────────────
        self.dist: Dict[str, Dict[str, float]] = {}
        nodes = instance["nodes"]
        for i, ni in nodes.items():
            self.dist[i] = {}
            xi, yi = ni["x"], ni["y"]
            for j, nj in nodes.items():
                self.dist[i][j] = math.hypot(xi - nj["x"], yi - nj["y"])
        # travel time = distance / v;   energy   = distance (because r = 1)

    # ───────────────────────────────────────────────────────────────
    def evaluate(self, sol: Solution) -> Dict[str, Any]:
        """
        Returns
        -------
        dict( cost = float,
              feasible = bool,
              violations = {cap, battery, time, stock}  )
        """

        v = dict(cap=0, battery=0, time=0, stock=0)
        cost = 0.0

        # -----------------------------------------------------------
        #  FIRST  ECHELON  (LVs)
        # -----------------------------------------------------------
        # we also build a list  (satellite, arrival time, qty delivered)
        delivery_events: Dict[str, List[Tuple[float, float]]] = {}

        for route in sol.lv_routes:
            if not self._check_endpoints(route, want_type="d"):
                raise ValueError("LV route must start & end at depot.")

            load_on_truck = 0.0
            t = 0.0
            prev = route[0]

            for nxt in route[1:]:
                d = self.dist[prev][nxt]
                cost += d
                t += d / self.SPEED

                # if we arrive at satellite we unload *everything that
                # this LV still carries* (simplest policy)
                if self.data["nodes"][nxt]["Type"] == "s":
                    if load_on_truck > 0:
                        delivery_events.setdefault(nxt, []).append((t, load_on_truck))
                        load_on_truck = 0.0
                prev = nxt

            # capacity check
            if load_on_truck > self.LV_CAP:
                v["cap"] += 1

        # -----------------------------------------------------------
        #  SECOND  ECHELON  (EVs)
        # -----------------------------------------------------------
        # we need pickup events for satellite stock test
        pickup_events: Dict[str, List[Tuple[float, float]]] = {}

        for sat, routes in sol.ev_routes.items():
            if self.data["nodes"][sat]["Type"] != "s":
                raise ValueError(f"Key '{sat}' in ev_routes is not a satellite.")

            for route in routes:
                if route[0] != sat or route[-1] != sat:
                    raise ValueError("EV route must start & end at its satellite.")

                load = 0.0
                soc = self.BAT_CAP
                t = 0.0
                prev = sat

                # departure   → register pickup *at departure time 0*
                qty = self._route_delivery_demand(route)
                pickup_events.setdefault(sat, []).append((0.0, -qty))

                for nxt in route[1:]:
                    d = self.dist[prev][nxt]
                    e = d                        # r = 1
                    tt = d / self.SPEED

                    t += tt
                    soc -= e
                    if soc < 0:
                        v["battery"] += 1

                    ntype = self.data["nodes"][nxt]["Type"]

                    if ntype == "f":            # charging station
                        t += (self.BAT_CAP - soc) * self.INV_RECHARGE
                        soc = self.BAT_CAP

                    elif ntype == "c":
                        node = self.data["nodes"][nxt]

                        # time window
                        if t < node["ReadyTime"]:
                            t = node["ReadyTime"]
                        if t > node["DueDate"]:
                            v["time"] += 1
                        t += node["ServiceTime"]

                        # load
                        load += node["DeliveryDemand"]    # only deliveries here
                        if load > self.EV_CAP:
                            v["cap"] += 1

                    prev = nxt
                    cost += d

        # -----------------------------------------------------------
        #  SATELLITE  STOCK  CONSISTENCY
        # -----------------------------------------------------------
        if self.check_sat_inventory:
            for s in self.data["satellites"]:
                events = delivery_events.get(s, []) + pickup_events.get(s, [])
                if not events:
                    continue
                events.sort(key=lambda x: x[0])   # by time
                stock = 0.0
                for _, delta in events:
                    stock += delta
                    if stock < -1e-6:
                        v["stock"] += 1
                        break

        feasible = not any(v.values())
        return dict(cost=cost, feasible=feasible, violations=v)

    # ───────────────────────────────────────────────────────────────
    #  helper
    # ───────────────────────────────────────────────────────────────
    def _route_delivery_demand(self, route: List[str]) -> float:
        """Sum of DeliveryDemand for all customers in the given route."""
        total = 0.0
        for nid in route:
            if self.data["nodes"][nid]["Type"] == "c":
                total += self.data["nodes"][nid]["DeliveryDemand"]
        return total

    @staticmethod
    def _check_endpoints(route: List[str], *, want_type: str) -> bool:
        """Validate that the first *and* last node have the desired type."""
        return (
            len(route) >= 2
            and route[0] == route[-1]
            and route[0][0].lower() == want_type
        )
