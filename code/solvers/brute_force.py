# ------------------------------------------------------------------
#  Brute-force search that also decides whether to insert *one*
#  charging station (or none) in every gap between two visits.
#
#  Search space per satellite
#     =  (customer permutations)  ×  (station/no-station)^(gaps)
# ------------------------------------------------------------------

from __future__ import annotations
from collections import defaultdict
from itertools import permutations, product
from math import hypot
from typing import Dict, Any, List, Tuple

from core.evaluator import Evaluator, Solution


class BruteForce:
    """Exhaustive enumeration for tiny 2E-EVRP instances.

       • Customers are assigned to their nearest satellite (fixed).
       • One LV      :  D → S → D    for every satellite
       • One EV      :  S → (…) → S  for every satellite
         where “(…)” is
               C₁, (opt F), C₂, (opt F), …, C_k, (opt F)
         so that each gap may or may not contain a single
         charging station.

       Works only for very small instances (≤ 6-7 customers).
    """

    # ------------------------------------------------------------------
    def solve(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        evaluator = Evaluator(instance, check_sat_inventory=False, check_time_windows=False)

        depot = self._single(instance["depots"], "depot")
        stations = instance["stations"]
        satellites = instance["satellites"]

        # ---------- 1. nearest-satellite assignment --------------------
        sat_of_cust = self._assign_to_nearest_sat(instance)

        customers_by_sat = defaultdict(list)
        for c, s in sat_of_cust.items():
            customers_by_sat[s].append(c)

        # ---------- 2. build *all* EV routes per satellite -------------
        routes_by_sat: Dict[str, List[List[str]]] = {}

        none_plus_stations: Tuple[str | None, ...] = (None, *stations)

        for s in satellites:
            clist = customers_by_sat[s]
            if not clist:                        # satellite without cust
                routes_by_sat[s] = [[s, s]]      # dummy round-trip
                continue

            gap_count = len(clist) + 1
            routes_here: List[List[str]] = []

            for perm in permutations(clist):
                # iterate over all (station/none)^gap_count combinations
                for choices in product(none_plus_stations, repeat=gap_count):
                    seq = [s]                                   # start
                    # interleave  (choice, customer) … (choice) pattern
                    for idx, cust in enumerate(perm):
                        if choices[idx]:
                            seq.append(choices[idx])
                        seq.append(cust)
                    # gap after last customer
                    if choices[-1]:
                        seq.append(choices[-1])
                    seq.append(s)                               # end
                    routes_here.append(seq)

            routes_by_sat[s] = routes_here

        # ---------- 3. cartesian product over satellites --------------
        sat_ids = list(satellites)
        iterator = product(*(routes_by_sat[s] for s in sat_ids))

        best_cost = float("inf")
        best_sol: Solution | None = None
        evaluated = 0

        for combo in iterator:
            evaluated += 1
            lv_routes = [[depot, s, depot] for s in sat_ids]
            ev_routes = {s: [combo[idx]] for idx, s in enumerate(sat_ids)}
            sol = Solution(lv_routes=lv_routes, ev_routes=ev_routes)

            res = evaluator.evaluate(sol)
            if res["feasible"] and res["cost"] < best_cost:
                best_cost = res["cost"]
                best_sol = sol

        return dict(
            best_cost=best_cost,
            best_solution=best_sol,
            evaluated=evaluated,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _single(lst, what):
        if not lst:
            raise ValueError(f"No {what} in instance.")
        if len(lst) > 1:
            print(f"Warning: multiple {what}s – using {lst[0]}")
        return lst[0]

    @staticmethod
    def _assign_to_nearest_sat(inst) -> Dict[str, str]:
        """Return mapping  customer-id → nearest satellite-id."""
        sat_coords = {
            s: (inst["nodes"][s]["x"], inst["nodes"][s]["y"])
            for s in inst["satellites"]
        }
        mapping = {}
        for cid in inst["customers"]:
            cx, cy = inst["nodes"][cid]["x"], inst["nodes"][cid]["y"]
            best_s = min(
                sat_coords,
                key=lambda s: hypot(cx - sat_coords[s][0], cy - sat_coords[s][1]),
            )
            mapping[cid] = best_s
        return mapping
