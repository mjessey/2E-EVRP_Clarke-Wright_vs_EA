# ────────────────────────────────────────────────────────────────────
#  Brute-force search for tiny instances that may have several
#  satellites, LVs and EVs.  The enumeration space is
#
#       ∏sat  ( |C_sat|! )
#
#  i.e. all permutations of the customer set of *each* satellite.
#  The mapping  Customer → Satellite  is fixed to the nearest
#  satellite (you can change that policy easily).
# ────────────────────────────────────────────────────────────────────

from __future__ import annotations
from itertools import permutations, product
from collections import defaultdict
from math import hypot, factorial
from typing import Dict, Any, List, Tuple

from evaluator import Evaluator, Solution


class BruteForceGeneral:
    """
    Exhaustively tries every possible *order* in which the customers
    assigned to a satellite can be visited by the EV that belongs to
    that satellite.

    • Customer → Satellite assignment:   nearest satellite
    • LV routes:  one per satellite     D → S → D
    • EV routes:  one per satellite     S → C* → S

    Warning:  combinatorial explosion – only use for very tiny data
    sets (≤ 8 customers per satellite is a good rule of thumb).
    """

    # ───────────────────────────────────────────────────────────────
    def solve(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        evaluator = Evaluator(instance)
        depot = self._single(instance["depots"], "depot")

        # -----------------------------------------------------------
        # 1. assign every customer to its nearest satellite
        # -----------------------------------------------------------
        sat_of_cust: Dict[str, str] = {}
        sat_coords = {s: (instance["nodes"][s]["x"], instance["nodes"][s]["y"])
                      for s in instance["satellites"]}

        for cid in instance["customers"]:
            cx, cy = instance["nodes"][cid]["x"], instance["nodes"][cid]["y"]
            best_s = min(sat_coords,
                         key=lambda s: hypot(cx - sat_coords[s][0],
                                             cy - sat_coords[s][1]))
            sat_of_cust[cid] = best_s

        customers_by_sat = defaultdict(list)
        for c, s in sat_of_cust.items():
            customers_by_sat[s].append(c)

        # -----------------------------------------------------------
        # 2. pre-compute all permutations per satellite
        # -----------------------------------------------------------
        perms_by_sat: Dict[str, List[Tuple[str, ...]]] = {}
        for s, clist in customers_by_sat.items():
            perms_by_sat[s] = list(permutations(clist)) or [()]   # [] for satellites without customers

        # -----------------------------------------------------------
        # 3. cartesian product over satellites  (gigantic!)
        # -----------------------------------------------------------
        sat_ids = list(perms_by_sat.keys())
        iterator = product(*(perms_by_sat[s] for s in sat_ids))

        best_cost = float("inf")
        best_solution = None
        evaluated = 0

        for orders in iterator:
            evaluated += 1

            # build one LV + one EV route per satellite
            lv_routes = [[depot, s, depot] for s in sat_ids]
            ev_routes = {
                s: [[s, *orders[idx], s]]      # idx matches sat_ids order
                for idx, s in enumerate(sat_ids)
            }
            sol = Solution(lv_routes=lv_routes, ev_routes=ev_routes)

            res = evaluator.evaluate(sol)
            if res["feasible"] and res["cost"] < best_cost:
                best_cost = res["cost"]
                best_solution = sol

        return {
            "best_cost": best_cost,
            "best_solution": best_solution,
            "evaluated": evaluated,
            "enumeration_size": self._enum_size(perms_by_sat),
        }

    # ───────────────────────────────────────────────────────────────
    # helpers
    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def _single(lst, what):
        if not lst:
            raise ValueError(f"No {what} found in instance.")
        if len(lst) > 1:
            # If more than one depot, pick the first for this toy solver
            print(f"Warning: Multiple {what}s found – using '{lst[0]}'.")
        return lst[0]

    @staticmethod
    def _enum_size(perms_by_sat) -> int:
        n = 1
        for p in perms_by_sat.values():
            n *= len(p)
        return n
