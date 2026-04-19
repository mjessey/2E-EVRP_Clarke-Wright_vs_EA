# -----------------------------------------------------------
#   Brute-force (permutation) solver for the tiny toy version
#   of the 2E-EVRP instance provided in the README.
# -----------------------------------------------------------

from itertools import permutations
from math import sqrt
from typing import Dict, List, Any, Tuple


class BruteForce:
    """
    Solve the *toy* variant of 2E-EVRP in which
        • one depot exists          (Type == 'd')
        • every customer is visited once
        • objective = shortest Euclidean distance
    The algorithm enumerates *all* permutations of the
    customer set, so it is feasible only for n ≤ 10-11.
    """

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
    def solve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        data : dict
            Output of the Parser class (“instance data”).

        Returns
        -------
        dict with keys
        • 'best_cost'   : float
        • 'best_route'  : list of node-ids  (starting/ending at depot)
        • 'evaluated'   : number of permutations inspected
        """

        depot_id = self._get_single_depot(data)
        customers = data["customers"]
        coords = {nid: (data["nodes"][nid]["x"], data["nodes"][nid]["y"])
                  for nid in [depot_id] + customers}

        # -------------------------------------------------
        # Pre-compute distance matrix (symmetric Euclidean)
        # -------------------------------------------------
        dist = {}
        for i in coords:
            xi, yi = coords[i]
            dist[i] = {}
            for j in coords:
                xj, yj = coords[j]
                dist[i][j] = sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

        # -------------------------------------------------
        # brute-force enumeration
        # -------------------------------------------------
        best_cost = float("inf")
        best_route: List[str] = []
        evaluated = 0

        for perm in permutations(customers):
            evaluated += 1
            route = (depot_id,) + perm + (depot_id,)
            cost = sum(dist[route[k]][route[k + 1]] for k in range(len(route) - 1))
            if cost < best_cost:
                best_cost = cost
                best_route = list(route)

        return {"best_cost": best_cost,
                "best_route": best_route,
                "evaluated": evaluated}

    # -----------------------------------------------------
    # private helpers
    # -----------------------------------------------------
    @staticmethod
    def _get_single_depot(data: Dict[str, Any]) -> str:
        depots = data.get("depots", [])
        if not depots:
            raise ValueError("No depot defined in instance.")
        if len(depots) > 1:
            raise ValueError("BruteForce solver works only with ONE depot.")
        return depots[0]
