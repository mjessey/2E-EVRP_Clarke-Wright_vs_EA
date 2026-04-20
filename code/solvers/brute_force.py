# ---------------------------------------------------------------
#  ONE-FILE implementation that
#     • enumerates every feasible EV route for each satellite
#     • chooses the minimum number of EVs with a MILP
#     • returns the best Solution object
#
#  External dependencies:
#     –  evaluator.Solution, evaluator.Evaluator  (core package)
#     –  OR-Tools           (pip install ortools)  *or* PuLP
# ---------------------------------------------------------------

from __future__ import annotations
from itertools import combinations, permutations, product
from math import hypot
from typing import Dict, Any, List, Tuple, Optional
import math
from tqdm import tqdm

# ----------------------------------------------------------------
#  optional solver back-ends
try:
    from ortools.linear_solver import pywraplp          # preferred
    _MILP_BACKEND = "ortools"
except ModuleNotFoundError:
    try:
        import pulp as pl                               # fallback
        _MILP_BACKEND = "pulp"
    except ModuleNotFoundError:
        raise ImportError(
            "You need either OR-Tools (`pip install ortools`) or "
            "PuLP (`pip install pulp`) to run the BruteForce solver."
        )

# project-internal
from core.evaluator import Evaluator, Solution


# ================================================================
#  public entry class   BruteForce
# ================================================================
class BruteForce:
    """
    solve(instance_dict)  ->
        { 'solution' : Solution  | None,
          'evs'      : int,
          'distance' : float,
          'routes'   : int,          # EV routes enumerated
          'milp_vars': int,          # columns in set-partitioning
          'milp_time': float }       # seconds (if backend supports)
    """

    BIG_M = 10_000        # cost for *one* EV in lexicographic objective
    EPS   = 1e-3          # distance weight (must be < 1)

    # ------------------------------------------------------------
    def solve(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        # ---------- enumerate every feasible EV route -----------
        rg = _RouteGenerator(instance)
        routes_by_sat: Dict[str, List[List[str]]] = {}
        satellites = instance["satellites"]
        for s in tqdm(satellites, desc="Enumerating EV routes",
                    unit="sat", colour="green"):
            routes_by_sat[s] = rg.enumerate_for_satellite(s)

        # ---------- build & solve MILP --------------------------
        milp = _SetPartitioningMILP(instance,
                                    routes_by_sat,
                                    self.BIG_M,
                                    self.EPS)
        sol, stats = milp.solve()

        if sol is None:
            return {"solution": None, **stats}

        # ---------- verification (optional) ---------------------
        ev = Evaluator(instance, check_sat_inventory=False)
        res = ev.evaluate(sol)
        assert res["feasible"], "MILP returned infeasible plan"

        return {
            "solution":  sol,
            "evs":       ev._count_evs(sol),
            "distance":  res["cost"],
            **stats,
        }


# ================================================================
#  helper class  –  enumerates *feasible* EV routes for one Sat
# ================================================================
class _RouteGenerator:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.eval = Evaluator(data,
                              check_sat_inventory=False,
                              check_time_windows=True)

        self.stations = data["stations"]
        self.bat_cap  = data["params"]["Q"]
        self.speed    = data["params"]["v"]

        # distance cache
        coords = {n: (self.data["nodes"][n]["x"], self.data["nodes"][n]["y"])
                  for n in self.data["nodes"]}
        self.dist = {(i, j): math.hypot(coords[i][0] - coords[j][0],
                                        coords[i][1] - coords[j][1])
                     for i in coords for j in coords}

    # ------------------------------------------------------------
    def enumerate_for_satellite(self, sat: str) -> List[List[str]]:
        custs = self._customers_of_sat(sat)
        n_cust = len(custs)
        m_sta  = len(self.stations)
        routes: List[List[str]] = []

        # ---------- pre-compute total candidates -------------------
        total = 0
        comb  = math.comb           # alias
        for k in range(1, n_cust + 1):
            total += comb(n_cust, k) * math.factorial(k) * (m_sta + 1) ** (k + 1)

        bar = tqdm(total=total, desc=f"  {sat}", unit="route",
                colour="yellow", leave=False)

        # ---------- enumeration -----------------------------------
        for k in range(1, n_cust + 1):
            for subset in combinations(custs, k):
                for perm in permutations(subset):
                    gap_cnt = len(perm) + 1
                    for pattern in product((None, *self.stations), repeat=gap_cnt):
                        bar.update(1)                              # <-- progress

                        seq = [sat]
                        for i, c in enumerate(perm):
                            if pattern[i]:
                                seq.append(pattern[i])
                            seq.append(c)
                        if pattern[-1]:
                            seq.append(pattern[-1])
                        seq.append(sat)

                        if self._battery_ok(seq) and self._evaluator_ok(sat, seq):
                            routes.append(seq)

        bar.close()

        # remove duplicates (can happen)
        uniq, seen = [], set()
        for r in routes:
            t = tuple(r)
            if t not in seen:
                uniq.append(r)
                seen.add(t)
        return uniq

    # ------------------------------------------------------------
    def _customers_of_sat(self, sat) -> List[str]:
        # nearest satellite assignment (same as earlier code)
        sat_coords = {
            s: (self.data["nodes"][s]["x"], self.data["nodes"][s]["y"])
            for s in self.data["satellites"]
        }
        lst = []
        for c in self.data["customers"]:
            cx, cy = self.data["nodes"][c]["x"], self.data["nodes"][c]["y"]
            best = min(sat_coords,
                       key=lambda s: hypot(cx - sat_coords[s][0],
                                           cy - sat_coords[s][1]))
            if best == sat:
                lst.append(c)
        return lst

    # quick battery filter (distance between recharges ≤ Q)
    def _battery_ok(self, route) -> bool:
        remaining = self.bat_cap
        prev = route[0]
        for nxt in route[1:]:
            d = self.dist[prev, nxt]
            if d > self.bat_cap:          # impossible leg
                return False
            remaining -= d
            if remaining < 0:
                return False
            if self.data["nodes"][nxt]["Type"] == "f":
                remaining = self.bat_cap
            prev = nxt
        return True

    # call full evaluator with TW, capacity, battery etc.
    def _evaluator_ok(self, sat: str, ev_route: List[str]) -> bool:
        sol = Solution(
            lv_routes=[[self.data["depots"][0], sat, self.data["depots"][0]]],
            ev_routes={sat: [ev_route]},
        )
        return self.eval.evaluate(sol)["feasible"]


# ================================================================
#  helper class  –  MILP: minimum #EVs  +  ϵ·distance
# ================================================================
class _SetPartitioningMILP:
    def __init__(self,
                 data: Dict[str, Any],
                 routes_by_sat: Dict[str, List[List[str]]],
                 M: float,
                 eps: float):
        self.data = data
        self.routes_by_sat = routes_by_sat
        self.M = M
        self.eps = eps
        # distance cache
        coords = {n: (data["nodes"][n]["x"], data["nodes"][n]["y"])
                  for n in data["nodes"]}
        self.dist = {(i, j): math.hypot(coords[i][0] - coords[j][0],
                                        coords[i][1] - coords[j][1])
                     for i in coords for j in coords}

        # demand cache per route  (delivery only, not pickup)
        self.demand = {}
        for s, routes in routes_by_sat.items():
            self.demand[s] = [
                sum(
                    self.data["nodes"][nid]["DeliveryDemand"]
                    for nid in r
                    if self.data["nodes"][nid]["Type"] == "c"
                )
                for r in routes
            ]

    # ------------------------------------------------------------
    def _route_distance(self, r: List[str]) -> float:
        return sum(self.dist[a, b] for a, b in zip(r, r[1:]))

    # ------------------------------------------------------------
    def solve(self) -> Tuple[Optional[Solution], Dict[str, Any]]:
        if _MILP_BACKEND == "ortools":
            return self._solve_ortools()
        else:
            return self._solve_pulp()

    # ===== OR-Tools backend =====================================
    def _solve_ortools(self):
        solver = pywraplp.Solver.CreateSolver("CBC")
        x = {}   # (sat, idx) -> binary var
        for s, rts in self.routes_by_sat.items():
            for i in range(len(rts)):
                x[s, i] = solver.BoolVar(f"x_{s}_{i}")

        # cover each customer exactly once
        for cid in self.data["customers"]:
            cst = solver.RowConstraint(1, 1, f"cover_{cid}")
            for s, rts in self.routes_by_sat.items():
                for i, r in enumerate(rts):
                    if cid in r:
                        cst.SetCoefficient(x[s, i], 1)

        # ---- LV capacity per satellite ---------------------------------
        LV_CAP = self.data["params"]["L"]
        for s, routes in self.routes_by_sat.items():
            cst = solver.RowConstraint(0, LV_CAP, f"lv_cap_{s}")
            for i, _ in enumerate(routes):
                cst.SetCoefficient(x[s, i], self.demand[s][i])

        # objective
        obj = solver.Objective()
        for (s, i), var in x.items():
            dist = self._route_distance(self.routes_by_sat[s][i])
            obj.SetCoefficient(var, self.M + self.eps * dist)
        obj.SetMinimization()

        status = solver.Solve()
        stats = dict(
            milp_vars=len(x),
            milp_time=solver.wall_time()/1000 if status == 0 else None
        )
        if status != pywraplp.Solver.OPTIMAL:
            return None, stats

        ev_routes = {}
        for (s, i), var in x.items():
            if var.solution_value() > 0.5:
                ev_routes.setdefault(s, []).append(self.routes_by_sat[s][i])

        lv_routes = [[self.data["depots"][0], s, self.data["depots"][0]]
                     for s in self.data["satellites"]]
        return Solution(lv_routes, ev_routes), stats

    # ===== PuLP backend =========================================
    def _solve_pulp(self):
        prob = pl.LpProblem("2E-EVRP-SP", pl.LpMinimize)
        x = {}
        for s, rts in self.routes_by_sat.items():
            for i in range(len(rts)):
                x[s, i] = pl.LpVariable(f"x_{s}_{i}", lowBound=0, upBound=1, cat="Binary")

        # cover constraints
        for cid in self.data["customers"]:
            prob += pl.lpSum(
                x[s, i]
                for s, rts in self.routes_by_sat.items()
                for i, r in enumerate(rts)
                if cid in r
            ) == 1

        # ---- LV capacity per satellite ---------------------------------
        LV_CAP = self.data["params"]["L"]
        for s, routes in self.routes_by_sat.items():
            prob += pl.lpSum(
                self.demand[s][i] * x[s, i]
                for i in range(len(routes))
            ) <= LV_CAP

        # objective
        prob += pl.lpSum(
            (self.M + self.eps * self._route_distance(self.routes_by_sat[s][i]))
            * x[s, i]
            for (s, i) in x
        )

        prob.solve(pl.PULP_CBC_CMD(msg=False))
        stats = dict(milp_vars=len(x), milp_time=None)

        if pl.LpStatus[prob.status] != "Optimal":
            return None, stats

        ev_routes = {}
        for (s, i), var in x.items():
            if var.value() > 0.5:
                ev_routes.setdefault(s, []).append(self.routes_by_sat[s][i])

        lv_routes = [[self.data["depots"][0], s, self.data["depots"][0]]
                     for s in self.data["satellites"]]
        return Solution(lv_routes, ev_routes), stats
