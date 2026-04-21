# ---------------------------------------------------------------
#  Stage 2: Adaptive Local Search
#
#  Based on Section 4.3 of:
#  "A Memetic Algorithm for the Green Vehicle Routing Problem"
#  Peng et al. (2019)
#
#  8 neighborhood moves
#  --------------------
#  Intra-route (within same EV route):
#    M1 - node insertion  : move one customer to another position
#    M2 - node swap       : swap two customers
#    M3 - arc insertion   : move two consecutive customers together
#    M4 - arc swap        : swap two pairs of consecutive customers
#
#  Inter-route (between different EV routes):
#    M5 - node insertion  : move one customer to a different route
#    M6 - node swap       : swap customers between two routes
#    M7 - arc insertion   : move a customer pair to a different route
#    M8 - arc swap        : swap customer pairs between two routes
#
#  Reward/Punishment mechanism
#  ---------------------------
#  Each move has a score. Move k is selected by roulette wheel:
#
#    P(k) = sc_k / sum(sc_i)
#
#  After each iteration:
#    new best solution  -> sc += α * β1   (reward heavily)
#    better solution    -> sc += α * β2   (reward lightly)
#    worse solution     -> sc  = γ * sc   (punish)
#
#  Parameters from paper (Table 1):
#    α  = 0.2   reaction factor
#    β1 = 5     reward for new best
#    β2 = 1     reward for improvement
#    γ  = 0.9   punishment multiplier
#
#  Termination
#  -----------
#  Stop when no move improves the solution across a full pass
#  of all 8 moves (no_improve_iter >= n*).
#  Then apply perturbation and restart.
# ---------------------------------------------------------------

from __future__ import annotations

import math
import random
import copy
from typing import Dict, Any, List, Optional, Tuple

from core.evaluator import Evaluator, Solution


class AdaptiveLocalSearch:
    """
    Adaptive local search with 8 neighborhood moves and
    reward/punishment mechanism from Peng et al. (2019).

    Parameters match Table 1 of the paper.
    """

    # ---- parameters from paper Table 1 -------------------------
    ALPHA   = 0.2    # reaction factor
    BETA1   = 5.0    # reward: new best solution
    BETA2   = 1.0    # reward: better than current
    GAMMA   = 0.9    # punishment multiplier
    SC0     = 1.0    # initial score for each move
    N_STAR  = 50     # max iterations without improvement
    ZETA    = 5      # perturbation removes n/zeta customers

    # ------------------------------------------------------------
    def __init__(self, instance: Dict[str, Any]) -> None:
        self.data    = instance
        self.ev      = Evaluator(instance, check_sat_inventory=False)
        self.params  = instance["params"]
        self.EV_CAP  = self.params["C"]
        self.BAT_CAP = self.params["Q"]
        self.SPEED   = self.params["v"]
        self.INV_G   = self.params["g"]

        # distance cache
        coords = {n: (instance["nodes"][n]["x"], instance["nodes"][n]["y"])
                  for n in instance["nodes"]}
        self.dist: Dict[Tuple[str, str], float] = {
            (i, j): math.hypot(coords[i][0] - coords[j][0],
                               coords[i][1] - coords[j][1])
            for i in coords for j in coords
        }

        # move registry — order matches M1..M8
        self.moves = [
            self._m1_intra_node_insertion,
            self._m2_intra_node_swap,
            self._m3_intra_arc_insertion,
            self._m4_intra_arc_swap,
            self._m5_inter_node_insertion,
            self._m6_inter_node_swap,
            self._m7_inter_arc_insertion,
            self._m8_inter_arc_swap,
        ]
        self.move_names = [
            "M1 intra-node-insert",
            "M2 intra-node-swap",
            "M3 intra-arc-insert",
            "M4 intra-arc-swap",
            "M5 inter-node-insert",
            "M6 inter-node-swap",
            "M7 inter-arc-insert",
            "M8 inter-arc-swap",
        ]

    # ============================================================
    #  Public interface
    # ============================================================

    def run(self, sol: Solution, time_limit: float = 30.0, quick: bool = False) -> Solution:
        import time
        t_start = time.time()
        n_star          = 1 if quick else self.N_STAR
        scores          = [self.SC0] * 8
        current         = copy.deepcopy(sol)
        current_cost    = self._cost(current)
        best            = copy.deepcopy(current)
        best_cost       = current_cost
        no_improve_iter = 0

        while no_improve_iter < n_star:
            if time.time() - t_start > time_limit:
                break
            # active move set resets each outer iteration
            active = list(range(8))
            improved_this_pass = False

            while active:
                # select move by roulette wheel over active moves
                idx = self._roulette(scores, active)
                move_fn = self.moves[idx]

                # apply move — get best neighbour
                candidate = move_fn(copy.deepcopy(current))

                if candidate is None:
                    # move produced nothing feasible
                    active.remove(idx)
                    scores[idx] = self.GAMMA * scores[idx]
                    continue

                candidate_cost = self._cost(candidate)

                if candidate_cost < current_cost:
                    # improvement — reward and reset active set
                    current      = candidate
                    current_cost = candidate_cost
                    active       = list(range(8))
                    improved_this_pass = True

                    if candidate_cost < best_cost:
                        best      = copy.deepcopy(candidate)
                        best_cost = candidate_cost
                        scores[idx] += self.ALPHA * self.BETA1
                    else:
                        scores[idx] += self.ALPHA * self.BETA2

                else:
                    # no improvement — remove from active, punish
                    active.remove(idx)
                    scores[idx] = self.GAMMA * scores[idx]

                # keep scores positive
                scores = [max(s, 0.01) for s in scores]

            if improved_this_pass:
                no_improve_iter = 0
            else:
                no_improve_iter += 1

            # perturbation when fully stuck
            if no_improve_iter >= self.N_STAR:
                perturbed = self._perturbation(best)
                if perturbed is not None:
                    current      = perturbed
                    current_cost = self._cost(current)

        return best

    # ============================================================
    #  M1 — Intra-route node insertion
    #  Remove customer i from position p, reinsert at position q
    #  within the same route. Try all (p, q) pairs.
    # ============================================================

    def _m1_intra_node_insertion(
        self, sol: Solution
    ) -> Optional[Solution]:
        best_delta = 0.0
        best_sol   = None

        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                custs = [i for i, n in enumerate(route)
                         if self.data["nodes"][n]["Type"] == "c"]
                if len(custs) < 2:
                    continue

                for p in custs:
                    node = route[p]
                    # cost of removing node from position p
                    prev, nxt = route[p-1], route[p+1]
                    delta_remove = (- self.dist[prev, node]
                                    - self.dist[node, nxt]
                                    + self.dist[prev, nxt])

                    # try reinserting at every other position
                    new_route = route[:p] + route[p+1:]
                    for q in range(1, len(new_route)):
                        if q == p:
                            continue
                        candidate_route = (new_route[:q]
                                           + [node]
                                           + new_route[q:])
                        if not self._route_feasible(sat, candidate_route):
                            continue
                        a, b = new_route[q-1], new_route[q]
                        delta_insert = (self.dist[a, node]
                                        + self.dist[node, b]
                                        - self.dist[a, b])
                        delta = delta_remove + delta_insert
                        if delta < best_delta:
                            best_delta = delta
                            new_sol = copy.deepcopy(sol)
                            new_sol.ev_routes[sat][r_idx] = candidate_route
                            best_sol = new_sol

        return best_sol

    # ============================================================
    #  M2 — Intra-route node swap
    #  Swap positions of two customers i and j in the same route.
    # ============================================================

    def _m2_intra_node_swap(
        self, sol: Solution
    ) -> Optional[Solution]:
        best_delta = 0.0
        best_sol   = None

        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                custs = [i for i, n in enumerate(route)
                         if self.data["nodes"][n]["Type"] == "c"]
                if len(custs) < 2:
                    continue

                for pi in range(len(custs)):
                    for pj in range(pi + 1, len(custs)):
                        p, q = custs[pi], custs[pj]
                        candidate_route = list(route)
                        candidate_route[p], candidate_route[q] = (
                            candidate_route[q], candidate_route[p]
                        )
                        if not self._route_feasible(sat, candidate_route):
                            continue
                        delta = (self._route_cost(candidate_route)
                                 - self._route_cost(route))
                        if delta < best_delta:
                            best_delta = delta
                            new_sol = copy.deepcopy(sol)
                            new_sol.ev_routes[sat][r_idx] = candidate_route
                            best_sol = new_sol

        return best_sol

    # ============================================================
    #  M3 — Intra-route arc insertion
    #  Remove consecutive pair (i, j) and reinsert elsewhere
    #  in the same route.
    # ============================================================

    def _m3_intra_arc_insertion(
        self, sol: Solution
    ) -> Optional[Solution]:
        best_delta = 0.0
        best_sol   = None

        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                custs = [i for i, n in enumerate(route)
                         if self.data["nodes"][n]["Type"] == "c"]
                if len(custs) < 3:
                    continue

                # find consecutive customer pairs
                for pi in range(len(custs) - 1):
                    p, q = custs[pi], custs[pi + 1]
                    if q != p + 1:
                        continue   # not consecutive in route

                    arc = [route[p], route[q]]
                    new_route = route[:p] + route[q+1:]

                    for ins in range(1, len(new_route)):
                        candidate_route = (new_route[:ins]
                                           + arc
                                           + new_route[ins:])
                        if not self._route_feasible(sat, candidate_route):
                            continue
                        delta = (self._route_cost(candidate_route)
                                 - self._route_cost(route))
                        if delta < best_delta:
                            best_delta = delta
                            new_sol = copy.deepcopy(sol)
                            new_sol.ev_routes[sat][r_idx] = candidate_route
                            best_sol = new_sol

        return best_sol

    # ============================================================
    #  M4 — Intra-route arc swap
    #  Swap two consecutive customer pairs within same route.
    # ============================================================

    def _m4_intra_arc_swap(
        self, sol: Solution
    ) -> Optional[Solution]:
        best_delta = 0.0
        best_sol   = None

        for sat, routes in sol.ev_routes.items():
            for r_idx, route in enumerate(routes):
                custs = [i for i, n in enumerate(route)
                         if self.data["nodes"][n]["Type"] == "c"]
                if len(custs) < 4:
                    continue

                for pi in range(len(custs) - 1):
                    for pj in range(pi + 2, len(custs) - 1):
                        p1, p2 = custs[pi],     custs[pi + 1]
                        p3, p4 = custs[pj],     custs[pj + 1]
                        if p2 != p1 + 1 or p4 != p3 + 1:
                            continue
                        candidate_route = list(route)
                        # swap the two arcs
                        (candidate_route[p1], candidate_route[p2],
                         candidate_route[p3], candidate_route[p4]) = (
                            candidate_route[p3], candidate_route[p4],
                            candidate_route[p1], candidate_route[p2]
                        )
                        if not self._route_feasible(sat, candidate_route):
                            continue
                        delta = (self._route_cost(candidate_route)
                                 - self._route_cost(route))
                        if delta < best_delta:
                            best_delta = delta
                            new_sol = copy.deepcopy(sol)
                            new_sol.ev_routes[sat][r_idx] = candidate_route
                            best_sol = new_sol

        return best_sol

    # ============================================================
    #  M5 — Inter-route node insertion
    #  Remove customer i from route 1, insert into route 2.
    # ============================================================

    def _m5_inter_node_insertion(self, sol):
        best_delta, best_sol = 0.0, None
        rlist = [(sat, ri, r)
                for sat, routes in sol.ev_routes.items()
                for ri, r in enumerate(routes)]

        for sat1, r1, route1 in rlist:
            custs1 = [i for i, n in enumerate(route1)
                    if self.data["nodes"][n]["Type"] == "c"]

            for p in custs1:
                node = route1[p]
                prev, nxt = route1[p-1], route1[p+1]
                d_rem = (- self.dist[prev, node]
                        - self.dist[node, nxt]
                        + self.dist[prev, nxt])
                nr1      = route1[:p] + route1[p+1:]
                r1_has_c = any(
                    self.data["nodes"][n]["Type"] == "c"
                    for n in nr1
                )

                for sat2, r2, route2 in rlist:
                    if sat1 == sat2 and r1 == r2:
                        continue

                    for q in range(1, len(route2)):
                        nr2  = route2[:q] + [node] + route2[q:]
                        load = sum(
                            self.data["nodes"][n]["DeliveryDemand"]
                            for n in nr2
                            if self.data["nodes"][n]["Type"] == "c"
                        )
                        if load > self.EV_CAP:
                            continue
                        if not self._route_feasible(sat2, nr2):
                            continue
                        a, b  = route2[q-1], route2[q]
                        delta = d_rem + (
                            self.dist[a, node]
                            + self.dist[node, b]
                            - self.dist[a, b]
                        )
                        if delta < best_delta:
                            best_delta = delta
                            # rebuild ev_routes cleanly from scratch
                            new_ev = {}
                            for sat, routes in sol.ev_routes.items():
                                new_routes = []
                                for ri, r in enumerate(routes):
                                    if sat == sat1 and ri == r1:
                                        if r1_has_c:
                                            new_routes.append(nr1)
                                        # else drop this route entirely
                                    elif sat == sat2 and ri == r2:
                                        new_routes.append(nr2)
                                    else:
                                        new_routes.append(list(r))
                                if new_routes:
                                    new_ev[sat] = new_routes
                            best_sol = Solution(
                                lv_routes=sol.lv_routes,
                                ev_routes=new_ev
                            )
        return best_sol

    # ============================================================
    #  M6 — Inter-route node swap
    #  Swap customer i from route 1 with customer j from route 2.
    # ============================================================

    def _m6_inter_node_swap(
        self, sol: Solution
    ) -> Optional[Solution]:
        best_delta = 0.0
        best_sol   = None

        route_list = [
            (sat, r_idx, route)
            for sat, routes in sol.ev_routes.items()
            for r_idx, route in enumerate(routes)
        ]

        for si in range(len(route_list)):
            sat1, r1, route1 = route_list[si]
            custs1 = [i for i, n in enumerate(route1)
                      if self.data["nodes"][n]["Type"] == "c"]

            for sj in range(si + 1, len(route_list)):
                sat2, r2, route2 = route_list[sj]
                custs2 = [i for i, n in enumerate(route2)
                          if self.data["nodes"][n]["Type"] == "c"]

                for p in custs1:
                    for q in custs2:
                        candidate1 = list(route1)
                        candidate2 = list(route2)
                        candidate1[p], candidate2[q] = (
                            candidate2[q], candidate1[p]
                        )

                        load1 = sum(
                            self.data["nodes"][n]["DeliveryDemand"]
                            for n in candidate1
                            if self.data["nodes"][n]["Type"] == "c"
                        )
                        load2 = sum(
                            self.data["nodes"][n]["DeliveryDemand"]
                            for n in candidate2
                            if self.data["nodes"][n]["Type"] == "c"
                        )
                        if load1 > self.EV_CAP or load2 > self.EV_CAP:
                            continue
                        if (not self._route_feasible(sat1, candidate1)
                                or not self._route_feasible(sat2, candidate2)):
                            continue

                        delta = (self._route_cost(candidate1)
                                 + self._route_cost(candidate2)
                                 - self._route_cost(route1)
                                 - self._route_cost(route2))
                        if delta < best_delta:
                            best_delta = delta
                            new_sol = copy.deepcopy(sol)
                            new_sol.ev_routes[sat1][r1] = candidate1
                            new_sol.ev_routes[sat2][r2] = candidate2
                            best_sol = new_sol

        return best_sol

    # ============================================================
    #  M7 — Inter-route arc insertion
    #  Move consecutive pair (i,j) from route 1 into route 2.
    # ============================================================

    def _m7_inter_arc_insertion(
            self, sol: Solution
        ) -> Optional[Solution]:
            best_delta = 0.0
            best_sol   = None

            rlist = [
                (sat, r_idx, route)
                for sat, routes in sol.ev_routes.items()
                for r_idx, route in enumerate(routes)
            ]

            for sat1, r1, route1 in rlist:
                custs1 = [i for i, n in enumerate(route1)
                        if self.data["nodes"][n]["Type"] == "c"]

                for pi in range(len(custs1) - 1):
                    p, q = custs1[pi], custs1[pi + 1]
                    if q != p + 1:
                        continue

                    arc        = [route1[p], route1[q]]
                    new_route1 = route1[:p] + route1[q+1:]
                    r1_has_c   = any(
                        self.data["nodes"][n]["Type"] == "c"
                        for n in new_route1
                    )

                    for sat2, r2, route2 in rlist:
                        if sat1 == sat2 and r1 == r2:
                            continue

                        for ins in range(1, len(route2)):
                            new_route2 = route2[:ins] + arc + route2[ins:]
                            load2 = sum(
                                self.data["nodes"][n]["DeliveryDemand"]
                                for n in new_route2
                                if self.data["nodes"][n]["Type"] == "c"
                            )
                            if load2 > self.EV_CAP:
                                continue
                            if not self._route_feasible(sat2, new_route2):
                                continue

                            delta = (self._route_cost(new_route1)
                                    + self._route_cost(new_route2)
                                    - self._route_cost(route1)
                                    - self._route_cost(route2))
                            if delta < best_delta:
                                best_delta = delta
                                new_ev = {}
                                for sat, routes in sol.ev_routes.items():
                                    new_routes = []
                                    for ri, r in enumerate(routes):
                                        if sat == sat1 and ri == r1:
                                            if r1_has_c:
                                                new_routes.append(new_route1)
                                        elif sat == sat2 and ri == r2:
                                            new_routes.append(new_route2)
                                        else:
                                            new_routes.append(list(r))
                                    if new_routes:
                                        new_ev[sat] = new_routes
                                best_sol = Solution(
                                    lv_routes=sol.lv_routes,
                                    ev_routes=new_ev
                                )

            return best_sol

    # ============================================================
    #  M8 — Inter-route arc swap
    #  Swap consecutive pair (i,j) in route 1 with (k,l) in route 2.
    # ============================================================

    def _m8_inter_arc_swap(
        self, sol: Solution
    ) -> Optional[Solution]:
        best_delta = 0.0
        best_sol   = None

        route_list = [
            (sat, r_idx, route)
            for sat, routes in sol.ev_routes.items()
            for r_idx, route in enumerate(routes)
        ]

        for si in range(len(route_list)):
            sat1, r1, route1 = route_list[si]
            custs1 = [i for i, n in enumerate(route1)
                      if self.data["nodes"][n]["Type"] == "c"]

            for sj in range(si + 1, len(route_list)):
                sat2, r2, route2 = route_list[sj]
                custs2 = [i for i, n in enumerate(route2)
                          if self.data["nodes"][n]["Type"] == "c"]

                for pi in range(len(custs1) - 1):
                    p1, p2 = custs1[pi], custs1[pi + 1]
                    if p2 != p1 + 1:
                        continue

                    for pj in range(len(custs2) - 1):
                        p3, p4 = custs2[pj], custs2[pj + 1]
                        if p4 != p3 + 1:
                            continue

                        candidate1 = list(route1)
                        candidate2 = list(route2)

                        # swap the arcs between routes
                        (candidate1[p1], candidate1[p2],
                         candidate2[p3], candidate2[p4]) = (
                            candidate2[p3], candidate2[p4],
                            candidate1[p1], candidate1[p2]
                        )

                        load1 = sum(
                            self.data["nodes"][n]["DeliveryDemand"]
                            for n in candidate1
                            if self.data["nodes"][n]["Type"] == "c"
                        )
                        load2 = sum(
                            self.data["nodes"][n]["DeliveryDemand"]
                            for n in candidate2
                            if self.data["nodes"][n]["Type"] == "c"
                        )
                        if load1 > self.EV_CAP or load2 > self.EV_CAP:
                            continue
                        if (not self._route_feasible(sat1, candidate1)
                                or not self._route_feasible(sat2, candidate2)):
                            continue

                        delta = (self._route_cost(candidate1)
                                 + self._route_cost(candidate2)
                                 - self._route_cost(route1)
                                 - self._route_cost(route2))
                        if delta < best_delta:
                            best_delta = delta
                            new_sol = copy.deepcopy(sol)
                            new_sol.ev_routes[sat1][r1] = candidate1
                            new_sol.ev_routes[sat2][r2] = candidate2
                            best_sol = new_sol

        return best_sol

    # ============================================================
    #  Perturbation
    #  Remove n/zeta random customers and reinsert greedily.
    # ============================================================

    def _perturbation(self, sol: Solution) -> Optional[Solution]:
        all_custs = [
            n
            for routes in sol.ev_routes.values()
            for r in routes
            for n in r
            if self.data["nodes"][n]["Type"] == "c"
        ]
        n_remove = max(1, len(all_custs) // self.ZETA)
        to_remove = random.sample(all_custs, min(n_remove, len(all_custs)))

        # remove customers
        new_ev = {}
        for sat, routes in sol.ev_routes.items():
            new_routes = []
            for r in routes:
                new_r = [n for n in r if n not in to_remove]
                if any(self.data["nodes"][n]["Type"] == "c" for n in new_r):
                    new_routes.append(new_r)
            if new_routes:
                new_ev[sat] = new_routes
        partial = Solution(lv_routes=sol.lv_routes, ev_routes=new_ev)

        # reinsert greedily
        for cust in to_remove:
            best_cost = math.inf
            best_sat  = None
            best_ridx = None
            best_pos  = None

            for sat, routes in partial.ev_routes.items():
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
                        cost = self._route_cost(candidate)
                        if cost < best_cost:
                            best_cost = cost
                            best_sat  = sat
                            best_ridx = r_idx
                            best_pos  = pos

            if best_sat is not None:
                r = partial.ev_routes[best_sat][best_ridx]
                partial.ev_routes[best_sat][best_ridx] = (
                    r[:best_pos] + [cust] + r[best_pos:]
                )
            else:
                # open new spoke at nearest satellite
                nearest = min(
                    self.data["satellites"],
                    key=lambda s: self.dist[cust, s]
                )
                partial.ev_routes.setdefault(nearest, []).append(
                    [nearest, cust, nearest]
                )

        return partial

    # ============================================================
    #  Feasibility check
    # ============================================================

    def _route_feasible(self, sat: str, route: List[str]) -> bool:
        """Check battery and time windows for a single EV route."""
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
    #  Cost helpers
    # ============================================================

    def _cost(self, sol: Solution) -> float:
        return self.ev.evaluate(sol)["cost_with_M"]

    def _route_cost(self, route: List[str]) -> float:
        return sum(
            self.dist[a, b] for a, b in zip(route, route[1:])
        )

    # ============================================================
    #  Roulette wheel selection over active move indices
    # ============================================================

    @staticmethod
    def _roulette(scores: List[float], active: List[int]) -> int:
        active_scores = [scores[i] for i in active]
        total = sum(active_scores)
        r     = random.random() * total
        cum   = 0.0
        for i, s in zip(active, active_scores):
            cum += s
            if r <= cum:
                return i
        return active[-1]