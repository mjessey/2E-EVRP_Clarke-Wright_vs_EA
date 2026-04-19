# ---------------------------------------------------------------
#   Light-weight visualiser for Two-Echelon EVRP instances
#   – shows every node at the proper Cartesian position
#   – colour / shape encode node type
#   – mouse-hover reveals a tooltip with full information
# ---------------------------------------------------------------

import pygame as pg
from typing import Dict, Any, Tuple


class UI:
    # ---------- colours --------------------------------------------------
    COLOURS = {
        "bg":        (245, 245, 245),
        "grid":      (220, 220, 220),
        "d":         (200,  30,  30),    # depot
        "s":         ( 30,  30, 200),    # satellite
        "f":         ( 40, 160,  40),    # charging station
        "c":         (230, 170,   0),    # customer
        "tooltip_bg": (50, 50, 50),
        "tooltip_fg": (255, 255, 255)
    }

    # ------------------------------------------------------------------
    def __init__(
        self,
        data: Dict[str, Any],
        size: Tuple[int, int] = (900, 700),
        margin: int = 60,
        node_radius: int = 7,
        grid_steps: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        data   : dictionary returned by Parser
        size   : window width, height
        margin : empty border around drawing area
        """
        self.data         = data
        self.W, self.H    = size
        self.margin       = margin
        self.node_radius  = node_radius
        self.grid_steps   = grid_steps

        # init pygame
        pg.init()
        pg.display.set_caption("2E-EVRP instance viewer")
        self.screen = pg.display.set_mode((self.W, self.H))
        self.font   = pg.font.SysFont("arial", 14)

        # build coordinate transform  -----------------------------
        xs = [node["x"] for node in data["nodes"].values()]
        ys = [node["y"] for node in data["nodes"].values()]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)

        span_x = self.x_max - self.x_min or 1
        span_y = self.y_max - self.y_min or 1

        avail_w = self.W - 2 * margin
        avail_h = self.H - 2 * margin
        self.scale = min(avail_w / span_x, avail_h / span_y)

        # pre-compute screen coordinates for all nodes
        self.pos_px: Dict[str, Tuple[int, int]] = {}
        for nid, node in data["nodes"].items():
            self.pos_px[nid] = self._world_to_screen(node["x"], node["y"])

    # --------------------------------------------------------------
    # coordinate conversion helpers
    # --------------------------------------------------------------
    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = self.margin + (x - self.x_min) * self.scale
        sy = self.H - (self.margin + (y - self.y_min) * self.scale)  # flip y-axis
        return int(sx), int(sy)

    # --------------------------------------------------------------
    def run(self) -> None:
        """Main event loop."""
        clock = pg.time.Clock()
        running = True

        while running:
            clock.tick(60)                 # limit FPS
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False

            self._draw_scene()
            pg.display.flip()

        pg.quit()

    # --------------------------------------------------------------
    def _draw_scene(self) -> None:
        self.screen.fill(self.COLOURS["bg"])
        self._draw_grid()
        self._draw_nodes()
        self._draw_hover_tooltip()

    # --------------------------------------------------------------
    def _draw_grid(self):
        step_x = (self.W - 2 * self.margin) / self.grid_steps
        step_y = (self.H - 2 * self.margin) / self.grid_steps

        for i in range(self.grid_steps + 1):
            # vertical
            x = self.margin + i * step_x
            pg.draw.line(
                self.screen, self.COLOURS["grid"],
                (x, self.margin), (x, self.H - self.margin), 1
            )
            # horizontal
            y = self.margin + i * step_y
            pg.draw.line(
                self.screen, self.COLOURS["grid"],
                (self.margin, y), (self.W - self.margin, y), 1
            )

    # --------------------------------------------------------------
    def _draw_nodes(self):
        for nid, node in self.data["nodes"].items():
            x, y = self.pos_px[nid]
            col = self.COLOURS[node["Type"]]
            radius = self.node_radius + (3 if node["Type"] == "d" else 0)
            pg.draw.circle(self.screen, col, (x, y), radius)
            # small id label
            label = self.font.render(nid, True, (0, 0, 0))
            self.screen.blit(label, (x + radius + 2, y - radius - 2))

    # --------------------------------------------------------------
    def _draw_hover_tooltip(self):
        mx, my = pg.mouse.get_pos()
        hovered = None
        for nid, (x, y) in self.pos_px.items():
            if (mx - x) ** 2 + (my - y) ** 2 <= (self.node_radius + 2) ** 2:
                hovered = nid
                break

        if hovered:
            info_lines = self._make_node_info_lines(hovered)
            padding = 5
            # compute tooltip size
            texts = [self.font.render(line, True, self.COLOURS["tooltip_fg"])
                     for line in info_lines]
            w = max(t.get_width() for t in texts) + 2 * padding
            h = sum(t.get_height() for t in texts) + 2 * padding

            # keep tooltip inside window
            pos_x = mx + 20
            pos_y = my + 20
            if pos_x + w > self.W:
                pos_x = mx - w - 20
            if pos_y + h > self.H:
                pos_y = my - h - 20

            # draw rectangle
            pg.draw.rect(self.screen, self.COLOURS["tooltip_bg"],
                         (pos_x, pos_y, w, h), border_radius=4)
            # blit every line
            cur_y = pos_y + padding
            for surface in texts:
                self.screen.blit(surface, (pos_x + padding, cur_y))
                cur_y += surface.get_height()

    # --------------------------------------------------------------
    def _make_node_info_lines(self, nid: str):
        n = self.data["nodes"][nid]
        return [
            f"{nid}  (type '{n['Type']}')",
            f"Coord: ({n['x']}, {n['y']})",
            f"Delivery : {n['DeliveryDemand']}",
            f"Pickup   : {n['PickupDemand']}",
            f"Div. rate: {n['DivisionRate']}%",
            f"TW: {n['ReadyTime']} – {n['DueDate']}",
            f"Service  : {n['ServiceTime']}",
        ]
