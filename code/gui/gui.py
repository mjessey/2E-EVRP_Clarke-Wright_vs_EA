# gui.py
# ---------------------------------------------------------------
#  Graphical viewer for 2E-EVRP instances + routes
#  + image export support
#     - auto-save on startup
#     - press S to save again manually
# ---------------------------------------------------------------

import pygame as pg
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from core.evaluator import Solution


class GUI:
    # -------- colours -------------------------------------------------
    COLOURS = {
        "bg":         (245, 245, 245),
        "grid":       (220, 220, 220),
        "d":          (200,  30,  30),   # depot
        "s":          ( 30,  30, 200),   # satellite
        "f":          ( 40, 160,  40),   # charging station
        "c":          (230, 170,   0),   # customer
        "lv_route":   (  0,   0,   0),   # black
        "tooltip_bg": ( 50,  50,  50),
        "tooltip_fg": (255, 255, 255),
    }

    EV_PALETTE = [
        (227,  26,  28),
        ( 51, 160,  44),
        ( 31, 120, 180),
        (166, 206,  57),
        (255, 127,   0),
        (106,  61, 154),
        (153, 153, 153),
        (177,  89,  40),
    ]

    def __init__(
        self,
        data: Dict[str, Any],
        solution: Optional[Solution] = None,
        size: Tuple[int, int] = (1000, 1000),
        margin: int = 50,
        node_radius: int = 5,
        grid_steps: int = 10,
        algorithm_name: str = "solution",
        instance_name: str = "instance",
        save_dir: str | Path = ".",
        save_on_start: bool = True,
    ) -> None:
        self.data         = data
        self.solution     = solution
        self.W, self.H    = size
        self.margin       = margin
        self.node_radius  = node_radius
        self.grid_steps   = grid_steps

        self.algorithm_name = algorithm_name
        self.instance_name  = instance_name
        self.save_dir       = Path(save_dir)
        self.save_on_start  = save_on_start
        self._saved_once    = False

        pg.init()
        pg.display.set_caption("2E-EVRP viewer")
        self.screen = pg.display.set_mode((self.W, self.H))
        self.font   = pg.font.SysFont("arial", 14)

        xs = [n["x"] for n in data["nodes"].values()]
        ys = [n["y"] for n in data["nodes"].values()]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)

        span_x = self.x_max - self.x_min or 1
        span_y = self.y_max - self.y_min or 1
        avail_w = self.W - 2 * margin
        avail_h = self.H - 2 * margin
        self.scale = min(avail_w / span_x, avail_h / span_y)

        self.pos_px: Dict[str, Tuple[int, int]] = {
            nid: self._world_to_screen(n["x"], n["y"])
            for nid, n in data["nodes"].items()
        }

        self.route_colours = {}
        if solution:
            colour_cursor = 0
            for _, rlist in solution.ev_routes.items():
                for r in rlist:
                    self.route_colours[id(r)] = self.EV_PALETTE[colour_cursor % len(self.EV_PALETTE)]
                    colour_cursor += 1

    # ------------------------------------------------------------------
    def run(self) -> None:
        clock = pg.time.Clock()
        running = True

        while running:
            clock.tick(60)
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_s:
                        path = self.save_image()
                        print(f"Image saved to: {path}")

            self._draw_scene()
            pg.display.flip()

            if self.save_on_start and not self._saved_once:
                path = self.save_image()
                print(f"Image saved to: {path}")
                self._saved_once = True

        pg.quit()

    # ------------------------------------------------------------------
    def save_image(self) -> Path:
        self._draw_scene()
        pg.display.flip()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{self.algorithm_name}_{self.instance_name}.png"
        pg.image.save(self.screen, str(path))
        return path

    # ------------------------------------------------------------------
    def _draw_scene(self):
        self.screen.fill(self.COLOURS["bg"])
        self._draw_grid()
        if self.solution:
            self._draw_routes()
        self._draw_nodes()
        self._draw_hover_tooltip()
        self._draw_mouse_coordinates()

    def _draw_grid(self):
        step_x = (self.W - 2 * self.margin) / self.grid_steps
        step_y = (self.H - 2 * self.margin) / self.grid_steps
        for i in range(self.grid_steps + 1):
            x = self.margin + i * step_x
            pg.draw.line(self.screen, self.COLOURS["grid"],
                         (x, self.margin), (x, self.H - self.margin), 1)
            y = self.margin + i * step_y
            pg.draw.line(self.screen, self.COLOURS["grid"],
                         (self.margin, y), (self.W - self.margin, y), 1)

    def _draw_routes(self):
        for route in self.solution.lv_routes:
            pts = [self.pos_px[nid] for nid in route]
            pg.draw.lines(self.screen, self.COLOURS["lv_route"], False, pts, 4)

        for routes in self.solution.ev_routes.values():
            for r in routes:
                col = self.route_colours[id(r)]
                pts = [self.pos_px[nid] for nid in r]
                pg.draw.lines(self.screen, col, False, pts, 2)

    def _draw_nodes(self):
        for nid, node in self.data["nodes"].items():
            x, y = self.pos_px[nid]
            col = self.COLOURS[node["Type"]]
            radius = self.node_radius + (3 if node["Type"] == "d" else 0)
            pg.draw.circle(self.screen, col, (x, y), radius)
            label = self.font.render(nid, True, (0, 0, 0))
            self.screen.blit(label, (x + radius + 2, y - radius - 2))

    def _draw_hover_tooltip(self):
        mx, my = pg.mouse.get_pos()
        hovered = None
        for nid, (x, y) in self.pos_px.items():
            if (mx - x) ** 2 + (my - y) ** 2 <= (self.node_radius + 2) ** 2:
                hovered = nid
                break
        if not hovered:
            return

        info = self._make_node_info_lines(hovered)
        surfaces = [self.font.render(t, True, self.COLOURS["tooltip_fg"]) for t in info]
        w = max(s.get_width() for s in surfaces) + 10
        h = sum(s.get_height() for s in surfaces) + 10
        px, py = mx + 20, my + 20
        if px + w > self.W:
            px = mx - w - 20
        if py + h > self.H:
            py = my - h - 20
        pg.draw.rect(self.screen, self.COLOURS["tooltip_bg"], (px, py, w, h), border_radius=4)
        y = py + 5
        for s in surfaces:
            self.screen.blit(s, (px + 5, y))
            y += s.get_height()

    def _make_node_info_lines(self, nid: str) -> List[str]:
        n = self.data["nodes"][nid]
        return [
            f"{nid} (type '{n['Type']}')",
            f"({n['x']}, {n['y']})",
            f"Delivery {n['DeliveryDemand']}",
            f"Pickup   {n['PickupDemand']}",
            f"DivRate  {n['DivisionRate']}%",
            f"TW {n['ReadyTime']}–{n['DueDate']}",
            f"Service  {n['ServiceTime']}",
        ]

    def _draw_mouse_coordinates(self):
        mx, my = pg.mouse.get_pos()
        wx, wy = self._screen_to_world(mx, my)
        text = f"{wx:.1f}, {wy:.1f}"
        surf = self.font.render(text, True, (0, 0, 0))
        rect = surf.get_rect(topleft=(8, 8)).inflate(4, 2)
        pg.draw.rect(self.screen, (220, 220, 220), rect)
        self.screen.blit(surf, (10, 9))

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = self.margin + (x - self.x_min) * self.scale
        sy = self.H - (self.margin + (y - self.y_min) * self.scale)
        return int(sx), int(sy)

    def _screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        x = (sx - self.margin) / self.scale + self.x_min
        y = (self.H - sy - self.margin) / self.scale + self.y_min
        return x, y
