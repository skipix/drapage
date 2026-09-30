"""Visualiseur interactif du drapé fishnet, ligne par ligne le long de l'axe X du tissu.

Touches :
    → / ←        ligne suivante / précédente
    Espace       lecture / pause
    Début / Fin  première / dernière ligne
    + / -        accélérer / ralentir la lecture
    c            colorer les fils selon le cisaillement

Usage : python viewer.py [sphere|cylindre|cylindre_u] [--shape N M] [--spacing D] [--bend R]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from fishnet import drape
from meshes import bent_cylinder_mesh, cylinder_mesh, sphere_mesh

WARP_COLOR, WEFT_COLOR = "tab:blue", "tab:red"
MAX_SHEAR = 45.0  # écart à 90° correspondant au rouge de la palette


def node_shear(P):
    """Écart à 90° (degrés) de l'angle chaîne/trame à chaque nœud, NaN si indéfini."""
    warp = np.gradient(P, axis=0)
    weft = np.gradient(P, axis=1)
    cos = np.sum(warp * weft, -1) / (np.linalg.norm(warp, axis=-1) * np.linalg.norm(weft, axis=-1))
    return np.abs(90.0 - np.degrees(np.arccos(np.clip(cos, -1, 1))))


class DrapeViewer:
    """`display_mesh` : maillage (V×3, F×3) plus léger pour dessiner l'objet ; par défaut le même."""

    def __init__(self, vertices, faces, P, history, display_mesh=None):
        self.P, self.history = P, history
        n, m = P.shape[:2]

        # Étape à laquelle chaque ligne j (colonne P[:, j]) est posée.
        self.line_step = np.full(m, np.inf)
        for k, step in enumerate(history):
            self.line_step[step.line] = k

        # Toutes les arêtes, avec l'étape à partir de laquelle elles sont visibles.
        edges, kinds = [], []
        for i in range(n):
            for j in range(m):
                if i + 1 < n:
                    edges.append(((i, j), (i + 1, j)))
                    kinds.append(0)
                if j + 1 < m:
                    edges.append(((i, j), (i, j + 1)))
                    kinds.append(1)
        self.segments = np.array([[P[p], P[q]] for p, q in edges])
        self.edge_step = np.array([max(self.line_step[p[1]], self.line_step[q[1]]) for p, q in edges])
        self.edge_kind = np.array(kinds)
        shear = node_shear(P)
        sp = np.array([shear[p] for p, _ in edges])
        sq = np.array([shear[q] for _, q in edges])
        self.edge_shear = np.where(np.isnan(sp), sq, np.where(np.isnan(sq), sp, (sp + sq) / 2))
        self.cmap = plt.get_cmap("RdYlGn_r")

        self.k = 0
        self.playing = False
        self.interval = 300  # ms entre deux lignes en lecture
        self.color_shear = False

        for key in ("keymap.back", "keymap.forward", "keymap.home", "keymap.fullscreen"):
            plt.rcParams[key] = []
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0, 0.1, 1, 0.85], projection="3d")
        # Le rendu 3D de matplotlib est lent : surface allégée et axes masqués.
        dv, df = display_mesh if display_mesh is not None else (vertices, faces)
        self.ax.plot_trisurf(*dv.T, triangles=df, color="lightgray", alpha=0.3, linewidth=0)
        self.ax.set_box_aspect(np.ptp(vertices, axis=0))
        self.ax.set_axis_off()

        self.lines = Line3DCollection(self.segments, linewidths=1.0)
        self.ax.add_collection3d(self.lines)
        (self.current,) = self.ax.plot([], [], [], "-o", color="black", lw=2.5, ms=4, zorder=10)
        self.links = Line3DCollection(self.segments, colors="orange", linewidths=2.0, zorder=9)
        self.ax.add_collection3d(self.links)
        self.info = self.fig.text(0.02, 0.97, "", va="top", family="monospace")
        self.fig.text(
            0.98, 0.97,
            "→/← ligne   Espace lecture\nDébut/Fin   +/- vitesse   c cisaillement",
            va="top", ha="right", fontsize=9, color="gray",
        )

        slider_ax = self.fig.add_axes([0.12, 0.03, 0.76, 0.03])
        self.slider = Slider(slider_ax, "ligne", 1, len(history), valinit=1, valstep=1)
        self.slider.drawon = False  # la figure entière est redessinée par draw()
        self.slider.on_changed(lambda v: self.goto(int(v) - 1))

        # Minuteur à un coup, relancé seulement quand l'image précédente est dessinée :
        # la lecture ne peut jamais aller plus vite que le rendu ni saturer la file d'événements.
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.single_shot = True
        self.timer.add_callback(self._tick)
        self.tick_pending = False
        self.fig.canvas.mpl_connect("draw_event", self._on_drawn)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.goto(0)

    def goto(self, k):
        self.k = int(np.clip(k, 0, len(self.history) - 1))
        if self.slider.val != self.k + 1:
            self.slider.eventson = False  # évite que le curseur rappelle goto
            self.slider.set_val(self.k + 1)
            self.slider.eventson = True
        self.draw()

    def draw(self):
        k, P = self.k, self.P
        visible = self.edge_step <= k
        self.lines.set_segments(self.segments[visible])
        if self.color_shear:
            colors = self.cmap(np.nan_to_num(self.edge_shear[visible]) / MAX_SHEAR)
        else:
            colors = np.where(self.edge_kind[visible] == 0, WARP_COLOR, WEFT_COLOR)
        self.lines.set_color(colors)

        # Ligne courante en noir, fils de trame qui la relient à la précédente en orange.
        step = self.history[k]
        line = P[:, step.line]
        self.current.set_data_3d(line[:, 0], line[:, 1], line[:, 2])
        self.links.set_segments(self.segments[(self.edge_step == k) & (self.edge_kind == 1)])

        placed = ~np.isnan(line).any(axis=1)
        shear = self._line_shear(k)
        shear_text = "—" if np.isnan(shear).all() else f"max {np.nanmax(shear):.1f}°, moyen {np.nanmean(shear):.1f}°"
        self.info.set_text(
            f"ligne {k + 1}/{len(self.history)}   (indice j = {step.line})\n"
            f"phase : {step.phase}\n"
            f"nœuds posés sur la ligne : {placed.sum()}/{len(line)}\n"
            f"cisaillement : {shear_text}\n"
            f"{'lecture' if self.playing else 'pause'} : {1000 / self.interval:.1f} lignes/s"
        )
        self.fig.canvas.draw_idle()

    def _line_shear(self, k):
        """Écart à 90° entre la ligne k et les fils de trame qui la relient à la ligne précédente."""
        j = self.history[k].line
        if k == 0:
            return np.full(self.P.shape[0], np.nan)
        jc = self.history[0].line
        prev = j - int(np.sign(j - jc))  # ligne sur laquelle s'appuie j
        line = self.P[:, j]
        warp = np.gradient(line, axis=0)
        weft = line - self.P[:, prev]
        cos = np.sum(warp * weft, -1) / (np.linalg.norm(warp, axis=-1) * np.linalg.norm(weft, axis=-1))
        return np.abs(90.0 - np.degrees(np.arccos(np.clip(cos, -1, 1))))

    def _on_drawn(self, _event):
        if self.playing and not self.tick_pending:
            self.tick_pending = True
            self.timer.start()

    def _tick(self):
        self.tick_pending = False
        if not self.playing:
            return
        if self.k >= len(self.history) - 1:
            self.set_playing(False)
        else:
            self.goto(self.k + 1)

    def set_playing(self, playing):
        self.playing = playing
        if not playing:
            self.timer.stop()
            self.tick_pending = False
        self.draw()  # le prochain draw_event relance le minuteur

    def toggle_play(self):
        if not self.playing and self.k >= len(self.history) - 1:
            self.goto(0)  # relance depuis le début
        self.set_playing(not self.playing)

    def set_speed(self, factor):
        self.interval = int(np.clip(self.interval / factor, 20, 5000))
        self.timer.interval = self.interval
        self.draw()

    def toggle_color(self):
        self.color_shear = not self.color_shear
        self.draw()

    def on_key(self, event):
        actions = {
            "right": lambda: self.goto(self.k + 1),
            "left": lambda: self.goto(self.k - 1),
            " ": self.toggle_play,
            "home": lambda: self.goto(0),
            "end": lambda: self.goto(len(self.history) - 1),
            "+": lambda: self.set_speed(2),
            "-": lambda: self.set_speed(0.5),
            "c": self.toggle_color,
        }
        if event.key in actions:
            actions[event.key]()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("objet", nargs="?", default="sphere", choices=["sphere", "cylindre", "cylindre_u"])
    parser.add_argument("--shape", type=int, nargs=2, default=None, metavar=("N", "M"))
    parser.add_argument("--spacing", type=float, default=None)
    parser.add_argument("--bend", type=float, default=6.0, help="rayon de courbure de cylindre_u")
    args = parser.parse_args()

    if args.objet == "sphere":
        vertices, faces = sphere_mesh(radius=1.0)
        shape, spacing = (21, 21), 0.1
        display = sphere_mesh(radius=1.0, n_lat=16, n_lon=32)
    elif args.objet == "cylindre":
        vertices, faces = cylinder_mesh(radius=1.0, height=4.0)
        shape, spacing = (25, 25), 0.12
        display = cylinder_mesh(radius=1.0, height=4.0, n_theta=32, n_h=4)
    else:
        vertices, faces = bent_cylinder_mesh(radius=1.0, length=4.0, bend_radius=args.bend)
        shape, spacing = (25, 25), 0.12
        display = bent_cylinder_mesh(radius=1.0, length=4.0, bend_radius=args.bend, n_theta=32, n_h=12)
    shape = tuple(args.shape) if args.shape else shape
    spacing = args.spacing or spacing

    print(f"Drapé en cours ({args.objet}, grille {shape}, pas {spacing})…")
    P, history = drape(vertices, faces, shape, spacing, return_history=True)
    DrapeViewer(vertices, faces, P, history, display_mesh=display)
    plt.show()


if __name__ == "__main__":
    main()
