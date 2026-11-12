"""Interactive mesh reader / editor.

The mesh is read and displayed element by element (triangles, in file order). The user
can interrupt the reading at any moment, slightly move some nodes, then continue: the
next elements are displayed with the edited geometry.

Reading:
    space          play / pause (selecting or moving a node also pauses)
    right / left   next / previous element, selecting only its newest node
    home / end     first / last element (end = whole mesh)
    + / -          faster / slower

Mouse:
    click          select the displayed node nearest to the cursor
    shift + click  add / remove a node from the selection
    drag           rotate the view

Editing:
    tab          select the next node of the current element
    x / X        move the selected nodes by +step / -step along x   (same for y / Y, z / Z)
    n / N        move each selected node along its own normal (outwards / inwards)
    ] / [        double / halve the step
    u            undo the last move
    r            reset the selected nodes to their original position
    escape       clear the selection
    v            show / hide the nodes
    s            save the edited mesh (OBJ) and print the displacements

Usage: python mesh_editor.py [file.obj|.off|.stl | sphere|cylinder|bent_cylinder]
                             [--rate ELEMENTS_PER_S] [--out FILE.obj]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mesh_io import read_mesh, write_obj
from meshes import bent_cylinder_mesh, cylinder_mesh, sphere_mesh

BASE_COLOR = np.array([0.69, 0.77, 0.87])
MODIFIED_COLOR = np.array([0.98, 0.60, 0.50])  # faces touching a moved node
CURRENT_COLOR = "orange"                        # last element read
LIGHT = np.array([0.4, -0.5, 0.77])
PICK_RADIUS = 25    # pixels: max distance between the click and the selected node
DRAG_THRESHOLD = 6  # pixels: a press/release further apart than this is a rotation
MIN_INTERVAL = 40   # ms between two frames during playback
MOVES = {"x": (0, 1), "X": (0, -1), "y": (1, 1), "Y": (1, -1), "z": (2, 1), "Z": (2, -1),
         "n": ("normal", 1), "N": ("normal", -1)}

BUILTIN = {
    "sphere": lambda: sphere_mesh(radius=1.0, n_lat=12, n_lon=24),
    "cylinder": lambda: cylinder_mesh(radius=1.0, height=4.0, n_theta=24, n_h=8),
    "bent_cylinder": lambda: bent_cylinder_mesh(radius=1.0, length=4.0, bend_radius=6.0, n_theta=24, n_h=16),
}


class MeshEditor:
    def __init__(self, vertices, faces, out_path="mesh_edited.obj", rate=None, play=True):
        self.V0 = np.asarray(vertices, dtype=float)
        self.V = self.V0.copy()
        self.F = np.asarray(faces, dtype=int)
        self.out_path = out_path
        n_faces = len(self.F)

        edges = self.V[self.F] - self.V[np.roll(self.F, 1, axis=1)]
        self.edge_length = float(np.mean(np.linalg.norm(edges, axis=-1)))
        self.step = 0.1 * self.edge_length  # "slightly": 10% of the mean edge length

        # Newest node of each element: the one that appeared last in the reading order
        # (ties: the last one listed in the element). Selected when stepping with the arrows.
        first_seen = np.full(len(self.V0), n_faces)
        np.minimum.at(first_seen, self.F.ravel(), np.repeat(np.arange(n_faces), 3))
        rank = first_seen[self.F] * 3 + np.arange(3)
        self.newest_node = self.F[np.arange(n_faces), np.argmax(rank, axis=1)]

        self.k = 1             # number of displayed elements; the current one is F[k-1]
        self.playing = False
        self.rate = rate or float(np.clip(n_faces / 30, 2, 200))  # elements/s (~30 s in total)
        self.selected = []     # selected node indices, in selection order
        self.undo_stack = []   # (node indices, previous positions)
        self.show_nodes = True
        self.status = None     # one-shot message (e.g. after saving)
        self.press_xy = None

        for key in ("keymap.back", "keymap.forward", "keymap.home", "keymap.fullscreen",
                    "keymap.save", "keymap.xscale", "keymap.yscale", "keymap.grid",
                    "keymap.grid_minor", "keymap.pan", "keymap.zoom"):
            plt.rcParams[key] = []
        self.fig = plt.figure(figsize=(10, 8))
        # Matplotlib only keeps weak references to bound-method callbacks: without this,
        # an editor created without being stored would be garbage-collected and the
        # window would ignore every click and key press.
        self.fig.mesh_editor = self
        self.ax = self.fig.add_axes([0, 0.08, 1, 0.87], projection="3d")
        lo, hi = self.V0.min(axis=0), self.V0.max(axis=0)
        margin = 0.1 * (hi - lo).max()
        self.ax.set_xlim(lo[0] - margin, hi[0] + margin)
        self.ax.set_ylim(lo[1] - margin, hi[1] + margin)
        self.ax.set_zlim(lo[2] - margin, hi[2] + margin)
        self.ax.set_box_aspect(hi - lo + 2 * margin)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.surface = Poly3DCollection([], edgecolors=(0.25, 0.25, 0.3, 0.6), linewidths=0.5)
        self.ax.add_collection3d(self.surface)
        self.current = Poly3DCollection([], facecolors=CURRENT_COLOR, edgecolors="black", linewidths=2)
        self.ax.add_collection3d(self.current)
        (self.node_pts,) = self.ax.plot([], [], [], ".", color="0.3", ms=5, zorder=9)
        (self.modified_pts,) = self.ax.plot([], [], [], "o", color="red", ms=4, zorder=10)
        (self.selected_pts,) = self.ax.plot([], [], [], "o", color="yellow", mec="black", mew=1.5,
                                            ms=9, zorder=11)
        (self.trails,) = self.ax.plot([], [], [], "--", color="black", lw=1, zorder=11)

        self.info = self.fig.text(0.02, 0.98, "", va="top", family="monospace", fontsize=9)
        self.fig.text(
            0.98, 0.98,
            "space play/pause   →/← element (selects its newest node)   home/end   +/- speed\n"
            "click select   shift+click add/remove   tab next node of element   drag rotate\n"
            "x y z n move (shift: opposite)   ]/[ step   u undo   r reset   esc clear   s save",
            va="top", ha="right", fontsize=8, color="gray",
        )
        slider_ax = self.fig.add_axes([0.12, 0.025, 0.76, 0.03])
        self.slider = Slider(slider_ax, "element", 1, n_faces, valinit=1, valstep=1)
        self.slider.drawon = False  # the whole figure is redrawn by draw()
        self.slider.on_changed(self._on_slider)

        # One-shot timer restarted only once the previous frame is drawn: playback can
        # never go faster than rendering nor flood the event queue.
        self.timer = self.fig.canvas.new_timer(interval=MIN_INTERVAL)
        self.timer.single_shot = True
        self.timer.add_callback(self._tick)
        self.tick_pending = False
        self.fig.canvas.mpl_connect("draw_event", self._on_drawn)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.playing = play
        self.draw()

    # --- geometry ----------------------------------------------------------------

    def face_normals(self):
        tri = self.V[self.F]
        return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # norm = 2 × area

    def vertex_normals(self, nodes):
        """Area-weighted average of the adjacent face normals, one row per node."""
        fn = self.face_normals()
        out = []
        for v in nodes:
            n = fn[np.any(self.F == v, axis=1)].sum(axis=0)
            norm = np.linalg.norm(n)
            out.append(n / norm if norm > 0 else np.array([0.0, 0.0, 1.0]))
        return np.array(out)

    def modified_nodes(self):
        return np.nonzero(np.any(self.V != self.V0, axis=1))[0]

    def displayed_nodes(self):
        return np.unique(self.F[:self.k])

    # --- reading -----------------------------------------------------------------

    def goto(self, k):
        self.k = int(np.clip(k, 1, len(self.F)))
        if self.slider.val != self.k:
            self.slider.eventson = False  # avoid the slider calling goto back
            self.slider.set_val(self.k)
            self.slider.eventson = True
        self.draw()

    def step_element(self, delta):
        """Previous / next element, selecting only its newest node."""
        self.pause()
        self.k = int(np.clip(self.k + delta, 1, len(self.F)))
        self.selected = [int(self.newest_node[self.k - 1])]
        self.goto(self.k)

    def _on_slider(self, value):
        self.pause()
        self.goto(int(value))

    def _on_drawn(self, _event):
        if self.playing and not self.tick_pending:
            self.tick_pending = True
            self.timer.interval = int(max(MIN_INTERVAL, 1000 / self.rate))
            self.timer.start()

    def _tick(self):
        self.tick_pending = False
        if not self.playing:
            return
        if self.k >= len(self.F):
            self.set_playing(False)
        else:
            self.goto(self.k + max(1, round(self.rate * self.timer.interval / 1000)))

    def set_playing(self, playing):
        self.playing = playing
        if not playing:
            self.timer.stop()
            self.tick_pending = False
        self.draw()  # the next draw_event restarts the timer

    def toggle_play(self):
        if not self.playing and self.k >= len(self.F):
            self.goto(1)  # replay from the start
        self.set_playing(not self.playing)

    def pause(self):
        if self.playing:
            self.set_playing(False)

    def set_rate(self, factor):
        self.rate = float(np.clip(self.rate * factor, 0.25, 5000))
        self.draw()

    # --- editing -----------------------------------------------------------------

    def move(self, axis, sign):
        self.pause()  # editing interrupts the reading
        if not self.selected:
            self.selected = [int(self.F[self.k - 1][0])]  # default: a node of the current element
        nodes = np.array(self.selected)
        direction = self.vertex_normals(nodes) if axis == "normal" else np.eye(3)[axis]
        self.undo_stack.append((nodes, self.V[nodes].copy()))
        self.V[nodes] += sign * self.step * direction
        self.draw()

    def cycle_node(self):
        self.pause()
        element = [int(v) for v in self.F[self.k - 1]]
        last = self.selected[-1] if self.selected else None
        self.selected = [element[(element.index(last) + 1) % 3] if last in element else element[0]]
        self.draw()

    def undo(self):
        if self.undo_stack:
            nodes, positions = self.undo_stack.pop()
            self.V[nodes] = positions
            self.selected = [int(v) for v in nodes]
            self.draw()

    def reset_selected(self):
        nodes = np.array(self.selected, dtype=int)
        if len(nodes) and np.any(self.V[nodes] != self.V0[nodes]):
            self.undo_stack.append((nodes, self.V[nodes].copy()))
            self.V[nodes] = self.V0[nodes]
            self.draw()

    def scale_step(self, factor):
        self.step *= factor
        self.draw()

    def clear_selection(self):
        self.selected = []
        self.draw()

    def toggle_nodes(self):
        self.show_nodes = not self.show_nodes
        self.draw()

    def save(self):
        write_obj(self.out_path, self.V, self.F)
        moved = self.modified_nodes()
        print(f"Saved: {self.out_path} ({len(moved)} node(s) moved)", flush=True)
        for v in moved:
            d = self.V[v] - self.V0[v]
            print(f"  node {v}: displacement ({d[0]:+.4g}, {d[1]:+.4g}, {d[2]:+.4g}), "
                  f"|d| = {np.linalg.norm(d):.4g}", flush=True)
        self.status = f"saved to {self.out_path}"
        self.draw()

    # --- mouse selection ---------------------------------------------------------

    def _on_press(self, event):
        self.press_xy = (event.x, event.y) if event.inaxes is self.ax and event.button == 1 else None

    def _on_release(self, event):
        # A click without dragging selects a node; dragging rotates the view.
        if self.press_xy is None or event.inaxes is not self.ax:
            return
        moved = np.hypot(event.x - self.press_xy[0], event.y - self.press_xy[1])
        if moved > DRAG_THRESHOLD:
            return  # rotation
        v = self.pick(event.x, event.y)
        if v is None:
            self.status = f"no displayed node within {PICK_RADIUS} px of the click"
        else:
            self.pause()  # selecting a node interrupts the reading
            if event.key == "shift":
                if v in self.selected:
                    self.selected.remove(v)
                    self.status = f"node {v} removed from the selection"
                else:
                    self.selected.append(v)
                    self.status = f"node {v} added to the selection"
            else:
                self.selected = [v]
                self.status = f"node {v} selected"
        print(f"click: {self.status}", flush=True)
        self.draw()

    def pick(self, x, y):
        """Displayed node nearest to the screen point (x, y), or None."""
        shown = self.displayed_nodes()
        P = self.V[shown]
        xs, ys, _ = proj3d.proj_transform(P[:, 0], P[:, 1], P[:, 2], self.ax.get_proj())
        screen = self.ax.transData.transform(np.column_stack([xs, ys]))
        dist = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
        near = dist <= PICK_RADIUS
        if not near.any():
            return None
        elev, azim = np.radians(self.ax.elev), np.radians(self.ax.azim)
        eye = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        # Drop hidden nodes (more than one edge behind the frontmost candidate),
        # then take the one nearest to the click.
        depth = P @ eye
        front = near & (depth >= depth[near].max() - self.edge_length)
        candidates = np.nonzero(front)[0]
        return int(shown[candidates[np.argmin(dist[candidates])]])

    # --- rendering ---------------------------------------------------------------

    def draw(self):
        k, V, F = self.k, self.V, self.F
        normals = self.face_normals()[:k]
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
        shade = 0.55 + 0.45 * np.abs(normals @ LIGHT)[:, None]
        moved = self.modified_nodes()
        touched = np.isin(F[:k], moved).any(axis=1)[:, None]
        colors = np.where(touched, MODIFIED_COLOR, BASE_COLOR) * shade
        self.surface.set_verts(list(V[F[:k]]))
        self.surface.set_facecolor(np.column_stack([colors, np.full(k, 0.9)]))
        self.current.set_verts([V[F[k - 1]]])

        shown = self.displayed_nodes()
        self.node_pts.set_data_3d(*(V[shown].T if self.show_nodes else ([], [], [])))
        self.modified_pts.set_data_3d(*V[moved].T)
        sel = np.array(self.selected, dtype=int)
        self.selected_pts.set_data_3d(*V[sel].T)
        # Dashed line from each selected node's original position to its current one.
        trail = np.full((3 * len(sel), 3), np.nan)
        trail[0::3], trail[1::3] = self.V0[sel], V[sel]
        self.trails.set_data_3d(*trail.T)

        element = F[k - 1]
        lines = [
            f"element {k}/{len(F)}   nodes {element[0]}, {element[1]}, {element[2]}",
            f"{'playing' if self.playing else 'paused'}: {self.rate:.3g} elements/s",
            f"step: {self.step:.4g}  ({self.step / self.edge_length:.0%} of an edge)",
            f"moved nodes: {len(moved)}   selected: {len(sel)}",
        ]
        for v in sel[:6]:
            p, d = V[v], V[v] - self.V0[v]
            lines.append(f"  node {v:<5} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})"
                         f"  |d| = {np.linalg.norm(d):.4g}")
        if len(sel) > 6:
            lines.append(f"  … and {len(sel) - 6} more")
        if self.status:
            lines.append(self.status)
            self.status = None
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in MOVES:
            self.move(*MOVES[event.key])
            return
        actions = {
            " ": self.toggle_play,
            "right": lambda: self.step_element(1),
            "left": lambda: self.step_element(-1),
            "home": lambda: self.goto(1),
            "end": lambda: self.goto(len(self.F)),
            "+": lambda: self.set_rate(2),
            "-": lambda: self.set_rate(0.5),
            "tab": self.cycle_node,
            "]": lambda: self.scale_step(2),
            "[": lambda: self.scale_step(0.5),
            "u": self.undo,
            "r": self.reset_selected,
            "escape": self.clear_selection,
            "v": self.toggle_nodes,
            "s": self.save,
        }
        if event.key in actions:
            actions[event.key]()


def load(source):
    if source in BUILTIN:
        return BUILTIN[source]()
    return read_mesh(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", nargs="?", default="sphere",
                        help=f".obj/.off/.stl file or generated mesh: {', '.join(BUILTIN)}")
    parser.add_argument("--rate", type=float, default=None, help="elements displayed per second")
    parser.add_argument("--out", default=None, help="output OBJ file (key s)")
    args = parser.parse_args()

    vertices, faces = load(args.mesh)
    out = args.out or f"{Path(args.mesh).stem}_edited.obj"
    print(f"{args.mesh}: {len(vertices)} nodes, {len(faces)} elements", flush=True)
    editor = MeshEditor(vertices, faces, out_path=out, rate=args.rate)  # noqa: F841 (kept alive during show)
    plt.show()


if __name__ == "__main__":
    main()
