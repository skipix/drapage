"""Headless tests: python test_mesh_editor.py"""

import tempfile
from pathlib import Path
from types import SimpleNamespace as Event

import matplotlib
import matplotlib.backend_bases

matplotlib.use("Agg")
import numpy as np
from mpl_toolkits.mplot3d import proj3d

import mesh_editor
from mesh_io import read_mesh, write_obj
from meshes import sphere_mesh


def test_io(tmp):
    V, F = sphere_mesh(n_lat=12, n_lon=24)
    write_obj(tmp / "s.obj", V, F)
    V2, F2 = read_mesh(tmp / "s.obj")
    assert np.allclose(V, V2) and (F == F2).all()

    off = f"OFF\n{len(V)} {len(F)} 0\n" + "".join(f"{a} {b} {c}\n" for a, b, c in V)
    off += "".join(f"3 {a} {b} {c}\n" for a, b, c in F)
    (tmp / "s.off").write_text(off)
    V3, F3 = read_mesh(tmp / "s.off")
    assert np.allclose(V, V3) and (F == F3).all()

    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = V[F]
    (tmp / "s.stl").write_bytes(b"\0" * 80 + np.uint32(len(F)).tobytes() + rec.tobytes())
    V4, F4 = read_mesh(tmp / "s.stl")
    assert len(V4) == len(V) and np.allclose(V4[F4], V[F].astype(np.float32))

    ascii_stl = "solid t\n" + "".join(
        "facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t)
        + "endloop\nendfacet\n" for t in V[F]) + "endsolid t\n"
    (tmp / "a.stl").write_text(ascii_stl)
    V5, F5 = read_mesh(tmp / "a.stl")
    assert len(V5) == len(V) and np.allclose(V5[F5], V[F])

    (tmp / "q.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\nf -4 -3 -1\n")
    assert read_mesh(tmp / "q.obj")[1].tolist() == [[0, 1, 2], [0, 2, 3], [0, 1, 3]]


def click(editor, node, shift=False):
    p = editor.V[node]
    xs, ys, _ = proj3d.proj_transform([p[0]], [p[1]], [p[2]], editor.ax.get_proj())
    sx, sy = editor.ax.transData.transform([[xs[0], ys[0]]])[0]
    event = Event(x=sx, y=sy, inaxes=editor.ax, button=1, key="shift" if shift else None)
    editor._on_press(event)
    editor._on_release(event)


def visible_nodes(editor, count):
    """Nodes facing the viewer, far enough apart on screen to be clicked unambiguously."""
    elev, azim = np.radians(editor.ax.elev), np.radians(editor.ax.azim)
    eye = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    order = np.argsort(-(editor.V @ eye))
    return [int(v) for v in order[: 4 * count : 4]]


def test_editor(tmp):
    V, F = mesh_editor.load("sphere")
    ed = mesh_editor.MeshEditor(V, F, out_path=tmp / "out.obj", play=False)
    key = lambda k: ed.on_key(Event(key=k))
    key("end")  # whole mesh displayed
    ed.fig.canvas.draw()

    key("x")  # nothing selected: moves the first node of the current element
    assert ed.modified_nodes().tolist() == [F[-1][0]]
    key("u"); key("escape")
    assert len(ed.modified_nodes()) == 0

    a, b, c = visible_nodes(ed, 3)
    click(ed, a)
    assert ed.selected == [a]
    click(ed, b, shift=True)
    click(ed, c, shift=True)
    assert ed.selected == [a, b, c]
    click(ed, b, shift=True)  # shift+click again removes
    assert ed.selected == [a, c]

    key("x"); key("x"); key("Y")
    d = ed.V[[a, c]] - ed.V0[[a, c]]
    assert np.allclose(d, [[2 * ed.step, -ed.step, 0]] * 2)

    key("]"); key("n")  # along each node's normal (outwards on a sphere)
    radial = np.einsum("ij,ij->i", ed.V[[a, c]] - ed.V0[[a, c]], ed.V0[[a, c]])
    assert (radial > 0).all()

    key("u")
    assert np.allclose(ed.V[[a, c]] - ed.V0[[a, c]], [[2 * ed.step / 2, -ed.step / 2, 0]] * 2)

    key("escape")
    assert ed.selected == []
    click(ed, a)
    key("r")
    assert np.allclose(ed.V[a], ed.V0[a]) and ed.modified_nodes().tolist() == [c]

    key("s")
    Vs, Fs = read_mesh(tmp / "out.obj")
    assert (Fs == F).all() and np.allclose(Vs, ed.V, atol=1e-8)

    # Clicking where a front node hides a back node must pick the front one.
    ed.fig.canvas.draw()
    elev, azim = np.radians(ed.ax.elev), np.radians(ed.ax.azim)
    eye = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    for v in range(len(V)):
        p = ed.V[v]
        xs, ys, _ = proj3d.proj_transform([p[0]], [p[1]], [p[2]], ed.ax.get_proj())
        sx, sy = ed.ax.transData.transform([[xs[0], ys[0]]])[0]
        assert ed.V[ed.pick(sx, sy)] @ eye >= p @ eye - ed.edge_length
    ed.fig.savefig(tmp / "editor.png")


def test_playback(tmp):
    """Read element by element, interrupt, move a node, continue."""
    V, F = mesh_editor.load("sphere")
    ed = mesh_editor.MeshEditor(V, F)
    key = lambda k: ed.on_key(Event(key=k))
    assert ed.playing and ed.k == 1  # reading starts on its own

    for _ in range(10):
        ed._tick()
    k_interrupt = ed.k
    assert k_interrupt > 1 and ed.playing

    # Only nodes of displayed elements can be clicked.
    ed.fig.canvas.draw()
    hidden = int(np.setdiff1d(np.arange(len(V)), ed.displayed_nodes())[0])
    assert ed.pick(*screen_xy(ed, hidden)) != hidden

    # Clicking a displayed node interrupts the reading.
    node = int(F[k_interrupt - 1][1])
    click(ed, node)
    assert not ed.playing and ed.selected == [node] and ed.k == k_interrupt
    key("z"); key("z")
    assert np.isclose(ed.V[node][2] - ed.V0[node][2], 2 * ed.step)

    # Moving while playing also interrupts.
    key(" ")
    assert ed.playing
    key("tab")  # next node of the current element
    assert not ed.playing and ed.selected == [int(F[ed.k - 1][2])]
    key("N")

    # Continue: elements displayed after the interruption use the edited geometry.
    drawn = []
    set_verts = ed.surface.set_verts
    ed.surface.set_verts = lambda verts: (drawn.append(np.array(verts)), set_verts(verts))
    key(" ")
    for _ in range(1000):
        ed._tick()
    assert ed.k == len(F) and not ed.playing  # stops at the end
    later = next(f for f in range(k_interrupt, len(F)) if node in F[f])
    corner = list(F[later]).index(node)
    assert np.allclose(drawn[-1][later][corner], ed.V[node])
    assert not np.allclose(drawn[-1][later][corner], ed.V0[node])
    assert sorted(ed.modified_nodes().tolist()) == sorted({node, int(F[k_interrupt - 1][2])})

    key(" ")  # at the end, space replays from the start
    assert ed.k == 1 and ed.playing


def test_arrows_select_newest_node():
    V, F = mesh_editor.load("sphere")
    ed = mesh_editor.MeshEditor(V, F)
    key = lambda k: ed.on_key(Event(key=k))
    ed.selected = [5, 6, 7]
    seen = set(F[0])
    for _ in range(40):
        key("right")
        new = [int(v) for v in F[ed.k - 1] if v not in seen]
        seen |= set(F[ed.k - 1])
        assert not ed.playing and len(ed.selected) == 1
        if new:  # the element brings a new node: that one is selected
            assert ed.selected == [new[-1]], (ed.k, F[ed.k - 1], new, ed.selected)
        assert ed.selected[0] in F[ed.k - 1]
    k = ed.k
    key("left")
    assert ed.k == k - 1 and ed.selected == [int(ed.newest_node[k - 2])]
    for _ in range(100):
        key("left")
    assert ed.k == 1 and ed.selected[0] in F[0]


def screen_xy(editor, node):
    p = editor.V[node]
    xs, ys, _ = proj3d.proj_transform([p[0]], [p[1]], [p[2]], editor.ax.get_proj())
    return editor.ax.transData.transform([[xs[0], ys[0]]])[0]


def test_editor_survives_without_reference():
    """Regression: main() used to drop the editor, so the window ignored all input."""
    import gc

    import matplotlib.pyplot as plt

    mesh_editor.MeshEditor(*mesh_editor.load("sphere"))
    gc.collect()
    fig = plt.gcf()
    fig.canvas.draw()
    p = fig.mesh_editor.V[0]
    xs, ys, _ = proj3d.proj_transform([p[0]], [p[1]], [p[2]], fig.axes[0].get_proj())
    sx, sy = fig.axes[0].transData.transform([[xs[0], ys[0]]])[0]
    for name in ("button_press_event", "button_release_event"):
        event = matplotlib.backend_bases.MouseEvent(name, fig.canvas, sx, sy, button=1)
        fig.canvas.callbacks.process(name, event)
    assert fig.mesh_editor.selected, "click through the canvas did not select a node"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_io(Path(d))
        print("io ok")
        test_editor(Path(d))
        print("editor ok")
        test_playback(Path(d))
        print("playback ok")
        test_arrows_select_newest_node()
        print("arrows ok")
        test_editor_survives_without_reference()
        print("regression ok")
