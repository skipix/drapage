"""Démo : drapé fishnet d'un tissu sur une sphère et sur un cylindre."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fishnet import drape, shear_angles
from meshes import cylinder_mesh, sphere_mesh


def report(name, P, spacing):
    ok = ~np.isnan(P).any(axis=-1)
    warp = np.linalg.norm(np.diff(P, axis=0), axis=-1)
    weft = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    edges = np.concatenate([warp[~np.isnan(warp)], weft[~np.isnan(weft)]])
    shear = shear_angles(P)
    shear = shear[~np.isnan(shear)]
    print(f"[{name}] nœuds posés : {ok.sum()}/{ok.size}")
    print(f"  longueur des fils : {edges.min():.5f} .. {edges.max():.5f} (cible {spacing})")
    print(f"  angle chaîne/trame : {shear.min():.1f}° .. {shear.max():.1f}°")


def plot(ax, vertices, faces, P, title):
    ax.plot_trisurf(*vertices.T, triangles=faces, color="lightgray", alpha=0.4, linewidth=0)
    for i in range(P.shape[0]):
        ax.plot(*P[i].T, color="tab:blue", lw=0.8)
    for j in range(P.shape[1]):
        ax.plot(*P[:, j].T, color="tab:red", lw=0.8)
    ax.set_title(title)
    ax.set_box_aspect(np.ptp(vertices, axis=0))


def main():
    cases = [
        ("sphère", *sphere_mesh(radius=1.0), (21, 21), 0.1),
        ("cylindre", *cylinder_mesh(radius=1.0, height=4.0), (25, 25), 0.12),
    ]
    fig = plt.figure(figsize=(12, 6))
    for k, (name, vertices, faces, shape, spacing) in enumerate(cases):
        t0 = time.perf_counter()
        P = drape(vertices, faces, shape, spacing)
        print(f"[{name}] {time.perf_counter() - t0:.1f} s, sortie {P.shape}")
        report(name, P, spacing)
        plot(fig.add_subplot(1, 2, k + 1, projection="3d"), vertices, faces, P, name)
    fig.tight_layout()
    fig.savefig("drape.png", dpi=120)
    print("image : drape.png")


if __name__ == "__main__":
    main()
