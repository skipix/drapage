"""Génération de maillages triangulaires simples (sommets V×3, faces F×3)."""

import numpy as np


def sphere_mesh(radius=1.0, n_lat=40, n_lon=80):
    """Sphère UV centrée à l'origine, faces orientées vers l'extérieur."""
    theta = np.linspace(0, np.pi, n_lat + 1)[1:-1]          # latitudes sans les pôles
    phi = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    ring = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)
    vertices = np.vstack([[0, 0, 1], ring.reshape(-1, 3), [0, 0, -1]]) * radius

    def idx(i, j):  # i: anneau (0..n_lat-2), j: longitude
        return 1 + i * n_lon + (j % n_lon)

    faces = []
    south = len(vertices) - 1
    for j in range(n_lon):
        faces.append([0, idx(0, j), idx(0, j + 1)])
        faces.append([south, idx(n_lat - 2, j + 1), idx(n_lat - 2, j)])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            faces += [[a, c, d], [a, d, b]]
    return vertices, np.array(faces)


def cylinder_mesh(radius=1.0, height=4.0, n_theta=80, n_h=40):
    """Cylindre fermé d'axe x, centré à l'origine, faces orientées vers l'extérieur."""
    phi = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    xs = np.linspace(-height / 2, height / 2, n_h + 1)
    x, p = np.meshgrid(xs, phi, indexing="ij")
    side = np.stack([x, radius * np.sin(p), radius * np.cos(p)], axis=-1).reshape(-1, 3)
    vertices = np.vstack([side, [-height / 2, 0, 0], [height / 2, 0, 0]])
    c0, c1 = len(vertices) - 2, len(vertices) - 1

    def idx(i, j):
        return i * n_theta + (j % n_theta)

    faces = []
    for i in range(n_h):
        for j in range(n_theta):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            faces += [[a, c, d], [a, d, b]]
    for j in range(n_theta):
        faces.append([c0, idx(0, j), idx(0, j + 1)])
        faces.append([c1, idx(n_h, j + 1), idx(n_h, j)])
    return vertices, np.array(faces)


def bent_cylinder_mesh(radius=1.0, length=4.0, bend_radius=6.0, n_theta=80, n_h=60):
    """Cylindre fermé incurvé en U : son axe est un arc de cercle de rayon `bend_radius`.

    L'axe part de l'origine selon x et ses extrémités remontent vers +z ;
    `length` est la longueur de l'axe. Faces orientées vers l'extérieur.
    """
    half = length / (2 * bend_radius)                    # demi-angle de l'arc
    t = np.linspace(-half, half, n_h + 1)
    axis = np.stack([bend_radius * np.sin(t), np.zeros_like(t), bend_radius * (1 - np.cos(t))], -1)
    up = np.stack([-np.sin(t), np.zeros_like(t), np.cos(t)], -1)   # vers le centre de courbure
    side = np.array([0.0, 1.0, 0.0])
    phi = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ring = (np.sin(phi)[None, :, None] * side + np.cos(phi)[None, :, None] * up[:, None, :]) * radius
    vertices = np.vstack([(axis[:, None, :] + ring).reshape(-1, 3), axis[0], axis[-1]])
    c0, c1 = len(vertices) - 2, len(vertices) - 1

    def idx(i, j):
        return i * n_theta + (j % n_theta)

    faces = []
    for i in range(n_h):
        for j in range(n_theta):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            faces += [[a, c, d], [a, d, b]]
    for j in range(n_theta):
        faces.append([c0, idx(0, j), idx(0, j + 1)])
        faces.append([c1, idx(n_h, j + 1), idx(n_h, j)])
    return vertices, np.array(faces)
