"""Drapé géométrique d'un tissu sur un maillage par l'algorithme "fishnet".

Le tissu est modélisé comme un filet de Tchebychev : une grille N×M de nœuds
dont les fils (chaîne selon l'axe 0, trame selon l'axe 1) sont inextensibles,
de longueur `spacing` entre deux nœuds voisins. Seul l'angle entre les fils
peut varier (cisaillement).

Étapes :
1. Poser un nœud de départ sur la surface.
2. Tracer deux fils générateurs (la ligne et la colonne passant par ce nœud)
   le long de géodésiques de la surface.
3. Remplir chaque cellule : connaissant A = P[i-1, j], B = P[i, j-1] et
   C = P[i-1, j-1], le nœud P[i, j] est le point de la surface situé à la
   distance `spacing` de A et de B, du côté opposé à C.

Limites : le signe de la distance au maillage utilise la normale de la face
la plus proche, ce qui suppose un maillage fermé, bien orienté (normales
vers l'extérieur) et raisonnablement convexe (sphère, cylindre…).
"""

from dataclasses import dataclass

import numpy as np


class MeshSurface:
    """Maillage triangulaire avec requêtes de point le plus proche et de distance signée."""

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=int)
        self.a, self.b, self.c = (self.vertices[self.faces[:, k]] for k in range(3))
        normals = np.cross(self.b - self.a, self.c - self.a)
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        self.centroids = (self.a + self.b + self.c) / 3
        # Rayon englobant de chaque triangle autour de son centroïde.
        self.face_radius = np.max(
            np.linalg.norm(np.stack([self.a, self.b, self.c]) - self.centroids, axis=-1), axis=0
        )

    def faces_near(self, center, radius):
        """Indices des faces pouvant contenir un point à moins de `radius` de `center`."""
        dist = np.linalg.norm(self.centroids - center, axis=1)
        return np.nonzero(dist <= radius + self.face_radius)[0]

    def top_point(self, x, y):
        """Point le plus haut du maillage sur la verticale (x, y), ou None."""
        a, b, c = self.a[:, :2], self.b[:, :2], self.c[:, :2]
        p = np.array([x, y])
        # Coordonnées barycentriques de (x, y) dans la projection de chaque face.
        det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = ((b[:, 1] - c[:, 1]) * (p[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[1] - c[:, 1])) / det
            l2 = ((c[:, 1] - a[:, 1]) * (p[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[1] - c[:, 1])) / det
            l3 = 1 - l1 - l2
            z = l1 * self.a[:, 2] + l2 * self.b[:, 2] + l3 * self.c[:, 2]
        hit = (det != 0) & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not hit.any():
            return None
        return np.array([x, y, np.max(z[hit])])

    def closest_point(self, points, face_idx=None):
        """Point le plus proche sur le maillage et normale de la face correspondante.

        points : (P, 3) -> (P, 3) points, (P, 3) normales.
        """
        points = np.atleast_2d(points)
        if face_idx is None or len(face_idx) == 0:
            face_idx = np.arange(len(self.faces))
        a, b, c = self.a[face_idx], self.b[face_idx], self.c[face_idx]
        q = _closest_on_triangles(points[:, None, :], a[None], b[None], c[None])
        d2 = np.sum((q - points[:, None, :]) ** 2, axis=-1)
        best = np.argmin(d2, axis=1)
        rows = np.arange(len(points))
        return q[rows, best], self.normals[face_idx[best]]

    def signed_distance(self, points, face_idx=None):
        """Distance signée (> 0 à l'extérieur, < 0 à l'intérieur)."""
        points = np.atleast_2d(points)
        q, n = self.closest_point(points, face_idx)
        diff = points - q
        return np.sign(np.sum(diff * n, axis=1)) * np.linalg.norm(diff, axis=1)


def _closest_on_triangles(p, a, b, c):
    """Point le plus proche de p sur chaque triangle (Ericson, vectorisé par diffusion)."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = np.sum(ab * ap, -1), np.sum(ac * ap, -1)
    d3, d4 = np.sum(ab * bp, -1), np.sum(ac * bp, -1)
    d5, d6 = np.sum(ab * cp, -1), np.sum(ac * cp, -1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v = (vb / denom)[..., None]
        w = (vc / denom)[..., None]
        res = a + ab * v + ac * w  # intérieur du triangle

        # Arêtes, puis sommets (régions de Voronoï disjointes).
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]
        res = np.where(m[..., None], b + (c - b) * t, res)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = (d2 / (d2 - d6))[..., None]
        res = np.where(m[..., None], a + ac * t, res)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = (d1 / (d1 - d3))[..., None]
        res = np.where(m[..., None], a + ab * t, res)

    shape = res.shape
    res = np.where(((d6 >= 0) & (d5 <= d6))[..., None], np.broadcast_to(c, shape), res)
    res = np.where(((d3 >= 0) & (d4 <= d3))[..., None], np.broadcast_to(b, shape), res)
    res = np.where(((d1 <= 0) & (d2 <= 0))[..., None], np.broadcast_to(a, shape), res)
    return res


def _circle_surface_point(surface, center, radius, e1, e2, guess, n_samples=72, n_bisect=30):
    """Intersection du cercle (center, radius, plan (e1, e2)) avec la surface.

    Renvoie la racine la plus proche de `guess`, ou None s'il n'y en a pas.
    """
    # Le point de surface le plus proche d'un point du cercle est à moins de 2·radius du centre.
    face_idx = surface.faces_near(center, 2 * radius + 1e-9)
    if len(face_idx) == 0:
        return None

    def point(theta):
        theta = np.atleast_1d(theta)[:, None]
        return center + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)

    thetas = np.linspace(-np.pi, np.pi, n_samples + 1)
    sd = surface.signed_distance(point(thetas), face_idx)
    brackets = np.nonzero(np.sign(sd[:-1]) * np.sign(sd[1:]) <= 0)[0]
    if len(brackets) == 0:
        return None

    roots = []
    for k in brackets:
        lo, hi, f_lo = thetas[k], thetas[k + 1], sd[k]
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f_mid = surface.signed_distance(point(mid), face_idx)[0]
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        roots.append(point(0.5 * (lo + hi))[0])
    roots = np.array(roots)
    return roots[np.argmin(np.linalg.norm(roots - guess, axis=1))]


def _unit(v):
    return v / np.linalg.norm(v)


def _geodesic_line(surface, start, direction, spacing, n_steps):
    """Suit une géodésique discrète : pas de corde `spacing`, direction la plus droite possible."""
    points = []
    prev = start
    _, normal = surface.closest_point(prev)
    tangent = _unit(direction - np.dot(direction, normal[0]) * normal[0])
    for _ in range(n_steps):
        # Le nouveau point reste dans le plan (tangente, normale) : pas de déviation latérale.
        q = _circle_surface_point(
            surface, prev, spacing, tangent, normal[0], guess=prev + spacing * tangent
        )
        if q is None:
            break
        chord = _unit(q - prev)
        _, normal = surface.closest_point(q)
        tangent = _unit(chord - np.dot(chord, normal[0]) * normal[0])
        points.append(q)
        prev = q
    points += [np.full(3, np.nan)] * (n_steps - len(points))
    return points


def _fishnet_node(surface, a, b, c, spacing):
    """Nœud à distance `spacing` de a et b sur la surface, à l'opposé de c."""
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return np.full(3, np.nan)
    ab = b - a
    length = np.linalg.norm(ab)
    if length >= 2 * spacing or length == 0:
        return np.full(3, np.nan)
    u = ab / length
    mid = 0.5 * (a + b)
    radius = np.sqrt(spacing**2 - (length / 2) ** 2)
    away = mid - c
    e1 = _unit(away - np.dot(away, u) * u)
    e2 = np.cross(u, e1)
    q = _circle_surface_point(surface, mid, radius, e1, e2, guess=a + b - c)
    return np.full(3, np.nan) if q is None else q


@dataclass
class Step:
    """Pose d'une ligne du tissu le long de l'axe X (axe 0) : la colonne P[:, j]."""

    line: int
    phase: str


def drape(
    vertices,
    faces,
    shape,
    spacing,
    start=None,
    warp_direction=(1.0, 0.0, 0.0),
    return_history=False,
):
    """Drape une grille de tissu `shape` = (N, M) sur le maillage.

    vertices, faces : maillage de l'objet (V×3, F×3).
    spacing         : distance entre deux nœuds voisins du tissu.
    start           : point de départ (projeté sur la surface) ; par défaut le
                      point le plus haut de l'objet à la verticale du centre
                      de sa boîte englobante.
    warp_direction  : direction des fils de chaîne (axe 0 de la grille) au départ.
    return_history  : renvoie aussi la liste des `Step`, une par ligne, dans
                      l'ordre de pose.

    Le tissu est construit ligne par ligne le long de son axe X (axe 0 de la
    grille, fils de chaîne) : d'abord la ligne centrale P[:, jc], puis les
    lignes jc+1 … M-1, puis jc-1 … 0. Chaque ligne s'appuie sur la précédente.

    Renvoie un array (N, M, 3) ; NaN pour les nœuds qui n'ont pas pu être posés.
    """
    surface = MeshSurface(vertices, faces)
    n, m = shape
    ic, jc = n // 2, m // 2
    P = np.full((n, m, 3), np.nan)
    history = []

    if start is None:
        lo, hi = surface.vertices.min(axis=0), surface.vertices.max(axis=0)
        center = (lo + hi) / 2
        start = surface.top_point(center[0], center[1])
        if start is None:
            start = [center[0], center[1], hi[2]]
    p0, normal = surface.closest_point(np.asarray(start, dtype=float))
    p0, normal = p0[0], normal[0]

    warp = np.asarray(warp_direction, dtype=float)
    warp = _unit(warp - np.dot(warp, normal) * normal)
    weft = np.cross(normal, warp)

    # Ligne centrale : fil de chaîne générateur le long de l'axe X.
    P[ic, jc] = p0
    P[ic + 1:, jc] = _geodesic_line(surface, p0, warp, spacing, n - 1 - ic)
    P[:ic, jc] = _geodesic_line(surface, p0, -warp, spacing, ic)[::-1]
    history.append(Step(jc, "ligne centrale (générateur)"))

    # Fil de trame générateur : il donne le nœud central (ic, j) de chaque ligne.
    spine = {
        1: _geodesic_line(surface, p0, weft, spacing, m - 1 - jc),
        -1: _geodesic_line(surface, p0, -weft, spacing, jc),
    }

    # Lignes suivantes : nœud central puis propagation vers les deux bouts.
    for dj in (1, -1):
        for k, j in enumerate(range(jc + dj, m if dj > 0 else -1, dj)):
            P[ic, j] = spine[dj][k]
            for di in (1, -1):
                for i in range(ic + di, n if di > 0 else -1, di):
                    P[i, j] = _fishnet_node(
                        surface, P[i - di, j], P[i, j - dj], P[i - di, j - dj], spacing
                    )
            history.append(Step(j, f"ligne côté {'+' if dj > 0 else '−'}"))
    return (P, history) if return_history else P


def shear_angles(P):
    """Angle (degrés) entre chaîne et trame à chaque nœud intérieur ; 90° = pas de cisaillement."""
    warp = P[2:, 1:-1] - P[:-2, 1:-1]
    weft = P[1:-1, 2:] - P[1:-1, :-2]
    cos = np.sum(warp * weft, -1) / (np.linalg.norm(warp, axis=-1) * np.linalg.norm(weft, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))
