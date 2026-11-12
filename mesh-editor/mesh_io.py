"""Lecture / écriture de maillages triangulaires (OBJ, OFF, STL) sans dépendance externe.

Chaque lecteur renvoie (vertices V×3 float, faces F×3 int). Les polygones à plus de
trois sommets sont découpés en éventail ; l'ordre des faces du fichier est conservé.
"""

from pathlib import Path

import numpy as np


def _fan(polygon):
    return [[polygon[0], polygon[k], polygon[k + 1]] for k in range(1, len(polygon) - 1)]


def read_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            # "f 1 2 3", "f 1/1/1 2/2/2 …" ; indices à partir de 1, négatifs = relatifs.
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
            faces += _fan(idx)
    return np.array(vertices, dtype=float), np.array(faces, dtype=int).reshape(-1, 3)


def read_off(path):
    tokens = []
    for line in Path(path).read_text().splitlines():
        tokens += line.split("#")[0].split()
    if not tokens[0].endswith("OFF"):
        raise ValueError(f"{path} : en-tête OFF manquant")
    pos = 1
    n_v, n_f = int(tokens[pos]), int(tokens[pos + 1])
    pos += 3
    vertices = np.array(tokens[pos:pos + 3 * n_v], dtype=float).reshape(n_v, 3)
    pos += 3 * n_v
    faces = []
    for _ in range(n_f):
        k = int(tokens[pos])
        faces += _fan([int(t) for t in tokens[pos + 1:pos + 1 + k]])
        pos += 1 + k
    return vertices, np.array(faces, dtype=int).reshape(-1, 3)


def read_stl(path):
    data = Path(path).read_bytes()
    n_bin = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if len(data) == 84 + 50 * n_bin:  # STL binaire : la taille correspond exactement
        record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        corners = np.frombuffer(data, dtype=record, count=n_bin, offset=84)["v"].astype(float)
    else:
        corners = np.array(
            [line.split()[1:4] for line in data.decode().splitlines() if line.strip().startswith("vertex")],
            dtype=float,
        ).reshape(-1, 3, 3)
    # STL ne partage pas les sommets : on fusionne les coins identiques.
    vertices, inverse = np.unique(corners.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


READERS = {".obj": read_obj, ".off": read_off, ".stl": read_stl}


def read_mesh(path):
    suffix = Path(path).suffix.lower()
    if suffix not in READERS:
        raise ValueError(f"format non reconnu : {suffix} (attendu : {', '.join(READERS)})")
    vertices, faces = READERS[suffix](path)
    if len(faces) == 0:
        raise ValueError(f"{path} : aucune face")
    return vertices, faces


def write_obj(path, vertices, faces):
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    Path(path).write_text("\n".join(lines) + "\n")
