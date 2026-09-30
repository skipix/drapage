# drapage

Drapé géométrique d'un tissu sur une surface 3D (algorithme *fishnet* / filet de Tchebychev).

Le tissu est une grille N×M de nœuds reliés par des fils inextensibles : seule la
variation d'angle entre chaîne et trame (cisaillement) est permise. L'objet est un
maillage triangulaire (`vertices` V×3, `faces` F×3). La sortie est la position 3D
de chaque nœud du tissu posé sur l'objet.

```python
from fishnet import drape
P = drape(vertices, faces, shape=(21, 21), spacing=0.1)  # -> (21, 21, 3), NaN si non posé
```

## Démo

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python demo.py   # sphère + cylindre -> drape.png
```

![drapé](drape.png)

## Limites

Maillage fermé, normales vers l'extérieur, forme à peu près convexe (sphère, cylindre…).
