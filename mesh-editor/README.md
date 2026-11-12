# mesh-editor

Interactive mesh reader / editor. The mesh is read and displayed element by element
(triangles, in file order). You can interrupt the reading at any moment, slightly move
some nodes, then continue: the next elements are displayed with the edited geometry.

Reads `.obj`, `.off` and `.stl` (ASCII or binary) with numpy only (polygons are split into
triangles), or generates a sphere, a cylinder or a bent cylinder. The edited mesh is saved as OBJ.

```bash
uv venv --python 3.14 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python mesh_editor.py                     # sphere
.venv/bin/python mesh_editor.py part.stl --rate 50 --out part_fixed.obj
.venv/bin/python test_mesh_editor.py                # headless tests
```

The Python used must have `tkinter` (the interpreters installed by uv do).

### Reading

| Input | Action |
|---|---|
| `space` | play / pause (reading starts on its own; at the end, replays from the start) |
| `→` / `←` | next / previous element; selects only that element's newest node (the one that appeared last), replacing the selection |
| `home` / `end` | first / last element (`end` shows the whole mesh) |
| `+` / `-` | faster / slower (the whole mesh takes ~30 s by default) |
| slider | jump to an element |

### Editing (selecting or moving a node interrupts the reading)

| Input | Action |
|---|---|
| click | select the displayed node nearest to the cursor |
| shift + click | add / remove a node from the selection |
| drag | rotate the view |
| `tab` | select the next node of the current element |
| `x` `y` `z` / `X` `Y` `Z` | move the selected nodes by +step / −step along the axis |
| `n` / `N` | move each selected node along its own normal (outwards / inwards) |
| `]` / `[` | double / halve the step (10% of the mean edge length at start) |
| `u` | undo the last move |
| `r` | reset the selected nodes to their original position |
| `esc` | clear the selection |
| `v` | show / hide the nodes |
| `s` | save the edited mesh (`<name>_edited.obj` by default) and print the displacements |

The last element read is orange. Selected nodes are yellow, with a dashed line from their
original position; moved nodes are red and the faces touching them salmon.
