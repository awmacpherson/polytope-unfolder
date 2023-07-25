# Hacking polytope-unfolder

## Structure

The project may be divided into two parts.

### Backend

This consists of Python library code contained in the `/tope` directory plus unit/integration tests in `/tests`. In more detail:

- `tope.py`. Definition of `Tope` object which is the centrepiece of the library.
- `graph.py`. `Graph` object + methods for finding a spanning tree.
- `net.py`. `Net` object, which intuitively should be a rooted tree (implemented as a `Graph`) with nodes labelled by `Topes` in a common ambient space and edges labelled by their intersections. Most important method here is `unfold()`. 

  The actual implementation is a bit of a cludge which also carries a reference to the `Tope` from which it is constructed. The goal is to migrate to something looking more like the `Net2` class, which is currently unused, at some point.
- `orth.py`. All the painful linear algebra stuff is in here. It's currently a big mess of deprecated functions that need to be removed.
- `affine.py`. Not used.
- `pcas.py`. Methods for accessing the public polytope database using the Fanosearch [pcas](https://www.pcas.xyz) package. Currently this gateway isn't accessible so this is useless.
- `plot.py`. Convenience methods for matplotlib.
- `stl.py`. Convenience methods for exporting STL files.

The dependencies are nearly all very standard libraries used in data science or scientific computing. The exception is perhaps `pypoman`, a polytope library based on `cvxopt` and `pycddlib`. The latter is a set of Python bindings to the widely used C library `cddlib` which implements the "double description method" for enumerating supporting halfspaces of the convex hull of a set of vertices (or dually, enumerating vertices of the complement of a set of halfspaces). This algorithm is exposed through the `pypoman` function `compute_polytope_halfspaces(vertices)` which is used in the constructor `Tope.from_vertices()` as a starting point for computing the combinatorics of faces.

### Frontend

Consists of the file `notebooks/combined.ipynb`. Other notebooks are older stuff that should be deleted once I'm sure we got everything worth salvaging from there. 

This repo uses the Jupyter plugin [Jupytext](https://jupytext.readthedocs.io) to maintain an ordinary Python script `notebooks/combined.py`. It has the same functionality as the notebook but can be passed directly to the Python interpreter. This is used by the integration test `tests/run_notebook.py`. During development in the Jupyter web UI, you should have Jupytext installed and the "paired percent script" option checked in the UI to ensure that jupyter automatically updates `combined.py` when you save.

The animation export in combined.ipynb requires `ffmpeg`. The integration test `tests/run_notebook.py` will fail if this is not installed.

To avoid having to debug local environment issues, one can run the notebook on `mybinder.org`, a public BinderHub instance that pulls in notebooks from a GitHub repo. (This is useful for users but not ideal for testing as one must commit + push changes before they are visible to the server.)

BinderHub is a derivative of JupyterHub that runs in a Kubernetes pod. The file `/apt.txt` is an environment definition file for BinderHub.

## Virtual environment

The project is configured using `pyproject.toml` to support using [PDM](https://pdm.fming.dev/latest/) to manage virtual environments and validate that (Python) dependencies are solvable.

You can test notebooks in the virtual environment with `pdm run juypter notebook`. Since jupyter is declared as a dependency this will invoke the local (to the virtual env) jupyter installation, not the system one. (It would also be possible, but a bit more fiddly, to use the system jupyter with a local Python kernel.)

Notebooks on the mybinder.org portal run in a conda environment.

## Test + deploy process
1. Run `tox run`. (Get comfortable as this can take a while.)
2. Commit + push to GitHub.
3. Load combined.ipynb on mybinder.org and do one full runthrough.

Don't forget to add your name to the authors list in `pyproject.toml` when you make your first contribution!

Ideas to improve this process:
1. Have these tests triggered by git hooks so testing happens automatically on commit.
