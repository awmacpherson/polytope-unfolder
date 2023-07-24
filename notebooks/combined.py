# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Table of Contents
#
# 0. Preface
#   - Instructions
#   - Installer
#   - Imports
#   - Colour scheme preview
# 1. 4D 
#   - Animated rotating wireframe image (projected)
#   - Superposed animation frames (projected)
# 2. 3D Net
#   - Wireframe image (projected)
#   - Solid cell-shaded image (projected)
#   - STL
# 3. 2D facet nets
#   - Plain
#   - Solid colour (1 per facet) (WIP)

# %% [markdown]
# ## Instructions
#
# 1. Run the initialisation cells (down to **Colour scheme preview**) exactly once, before doing anything else.
# 2. In the **Colour scheme preview** section, run the first cell to display a list of colour schemes. Change the value of `PREVIEW_COLOR_SCHEME` in the second cell and run it to get a preview of that colour scheme.
# 3. The first half of the **Parameters** cell is where you can set all the options for the output. They should be more or less self explanatory; more will be added later. Note that strange things can happen with the projection is `PERSPECTIVE_DISTANCE` is set too low.
# 4. Run the **Randomization** cell each time you want to generate a new viewing angle.
# 5. In the **Output area**, each type of output has a *preview pane* cell and a *save files* cell. Run the preview pane cell and adjust **Parameters** until you find a view you like, then run the save files cell. 
# 6. Files are saved to `SAVE_DIRECTORY` (which is `./output` by default). You can add a custom suffix to the filename by setting the `TAG` parameter. If `TAG` is set to the empty string, i.e. `TAG = ""`, a 4 character suffix is generated automatically. This suffix automatically changes whenver you change the parameters or regenerate the viewing angle.
# 7. Note that it is not possible to preview animations or STL files. For animations, ffmpeg must be installed (this should happen automatically if you are running the notebook on mybinder.org).

# %%
# Installer
import sys, os

# IMPORTS
sys.path.append("..")
try:
    from tope import Tope
except ImportError:
    # # !{sys.executable} -m pip install ..
    from tope import Tope
from tope.net import *
from tope.orth import *
from tope.graph import Graph
from tope.plot import *

import numpy as np
rng = np.random.default_rng()

import matplotlib as mpl
import matplotlib.pyplot as plt

import json, os
import itertools

DATA_DIR = os.environ.get("POLYTOPE_UNFOLDER_DATA_DIRECTORY") or "../data"

# import prepackaged data
with open(os.path.join(DATA_DIR, "polys2.json")) as fd: 
    polys = json.load(fd)
    
# and the rest
import gzip
with gzip.open(os.path.join(DATA_DIR, "474polys.json.gz"), "rt") as fd:
    polys.update(json.load(fd))

with gzip.open(os.path.join(DATA_DIR, "d30.json.gz"), "rt") as fd:
    d30 = json.load(fd)
polys.update({f"d30-{record['ID']}": record["Vertices"] for record in d30})

logger.remove()


# %%
def create_lc(edges, color_map = "Set2", color_range=(0.25,0.75), lw=1):
    segments = []
    colors = []
    cmap = mpl.colormaps.get(color_map)
    for i, edge in enumerate(edges):
        segments.append(edge)
        crange_start = color_range[0]
        crange_step = (color_range[1]-color_range[0]) / len(edges)
        colors.append(cmap(crange_start + i*crange_step))
    return mpl.collections.LineCollection(segments, color=colors, linewidth=lw)

def get_wireframe(P: Tope, rotator, perspective_distance=10):
    rotate  = lambda e: e @ rotator
    project = lambda e: perspective_project(perspective_project(e, perspective_distance), perspective_distance)
    return list(map(project, map(rotate, P.iter_faces_as_arrays(dim=1))))

def generate_rotators(N: int, num_steps: int = 10) -> np.ndarray:
    """
    Generate num_steps evenly spaced rotations of stacked vectors.
    """
    return rotator_nd(np.arange(0, 2*np.pi, 2*np.pi / num_steps), N)


# %%
# Special rotation

Q4a_d = {
    "standard": np.eye(4),
    "random": random_orth(4),
    "diagonal": np.array([
        [-1,0,1,0],
        [0,-1,1,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
}

Q4b_d = {
    "obverse": np.eye(4),
    "lateral": np.array([
        [0,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
        [0,1,0,0]
    ]),
    "mixed": np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ]),
    "random": random_orth(4)
}

# %% [markdown]
# ## Polytope ID listing

# %%
print("Available polytopes:")
print("--------------------")
for i, cm in enumerate(polys): 
    print(f"{cm:14}", end="")
    if i%8==7: print()
print()
print()
print("Note: all numbers less than 474000000 and ending in 123456 are available.")
print("Entries whose ID has the prefix 'd30-' have at least 30 facets.")

# %% [markdown]
# # Colour scheme preview

# %%
print("Available named colors:")
print("-----------------------")
count = 0
for cm in mpl.colors.get_named_colors_mapping(): 
    if cm.startswith("xkcd:"): continue
    print(f"{cm:24}", end="")
    count += 1
    if count%5==4: print()
print("-----------------------")
print("Note: b/g/r/c/m/y/k/w are short for blue/green/red/cyan/magenta/yellow/black/white.")

# %%
print("Available color schemes:")
print("------------------------")
count = 0
for cm in mpl.colors.get_named_colors_mapping(): 
    if cm.startswith("xkcd:"): continue
    print(f"{cm:20}", end="")
    count += 1
    if count%6==5: print()
print("-----------------------")
print("Note: b/g/r/c/m/y/k/w are short for blue/green/red/cyan/magenta/yellow/black/white.")

# %% [markdown]
# #### Paste one of these strings in between the quotation marks and run this cell (Shift+Enter) to preview a colour scheme!

# %%
PREVIEW_COLOR_SCHEME = "Spectral"
mpl.colormaps.get(PREVIEW_COLOR_SCHEME)

# %% [markdown]
# # Parameters

# %%
POLYTOPE     = "d30-86363435"

COLOR_SCHEME      = "Pastel1_r"
COLOR_RANGE_BEGIN = 0.25   # between 0 and 1
COLOR_RANGE_END   = 0.75   # between 0 and 1
BG_COLOR         = "black"

PERSPECTIVE_DISTANCE = 10

DPI           = 300
ANIMATION_DPI = 150

IMAGE_FORMAT   = "svg" # or "svg" or whatever
TAG            = "" # put nonempty string here to add custom text to filenames
                    # otherwise a new tag will be generated every time this cell is run

ROTATION_AXIS = "diagonal"    # "standard" or "diagonal" or "random"
PROJECTION = "obverse" # "lateral" or "obverse" or "mixed" or "random"

ANIMATION_FPS = 20 # frame per seconds in the animation
ANIMATION_LENGTH_SECS = 3 # total length of the animation in seconds


# don't change ##################
SAVE_DIRECTORY = os.environ.get("POLYTOPE_UNFOLDER_OUTPUT_DIRECTORY") or "output"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

DIR_4D = os.path.join(SAVE_DIRECTORY, "4d-wireframe")
DIR_ANIMATION = os.path.join(DIR_4D, "animated")
DIR_SMEARED = os.path.join(DIR_4D, "smeared")
DIR_3D = os.path.join(SAVE_DIRECTORY, "3d-net")
DIR_STL = os.path.join(DIR_3D, "stl")
DIR_NET_PROJECTION = os.path.join(DIR_3D, "projected")
DIR_SHADED_3D_NET = os.path.join(DIR_3D, "shaded")
DIR_2D = os.path.join(SAVE_DIRECTORY, "facet-nets")
P = Tope.from_vertices(polys[POLYTOPE])
P.vertices = P.vertices.astype(float) + rng.random(size=4) # random offset

configvars = [
    POLYTOPE, COLOR_SCHEME, COLOR_RANGE_BEGIN, BG_COLOR, PERSPECTIVE_DISTANCE, DPI, ANIMATION_DPI
]

def get_tag():
    return TAG or (sum(id(v) for v in configvars)%65536).to_bytes(2,"big").hex()

# pre-rotation
ROTATION_AXIS = ROTATION_AXIS.lower().strip()
PROJECTION = PROJECTION.lower().strip()
Q4a = random_orth(4) if "random".startswith(ROTATION_AXIS) else Q4a_d[ROTATION_AXIS]
Q4b = random_orth(4) if "random".startswith(PROJECTION) else Q4b_d[PROJECTION]

# other
Q3 = random_orth(3)


# %% [markdown]
# # Output area

# %% [markdown]
# # 4D

# %%
def get_frames(P, before, after, num_steps=10):
    return [create_lc(get_wireframe(P, before @ rotator_nd(theta, 4) @ after)) for theta in np.arange(0, 2*np.pi, 2*np.pi / num_steps)]


# %%
def plot_wireframe(
    wf: list[np.ndarray], # iterable yielding 2x2 arrays
    color_map = "Pastel1_r",
    color_range = (0.25,0.75),
    weight = 1,
    bg_color = "beige",
    border = False,
    **kwargs
):
    
    fig, ax = plt.subplots()
    
    lines = create_lc(
        wf,
        color_map = color_map,
        color_range = color_range,
        lw = weight
    )

    ax.add_collection(lines)
    
    ax.autoscale()
    ax = configure_axes(ax, bg=bg_color, border=border)

    return fig, ax


# %% [markdown]
# ### Preview 4d wireframe

# %%
fig, _ = plot_wireframe(get_wireframe(P, Q4b), border=True, color_map = "Spectral", color_range=(0.75,1), bg_color=BG_COLOR)
fig.set_size_inches(10,10)

# %%
frames = get_frames(P, Q4a, Q4b, num_steps=ANIMATION_LENGTH_SECS*ANIMATION_FPS)
bbox = get_tightbbox(*frames) # must compute before adding artists to axes!

fig, ax = plt.subplots()
for frame in frames:
    ax.add_artist(frame)
ax.dataLim = bbox
ax = configure_axes(ax, BG_COLOR)

# %% [markdown]
# ### Save output (smear and animation)
# Make sure to run the previous cell first!

# %%
TAG = get_tag()

# save smear
os.makedirs(DIR_SMEARED, exist_ok=True)
fig.savefig(os.path.join(DIR_SMEARED, f"{POLYTOPE}-{TAG}.{IMAGE_FORMAT}"), dpi=DPI)

# save animation
from matplotlib.animation import ArtistAnimation
os.makedirs(DIR_ANIMATION, exist_ok=True)
animation = ArtistAnimation(fig, [[frame] for frame in frames], interval=1000/ANIMATION_FPS)
animation.save(os.path.join(DIR_ANIMATION, f"{POLYTOPE}-{TAG}.mp4"), dpi=ANIMATION_DPI)

# %% [markdown]
# # 3D net

# %%
N = P.net().unfold().in_own_span()

# %% [markdown]
# ### Preview

# %%
fig, ax = plt.subplots(dpi=DPI)

cell_edges = [np.stack([F.vertices[sorted(e)] for e in F.iter_faces(1)]) for F in N.facets.values()]
cmaps = list(mpl.colormaps)

for n, cell in enumerate(cell_edges):
    edges = perspective_project(cell @ Q3, 10)
    lc = create_lc(edges, color_map = cmaps[n%len(cmaps)])
    ax.add_collection(lc)

ax = configure_axes(ax, bg=BG_COLOR)

fig.set_size_inches(20,20)

# %% [markdown]
# ### Save output
# Make sure to run the previous cell first!

# %%
TAG = get_tag()

os.makedirs(DIR_NET_PROJECTION, exist_ok=True)
fig.savefig(os.path.join(DIR_NET_PROJECTION, f"{POLYTOPE}-{TAG}.{IMAGE_FORMAT}"), dpi=DPI)

# %% [markdown]
# ## STL
# Export as STL

# %%
TAG = get_tag()

from tope.stl import create_stl
thing = create_stl(*N.facets.values())
assert thing.check()

os.makedirs(DIR_STL, exist_ok=True)
thing.save(os.path.join(DIR_STL, f"{POLYTOPE}-{TAG}.stl"))

# %% [markdown]
# ## Experimental: shaded 3d net

# %%
l = list(N.facets.values())
facet_colors = [mpl.colormaps[COLOR_SCHEME](k/len(l)) for k in range(len(l)) for _ in l[k].triangulate()]

# %% [markdown]
# ### Preview

# %%
import mpl_toolkits.mplot3d as mpl3d

ar = mpl3d.art3d.Poly3DCollection(thing.vectors, shade=True, lightsource=mpl.colors.LightSource(), facecolors=facet_colors)
fig = plt.figure(dpi=DPI)
ax = fig.add_subplot(projection='3d')
ax.add_artist(ar)

ax = configure_axes_3d(ax, thing.vectors, bg=BG_COLOR)

# %% [markdown]
# ### Save

# %%
TAG = get_tag()

os.makedirs(DIR_SHADED_3D_NET, exist_ok=True)
fig.savefig(os.path.join(DIR_SHADED_3D_NET, f"{POLYTOPE}-{TAG}.{IMAGE_FORMAT}"), dpi=DPI)

# %% [markdown]
# # 2d nets

# %%
facets = [P.get_facet(i) for i in range(len(P.faces[P.dim-1]))]
facet_nets = [F.net().unfold().in_own_span() for F in facets]

# %% [markdown]
# ### Preview pane

# %%
lcs = [create_lc(list(net.iter_edges())) for net in facet_nets]

# preview nets in approximately square grid
h = int(np.ceil(np.sqrt(len(facets))))
fig, axs = plt.subplots(h, h, figsize=(h*5,h*5))
axs = list(itertools.chain(*axs))

# hide and discard unused axes
for _ in range(len(axs)-len(lcs)):
    axs.pop().set_visible(False)

# now display
for ax, lc in zip(axs, lcs):
    ax.add_collection(lc)
    configure_axes(ax, bg=BG_COLOR)

# %% [markdown]
# ### Save PNGs

# %%
TAG = get_tag()

os.makedirs(DIR_2D, exist_ok=True)

savedir = os.path.join(DIR_2D, f"{POLYTOPE}-{TAG}")
os.makedirs(savedir, exist_ok=True)
for n, ax in enumerate(axs):
    save_subplot(fig, ax, os.path.join(savedir, f"{n}.{IMAGE_FORMAT}"), dpi=DPI)

# %%
