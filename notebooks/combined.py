# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
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
    # !{sys.executable} -m pip install ..
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

with open(os.path.join(DATA_DIR, "20s.json")) as fd:
    polys.update(json.load(fd))

with open(os.path.join(DATA_DIR, "10s-short.json")) as fd:
    polys.update(json.load(fd))
    
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
print("\n-----------------------")
print("Note: b/g/r/c/m/y/k/w are short for blue/green/red/cyan/magenta/yellow/black/white.")

# %%
print("Available color schemes:")
print("------------------------")
count = 0
for cm in mpl.colormaps: 
    print(f"{cm:20}", end="")
    count += 1
    if count%6==5: print()

# %% [markdown]
# #### Paste one of these strings in between the quotation marks and run this cell (Shift+Enter) to preview a colour scheme!

# %%
PREVIEW_COLOR_SCHEME = "PuRd"
mpl.colormaps.get(PREVIEW_COLOR_SCHEME)

# %% [markdown]
# # Parameters

# %%
POLYTOPE     = "d10-43226722"

COLOR_SCHEME      = "PuRd"
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

STL_THICKNESS = 15 # thickness of frame for stl with wireframe in mm
STL_BOX_SIZE = 1000 # side of box which countains the stl with wireframe in mm (rough estimate)


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
P.save_index()
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
# Export as STL (wireframe)

# %%
TAG = get_tag()

from tope.stl import create_stl_from_net, stl_dimensions, define_scale
from math import sqrt

SCALE = define_scale(N, STL_BOX_SIZE, STL_THICKNESS)

t = sqrt((STL_THICKNESS/SCALE)**2/2)

thing = create_stl_from_net(N, t, walls=False)
assert thing.check()

thing.vectors = SCALE*thing.vectors

os.makedirs(DIR_STL, exist_ok=True)
thing.save(os.path.join(DIR_STL, f"{POLYTOPE}-{TAG}-wireframe.stl"))

# Print the dimensions
stl_dimensions(thing)

# %% [markdown]
# X-ray STL plot

# %%
import mpl_toolkits.mplot3d as mpl3d

ar = mpl3d.art3d.Poly3DCollection(thing.vectors, lightsource=mpl.colors.LightSource(),facecolors='w', linewidths=1, alpha = 0.1)
fig = plt.figure(dpi=DPI)
ax = fig.add_subplot(projection='3d')
ax.add_artist(ar)

ax = configure_axes_3d(ax, thing.vectors, bg=BG_COLOR)

# %% [markdown]
# Export as STl (walls)

# %%
TAG = get_tag()

from tope.stl import create_stl_from_net
thing = create_stl_from_net(N, STL_THICKNESS, walls=True)
assert thing.check()

os.makedirs(DIR_STL, exist_ok=True)
thing.save(os.path.join(DIR_STL, f"{POLYTOPE}-{TAG}.stl"))

# %% [markdown]
# ### Plotly mesh object (navigable in notebook)

# %% [markdown]
# Define ad-hoc colour map

# %%
# Define ad-hoc colour map
import matplotlib.pyplot as plt

# Define the RAL colors and their HTML color codes
ral_colors = ["#d3b088", "#b38055", "#dfc49e", "#dc6452", "#ca8470", "#d17276", "#71342e", "#846181", "#924834", "#7c4d40"]

cmap = plt.cm.colors.ListedColormap(ral_colors)
cmap


# %%
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import math

# Define how many colours you want you want to interpolate by (so that the total is more that the max of different edges, vertices and faces)
num_intermediate_colors = math.ceil((max(len(P.faces[1]), max(len(P.faces[0]), len(P.faces[2])))-10)/9)+2

# Interpolate colour
color_segments = []
for i in range(len(ral_colors)-1):
    start_color = mpl.colors.hex2color(ral_colors[i])
    end_color = mpl.colors.hex2color(ral_colors[i + 1])
    
    r = np.linspace(start_color[0], end_color[0], num_intermediate_colors)[:-1]
    g = np.linspace(start_color[1], end_color[1], num_intermediate_colors)[:-1]
    b = np.linspace(start_color[2], end_color[2], num_intermediate_colors)[:-1]
    
    color_segment = np.column_stack((r, g, b))
    color_segments.append(color_segment)
color_segments.append(mpl.colors.hex2color(ral_colors[-1]))

interpolated_colors = np.vstack(color_segments)

# Define colour map from interpolated colours
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors=interpolated_colors, N=num_intermediate_colors * (len(ral_colors)))

# Ad-hoc functions for colour scheme
def color_by_numbers(x: Iterable, lim, cm):
    return [cm(n/lim) for n in x] 

def color_faces(F: Tope, lim, cm, perm,dim) -> list[tuple]: 
    return color_by_numbers((perm[face["index"]] for face in F.meta[dim]), lim, cm)

cmap

# %% [markdown]
# Define colours for edges and vertices of wireframe model.

# %%
from tope.stl import edge_prism, icosahedron_mesh

# Deal with vertices
tr_vert = []
colors_vert = []
n_vert = len(P.faces[0])

# Permutation for indices (so similar shades are not close together)
perm = np.random.permutation(n_vert)

# Loop over facets
for facet in N.facets.values():
    # Define colours 
    facet_colors = color_faces(facet, n_vert, cmap, perm,0)
    for i in range(len(facet.faces[0])):     # edge index
        face = facet.get_face(i, k=0) # polygon embedded in 3d

        # Get triangles for edge prism (t is the prescriped thickness)
        tri = icosahedron_mesh(face.vertices, t)

        # Add triangles and colours
        tr_vert.extend(tri)
        colors_vert.extend([facet_colors[i]] * len(tri)) # repeat ith color for number of triangles in the prism fr the edge

# Deal with edges
tr_edge = []
colors_edge = []
n_edges = len(P.faces[1])

# Permutation for indices (so similar shades are not close together)
perm = np.random.permutation(n_edges)

# Generate one random 3D vector
rv1 = np.random.rand(3)

# Loop over facets
for facet in N.facets.values():

    # Define colours 
    facet_colors = color_faces(facet, n_edges, cmap, perm,1)
    for i in range(len(facet.faces[1])):     # edge index
        face = facet.get_face(i, k=1) # polygon embedded in 3d

        # Get triangles for edge prism (t is the prescriped thickness)
        tri = edge_prism(face.vertices, t, rv1)

        # Add triangles and colours
        tr_edge.extend(tri)
        colors_edge.extend([facet_colors[i]] * len(tri)) # repeat ith color for number of triangles in the prism fr the edge


colors = colors_edge+colors_vert
triangles = tr_edge+tr_vert

# %% [markdown]
# Plot in plotly (wireframe)

# %%
import plotly
plotly.io.renderers.default = "iframe"
import plotly.graph_objects as go

verts = np.concatenate(triangles)
i = np.arange(len(triangles)) * 3

mesh = go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                 i=i, j=i+1, k=i+2, facecolor=colors)
go.Figure(mesh)

# %% [markdown]
# Define colours for mesh plot

# %%
triangles = []
colors = []
n_faces = len(P.faces[2])

# Permutation for indices (so similar shades are not close together)
perm = np.random.permutation(n_faces)

for facet in N.facets.values():
    facet_colors = color_faces(facet, n_faces, cmap, perm, 2)
    for i in range(len(facet.faces[2])):     # face index
        face = facet.get_face(i, k=2) # polygon embedded in 3d
        tri = list(face.triangulate())
        triangles.extend(tri)
        colors.extend([facet_colors[i]] * len(tri)) # repeat ith color for number of triangles in the face


# %% [markdown]
# Plot in plotly (mesh)

# %%
import plotly
plotly.io.renderers.default = "iframe"
import plotly.graph_objects as go

verts = np.concatenate(triangles)
i = np.arange(len(triangles)) * 3

mesh = go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                 i=i, j=i+1, k=i+2, facecolor=colors)
go.Figure(mesh)

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

# %%
LABEL_CFG = {"color": "white", "fontsize": "5", "ha": "center", "va": "center"}

def get_facet_label_artists(N: Net):
    labels = []
    for x in N.facets.values():
        pos = x.vertices.mean(axis=0)
        txt = str( x.meta[x.dim][0]["index"] )
        labels.append(Text(*pos, text=txt, **LABEL_CFG))
    return labels

facet_net_labels = [get_facet_label_artists(F) for F in facet_nets]

def get_facet_labels(N: Net):
    labels = []
    for i, x in N.facets.items():
        vertices = x.vertices
        labels.append((str(N.tope.meta[N.tope.dim-1][i]["index"]), vertices.mean(axis=0)))
    return labels

from typing import Iterable

def create_text_artists(*labels, **cfg) -> Iterable[Text]:
    cfg = LABEL_CFG | cfg # default values
    return [Text(*label[1], text=str(label[0]), **cfg) for label in labels]


# %% [markdown]
# ### Preview pane

# %%
list(facet_nets[0].iter_edges())[0]

# %%
lcs = [create_lc(list(net.iter_edges())) for net in facet_nets]

# preview nets in approximately square grid
h = int(np.ceil(np.sqrt(len(facets))))
fig, axs = plt.subplots(h, h, figsize=(h*5,h*5), sharex=True, sharey=True)
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
lcs = [create_lc(list(net.iter_edges())) for net in facet_nets]
facet_net_labels = [get_facet_label_artists(F) for F in facet_nets]

# preview nets in approximately square grid
h = int(np.ceil(np.sqrt(len(facets))))
fig, axs = plt.subplots(h, h, figsize=(h*5,h*5), sharex=True, sharey=True)
axs = list(itertools.chain(*axs))

# hide and discard unused axes
for _ in range(len(axs)-len(lcs)):
    axs.pop().set_visible(False)

# now display
for ax, lc, labels in zip(axs, lcs, facet_net_labels):
    ax.add_collection(lc)
    for label in labels:
        ax.add_artist(label)
    configure_axes(ax, bg=BG_COLOR)

# %% [markdown]
# ### Save PNGs

# %%
TAG = get_tag()

os.makedirs(DIR_2D, exist_ok=True)

savedir = os.path.join(DIR_2D, f"{POLYTOPE}-{TAG}-l")
os.makedirs(savedir, exist_ok=True)
for n, ax in enumerate(axs):
    save_subplot(fig, ax, os.path.join(savedir, f"{n}.{IMAGE_FORMAT}"), dpi=DPI)

# %% [markdown]
# ## Labels and mirror images

# %%
import matplotlib
MARGIN_FACTOR = 0.05

def get_net_for_facet(P: Tope, i: int, keys=[]) -> Net:
    P.save_index()
    F = P.get_facet(i, meta_keys=keys)
    N = F.net().unfold_with_meta(meta_keys=keys).in_own_span(meta_keys=keys)
    return N

FacetLabels = list[tuple[str, np.ndarray]] # label, position
EdgeList = list[np.ndarray] # list of 2xdim arrays

def get_facet_labels(N: Net) -> FacetLabels:
    labels = []
    for i, x in N.facets.items():
        vertices = x.vertices
        labels.append((N.tope.meta[N.tope.dim-1][i]["index"], vertices.mean(axis=0)))
    return labels

def get_edges(N: Net) -> EdgeList: # apply to unfolded Net
    edges = []
    for i, x in N.facets.items():
        vertices = x.vertices
        facet_template = N.tope.get_face(i) # has correct indices
        edges.extend((vertices[list(e)] for e in facet_template.faces[1]))
    return edges

def plot_nets(
    nets: list, 
    name: str = "P", 
    margin: float = MARGIN_FACTOR, 
    label_config: dict = {"fontsize": 5, "ha": "center", "va": "center"},
    title_config: dict = {"fontsize": 5, "pad": -14},
    label = True, 
    mirror = False
):
    # Compute common bounding box
    bbox = bounding_bbox_from_arrays(*(cell .vertices for N in nets for cell in N.facets.values()))
     
    # List of figures
    figs = []

    # Generate images
    for i, N in enumerate(nets):

        # Define artists
        artists = [LineCollection(get_edges(N), colors="white", linewidths=0.2)]    
        
        # Add labels
        if label: 
            artists.extend(Text(*pos, text=str(l), **label_config, color = 'white') for l, pos in get_facet_labels(N))
        
        # Draw the actual picture
        fig, ax = plot_artists_in_view(*artists, bbox = bbox, margin = margin)
        
        facet_name = f"{name}-facet-{i}"
        if mirror:
            facet_name = facet_name+"-mirror"
        if label:
            facet_name = facet_name+"-labels"
        
        # title with "default" positioning
        ax.set_title(facet_name, **title_config)

        # If mirror, mirror with respect to the x-axis
        if mirror:
            ax.invert_xaxis()
                
        # store and next()
        figs.append((facet_name, fig))
        matplotlib.pyplot.close()
    
    return figs

nfacets = len(P.faces[P.dim-1])
# Get the facet nets
nets = [get_net_for_facet(P, i) for i in range(nfacets)]

# No labels no mirror
figs_nl_nm = plot_nets(nets, mirror = False, label = False)

# Labels no mirror
figs_l_nm = plot_nets(nets, mirror = False, label = True)

# Labels mirror
figs_l_m = plot_nets(nets, mirror = True, label = True)

# No labels mirror
figs_nl_m = plot_nets(nets, mirror = True, label = False)


# %% [markdown]
# ### Preview

# %%
i = 0 # change this for different examples

display(figs_nl_nm[i][1])
display(figs_l_nm[i][1])
display(figs_nl_m[i][1])
display(figs_l_m[i][1])

# %% [markdown]
# ### Save zip

# %%
from tope.plot import save_figs_to_zip # THIS TAKES A LITTLE TIME
TAG = get_tag()

os.makedirs(DIR_2D, exist_ok=True)

savedir = os.path.join(DIR_2D, f"{POLYTOPE}-{TAG}")
os.makedirs(savedir, exist_ok=True)

save_figs_to_dir(figs_nl_nm, directory = savedir+'_nolabel_nomirror', force=True)
save_figs_to_dir(figs_l_nm, directory = savedir+'_label_nomirror', force=True)
save_figs_to_dir(figs_l_m, directory = savedir+'_label_mirror', force=True)
save_figs_to_dir(figs_nl_m, directory = savedir+'_nolabel_mirror', force=True)


# %% [markdown]
# =======
# ## Save bundle
#
# Run this cell once to bundle all outputs together for download!
# Note that it will overwrite previous runs.

# %%
import zipfile

with zipfile.ZipFile(f"{SAVE_DIRECTORY}.zip", "w") as ziph:
    for d, _, file_l in os.walk(SAVE_DIRECTORY):
        for f in file_l:
            ziph.write(os.path.join(d,f))

# %%
