# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %%
# %autosave 0
import sys
# !{sys.executable} -m pip install ..

# %%
from tope import Tope
from tope.net import *
from tope.orth import *
from tope.graph import Graph
from tope.plot import plot_artists_in_view

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import json, os

# import prepackaged data
with open("../data/polys2.json") as fd: 
    polys = json.load(fd)
    
logger.remove()

# %%
mpl.colormaps.get("YlOrRd")


# %%
def create_lc(edges, color_map = "Set2"):
    segments = []
    colors = []
    cmap = mpl.colormaps.get(color_map)
    for i, edge in enumerate(edges):
        segments.append(edge)
        colors.append(cmap(0.25+ 0.5*i/len(edges)))
    return mpl.collections.LineCollection(segments, color=colors)


# %% [markdown]
# # Parameters

# %%
print("Available color schemes:")
print("------------------------")
for i, cm in enumerate(mpl.colormaps): 
    print(f"{cm:20}", end="")
    if i%5==4: print()

# %%
POLYTOPE     = "24-cell"
FIG_FILENAME = "nets-unfolded.png"
STL_FILENAME = "thing24.stl"
COLOR_SCHEME = "Pastel1_r"

# %% [markdown]
# ## Randomization

# %%
Q = random_orth(3)

# %% [markdown]
# # Actually do the stuff

# %%
P = Tope.from_vertices(polys[POLYTOPE])
T = P.facet_graph().width_first_spanning_tree()
N = Net(P, T).unfold().in_own_span()

# %%
fig, ax = plt.subplots(dpi=300)

cell_edges = [np.stack([F.vertices[sorted(e)] for e in F.iter_faces(1)]) for F in N.facets.values()]
cmaps = list(mpl.colormaps)

for n, cell in enumerate(cell_edges):
    edges = perspective_project(cell @ Q, 10)
    lc = create_lc(edges, color_map = cmaps[n%len(cmaps)])
    ax.add_collection(lc)
#    for ax in axs[:n+1]:
#        lc = create_lc(edges, color_map = cmaps[n%len(cmaps)])
#        ax.add_collection(lc)
        
#bbox = axs[0].dataLim
        
#for ax in axs:
#    ax.dataLim = bbox
ax.set_aspect("equal")
ax.autoscale()

fig.set_size_inches(20,20)
fig.savefig("nets-builder1.png")

# %% [markdown]
# # Make STL

# %%
l = list(N.facets.values())

from tope.stl import create_stl
thing = create_stl(*l)
assert thing.check()
thing.save(STL_FILENAME)

# %% [markdown]
# ## Experimental: shaded 3d net

# %%
colors = [mpl.colormaps[COLOR_SCHEME](k/len(l)) for k in range(len(l)) for _ in l[k].triangulate()]

# %%
import mpl_toolkits.mplot3d as mpl3d

ar = mpl3d.art3d.Poly3DCollection(thing.vectors, shade=True, lightsource=mpl.colors.LightSource(), facecolors=colors)
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection='3d')
ax.add_artist(ar)
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(-3,3)
ax.set_aspect("equal")
ax.axis("off")

# %%
