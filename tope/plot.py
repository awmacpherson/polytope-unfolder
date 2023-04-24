import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.text import Text, Annotation

import os
from typing import Iterable
from zipfile import ZipFile
from tempfile import mkdtemp
from shutil import rmtree
from loguru import logger

# Axis prep

def bounding_bbox_from_arrays(*arrays): # arrays
    box = mpl.transforms.Bbox.null()
    for array in arrays:
        box.update_from_data_xy(array, ignore=False)
    return box

def get_tightbbox(*things):
    bboxes = [thing.get_tightbbox() for thing in things]
    return mpl.transforms.Bbox.union(bboxes)

def configure_axes_3d(ax: mpl.axes.Axes, triangles, bg="black", border=False):
    """
    Set explicit limits, 1:1 aspect ratio, turn off axes, and set background
    colour on a 3d Axes object.
    (Note that autoscaling doesn't work here, so explicit data is needed.)
    """
    ax.set_xlim(triangles[:,:,0].min(), triangles[:,:,0].max())
    ax.set_ylim(triangles[:,:,1].min(), triangles[:,:,1].max())
    ax.set_zlim(triangles[:,:,2].min(), triangles[:,:,2].max())

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_facecolor(bg)

    return ax

def configure_axes(ax: mpl.axes.Axes, bg="black", border=False):
    #ax.set_xticks([])
    #ax.set_xticklabels([])
    #ax.set_yticks([])
    #ax.set_yticklabels([])

    #if not border:
    #    ax.spines[:].set_visible(False)

    limits = ax.axis("scaled")
    ax.set_axis_off()
    ax.set_facecolor(bg)

    return ax

def plot_artists_in_view(
    *artists,
    bbox: mpl.transforms.Bbox = mpl.transforms.Bbox.null(),
    margin: float = 0.05
):
    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_axes((0,0,1,1))
    for ar in artists:
        ax.add_artist(ar)
    ax.dataLim = bbox
    configure_axes(ax)
    return fig, ax


# File operations

def save_figs_to_dir(
    figs: Iterable, 
    dpi: int = 300,
    directory: str = ".",
    force: bool = False,
    fmt: str = "png"
):
    # make directory
    if force:
        os.makedirs(directory, exist_ok=True) # don't warn if it's already present
    else:
        os.mkdir(directory) # cannot force without parents
    
    # save figs
    for name, fig in figs:
        fname = os.path.join(directory, name + "." + fmt)
        fig.savefig(fname, dpi=dpi) 
        # default behaviour is to overwrite
        # if we made it this far, either force = True or the directory didn't exist

def copy_dir_to_zip(
    path: str,
    directory: str
):
    """
    Copy contents of directory (non-recursively) to a zip archive.
    """
    with ZipFile(path, "w") as zfd:
        for fname in os.listdir(directory):
            zfd.write(os.path.join(directory, fname), arcname=fname)
    # to make recursive use os.walk()

def save_figs_to_zip(
    figs: Iterable, 
    path: str,
    dpi: int = 300,
    overwrite: bool = False,
    fmt: str = "png"
):
    # make directory
    tmpdir = mkdtemp()
    save_figs_to_dir(
        figs, 
        dpi=dpi, 
        directory=tmpdir, 
        force=True, # tmpdir already exists
        fmt=fmt
    )

    # compress and add to archive
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError("Path already exists.")
        if overwrite and os.path.isdir(path):
            logger.warning(f"Warning: removing directory tree at {path}.")
            rmtree(path)

    try:
        copy_dir_to_zip(path, tmpdir)
    except FileNotFoundError: # parents of path don't exist
        os.makedirs(os.path.dirname(path))
        copy_dir_to_zip(path, tmpdir)
    rmtree(tmpdir)


# Artist construction

from matplotlib.collections import LineCollection
from matplotlib import colormaps
# matplotlib.cm is mostly deprecated API
import numpy as np

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]

def new_color_mapped_line_collection(lines: Iterable[np.ndarray], cmap: str) -> LineCollection:
    cmap = colormaps[cmap]
    n = len(lines)
    color = (cmap( i/len(lines) ) for i in range(n))
    return LineCollection(*lines, color=color)
