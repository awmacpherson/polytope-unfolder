import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.text import Text, Annotation

import os
from typing import Iterable
from zipfile import ZipFile
from tempfile import mkdtemp
from shutil import rmtree

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

from matplotlib.transforms import Bbox

def bounding_bbox(*arrays): # arrays
    box = Bbox.null()
    for array in arrays:
        box.update_from_data_xy(array, ignore=False)
    return box

def plot_artists_in_view(
    *artists,
    bbox: Bbox = Bbox.null(),
    margin: float = 0.05
):
    fig, ax = plt.subplots()
    for ar in artists:
        ax.add_artist(ar)
    ax.dataLim = bbox
    ax.set_aspect(1)
    ax.set_xmargin(margin)
    ax.set_ymargin(margin)
    ax.autoscale()
    return fig, ax

