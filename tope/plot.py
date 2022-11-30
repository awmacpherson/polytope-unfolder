import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.text import Text, Annotation

import os, shutil, tempfile
from typing import Iterable


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
            
def save_figs_to_zip(
    figs: Iterable, 
    arcname: str,
    dpi: int = 300,
    force: bool = False, # force => parents
    as_zip: bool = True,
    fmt: str = "png"
):
    # make directory
    import tempfile
    tmpdir = tempfile.mkdtemp()
    save_figs_to_dir(figs, dpi, tmpdir, parents, overwrite)

    # compress and add to archive
    from zipfile import ZipFile
    with ZipFile(arcname, 'w') as zfd:
        for fname in os.listdir(tmppath):
            zfd.write(fname)
        
    from shutil import rmtree
    rmtree(tmppath)
