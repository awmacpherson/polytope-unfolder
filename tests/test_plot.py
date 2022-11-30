from tope.plot import *
import os, pytest

def test_save_figs_to_dir(tmp_path, tmp_path_factory):
    fig, _ = plt.subplots()

    assert os.path.exists(tmp_path)
    os.chdir(tmp_path)

    # default parameters
    with pytest.raises(FileExistsError):
        save_figs_to_dir([("a", fig)])
    # should throw error because parent directory "." already exists

    # new directory
    save_figs_to_dir(
        [("b", fig)],
        directory = "a",
        fmt = "png"
    )
    assert os.path.exists("a/b.png")
    # TODO: test that genuine PNG file is created
    
    # should fail to create parents
    with pytest.raises(FileNotFoundError):
        save_figs_to_dir(
            [("d", fig)],
            directory = "b/c",
            fmt = "png"
        )

    # force
    save_figs_to_dir(
        [("d", fig)],
        directory = "b/c",
        fmt = "png",
        force = True
    )
    assert os.path.exists("b/c/d.png")

def test_save_figs_to_zip():
    pass
