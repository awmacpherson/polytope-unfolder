from tope.plot import *
import os, pytest

def test_save_figs_to_dir(tmp_path):
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

def test_copy_dir_to_zip(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("a")
    open("a/b", "w").close()
    open("a/c", "w").close()

    copy_dir_to_zip("a.zip", "a")
    assert os.path.exists("a.zip")

def test_save_figs_to_zip(tmp_path):
    fig, _ = plt.subplots()
    os.chdir(tmp_path)

    # desired behaviour:
    # archive does not exist but parent exists YES
    # neither archive nor parent(s) exists. YES
    # archive exists as ordinary file YES if overwrite else NO
    # archive exists as directory YES WITH WARNING if overwrite else NO

    save_figs_to_zip(
        [("a", fig), ("b", fig)],
        path = "c.zip",
        fmt = "png",
        overwrite = False
    )
    assert os.path.exists("c.zip")
    with ZipFile("c.zip") as zfd:
        print(zfd.namelist())
        fd = zfd.open("a.png")
    fd.close()

    save_figs_to_zip(
        [("a", fig), ("b", fig)],
        path = "d/e.zip",
        fmt = "png",
        overwrite = False
    )
    assert os.path.exists("d/e.zip")
    with ZipFile("d/e.zip") as zfd:
        print(zfd.namelist())
        fd = zfd.open("a.png")
    fd.close()

def test_save_figs_to_zip_overwrite(tmp_path):
    # overwrite tests
    fig, _ = plt.subplots()
    os.chdir(tmp_path)

    # touch "c.zip"
    open("c.zip", "w").close()

    with pytest.raises(FileExistsError):
        save_figs_to_zip(
            [("a", fig), ("b", fig)],
            path = "c.zip",
            fmt = "png",
            overwrite = False
        )

    save_figs_to_zip(
        [("a", fig), ("b", fig)],
        path = "c.zip",
        fmt = "png",
        overwrite = True
    )
    assert os.path.exists("c.zip")
    with ZipFile("c.zip") as zfd:
        print(zfd.namelist())
        fd = zfd.open("a.png")
    fd.close()

def test_bbox():
    a = [[-1,-1], [0,-1], [-1,0]]
    b = [[1,0.5], [-0.5,1]]
    c = bounding_bbox(a,b)
    assert c.x0 == -1 and c.x1 == 1 and c.y0 == -1 and c.y1 == 1

def test_plot_artists_in_view():
    c = Bbox.from_extents(-1,-1,1,1)
    fig, ax = plot_artists_in_view(bbox=c, margin=0.1)
    b = ax.viewLim
    assert b.x0 == -1.2 and b.x1 == 1.2 and b.y0 == -1.2 and b.y1 == 1.2
    assert ax.get_aspect() == 1
