import sys, os, subprocess, tempfile

PATH_4D = "4d-wireframe"
PATH_ANIMATION = os.path.join(PATH_4D, "animated")
PATH_SMEARED = os.path.join(PATH_4D, "smeared")
PATH_3D = "3d-net"
PATH_STL = os.path.join(PATH_3D, "stl")
PATH_NET_PROJECTION = os.path.join(PATH_3D, "projected")
PATH_2D = "facet-nets"

PATHS = [
    PATH_ANIMATION, PATH_SMEARED, PATH_STL, PATH_NET_PROJECTION, PATH_2D
]

def run_script(path, env=dict(), **kwargs):
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    cmdline = [
        sys.executable, path
    ]
    return subprocess.run(cmdline, env=env, **kwargs)

env_vars = {
    "nt": ["USERPROFILE", "PATH"],
    "posix": ["HOME", "PATH"]
}

def get_env():
    return {k: os.environ.get(k) for k in env_vars[os.name]}

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Saving output to {tmpdir}.")
        
        env = get_env()
        env["POLYTOPE_UNFOLDER_OUTPUT_DIRECTORY"] = tmpdir
        this_dir = os.path.dirname(os.path.realpath(__file__))
        env["POLYTOPE_UNFOLDER_DATA_DIRECTORY"] = os.path.join(this_dir, "data")

        os.chdir("notebooks")
        result = run_script("combined.py", env=env, capture_output=True)

        if result.returncode:
            print(result.stderr.decode())

        for path in PATHS:
            if not os.path.exists(os.path.join(tmpdir, path)):
                raise Exception(f"{path} not found!")

