import sys, subprocess, os, tempfile, shutil

if not shutil.which("ffmpeg"):
    print("Warning: ffmpeg not found. Animation outputs will not work.")

NOTEBOOK_PATH = "notebooks/combined.py"
ENV = {"POLYTOPE_UNFOLDER_DATA_DIRECTORY": "./data"}

with tempfile.TemporaryDirectory() as output_dir:
    ENV["POLYTOPE_UNFOLDER_OUTPUT_DIRECTORY"] = output_dir
    task = subprocess.run(
        [sys.executable, NOTEBOOK_PATH],
        env = os.environ | ENV, 
        capture_output = True
    )
    try:
        task.check_returncode()
    except subprocess.CalledProcessError as e:
        print("FAILED: ", task.stderr)
        raise e


