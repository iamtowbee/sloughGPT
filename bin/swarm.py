import runpy
import sys
import os
from pathlib import Path

# Change to the repository root directory
os.chdir(ROOT_DIR)
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    runpy.run_module("packages.core.src.scripts.swarm", run_name="__main__")
