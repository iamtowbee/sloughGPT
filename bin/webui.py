import runpy
import sys
import os
from pathlib import Path

# Change to the repository root directory
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    runpy.run_module("packages.apps.apps.cerebro", run_name="__main__")
