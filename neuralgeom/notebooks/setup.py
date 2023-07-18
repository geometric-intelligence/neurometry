"""Script that sets up the file directory tree before running a notebook.

Usage:

import setup

setup.main()
"""

import os
import subprocess
import sys
import warnings


def main():
    warnings.filterwarnings("ignore")

    gitroot_path = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    )

    os.chdir(os.path.join(gitroot_path[:-1], "neuralgeom"))
    print("Working directory: ", os.getcwd())

    sys_dir = os.path.dirname(os.getcwd())
    sys.path.append(sys_dir)
    print("Directory added to path: ", sys_dir)
    sys.path.append(os.getcwd())
    print("Directory added to path: ", os.getcwd())

def get_data_dir():
    RAW_DIR = os.path.join(os.getcwd(), "data", "raw")
    print(f"The raw data is located in the directory:\n{RAW_DIR}.")
    BINNED_DIR = os.path.join(os.getcwd(), "data", "binned")
    print(f"The binned data is located in the directory:\n{BINNED_DIR}.")
    return RAW_DIR, BINNED_DIR
