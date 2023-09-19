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

    os.chdir(os.path.join(gitroot_path[:-1], "neurometry/neuralwarp"))
    print("Working directory: ", os.getcwd())

    sys_dir = os.path.dirname(os.getcwd())
    sys.path.append(gitroot_path[:-1])
    sys.path.append(sys_dir)
    print("Directory added to path: ", sys_dir)
    sys.path.append(os.getcwd())
    print("Directory added to path: ", os.getcwd())
    print(sys.path)
