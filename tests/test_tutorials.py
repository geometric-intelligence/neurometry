"""Unit tests for the tutorials."""

import subprocess
import tempfile

import pytest


def _exec_tutorial(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as temp_file:
        file_name = temp_file.name

    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1000",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        file_name,
        path,
    ]
    subprocess.check_call(args)


TUTORIALS_DIR = "tutorials"
paths = [f"{TUTORIALS_DIR}/01_methods_create_synthetic_data.ipynb"]


@pytest.mark.parametrize("path", paths)
def test_tutorial(path):
    """Test the tutorials."""
    _exec_tutorial(path)
