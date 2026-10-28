# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import subprocess
import sys

import pytest

from mmtwfs.config import WFS_DATA_DIR


@pytest.mark.parametrize("module", ["mmtwfs.config", "mmtwfs.mmtcell"])
def test_module_imports_on_its_own(module):
    # a fresh interpreter, so nothing else has already imported importlib.resources
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_data_dir_is_package_data():
    assert WFS_DATA_DIR.is_dir()
    assert (WFS_DATA_DIR / "actuator_coordinates.dat").is_file()
