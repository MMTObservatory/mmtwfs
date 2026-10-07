# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib
import shutil

import matplotlib.pyplot as plt

from mmtwfs.scripts.reanalyze import upgrade_cached_line, process_image, CSV_HEADER

WFS_DATA_DIR = importlib.resources.files("mmtwfs") / "data"


def test_upgrade_cached_line():
    ncols = len(CSV_HEADER.strip().split(","))
    old = ",".join(["1"] * (ncols - 1)) + "\n"
    new = upgrade_cached_line(old)
    assert new.endswith(",full\n")
    assert len(new.strip().split(",")) == ncols
    # lines that already have the method column pass through untouched
    assert upgrade_cached_line(new) == new


def test_retry_failed(tmp_path):
    f = tmp_path / "mmirs_wfs_0150.fits"
    shutil.copy(WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits", f)
    failed = tmp_path / "mmirs_wfs_0150.failed"
    failed.touch()  # marked as failed by an earlier run

    assert process_image(f) is None  # skipped by default
    line = process_image(f, retry_failed=True)
    plt.close("all")
    assert line is not None and line.endswith(",full\n")
    assert not failed.exists()  # a successful retry clears the marker
    assert (tmp_path / "mmirs_wfs_0150.output").exists()
