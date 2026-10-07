# Licensed under a 3-clause BSD style license - see LICENSE.rst

import concurrent.futures
import importlib
import shutil
import sys
from unittest.mock import patch

import astropy.units as u
import matplotlib.pyplot as plt

from mmtwfs.zernike import ZernikeVector
from mmtwfs.scripts.reanalyze import upgrade_cached_line, process_image, wfs_systems, CSV_HEADER, main

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


def _focus_only_results(center_measured=True, zernike=None):
    return {
        "slopes": None,
        "focus_only": True,
        "method": "periodicity",
        "grid": {"center": (230.0, 240.0), "center_measured": center_measured},
        "zernike": ZernikeVector(Z04=-500.0) if zernike is None else zernike,
        "pending_focus": 5.0 * u.um,
    }


def _copy_frame(tmp_path):
    f = tmp_path / "mmirs_wfs_0150.fits"
    shutil.copy(WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits", f)
    return f


def test_focus_only_nominal_center_not_recorded(tmp_path):
    # a pupil center that wasn't measured (centering failed, nominal used) must not be written as if it were
    f = _copy_frame(tmp_path)
    with patch.object(wfs_systems["mmirs"], "measure_slopes", return_value=_focus_only_results(False)):
        line = process_image(f)
    fields = line.strip().split(",")
    cols = CSV_HEADER.strip().split(",")
    assert fields[cols.index("method")] == "periodicity"
    assert fields[cols.index("xcen")] == "nan" and fields[cols.index("ycen")] == "nan"

    with patch.object(wfs_systems["mmirs"], "measure_slopes", return_value=_focus_only_results(True)):
        line = process_image(f, force=True)
    fields = line.strip().split(",")
    assert float(fields[cols.index("xcen")]) == 230.0


def test_focus_only_errors_mark_failed(tmp_path):
    # a problem writing focus-only output must not escape into pool.map and stop the whole run
    f = _copy_frame(tmp_path)
    bad = ZernikeVector(Z04=-500.0)
    with patch.object(bad, "save", side_effect=OSError("disk full")):
        with patch.object(wfs_systems["mmirs"], "measure_slopes", return_value=_focus_only_results(zernike=bad)):
            assert process_image(f) is None
    assert (tmp_path / "mmirs_wfs_0150.failed").exists()


def _old_cached_line():
    ncols = len(CSV_HEADER.strip().split(","))
    return ",".join(["1"] * (ncols - 1)) + "\n"


def test_cached_output_upgraded(tmp_path):
    # results cached before the method column existed are reused and gain the column
    f = tmp_path / "mmirs_wfs_0150.fits"
    f.touch()  # never read when there's a cached result
    (tmp_path / "mmirs_wfs_0150.output").write_text(_old_cached_line())
    assert process_image(f) == upgrade_cached_line(_old_cached_line())


def test_main_retry_failed(tmp_path):
    d = tmp_path / "20261006"
    d.mkdir()
    (d / "a.fits").touch()
    (d / "a.output").write_text(_old_cached_line())
    shutil.copy(WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits", d / "b.fits")
    (d / "b.failed").touch()
    csv = d / "reanalyze_results.csv"

    def run(*args):
        # forking this process, which already has numpy's threads running, can deadlock a worker (the default
        # on linux before python 3.14), so use threads; reanalyze itself starts from a fresh process
        with patch.object(sys, "argv", ["reanalyze", "-r", str(tmp_path), "-d", d.name, "-n", "1", *args]), \
                patch("concurrent.futures.ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor):
            main()

    run()
    lines = csv.read_text().splitlines(keepends=True)
    assert lines == [CSV_HEADER, upgrade_cached_line(_old_cached_line())]

    # an existing CSV is left alone unless asked
    csv.write_text("untouched")
    run()
    assert csv.read_text() == "untouched"

    # --retry-failed rebuilds the CSV and reanalyzes the failed file
    run("--retry-failed")
    lines = csv.read_text().splitlines(keepends=True)
    assert len(lines) == 3 and lines[2].endswith(",full\n")
    assert not (d / "b.failed").exists()
