# Licensed under a 3-clause BSD style license - see LICENSE.rst

import concurrent.futures
import importlib
import shutil
import sys
from unittest.mock import patch

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from mmtwfs.zernike import ZernikeVector
from mmtwfs.scripts.reanalyze import upgrade_cached_line, process_image, wfs_systems, CSV_HEADER, main

WFS_DATA_DIR = importlib.resources.files("mmtwfs") / "data"


COLS = CSV_HEADER.strip().split(",")


def _field(line, col):
    return line.strip().split(",")[COLS.index(col)]


def test_upgrade_cached_line():
    ncols = len(COLS)
    old = ",".join(["1"] * (ncols - 2)) + "\n"
    new = upgrade_cached_line(old)
    assert _field(new, "method") == "full"
    # with no saved wavefront there's nothing to compute the delivered IQ from
    assert _field(new, "delivered_iq") == "nan"
    assert len(new.strip().split(",")) == ncols
    # current lines pass through untouched
    assert upgrade_cached_line(new) == new


def _cached_line(method="full", raw_vlt_seeing=0.9):
    """a line cached before the delivered_iq column existed"""
    fields = ["1"] * (len(COLS) - 1)
    fields[COLS.index("wfs")] = "mmirs"
    fields[COLS.index("raw_vlt_seeing")] = str(raw_vlt_seeing)
    fields[COLS.index("method")] = method
    return ",".join(fields) + "\n"


def test_upgrade_backfills_delivered_iq(tmp_path):
    zfile = tmp_path / "frame.reanalyze.zernike"
    zv = ZernikeVector(Z04=500 * u.nm, Z07=200 * u.nm)
    zv.save(filename=zfile)
    expected, _ = wfs_systems["mmirs"].telescope.psf(zv, seeing=0.9 * u.arcsec, plot=False)

    new = upgrade_cached_line(_cached_line(), zfile=zfile)
    assert len(new.strip().split(",")) == len(COLS)
    assert new.endswith("\n")
    assert np.isclose(float(_field(new, "delivered_iq")), expected.delivered_fwhm.value)
    assert upgrade_cached_line(new, zfile=zfile) == new

    # focus-only rows have no seeing, and a missing or unusable seeing gives no delivered IQ
    for line in [_cached_line(method="periodicity"), _cached_line(raw_vlt_seeing="nan")]:
        assert _field(upgrade_cached_line(line, zfile=zfile), "delivered_iq") == "nan"
    # neither does a missing or unreadable wavefront
    assert _field(upgrade_cached_line(_cached_line(), zfile=tmp_path / "nope"), "delivered_iq") == "nan"
    zfile.write_text("{not json")
    assert _field(upgrade_cached_line(_cached_line(), zfile=zfile), "delivered_iq") == "nan"


def test_psf_failure_keeps_full_analysis(tmp_path):
    # a problem with the PSF calculation costs only the delivered IQ, not the rest of the frame's results
    f = _copy_frame(tmp_path)
    with patch.object(wfs_systems["mmirs"].telescope, "psf", side_effect=ValueError("bad PSF")):
        line = process_image(f)
    plt.close("all")
    assert _field(line, "method") == "full"
    assert _field(line, "delivered_iq") == "nan"
    assert not (tmp_path / "mmirs_wfs_0150.failed").exists()


def test_retry_failed(tmp_path):
    f = tmp_path / "mmirs_wfs_0150.fits"
    shutil.copy(WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits", f)
    failed = tmp_path / "mmirs_wfs_0150.failed"
    failed.touch()  # marked as failed by an earlier run

    assert process_image(f) is None  # skipped by default
    line = process_image(f, retry_failed=True)
    plt.close("all")
    assert line is not None and _field(line, "method") == "full"
    assert not failed.exists()  # a successful retry clears the marker
    assert (tmp_path / "mmirs_wfs_0150.output").exists()

    # the delivered IQ is the optics PSF at 500 nm convolved with the seeing as observed
    zv = ZernikeVector()
    zv.load(filename=tmp_path / "mmirs_wfs_0150.reanalyze.zernike")
    seeing = float(_field(line, "raw_vlt_seeing")) * u.arcsec
    expected, _ = wfs_systems["mmirs"].telescope.psf(zv, band="500nm", seeing=seeing, plot=False)
    iq = float(_field(line, "delivered_iq"))
    assert np.isclose(iq, expected.delivered_fwhm.value)
    assert 1.1 < iq < 1.2


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
    assert fields[cols.index("delivered_iq")] == "nan"
    assert len(fields) == len(cols)

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
    return ",".join(["1"] * (ncols - 2)) + "\n"


def test_cached_output_upgraded(tmp_path):
    # results cached before the method column existed are reused and gain the column
    f = tmp_path / "mmirs_wfs_0150.fits"
    f.touch()  # never read when there's a cached result
    (tmp_path / "mmirs_wfs_0150.output").write_text(_old_cached_line())
    assert process_image(f) == upgrade_cached_line(_old_cached_line())


def test_cached_output_backfilled_once(tmp_path):
    # results cached before the delivered_iq column existed get it from the saved wavefront, and the cache is
    # rewritten so the PSF isn't recomputed on every rebuild of the CSV
    f = tmp_path / "mmirs_wfs_0150.fits"
    f.touch()
    zv = ZernikeVector(Z04=500 * u.nm)
    zv.save(filename=tmp_path / "mmirs_wfs_0150.reanalyze.zernike")
    outfile = tmp_path / "mmirs_wfs_0150.output"
    outfile.write_text(_cached_line())

    line = process_image(f)
    assert float(_field(line, "delivered_iq")) > 0.9
    assert outfile.read_text() == line
    with patch.object(wfs_systems["mmirs"].telescope, "psf", side_effect=AssertionError("recomputed")):
        assert process_image(f) == line


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
    assert len(lines) == 3 and _field(lines[2], "method") == "full"
    assert not (d / "b.failed").exists()
