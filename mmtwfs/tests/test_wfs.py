# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import importlib
from unittest.mock import patch

import numpy as np
import astropy.units as u
import pytest

import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.io import fits

from mmtwfs.zernike import ZernikeVector
from mmtwfs.config import mmtwfs_config
from mmtwfs.wfs import WFSFactory, check_wfsdata, mk_wfs_mask, wfsfind, get_apertures
from mmtwfs.custom_exceptions import WFSConfigException, WFSCommandException, WFSAnalysisFailed


WFS_DATA_DIR = importlib.resources.files("mmtwfs") / "data"


@pytest.fixture(autouse=True)
def close_figures():
    # creating a WFS draws reference figures; don't let them pile up past matplotlib's open-figure warning
    yield
    plt.close("all")


def _analyze_image(wfs, test_file):
    results = wfs.measure_slopes(test_file)
    zresults = wfs.fit_wavefront(results)
    plt.close("all")
    return zresults


def test_check_wfsdata():
    try:
        check_wfsdata("bogus.fits")
    except WFSConfigException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False

    try:
        check_wfsdata([[1.0, 1.0], [1, 1]])
    except WFSConfigException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False

    try:
        arr = np.zeros((5, 5, 5))
        check_wfsdata(arr)
    except WFSConfigException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False


def test_wfses():
    for s in mmtwfs_config["wfs"]:
        wfs = WFSFactory(wfs=s, plot=True, test="foo")
        assert wfs.test == "foo"
    plt.close("all")


def test_bogus_wfs():
    try:
        WFSFactory(wfs="bazz")
    except WFSConfigException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False


def test_make_mask():
    test_file = WFS_DATA_DIR / "test_data" / "test_newf9.fits"
    mask = mk_wfs_mask(test_file, thresh_factor=4.0, outfile=None)
    assert mask.min() == 0.0


def test_mmirs_analysis(benchmark):
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    mmirs = WFSFactory(wfs="mmirs")
    zresults = benchmark(_analyze_image, mmirs, test_file)
    testval = int(zresults["zernike"]["Z10"].value)
    # recalibrated for photutils 3.0 (and the numpy/scipy/astropy upgrade), which shifts Z10 to ~398 nm
    assert (testval > 388) & (testval < 408)
    plt.close("all")


def test_mmirs_pacman():
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_rename_0566.fits"
    mmirs = WFSFactory(wfs="mmirs")
    results = mmirs.measure_slopes(test_file)
    testval = results["xcen"]
    assert (testval > 227) & (testval < 229)
    plt.close("all")


def test_mmirs_plotgrid_hdr():
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    mmirs = WFSFactory(wfs="mmirs")
    data, hdr = check_wfsdata(test_file, header=True)
    fig, ax = plt.subplots()
    ngood = mmirs.plotgrid_hdr(hdr, ax)
    assert ngood > 0
    plt.close("all")


def test_mmirs_pickoff_plots():
    mmirs = WFSFactory(wfs="mmirs")
    fig, ax = plt.subplots()
    mmirs.drawoutline(ax)
    # Some representative positions that vignette on different edges of the mirror
    mmirs.plotgrid(-50, -60, ax)
    mmirs.plotgrid(-45, -40, ax)
    mmirs.plotgrid(-7, -52, ax)
    mmirs.plotgrid(50, 60, ax)
    mmirs.plotgrid(45, 40, ax)
    mmirs.plotgrid(7, 52, ax)
    assert fig is not None
    plt.close("all")


def test_mmirs_bogus_pupil_mask():
    mmirs = WFSFactory(wfs="mmirs")
    hdr = {}
    fig, ax = plt.subplots()
    try:
        mmirs.plotgrid_hdr(hdr, ax)
    except WFSCommandException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False


def test_f9_analysis(benchmark):
    test_file = WFS_DATA_DIR / "test_data" / "TREX_p500_0000.fits"
    f9 = WFSFactory(wfs="f9")
    zresults = benchmark(_analyze_image, f9, test_file)
    testval = int(zresults["zernike"]["Z09"].value)
    assert (testval > 440) & (testval < 450)
    plt.close("all")


def test_newf9_analysis(benchmark):
    test_file = WFS_DATA_DIR / "test_data" / "test_newf9.fits"
    f9 = WFSFactory(wfs="newf9")
    zresults = benchmark(_analyze_image, f9, test_file)
    testval = int(zresults["zernike"]["Z09"].value)
    assert (testval > 109) & (testval < 129)
    plt.close("all")


def test_f5_analysis(benchmark):
    test_file = WFS_DATA_DIR / "test_data" / "auto_wfs_0037_ave.fits"
    f5 = WFSFactory(wfs="f5")
    zresults = benchmark(_analyze_image, f5, test_file)
    testval = int(zresults["zernike"]["Z10"].value)
    assert (testval > 76) & (testval < 96)
    plt.close("all")


def test_bino_analysis(benchmark):
    test_file = WFS_DATA_DIR / "test_data" / "wfs_ff_cal_img_2017.1113.111402.fits"
    wfs = WFSFactory(wfs="binospec")
    zresults = benchmark(_analyze_image, wfs, test_file)
    testval = int(zresults["zernike"]["Z10"].value)
    assert (testval > 163) & (testval < 183)
    plt.close("all")


def test_flwo_analysis(benchmark):
    test_file = WFS_DATA_DIR / "test_data" / "1195.star.p2m18.fits"
    wfs = WFSFactory(wfs="flwo15")
    zresults = benchmark(_analyze_image, wfs, test_file)
    testval = int(zresults["zernike"]["Z06"].value)
    assert (testval > 700) & (testval < 1000)
    plt.close("all")


def test_too_few_spots():
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_bogus.fits"
    mmirs = WFSFactory(wfs="mmirs")
    results = mmirs.measure_slopes(test_file)
    # too few spots for the full analysis, but the grid is clear enough for the focus-only fallback
    assert results["slopes"] is None
    assert results["focus_only"]
    assert results["method"] == "periodicity"
    assert abs(results["grid"]["scale"] - 0.9895) < 2e-3
    assert -6.0 < results["pending_focus"].to_value(u.um) < 3.0
    plt.close("all")


# def test_no_spots():
#     test_file = WFS_DATA_DIR / "test_data" / "mmirs_blank.fits"
#     mmirs = WFSFactory(wfs="mmirs")
#     results = mmirs.measure_slopes(test_file)
#     assert results["slopes"] is None
#     plt.close("all")


def test_frosted_donut():
    test_file = WFS_DATA_DIR / "test_data" / "f9wfs_20200225-205600.fits"
    wfs = WFSFactory(wfs="newf9")
    results = wfs.measure_slopes(test_file)
    # spot detection fails on the blurred donut, but the hex grid is visible and gives a focus-only correction
    assert results["slopes"] is None
    assert results["focus_only"]
    assert results["method"] == "periodicity"
    assert abs(results["grid"]["scale"] - 0.899) < 5e-3
    assert -40.0 < results["pending_focus"].to_value(u.um) < -24.0
    plt.close("all")


def test_correct_primary():
    wfs = WFSFactory(wfs="f5")
    zv = ZernikeVector(Z04=1000)
    f, m1f, zv_masked = wfs.calculate_primary(zv)
    assert m1f == 0.0


def test_correct_focus():
    wfs = WFSFactory(wfs="f5")
    zv = ZernikeVector()
    corr = wfs.calculate_focus(zv)
    assert corr == 0.0


def test_correct_coma():
    wfs = WFSFactory(wfs="f5")
    zv = ZernikeVector()
    cx, cy = wfs.calculate_cc(zv)
    assert cx == 0.0
    assert cy == 0.0


def test_recenter():
    test_file = WFS_DATA_DIR / "test_data" / "test_newf9.fits"
    f9 = WFSFactory(wfs="newf9")
    results = f9.measure_slopes(test_file, plot=False)
    az, el = f9.calculate_recenter(results)
    assert np.abs(az) > 0.0
    assert np.abs(el) > 0.0
    plt.close("all")


def test_f5_recenter():
    test_file = WFS_DATA_DIR / "test_data" / "auto_wfs_0037_ave.fits"
    f5 = WFSFactory(wfs="f5")
    results = f5.measure_slopes(test_file, plot=False)
    az, el = f5.calculate_recenter(results)
    assert np.abs(az) > 0.0
    assert np.abs(el) > 0.0
    plt.close("all")


def test_clear():
    wfs = WFSFactory(wfs="f5")
    clear_forces, clear_m1f, cmds = wfs.clear_corrections()
    assert clear_m1f == 0.0
    assert np.allclose(clear_forces["force"], 0.0)


def test_wfs_connect_disconnect():
    """Test WFS connect and disconnect methods"""
    wfs = WFSFactory(wfs="f5")
    assert wfs.connected is False

    with patch.object(wfs.telescope, "connect") as mock_tel_connect:
        with patch.object(wfs.secondary, "connect") as mock_sec_connect:
            wfs.telescope.connected = True
            wfs.secondary.connected = True
            wfs.connect()
            mock_tel_connect.assert_called_once()
            mock_sec_connect.assert_called_once()
            assert wfs.connected is True

    with patch.object(wfs.telescope, "disconnect") as mock_tel_disconnect:
        with patch.object(wfs.secondary, "disconnect") as mock_sec_disconnect:
            wfs.disconnect()
            mock_tel_disconnect.assert_called_once()
            mock_sec_disconnect.assert_called_once()
            assert wfs.connected is False


def test_wfs_connect_partial_failure():
    """Test WFS connect when one component fails"""
    wfs = WFSFactory(wfs="f5")

    with patch.object(wfs.telescope, "connect"):
        with patch.object(wfs.secondary, "connect"):
            wfs.telescope.connected = True
            wfs.secondary.connected = False
            wfs.connect()
            assert wfs.connected is False


def test_f9_connect_disconnect():
    """Test F9 WFS connect and disconnect with compmirror"""
    wfs = WFSFactory(wfs="f9")
    assert wfs.connected is False

    with patch.object(wfs.telescope, "connect"):
        with patch.object(wfs.secondary, "connect"):
            with patch.object(wfs.compmirror, "connect") as mock_cm_connect:
                wfs.telescope.connected = True
                wfs.secondary.connected = True
                wfs.connect()
                mock_cm_connect.assert_called_once()
                assert wfs.connected is True

    with patch.object(wfs.telescope, "disconnect"):
        with patch.object(wfs.secondary, "disconnect"):
            with patch.object(wfs.compmirror, "disconnect") as mock_cm_disconnect:
                wfs.disconnect()
                mock_cm_disconnect.assert_called_once()
                assert wfs.connected is False


def test_mmirs_pupil_mask():
    """Test MMIRS pupil_mask method"""
    mmirs = WFSFactory(wfs="mmirs")
    # Use non-zero values to avoid division by zero in onmirror check
    hdr = {"GUIDERX": 10.0, "GUIDERY": 10.0, "CA": 0.0, "CAMERA": 1}
    mask = mmirs.pupil_mask(hdr, npts=10)
    assert mask.shape[0] > 0
    assert mask.shape[1] > 0


def test_mmirs_pupil_mask_camera2():
    """Test MMIRS pupil_mask with camera 2 rotation"""
    mmirs = WFSFactory(wfs="mmirs")
    hdr = {"GUIDERX": 10.0, "GUIDERY": 10.0, "CA": 0.0, "CAMERA": 2}
    mask = mmirs.pupil_mask(hdr, npts=10)
    assert mask.shape[0] > 0


def test_mmirs_pupil_mask_no_position():
    """Test MMIRS pupil_mask with missing position"""
    mmirs = WFSFactory(wfs="mmirs")
    hdr = {"CA": 0.0, "CAMERA": 1}
    try:
        mmirs.pupil_mask(hdr)
    except WFSCommandException:
        assert True
    else:
        assert False


def test_mmirs_pupil_mask_no_ca():
    """Test MMIRS pupil_mask with missing camera rotation"""
    mmirs = WFSFactory(wfs="mmirs")
    hdr = {"GUIDERX": 0.0, "GUIDERY": 0.0, "CAMERA": 1}
    try:
        mmirs.pupil_mask(hdr)
    except WFSCommandException:
        assert True
    else:
        assert False


def test_wfsfind_no_spots():
    """Test wfsfind with no detectable spots"""
    import warnings
    from photutils.utils.exceptions import NoDetectionsWarning
    data = np.random.normal(0, 1, (100, 100))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDetectionsWarning)
        try:
            wfsfind(data, fwhm=5.0, threshold=100.0, plot=False)
        except WFSAnalysisFailed:
            assert True
        else:
            assert False


def test_wfsfind_few_spots():
    """Test wfsfind with too few spots"""
    import warnings
    from photutils.utils.exceptions import NoDetectionsWarning
    data = np.zeros((100, 100))
    # Add just a few spots (less than 5)
    data[25, 25] = 1000
    data[75, 75] = 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDetectionsWarning)
        try:
            wfsfind(data, fwhm=5.0, threshold=3.0, plot=False)
        except WFSAnalysisFailed:
            assert True
        else:
            assert False


def test_mk_wfs_mask_with_outfile(tmp_path):
    """Test mk_wfs_mask with output file"""
    test_file = WFS_DATA_DIR / "test_data" / "test_newf9.fits"
    outfile = tmp_path / "mask_output.fits"
    mask = mk_wfs_mask(test_file, thresh_factor=4.0, outfile=str(outfile))
    assert mask.min() == 0.0
    assert outfile.exists()


def test_get_apertures_background_region():
    # quiet region left of x=150, very noisy to the right. with cen=(100, 400) the 100x100 stats box is
    # entirely in the quiet region; the old typo (xcen - 50:ycen + 50) made it run to x=450.
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 1.0, (512, 512))
    data[:, 150:] = rng.normal(0.0, 100.0, (512, 362))
    captured = {}

    def fake_wfsfind(data, fwhm=7.0, threshold=5.0, plot=True, ap_radius=5.0, std=None):
        captured["std"] = std
        raise RuntimeError("stop after background stats")

    with patch("mmtwfs.wfs.wfsfind", side_effect=fake_wfsfind):
        with pytest.raises(RuntimeError):
            get_apertures(data, 20.0, cen=(100, 400))
    assert captured["std"] < 2.0


def test_get_apertures_background_box_size():
    # quiet within 30 px of cen, noisy beyond: a box that stays inside (like the central obscuration) sees only
    # the quiet part
    rng = np.random.default_rng(43)
    data = rng.normal(0.0, 100.0, (512, 512))
    data[226:286, 226:286] = rng.normal(0.0, 1.0, (60, 60))
    captured = {}

    def fake_wfsfind(data, fwhm=7.0, threshold=5.0, plot=True, ap_radius=5.0, std=None):
        captured["std"] = std
        raise RuntimeError("stop after background stats")

    with patch("mmtwfs.wfs.wfsfind", side_effect=fake_wfsfind):
        with pytest.raises(RuntimeError):
            get_apertures(data, 20.0, cen=(256, 256), box=28)
    assert captured["std"] < 2.0


def test_get_slopes_noise_box_inside_obscuration():
    # the noise box must fit inside the central obscuration, or the first ring of spots inflates the noise
    # estimate on blurry frames
    mmirs = WFSFactory(wfs="mmirs")
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    with patch("mmtwfs.wfs.get_apertures", side_effect=RuntimeError("stop")) as spy:
        with pytest.raises(WFSAnalysisFailed):
            mmirs.measure_slopes(test_file, plot=False)
    assert spy.call_args.kwargs["box"] == int(mmirs.pup_inner / np.sqrt(2.0))


def test_wfs_poor_seeing_defaults():
    for s in mmtwfs_config["wfs"]:
        wfs = WFSFactory(wfs=s)
        assert wfs.bkg_method == "background2d"
        assert wfs.periodicity_fallback
        assert wfs.m2_gain_periodicity == 0.5
    plt.close("all")


def test_reference_image_center_is_fixed():
    mmirs = WFSFactory(wfs="mmirs")
    ref = mmirs.modes["mmirs2"]["reference"]
    x0, y0 = ref.img_xcen, ref.img_ycen
    ref.adjust_center(x0 + 10.0, y0 - 5.0)
    assert (ref.img_xcen, ref.img_ycen) == (x0, y0)
    assert ref.xcen == x0 + 10.0


def test_find_pupil_center_falls_back():
    mmirs = WFSFactory(wfs="mmirs")
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    data, hdr = mmirs.process_image(test_file)
    xc, yc, measured = mmirs.find_pupil_center(data, mmirs.pupil_mask(hdr=hdr))
    assert measured
    assert np.hypot(xc - mmirs.cor_coords[0], yc - mmirs.cor_coords[1]) < mmirs.cen_tol
    xc, yc, measured = mmirs.find_pupil_center(np.zeros((10, 10)), mmirs.pupil_mask(hdr=hdr))
    assert not measured
    assert (xc, yc) == tuple(mmirs.cor_coords)


def test_mmirs_analysis_pupil_background():
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    mmirs = WFSFactory(wfs="mmirs", config={"bkg_method": "pupil"})
    zresults = _analyze_image(mmirs, test_file)
    testval = int(zresults["zernike"]["Z10"].value)
    # same window as test_mmirs_analysis: the new background must not change a good frame's wavefront
    assert (testval > 388) & (testval < 408)


def _mmirs_ready(config={}):
    mmirs = WFSFactory(wfs="mmirs", config=config)
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    data, hdr = check_wfsdata(test_file, header=True)
    mode = mmirs.get_mode(hdr)
    mmirs.prepare_reference(mode, hdr=hdr)
    return mmirs, mode, hdr


def test_focus_from_scale_matches_analytic():
    # pure defocus: slope = (s - 1) * r, and d(fringe Z04)/dr = 4 Z04 r, so Z04 = -tiltfactor * (s - 1) * R / 4
    mmirs, mode, hdr = _mmirs_ready()
    zref = mmirs.reference_aberrations(mode, hdr=hdr)
    k = -mmirs.tiltfactor * (mmirs.pup_size / 2.0) / 4.0
    zv, focus = mmirs.focus_from_scale(1.01, 1e-4, mode, 0.0 * u.deg, hdr=hdr)
    expected = k * 0.01 - zref["Z04"].value
    assert np.isclose(zv["Z04"].value, expected, rtol=0.02)
    assert np.isclose(zv.errorbars["Z04"].value, abs(k) * 1e-4, rtol=0.02)


def test_focus_from_scale_gain():
    mmirs, mode, hdr = _mmirs_ready()
    zv, focus = mmirs.focus_from_scale(0.99, 1e-5, mode, 0.0 * u.deg, hdr=hdr)
    assert np.isclose(focus.value, 0.5 * mmirs.calculate_focus(zv.copy()).value, atol=0.02)


def test_focus_from_scale_large_error_is_zero():
    mmirs, mode, hdr = _mmirs_ready()
    zv, focus = mmirs.focus_from_scale(0.99, 1.0, mode, 0.0 * u.deg, hdr=hdr)
    assert focus.value == 0.0


def test_focus_from_scale_clipped():
    mmirs, mode, hdr = _mmirs_ready({"m2_gain_periodicity": 1.0})
    zv, focus = mmirs.focus_from_scale(0.8, 1e-5, mode, 0.0 * u.deg, hdr=hdr)
    assert abs(focus.to_value(u.um)) == 300.0


def test_reference_grid_cached():
    mmirs, mode, hdr = _mmirs_ready()
    g = mmirs.reference_grid(mode)
    assert g is not None
    ref = mmirs.modes[mode]["reference"]
    assert np.allclose(g["spacing"], np.mean([ref.xspacing, ref.yspacing]), rtol=0.02)
    assert mmirs.reference_grid(mode) is g
    # the cache lives as long as the WFS object (e.g. in wfssrv); keep only the small arrays, not the spectrum
    assert "power" not in g and "freq_axis" not in g


def test_periodicity_no_reference_grid():
    # a reference image whose grid can't be measured gives no fallback rather than a bogus scale
    mmirs, mode, hdr = _mmirs_ready()
    ref = mmirs.modes[mode]["reference"]
    ref.data = np.zeros_like(ref.data)
    assert mmirs.reference_grid(mode) is None
    assert mmirs.periodicity_focus(ref.data, mode, (ref.img_xcen, ref.img_ycen), 0.0 * u.deg, plot=False) is None


def _blurred_mmirs(tmp_path, sigma):
    # gaussian blur of a good MMIRS frame: sigma >= 5 makes the spot analysis fail but leaves the grid visible
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    with fits.open(test_file) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header.copy()
    out = tmp_path / f"mmirs_blur{sigma}.fits"
    # the original header has lower-case keywords (ActualX) that need fixing to write
    fits.writeto(out, ndimage.gaussian_filter(data, sigma).astype(np.float32), hdr, output_verify="silentfix")
    return out


def test_periodicity_fallback(tmp_path):
    mmirs = WFSFactory(wfs="mmirs", config={"m2_gain_periodicity": 1.0})
    results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 6), plot=True)
    assert results["slopes"] is None
    assert results["focus_only"]
    assert results["method"] == "periodicity"
    # full analysis of the unblurred frame gives about -45 um; the fallback on this blur gave -43.5 um in a prototype
    assert -60.0 < results["pending_focus"].to_value(u.um) < -30.0
    assert results["grid"]["scale_err"] > 0.0
    assert results["grid"]["center_measured"]
    assert results["figures"]["periodicity"].get_label() == "Grid Periodicity"
    plt.close("all")


def test_periodicity_matches_full_fit(tmp_path):
    # sigma = 3 blur still passes the spot analysis, so both answers are available on the same frame
    mmirs = WFSFactory(wfs="mmirs", config={"m2_gain_periodicity": 1.0})
    results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 3), plot=False)
    assert results["slopes"] is not None
    full = mmirs.calculate_focus(mmirs.fit_wavefront(results, plot=False)["zernike"])
    fb = mmirs.periodicity_focus(
        results["data"], results["mode"], (results["xcen"], results["ycen"]), results["rotator"],
        hdr=results["header"], plot=False,
    )
    assert abs(fb["pending_focus"].to_value(u.um) - full.to_value(u.um)) < 10.0
    plt.close("all")


def test_periodicity_fallback_disabled(tmp_path):
    mmirs = WFSFactory(wfs="mmirs", config={"periodicity_fallback": False})
    results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 6), plot=False)
    assert results["slopes"] is None
    assert "focus_only" not in results


# make_spot_mask() warns on a blank frame; pytest's "error" filter would otherwise stop the analysis before
# it reaches the fallback.
@pytest.mark.filterwarnings("ignore::photutils.utils.exceptions.NoDetectionsWarning")
def test_periodicity_no_grid(tmp_path):
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    with fits.open(test_file) as hdul:
        hdr = hdul[0].header.copy()
        shape = hdul[0].data.shape
    noise = np.random.default_rng(9).normal(1250.0, 5.0, shape).astype(np.float32)
    out = tmp_path / "mmirs_noise.fits"
    fits.writeto(out, noise, hdr, output_verify="silentfix")
    mmirs = WFSFactory(wfs="mmirs")
    try:
        results = mmirs.measure_slopes(out, plot=False)
    except WFSAnalysisFailed:
        return  # also acceptable: no correction either way
    assert results["slopes"] is None
    assert not results.get("focus_only", False)


def test_periodicity_fallback_swallows_errors(tmp_path):
    mmirs = WFSFactory(wfs="mmirs")
    with patch.object(mmirs, "periodicity_focus", side_effect=RuntimeError("boom")):
        results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 6), plot=False)
    assert results["slopes"] is None
    assert "focus_only" not in results


def test_periodicity_hex_reference_spacing():
    # newf9 is a hex grid: xspacing (33.8 px) is the grid line spacing and yspacing (19.5 px) is half the row offset,
    # so their mean is not the grid period. the search must be centered on the measured reference grid instead.
    wfs = WFSFactory(wfs="newf9")
    mode = "spol"
    ref = wfs.prepare_reference(mode)
    scale = 1.05
    c = np.array([ref.img_ycen, ref.img_xcen])
    data = ref.data - np.median(ref.data)
    zoomed = ndimage.affine_transform(data, np.eye(2) / scale, offset=c - c / scale, order=1)
    fb = wfs.periodicity_focus(zoomed, mode, (ref.img_xcen, ref.img_ycen), 0.0 * u.deg, plot=False)
    assert fb is not None
    assert abs(fb["grid"]["scale"] - scale) < 3e-3


def test_pupil_background_failure_is_contained():
    # a pupil background that can't be fit (e.g. too few blocks outside the pupil) must not escape measure_slopes;
    # the analysis carries on with a constant background
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    mmirs = WFSFactory(wfs="mmirs", config={"bkg_method": "pupil"})
    with patch("mmtwfs.wfs.pupil_background", side_effect=ValueError("too few background blocks")):
        results = mmirs.measure_slopes(test_file, plot=False)
    assert results["slopes"] is not None or results.get("focus_only", False)


@pytest.mark.parametrize("method", ["Pupil", "bkg2d", None])
def test_bkg_method_validated(method):
    # a typo would otherwise silently mean no background subtraction at all
    with pytest.raises(WFSConfigException):
        WFSFactory(wfs="mmirs", config={"bkg_method": method})


def test_periodicity_calibration():
    # calibrated on the October 2026 MMIRS run: below grid SNR ~300 the frame-to-frame scatter of fallback focus
    # was 1.5-2.4x larger than the propagated errors; above it the errors needed scaling by 1.48. 250 keeps the
    # F/9 frosted-donut frame (SNR 262), whose correction looks right.
    assert WFSFactory(wfs="f5").period_snr_thresh == 250.0
    mmirs = WFSFactory(wfs="mmirs")
    assert mmirs.period_snr_thresh == 250.0
    assert mmirs.period_err_factor == 1.48
    assert mmirs.bkg_method == "background2d"


@pytest.mark.parametrize("enabled", [True, False])
def test_bkg_pedestal_option(enabled):
    # the config switch is bkg_pedestal, alongside bkg_method/bkg_box/bkg_order; pedestal() is the model function
    mmirs = WFSFactory(wfs="mmirs", config={"bkg_method": "pupil", "bkg_pedestal": enabled})
    assert not hasattr(mmirs, "pedestal")
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    data, hdr = mmirs.process_image(test_file)
    mode = mmirs.get_mode(hdr)
    with patch("mmtwfs.wfs.pedestal", return_value=np.zeros_like(data)) as ped:
        mmirs.subtract_pupil_background(data, mode, mmirs.cor_coords)
    assert ped.called == enabled


def test_focus_from_scale_f9_blue_not_clipped():
    # in-focus F/9 blue frames sit at grid scale ~0.93 because the reference carries ~8300 nm of Z04, so a real
    # -150 um correction (scale ~0.87, seen on 2024-2026 data) must pass the +/-300 um clip unchanged
    wfs = WFSFactory(wfs="newf9", config={"m2_gain_periodicity": 1.0})
    mode = "blue"
    wfs.prepare_reference(mode)
    zv, focus = wfs.focus_from_scale(0.87, 1e-5, mode, 0.0 * u.deg)
    unclipped = wfs.calculate_focus(zv.copy()).to_value(u.um)
    assert -300.0 < unclipped < -100.0
    assert np.isclose(focus.to_value(u.um), unclipped, atol=0.02)
