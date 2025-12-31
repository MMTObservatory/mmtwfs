# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import importlib
from unittest.mock import patch, MagicMock

import numpy as np

import matplotlib.pyplot as plt

from mmtwfs.zernike import ZernikeVector
from mmtwfs.config import mmtwfs_config
from mmtwfs.wfs import WFSFactory, check_wfsdata, mk_wfs_mask, wfsfind
from mmtwfs.custom_exceptions import WFSConfigException, WFSCommandException, WFSAnalysisFailed


WFS_DATA_DIR = importlib.resources.files("mmtwfs") / "data"


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
    assert (testval > 416) & (testval < 436)
    plt.close("all")


def test_mmirs_pacman():
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_rename_0566.fits"
    mmirs = WFSFactory(wfs="mmirs")
    results = mmirs.measure_slopes(test_file)
    testval = results["xcen"]
    assert (testval > 227) & (testval < 229)
    plt.close("all")


def test_mmirs_pupil_mask():
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
    assert results["slopes"] is None
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
    assert results["slopes"] is None
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
