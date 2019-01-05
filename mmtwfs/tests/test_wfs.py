# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import pkg_resources
import os

import numpy as np

from matplotlib.testing.decorators import cleanup

from ..zernike import ZernikeVector
from ..config import mmt_config
from ..wfs import WFSFactory, check_wfsdata
from ..custom_exceptions import WFSConfigException, WFSCommandException


def test_check_wfsdata():
    try:
        check_wfsdata("bogus.fits")
    except WFSConfigException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

    try:
        check_wfsdata([[1.0, 1.0], [1, 1]])
    except WFSConfigException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

    try:
        arr = np.zeros((5, 5, 5))
        check_wfsdata(arr)
    except WFSConfigException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_wfses():
    for s in mmt_config['wfs']:
        wfs = WFSFactory(wfs=s, plot=True, test="foo")
        assert(wfs.test == "foo")

def test_bogus_wfs():
    try:
        wfs = WFSFactory(wfs="bazz")
    except WFSConfigException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_connect():
    wfs = WFSFactory(wfs='f5')
    wfs.connect()
    wfs.disconnect()
    assert(not wfs.connected)  # can't always access systems...

@cleanup
def test_mmirs_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "mmirs_wfs_0150.fits"))
    mmirs = WFSFactory(wfs='mmirs')
    results = mmirs.measure_slopes(test_file)
    zresults = mmirs.fit_wavefront(results)
    testval = int(zresults['zernike']['Z10'].value)
    assert((testval > 205) & (testval < 220))

@cleanup
def test_f9_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "TREX_p500_0000.fits"))
    f9 = WFSFactory(wfs='f9')
    results = f9.measure_slopes(test_file)
    zresults = f9.fit_wavefront(results)
    testval = int(zresults['zernike']['Z09'].value)
    assert((testval > 475) & (testval < 495))

@cleanup
def test_newf9_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "test_newf9.fits"))
    f9 = WFSFactory(wfs='newf9')
    results = f9.measure_slopes(test_file)
    zresults = f9.fit_wavefront(results)
    testval = int(zresults['zernike']['Z09'].value)
    assert((testval > 150) & (testval < 170))

@cleanup
def test_f5_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "auto_wfs_0037_ave.fits"))
    f5 = WFSFactory(wfs='f5')
    results = f5.measure_slopes(test_file)
    zresults = f5.fit_wavefront(results)
    testval = int(zresults['zernike']['Z10'].value)
    assert((testval > -105) & (testval < -90))

@cleanup
def test_bino_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "wfs_ff_cal_img_2017.1113.111402.fits"))
    wfs = WFSFactory(wfs='binospec')
    results = wfs.measure_slopes(test_file, mode="binospec")
    zresults = wfs.fit_wavefront(results)
    testval = int(zresults['zernike']['Z10'].value)
    assert((testval > 180) & (testval < 200))

@cleanup
def test_too_few_spots():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "mmirs_bogus.fits"))
    mmirs = WFSFactory(wfs='mmirs')
    results = mmirs.measure_slopes(test_file)
    assert(results['slopes'] == None)

@cleanup
def test_no_spots():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "mmirs_blank.fits"))
    mmirs = WFSFactory(wfs='mmirs')
    results = mmirs.measure_slopes(test_file)
    assert(results['slopes'] == None)

def test_correct_primary():
    wfs = WFSFactory(wfs='f5')
    zv = ZernikeVector(Z04=1000)
    f, m1f, zv_masked = wfs.calculate_primary(zv)
    assert(m1f == 0.0)

def test_correct_focus():
    wfs = WFSFactory(wfs='f5')
    zv = ZernikeVector()
    corr = wfs.calculate_focus(zv)
    assert(corr == 0.0)

def test_correct_coma():
    wfs = WFSFactory(wfs='f5')
    zv = ZernikeVector()
    cx, cy = wfs.calculate_cc(zv)
    assert(cx == 0.0)
    assert(cy == 0.0)

def test_recenter():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "auto_wfs_0037_ave.fits"))
    f5 = WFSFactory(wfs='f5')
    results = f5.measure_slopes(test_file, plot=False)
    az, el = f5.calculate_recenter(results)
    assert(np.abs(az) > 0.0)
    assert(np.abs(el) > 0.0)

def test_clear():
    wfs = WFSFactory(wfs='f5')
    clear_forces, clear_m1f, cmds = wfs.clear_corrections()
    assert(clear_m1f == 0.0)
    assert(np.allclose(clear_forces['force'], 0.0))
