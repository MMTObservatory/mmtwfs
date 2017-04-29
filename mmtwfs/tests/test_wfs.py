# Licensed under GPL3 (see LICENSE)
# coding=utf-8

import pkg_resources
import os

import numpy as np

from ..zernike import ZernikeVector
from ..config import mmt_config
from ..wfs import WFSFactory
from ..custom_exceptions import WFSConfigException, WFSCommandException


def test_wfses():
    for s in mmt_config['wfs']:
        wfs = WFSFactory(wfs=s, test="foo")
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
    assert(wfs.connected)
    wfs.disconnect()
    assert(not wfs.connected)

def test_mmirs_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "mmirs_wfs_0150.fits"))
    mmirs = WFSFactory(wfs='mmirs')
    results = mmirs.measure_slopes(test_file, plot=False)
    zresults = mmirs.fit_wavefront(results, plot=False)
    assert(int(zresults['zernike']['Z10'].value) == -250)

def test_f9_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "TREX_p500_0000.fits"))
    f9 = WFSFactory(wfs='f9')
    results = f9.measure_slopes(test_file, plot=False)
    zresults = f9.fit_wavefront(results, plot=False)
    assert(int(zresults['zernike']['Z09'].value) == 474)

def test_f5_analysis():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "auto_wfs_0037_ave.fits"))
    f5 = WFSFactory(wfs='f5')
    results = f5.measure_slopes(test_file, plot=False)
    zresults = f5.fit_wavefront(results, plot=False)
    assert(int(zresults['zernike']['Z10'].value) == 82)

def test_too_few_spots():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "mmirs_bogus.fits"))
    mmirs = WFSFactory(wfs='mmirs')
    results = mmirs.measure_slopes(test_file, plot=False)
    assert(results == None)

def test_no_spots():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "mmirs_blank.fits"))
    mmirs = WFSFactory(wfs='mmirs')
    results = mmirs.measure_slopes(test_file, plot=False)
    assert(results == None)

def test_correct_primary():
    wfs = WFSFactory(wfs='f5')
    zv = ZernikeVector()
    f, m1f = wfs.correct_primary(zv)
    assert(m1f == 0.0)
    assert(np.allclose(f['force'], 0.0))

def test_correct_focus():
    wfs = WFSFactory(wfs='f5')
    zv = ZernikeVector()
    corr = wfs.correct_focus(zv)
    assert(corr == 0.0)

def test_correct_coma():
    wfs = WFSFactory(wfs='f5')
    zv = ZernikeVector()
    cx, cy = wfs.correct_coma(zv)
    assert(cx == 0.0)
    assert(cy == 0.0)

def test_recenter():
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "auto_wfs_0037_ave.fits"))
    f5 = WFSFactory(wfs='f5')
    results = f5.measure_slopes(test_file, plot=False)
    az, el = f5.recenter(results)
    assert(np.abs(az) > 0.0)
    assert(np.abs(el) > 0.0)

def test_clear():
    wfs = WFSFactory(wfs='f5')
    clear_forces, clear_m1f = wfs.clear_corrections()
    assert(clear_m1f == 0.0)
    assert(np.allclose(clear_forces['force'], 0.0))
