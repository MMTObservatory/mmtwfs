# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import os
import importlib
import filecmp

import matplotlib.pyplot as plt

import numpy as np
import astropy.units as u

from mmtwfs.config import mmtwfs_config
from mmtwfs.telescope import TelescopeFactory, MMT
from mmtwfs.zernike import ZernikeVector
from mmtwfs.custom_exceptions import WFSConfigException


def test_telescope():
    for s in mmtwfs_config["secondary"]:
        tel = mmtwfs_config["secondary"][s]["telescope"]
        t = TelescopeFactory(telescope=tel, secondary=s)
        assert t.secondary.diameter == mmtwfs_config["secondary"][s]["diameter"]


def test_pupil_mask():
    for s in mmtwfs_config["secondary"]:
        tel = mmtwfs_config["secondary"][s]["telescope"]
        t = TelescopeFactory(telescope=tel, secondary=s)
        mask = t.pupil_mask(size=400)
        assert mask.shape == (400, 400)
        assert mask.max() == 1.0
        assert mask.min() == 0.0


def test_bogus_pupil_mask():
    for s in mmtwfs_config["secondary"]:
        tel = mmtwfs_config["secondary"][s]["telescope"]
        t = TelescopeFactory(telescope=tel, secondary=s)
        try:
            t.pupil_mask(size=900)
        except WFSConfigException:
            assert True
        except Exception as e:
            assert e is not None
            assert False
        else:
            assert False


def test_psf():
    for s in mmtwfs_config["secondary"]:
        tel = mmtwfs_config["secondary"][s]["telescope"]
        t = TelescopeFactory(telescope=tel, secondary=s)
        zv = ZernikeVector(Z04=500 * u.nm)
        p, p_fig = t.psf(zv=zv)
        p_im = p[0].data
        assert p_im.max() < 1.0
        assert p_fig is not None
        plt.close("all")


def test_force_file():
    t = MMT()
    # define a zernike vector with AST45 of -1000 nm and check if the correction equals the forces required to bend
    # +1000 nm of AST45 into the mirror.
    zv = ZernikeVector(Z05=1000)
    f_table = t.bending_forces(zv=zv, gain=1.0)
    t.to_rcell(f_table, filename="forcefile")
    test_file = (
        importlib.resources.files("mmtwfs") / "data" / "test_data" / "AST45_p1000.frc"
    )
    assert filecmp.cmp("forcefile", test_file)
    os.remove("forcefile")


def test_correct_primary():
    t = MMT()
    zv = ZernikeVector(Z05=1000, Z11=250)
    force, focus, zv_masked = t.calculate_primary_corrections(zv)
    lforce, lfocus = t.correct_primary(force, focus)
    assert np.abs(focus) > 0.0
    uforce, ufocus = t.undo_last()
    assert ufocus == -1 * focus
    assert np.allclose(uforce["force"], -force["force"])
    nullforce, nullfocus = t.clear_forces()
    assert nullfocus == 0.0
    assert np.allclose(nullforce["force"].data, 0.0)


def test_plots():
    t = MMT()
    zv = ZernikeVector(Z05=1000, Z11=250)
    f_table = t.bending_forces(zv=zv)
    fig = t.plot_forces(f_table)
    assert fig is not None
    plt.close("all")
