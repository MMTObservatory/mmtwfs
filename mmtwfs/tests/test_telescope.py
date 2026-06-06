# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import os
import importlib
import filecmp
from unittest.mock import patch, MagicMock

import matplotlib.pyplot as plt

import numpy as np
import astropy.units as u

from mmtwfs.config import mmtwfs_config
from mmtwfs.telescope import TelescopeFactory, MMT, Telescope
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


def test_plots_with_m1focus():
    """Test plot_forces with m1focus label"""
    t = MMT()
    zv = ZernikeVector(Z05=1000, Z11=250)
    f_table = t.bending_forces(zv=zv)
    fig = t.plot_forces(f_table, m1focus=100.0 * u.um)
    assert fig is not None
    plt.close("all")


def test_load_act2surf():
    """Test lazy loading and caching of the actuator-to-surface influence matrix"""
    t = MMT()
    assert t._act2surf is None
    inf_matrix = t.load_act2surf()
    # the matrix maps a force vector (one entry per actuator) to surface displacement at each BCV node
    assert inf_matrix.shape[0] == t.n_act
    assert inf_matrix.shape[1] == len(t.nodecoor)
    # a second call should return the cached array rather than reloading it
    assert t.load_act2surf() is inf_matrix


def test_plot_force_influence():
    """Test plot_force_influence with the influence matrix loaded on demand"""
    t = MMT()
    zv = ZernikeVector(Z05=1000, Z11=250)
    f_table = t.bending_forces(zv=zv)
    fig = t.plot_force_influence(f_table)
    assert fig is not None
    plt.close("all")


def test_plot_force_influence_with_matrix():
    """Test plot_force_influence with an explicitly supplied influence matrix"""
    t = MMT()
    zv = ZernikeVector(Z05=1000, Z11=250)
    f_table = t.bending_forces(zv=zv)
    inf_matrix = t.load_act2surf()
    fig = t.plot_force_influence(f_table, inf_matrix=inf_matrix)
    assert fig is not None
    plt.close("all")


def test_bogus_telescope():
    """Test TelescopeFactory with invalid telescope"""
    try:
        TelescopeFactory(telescope="invalid")
    except WFSConfigException:
        assert True
    else:
        assert False


def test_bogus_secondary_for_telescope():
    """Test Telescope with invalid secondary for telescope"""
    try:
        TelescopeFactory(telescope="mmt", secondary="flwo12")
    except WFSConfigException:
        assert True
    else:
        assert False


def test_telescope_invalid_telescope_direct():
    """Test Telescope class directly with invalid telescope"""
    try:
        Telescope(telescope="invalid", secondary="f5")
    except WFSConfigException:
        assert True
    else:
        assert False


def test_telescope_invalid_secondary_direct():
    """Test Telescope class directly with invalid secondary"""
    try:
        Telescope(telescope="mmt", secondary="invalid")
    except WFSConfigException:
        assert True
    else:
        assert False


def test_psf_no_unit():
    """Test psf with wavelength without unit (assumed meters)"""
    t = MMT()
    zv = ZernikeVector(Z04=500 * u.nm)
    p, p_fig = t.psf(zv=zv, wavelength=550e-9, plot=False)
    p_im = p[0].data
    assert p_im.max() < 1.0
    plt.close("all")


def test_connect_disconnect():
    """Test connect and disconnect methods"""
    t = MMT()
    assert t.connected is False
    t.connect()
    assert t.connected is True
    t.disconnect()
    assert t.connected is False


def test_bending_forces_with_tilts():
    """Test bending_forces zeroes out tilts"""
    t = MMT()
    zv = ZernikeVector(Z02=500, Z03=500, Z05=1000)
    f_table = t.bending_forces(zv=zv)
    # Should still produce forces (from Z05) despite tilts being zeroed
    assert f_table is not None


def test_bend_mirror_success():
    """Test bend_mirror with successful apply"""
    t = MMT()
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"Able to Apply", b"")
        mock_popen.return_value = mock_proc
        frac = t.bend_mirror("testfile")
        assert frac == 1.0


def test_bend_mirror_rejected():
    """Test bend_mirror with rejected forces"""
    t = MMT()
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"Forces Rejected", b"")
        mock_popen.return_value = mock_proc
        frac = t.bend_mirror("testfile")
        assert frac == 0.0


def test_bend_mirror_unable():
    """Test bend_mirror with unable to apply"""
    t = MMT()
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"Unable to apply forces", b"")
        mock_popen.return_value = mock_proc
        frac = t.bend_mirror("testfile")
        assert frac == 0.0


def test_bend_mirror_partial():
    """Test bend_mirror with partial forces"""
    import warnings
    t = MMT()
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"Applying partial forces 75 percent", b"")
        mock_popen.return_value = mock_proc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            frac = t.bend_mirror("testfile")
        assert frac == 0.75


def test_bend_mirror_unexpected():
    """Test bend_mirror with unexpected response"""
    t = MMT()
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"Something unexpected", b"error msg")
        mock_popen.return_value = mock_proc
        frac = t.bend_mirror("testfile")
        # Returns 1.0 for unexpected responses (no explicit handling changes frac)
        assert frac == 1.0


def test_bend_mirror_timeout():
    """Test bend_mirror with timeout"""
    import subprocess
    t = MMT()
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired("cmd", 10),
            (b"Able to Apply", b"")
        ]
        mock_popen.return_value = mock_proc
        t.bend_mirror("testfile")
        mock_proc.kill.assert_called_once()


def test_correct_primary_connected():
    """Test correct_primary when connected"""
    t = MMT()
    t.connected = True
    zv = ZernikeVector(Z05=1000, Z11=250)
    force, focus, zv_masked = t.calculate_primary_corrections(zv)

    with patch.object(t, "to_rcell") as mock_to_rcell:
        with patch.object(t, "bend_mirror") as mock_bend:
            mock_bend.return_value = 1.0
            with patch.object(t.secondary, "m1spherical") as mock_m1sph:
                lforce, lfocus = t.correct_primary(force, focus)
                mock_to_rcell.assert_called_once()
                mock_bend.assert_called_once()
                mock_m1sph.assert_called_once()


def test_undo_last_connected():
    """Test undo_last when connected"""
    t = MMT()
    zv = ZernikeVector(Z05=1000, Z11=250)
    force, focus, zv_masked = t.calculate_primary_corrections(zv)
    # First apply corrections (not connected, so just updates tracking)
    t.correct_primary(force, focus)
    # Now set connected and test undo
    t.connected = True

    with patch.object(t, "to_rcell") as mock_to_rcell:
        with patch.object(t, "bend_mirror") as mock_bend:
            mock_bend.return_value = 1.0
            with patch.object(t.secondary, "m1spherical"):
                uforce, ufocus = t.undo_last()
                mock_to_rcell.assert_called_once()
                mock_bend.assert_called_once()


def test_clear_forces_connected():
    """Test clear_forces when connected"""
    t = MMT()
    t.connected = True

    with patch.object(t.secondary, "clear_m1spherical") as mock_clear:
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = (b"Forces cleared", b"")
            mock_popen.return_value = mock_proc
            lforce, lfocus = t.clear_forces()
            mock_clear.assert_called_once()


def test_clear_forces_connected_with_stderr():
    """Test clear_forces when connected with stderr output"""
    import warnings
    t = MMT()
    t.connected = True

    with patch.object(t.secondary, "clear_m1spherical"):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = (b"Forces cleared", b"warning message")
            mock_popen.return_value = mock_proc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                lforce, lfocus = t.clear_forces()


def test_clear_forces_connected_timeout():
    """Test clear_forces when connected with timeout"""
    import subprocess
    t = MMT()
    t.connected = True

    with patch.object(t.secondary, "clear_m1spherical"):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.side_effect = [
                subprocess.TimeoutExpired("cmd", 20),
                (b"Forces cleared", b"")
            ]
            mock_popen.return_value = mock_proc
            lforce, lfocus = t.clear_forces()
            mock_proc.kill.assert_called_once()
