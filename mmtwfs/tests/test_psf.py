# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import pytest
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from mmtwfs.zernike import ZernikeVector
from mmtwfs.telescope import MMT, TelescopeFactory
from mmtwfs.config import mmtwfs_config
from mmtwfs.custom_exceptions import WFSConfigException
from mmtwfs.psf import (
    PSF_BANDS,
    band_wavelengths,
    pupil_model,
    area_fwhm,
    ee50_diameter,
    broadband_psf,
    MAX_OPTICS_NPIX,
)

D = 6.5 * u.m


def lam_over_d(wave, diameter=D):
    return (wave.to(u.m) / diameter.to(u.m)).value * u.rad.to(u.arcsec) * u.arcsec


def clear_psf(**kwargs):
    """PSF of an unobscured, unaberrated circular aperture"""
    return broadband_psf(ZernikeVector(), diameter=D, obscuration_diameter=0 * u.m, **kwargs)


def test_bands():
    assert PSF_BANDS["500nm"] == (500 * u.nm, 0.1)
    for b in ["R", "I", "z", "J", "H", "K"]:
        assert b in PSF_BANDS
    waves = band_wavelengths("500nm", nwave=11)
    assert len(waves) == 11
    assert u.allclose(waves[[0, -1]], [450, 550] * u.nm)
    assert u.allclose(band_wavelengths((1 * u.um, 0.0), nwave=5), [1] * u.um)


def test_bogus_band():
    with pytest.raises(WFSConfigException):
        band_wavelengths("Q")


def test_pupil_model():
    pup, dp = pupil_model(512, D, 1.0 * u.m, n_supports=4, support_width=0.1 * u.m, support_offset=45 * u.deg)
    assert pup.shape == (512, 512)
    assert u.isclose(dp, D / 512)
    assert pup[256, 256] == 0  # behind the secondary
    assert pup[0, 0] == 0  # outside the primary
    open_area = pup.sum() * dp**2
    spiders = 4 * 0.1 * u.m * (D - 1.0 * u.m) / 2
    expected = np.pi * ((D / 2) ** 2 - (0.5 * u.m) ** 2) - spiders
    assert u.isclose(open_area, expected, rtol=0.01)


def test_area_fwhm():
    y, x = np.mgrid[-100:101, -100:101]
    sigma = 10.0
    im = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    fwhm = area_fwhm(im, 0.01 * u.arcsec)
    assert u.isclose(fwhm, 2.3548 * sigma * 0.01 * u.arcsec, rtol=0.02)


def test_ee50_diameter():
    y, x = np.mgrid[-100:101, -100:101]
    sigma = 10.0
    im = np.exp(-((x - 3) ** 2 + (y + 5) ** 2) / (2 * sigma**2))
    # for a gaussian, half the flux falls within the half-max radius. measured about the centroid.
    assert u.isclose(ee50_diameter(im, 0.01 * u.arcsec), 2.3548 * sigma * 0.01 * u.arcsec, rtol=0.02)


def test_ee50_tracks_aberrated_size():
    """half-max only sees the brightest speckle of a lumpy PSF while EE50 follows where the flux is"""
    zv = ZernikeVector(Z07=800 * u.nm, Z09=1000 * u.nm, Z10=-600 * u.nm)
    r = broadband_psf(zv, diameter=D, obscuration_diameter=0 * u.m)
    assert r.optics_ee50 > 2 * r.optics_fwhm
    assert r.delivered_ee50 is None
    r = broadband_psf(zv, diameter=D, obscuration_diameter=0 * u.m, seeing=0.8 * u.arcsec)
    assert r.delivered_ee50 > r.delivered_fwhm


def test_unaberrated_fwhm():
    r = clear_psf(band=(500 * u.nm, 0.0), nwave=1)
    assert u.isclose(r.optics_fwhm, 1.029 * lam_over_d(500 * u.nm), rtol=0.03)
    assert r.delivered is None
    assert r.delivered_fwhm is None
    assert r.delivered_fov is None


def test_flux_normalization():
    r = clear_psf()
    # fraction of total flux per pixel; the default field catches all but the far wings
    assert 0.97 < r.optics.sum() <= 1.0


def test_defocus_broadens():
    perfect = clear_psf()
    r = broadband_psf(ZernikeVector(Z04=500 * u.nm), diameter=D, obscuration_diameter=0 * u.m)
    assert r.optics_fwhm > 1.5 * perfect.optics_fwhm


def test_large_defocus_fits_in_field():
    """badly out of focus frames are common during setup and the field has to grow to hold them"""
    r = broadband_psf(ZernikeVector(Z04=3000 * u.nm), diameter=D, obscuration_diameter=0 * u.m)
    assert r.optics.sum() > 0.95
    assert r.optics_fwhm < r.optics_fov / 2


def test_huge_defocus_caps_optics_grid():
    """far out of focus the field outgrows the pixel budget, so the sampling coarsens rather than the field shrinking"""
    r = broadband_psf(ZernikeVector(Z04=8000 * u.nm), diameter=D, obscuration_diameter=0 * u.m, nwave=3)
    assert r.optics.shape[0] <= MAX_OPTICS_NPIX + 1
    assert r.optics_pixel_scale > lam_over_d(450 * u.nm) / 2
    assert r.optics.sum() > 0.95


def test_tilt_ignored_and_input_untouched():
    zv = ZernikeVector(Z02=5000 * u.nm, Z03=-5000 * u.nm, Z05=200 * u.nm)
    before = zv.copy()
    tilted = broadband_psf(zv, diameter=D, obscuration_diameter=0 * u.m)
    untilted = broadband_psf(ZernikeVector(Z05=200 * u.nm), diameter=D, obscuration_diameter=0 * u.m)
    assert u.isclose(tilted.optics_fwhm, untilted.optics_fwhm)
    assert np.allclose(tilted.optics, untilted.optics)
    assert zv.modestart == before.modestart
    assert dict(zv.coeffs) == dict(before.coeffs)


@pytest.mark.parametrize("band", ["500nm", "J", "K"])
def test_delivered_follows_seeing(band):
    """with seeing well above diffraction the delivered FWHM is the seeing scaled as lambda^-1/5"""
    r = clear_psf(band=band, seeing=1.0 * u.arcsec)
    center = PSF_BANDS[band][0]
    expected = 1.0 * u.arcsec * (center / (500 * u.nm)).decompose() ** -0.2
    assert u.isclose(r.seeing_fwhm, expected, rtol=0.01)
    assert u.isclose(r.delivered_fwhm, expected, rtol=0.03)
    assert np.isclose(r.delivered.sum(), r.optics.sum(), rtol=0.02)
    # the delivered field is sized to hold the seeing halo
    assert u.isclose(r.delivered_fov, r.delivered.shape[0] * r.delivered_pixel_scale)
    assert r.delivered_fov > 6 * r.seeing_fwhm


def test_delivered_includes_aberrations():
    seeing_only = clear_psf(seeing=0.5 * u.arcsec)
    r = broadband_psf(ZernikeVector(Z04=1000 * u.nm), diameter=D, obscuration_diameter=0 * u.m, seeing=0.5 * u.arcsec)
    assert r.delivered_fwhm > 1.1 * seeing_only.delivered_fwhm


def test_seeing_without_units():
    r = clear_psf(seeing=0.8)
    assert u.isclose(r.seeing_fwhm, 0.8 * u.arcsec, rtol=0.01)


def test_telescope_psf():
    for s in mmtwfs_config["secondary"]:
        tel = mmtwfs_config["secondary"][s]["telescope"]
        t = TelescopeFactory(telescope=tel, secondary=s)
        r, fig = t.psf(ZernikeVector(Z04=300 * u.nm), plot=False)
        assert fig is None
        assert r.optics_fwhm > lam_over_d(450 * u.nm, t.diameter)


def test_telescope_psf_default_wavefront():
    t = MMT()
    r, fig = t.psf(plot=False)
    assert fig is None
    assert r.optics_fwhm < 1.1 * lam_over_d(550 * u.nm, t.diameter)


def test_telescope_psf_plot():
    t = MMT()
    r, fig = t.psf(ZernikeVector(Z05=200 * u.nm), plot=True)
    assert len(fig.axes) == 2  # optics image and its colorbar
    assert "EE50" in fig.axes[0].get_title()
    r, fig = t.psf(ZernikeVector(Z05=200 * u.nm), band="H", seeing=0.7 * u.arcsec, plot=True)
    assert len(fig.axes) == 4
    plt.close("all")


def test_plot_window_holds_aberrated_psf():
    """the plot zoom has to follow the flux, not the FWHM, which underestimates the size of lumpy PSFs"""
    zv = ZernikeVector(Z07=800 * u.nm, Z09=1000 * u.nm, Z10=-600 * u.nm)
    r, fig = MMT().psf(zv, plot=True)
    half = fig.axes[0].get_xlim()[1]
    n = r.optics.shape[0]
    c = (np.arange(n) - (n - 1) / 2) * r.optics_pixel_scale.value
    inside = (np.abs(c)[np.newaxis, :] <= half) & (np.abs(c)[:, np.newaxis] <= half)
    assert r.optics[inside].sum() > 0.9 * r.optics.sum()
    plt.close("all")
