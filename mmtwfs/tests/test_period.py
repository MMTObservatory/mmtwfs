# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy import ndimage

from mmtwfs.period import measure_grid_period, grid_scale, plot_periodicity, _refine_peak
from mmtwfs.tests.synthetic import make_sh_image, gaussian_halo

CENTER = (256.0, 256.0)
RADIUS = 172.5
INNER = 40.0
PITCH = 22.7


def _scale(img, hexgrid=False, center=CENTER):
    ref, _ = make_sh_image(sigma=2.0, hexgrid=hexgrid)
    r = measure_grid_period(ref, CENTER, RADIUS, PITCH, inner=INNER, snr_thresh=0.0)
    m = measure_grid_period(img, center, RADIUS, PITCH, inner=INNER)
    return grid_scale(m, r)


@pytest.mark.parametrize("hexgrid", [False, True])
@pytest.mark.parametrize("truth", [0.97, 1.0, 1.03])
@pytest.mark.parametrize("sigma, tol", [(2.0, 5e-4), (6.0, 1e-3)])
def test_measure_grid_period_scale(hexgrid, truth, sigma, tol):
    rng = np.random.default_rng(1)
    img, _ = make_sh_image(spacing=PITCH * truth, sigma=sigma, hexgrid=hexgrid)
    img = img + rng.normal(0.0, 5.0, img.shape)
    g = _scale(img, hexgrid=hexgrid)
    assert abs(g["scale"] - truth) < tol
    assert abs(g["rotation"]) < 0.5


def test_measure_grid_period_rotated_hex():
    rng = np.random.default_rng(4)
    img, _ = make_sh_image(hexgrid=True, angle=10.0, sigma=3.0)
    ref, _ = make_sh_image(hexgrid=True, angle=10.0, sigma=2.0)
    r = measure_grid_period(ref, CENTER, RADIUS, PITCH, inner=INNER, snr_thresh=0.0)
    m = measure_grid_period(img + rng.normal(0.0, 5.0, img.shape), CENTER, RADIUS, PITCH, inner=INNER)
    assert abs(grid_scale(m, r)["scale"] - 1.0) < 5e-4


def test_measure_grid_period_halo():
    # a halo 30x brighter than the spot peaks; the built-in high-pass keeps it out of the search annulus
    rng = np.random.default_rng(5)
    img, _ = make_sh_image(spacing=PITCH * 1.03, sigma=6.0)
    img = img + gaussian_halo(img.shape, CENTER, 150.0, 300.0) + rng.normal(0.0, 5.0, img.shape)
    g = _scale(img)
    assert abs(g["scale"] - 1.03) < 3e-3


def test_measure_grid_period_noise_returns_none():
    rng = np.random.default_rng(6)
    noise = rng.normal(0.0, 5.0, (512, 512))
    assert measure_grid_period(noise, CENTER, RADIUS, PITCH, inner=INNER) is None


def test_measure_grid_period_blank_returns_none():
    # no noise power to measure SNR against
    assert measure_grid_period(np.zeros((512, 512)), CENTER, RADIUS, PITCH, inner=INNER) is None


def test_measure_grid_period_no_second_peak():
    # no two directions in the half-plane are more than 90 deg apart
    img, _ = make_sh_image(sigma=2.0)
    assert measure_grid_period(img, CENTER, RADIUS, PITCH, inner=INNER, min_sep=91.0) is None


def test_refine_peak_rejects_non_maximum():
    # n = 8 puts the (2, 2) bin off zero frequency, as peaks in the search annulus always are
    power = np.ones((5, 5))
    assert _refine_peak(power, 8, 2, 2, 1.0) is None
    power[2, 2] = 4.0
    assert _refine_peak(power, 8, 2, 2, 1.0) is not None


def test_measure_grid_period_off_edge():
    # pupil hanging 60 px off the left edge must not raise, and still find the grid
    rng = np.random.default_rng(7)
    center = (RADIUS - 60.0, 256.0)
    img, _ = make_sh_image(shape=(512, 640), center=(RADIUS + 68.0, 256.0), sigma=3.0)
    img = img[:, 128:] + rng.normal(0.0, 5.0, (512, 512))
    g = _scale(img, center=center)
    assert abs(g["scale"] - 1.0) < 2e-3


def test_scale_err_is_conservative():
    # the propagated error ignores correlations between zero-padded bins, so it overestimates the true
    # scatter. pin it between 1x and 5x; period_err_factor calibrates it on real data.
    rng = np.random.default_rng(8)
    ref, _ = make_sh_image(sigma=2.0)
    r = measure_grid_period(ref, CENTER, RADIUS, PITCH, inner=INNER, snr_thresh=0.0)
    img0, _ = make_sh_image(spacing=PITCH * 1.004, sigma=2.0)
    scales, errs = [], []
    for _ in range(40):
        m = measure_grid_period(img0 + rng.normal(0.0, 30.0, img0.shape), CENTER, RADIUS, PITCH, inner=INNER)
        g = grid_scale(m, r)
        scales.append(g["scale"])
        errs.append(g["scale_err_fit"])
    ratio = np.std(scales) / np.median(errs)
    assert 0.2 < ratio < 1.0


def test_plot_periodicity():
    img, _ = make_sh_image(sigma=3.0)
    m = measure_grid_period(img, CENTER, RADIUS, PITCH, inner=INNER)
    fig = plot_periodicity(m)
    assert fig.get_label() == "Grid Periodicity"
    # the stretch runs from the noise floor (median) to the strongest peak, not from the near-empty minimum, which
    # leaves the whole spectrum a dark, flat grey
    im = fig.axes[0].images[0]
    shown = np.asarray(im.get_array())
    assert im.get_clim() == pytest.approx((np.median(shown), shown.max()))
    plt.close("all")


@pytest.mark.parametrize("hexgrid", [False, True])
@pytest.mark.parametrize("a", [0.01, -0.01])
def test_measure_grid_period_astigmatism(hexgrid, a):
    # astigmatism stretches the grid along one axis and squeezes it along the other. that is a traceless
    # distortion with no defocus, so the scale must stay at 1. on a hex grid the mean length of two fundamentals
    # doesn't cancel it; the area of the frequency cell does.
    rng = np.random.default_rng(10)
    img, _ = make_sh_image(sigma=3.0, hexgrid=hexgrid)
    c = np.array([CENTER[1], CENTER[0]])
    m = np.diag([1.0 / (1.0 - a), 1.0 / (1.0 + a)])  # output -> input (y, x): x stretched by 1 + a
    img = ndimage.affine_transform(img, m, offset=c - m @ c, order=1) + rng.normal(0.0, 5.0, img.shape)
    g = _scale(img, hexgrid=hexgrid)
    assert abs(g["scale"] - 1.0) < 5e-4
