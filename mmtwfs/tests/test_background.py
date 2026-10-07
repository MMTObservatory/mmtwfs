# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from scipy import ndimage

from mmtwfs.background import pupil_footprint, pupil_background, pedestal
from mmtwfs.tests.synthetic import make_sh_image, gaussian_halo

SHAPE = (512, 512)
CENTER = (256.0, 256.0)
RADIUS = 172.5
INNER = 40.0
PITCH = 22.7


def test_pupil_footprint():
    fp = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER, margin=5.0)
    assert fp[256, 356]
    assert not fp[256, 256]  # central obscuration
    assert fp[256, 256 + 176]  # within the outer margin
    assert not fp[256, 256 + 180]  # beyond pupil + margin
    assert fp[256, 256 + 36]  # inner edge pulled in by the margin to r = 35
    assert not fp[256, 256 + 34]


@pytest.mark.parametrize("width, tol", [(400.0, 1.0), (250.0, 3.0)])
def test_pupil_background_halo(width, tol):
    rng = np.random.default_rng(3)
    spots, _ = make_sh_image(sigma=4.0)
    halo = gaussian_halo(SHAPE, CENTER, width, 300.0, tilt=0.2)
    img = spots + halo + rng.normal(0.0, 5.0, SHAPE)
    fp = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER, margin=PITCH)
    bkg = pupil_background(img, fp, box=16, order=4)
    inside = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER)
    resid = (bkg - halo)[inside]
    assert np.std(resid) < tol
    assert abs(np.mean(resid)) < 3.0 * tol


def test_pupil_background_too_few_samples():
    with pytest.raises(ValueError):
        pupil_background(np.zeros(SHAPE), np.ones(SHAPE, dtype=bool))


@pytest.mark.parametrize("sigma, max_offset", [(2.0, 0.5), (4.0, 0.5), (6.0, 2.0)])
def test_pedestal_flat_residual(sigma, max_offset):
    # a smooth floor under the spots must come out, leaving a flat residual so centroids can't shift.
    # for blurry spots the opening also takes the overlapping wings, hence the larger allowed offset.
    spots, _ = make_sh_image(sigma=sigma)
    inside = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER)
    floor = 50.0 * ndimage.gaussian_filter(inside.astype(float), 10.0)
    cleaned = spots + floor - pedestal(spots + floor, PITCH)
    deep = pupil_footprint(SHAPE, CENTER, RADIUS - 2 * PITCH, inner=INNER + 2 * PITCH)
    resid = (cleaned - spots)[deep]
    assert abs(np.median(resid)) < max_offset
    assert np.std(resid) < 0.1
