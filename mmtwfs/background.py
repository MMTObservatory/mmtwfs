# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Background models for Shack-Hartmann WFS images that don't depend on detecting spots. In poor seeing the spots
blend together, spot masks cover most of the pupil, and small-box background estimators end up absorbing spot
light. These models instead fit the scattered-light halo from pixels outside the pupil and remove the diffuse
floor between spots with a morphological opening.
"""

import warnings

import numpy as np
from scipy import ndimage

from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from astropy.stats import sigma_clip

__all__ = ["pupil_footprint", "pupil_background", "pedestal"]


def pupil_footprint(shape, center, outer, inner=0.0, margin=0.0):
    """
    Boolean image that is True where the pupil (padded by margin on both edges) illuminates the detector.

    Parameters
    ----------
    shape : tuple
        Image shape (ny, nx)
    center : list-like
        Pupil center (x, y) in pixels
    outer, inner : float
        Outer radius and central obscuration radius in pixels
    margin : float
        Padding added outside the outer edge and inside the inner edge
    """
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    r = np.hypot(xx - center[0], yy - center[1])
    return (r < outer + margin) & (r > max(inner - margin, 0.0))


def pupil_background(data, footprint, box=16, order=4):
    """
    Model large-scale background (scattered light halo, gradients) using only pixels outside the footprint,
    i.e. outside the pupil and inside the central obscuration, and interpolate it across the pupil.

    The image is binned into box x box blocks; blocks that are at least half background contribute their
    median to a sigma-clipped 2D polynomial fit.

    Parameters
    ----------
    data : 2D np.ndarray
    footprint : 2D bool np.ndarray
        True on pixels to exclude (the padded pupil), e.g. from `pupil_footprint`
    box : int
        Block size in pixels
    order : int
        Polynomial order

    Returns
    -------
    background : 2D np.ndarray, same shape as data
    """
    ny, nx = data.shape
    nby, nbx = ny // box, nx // box
    blocks = np.where(footprint, np.nan, data)[: nby * box, : nbx * box].reshape(nby, box, nbx, box)
    with warnings.catch_warnings():
        # all-NaN blocks inside the pupil are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(blocks, axis=(1, 3))
    frac = np.isfinite(blocks).mean(axis=(1, 3))
    yb, xb = (np.mgrid[:nby, :nbx] + 0.5) * box - 0.5
    good = (frac > 0.5) & np.isfinite(med)

    nterms = (order + 1) * (order + 2) // 2
    if good.sum() < 2 * nterms:
        raise ValueError(
            f"Only {good.sum()} background blocks available for an order {order} fit ({2 * nterms} needed)."
        )

    # fit in normalized coordinates to keep the polynomial well conditioned
    def norm(x, n):
        return 2.0 * x / (n - 1) - 1.0

    fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
    model, _ = fitter(Polynomial2D(order), norm(xb[good], nx), norm(yb[good], ny), med[good])

    yy, xx = np.mgrid[:ny, :nx]
    return model(norm(xx, nx), norm(yy, ny))


def pedestal(data, pitch, presmooth=1.5):
    """
    Estimate the diffuse light floor between WFS spots with a grey opening whose footprint is about one lenslet
    pitch, so every window reaches the gap between spots. A light pre-smoothing keeps the opening from locking onto
    negative noise excursions, and the result is smoothed by half a pitch.

    Parameters
    ----------
    data : 2D np.ndarray
    pitch : float
        Lenslet pitch (spot spacing) in pixels
    presmooth : float
        Sigma in pixels of gaussian smoothing applied before the opening

    Returns
    -------
    floor : 2D np.ndarray, same shape as data
    """
    size = int(np.ceil(pitch)) | 1  # odd so the footprint is centered
    floor = ndimage.grey_opening(ndimage.gaussian_filter(data, presmooth), size=(size, size))
    return ndimage.gaussian_filter(floor, pitch / 2.0)
