"""
This module provides tools for performing photometry on spots in Shack-Hartmann
WFS images.
"""

import numpy as np
from scipy import ndimage

from astropy.stats import SigmaClip

from photutils.segmentation.detect import detect_sources, detect_threshold


def make_spot_mask(
    data, nsigma, npixels, mask=None, sigclip_sigma=3.0, sigclip_iters=5, dilate_size=11
):
    """
    Make a WFS spot mask using source segmentation and binary dilation. This
    is a modified version of the deprecated function
    `~photutils.segmentation.detect.mask_source_mask`

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D array of the image.

        .. note::
           It is recommended that the user convolve the data with
           ``kernel`` and input the convolved data directly into the
           ``data`` parameter. In this case do not input a ``kernel``,
           otherwise the data will be convolved twice.

    nsigma : float
        The number of standard deviations per pixel above the
        ``background`` for which to consider a pixel as possibly being
        part of a source.

    npixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be detected.
        ``npixels`` must be a positive integer.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when computing the image background
        statistics.

    sigclip_sigma : float, optional
        The number of standard deviations to use as the clipping limit
        when calculating the image background statistics.

    sigclip_iters : int, optional
        The maximum number of iterations to perform sigma clipping, or
        `None` to clip until convergence is achieved (i.e., continue
        until the last iteration clips nothing) when calculating the
        image background statistics.

    dilate_size : int, optional
        The size of the square array used to dilate the segmentation
        image.

    Returns
    -------
    mask : 2D bool `~numpy.ndarray`
        A 2D boolean image containing the source mask.
    """
    sigma_clip = SigmaClip(sigma=sigclip_sigma, maxiters=sigclip_iters)
    threshold = detect_threshold(
        data, nsigma, background=None, error=None, mask=mask, sigma_clip=sigma_clip
    )

    segm = detect_sources(data, threshold, npixels)
    if segm is None:
        return np.zeros(data.shape, dtype=bool)

    footprint = np.ones((dilate_size, dilate_size))
    return ndimage.binary_dilation(segm.data.astype(bool), footprint)
