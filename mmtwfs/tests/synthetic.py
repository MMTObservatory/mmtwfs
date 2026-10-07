# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Synthetic Shack-Hartmann images for testing background and grid-period code.
"""

import numpy as np


def make_sh_image(
    shape=(512, 512),
    center=(256.0, 256.0),
    radius=172.5,
    inner=40.0,
    spacing=22.7,
    angle=0.0,
    hexgrid=False,
    sigma=2.0,
    flux=2000.0,
):
    """
    Render gaussian spots on a square or hexagonal grid inside an annular pupil.

    Returns
    -------
    image : 2D np.ndarray
    positions : (N, 2) np.ndarray of (x, y) spot centers
    """
    if hexgrid:
        basis = spacing * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    else:
        basis = spacing * np.array([[1.0, 0.0], [0.0, 1.0]])
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    n = int(2 * radius / spacing) + 3
    ij = np.array([(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)], dtype=float)
    pts = ij @ basis @ rot.T
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(r < radius) & (r > inner)] + np.asarray(center)

    image = np.zeros(shape)
    h = int(4 * sigma) + 1
    norm = flux / (2.0 * np.pi * sigma**2)
    for x, y in pts:
        x0, y0 = int(round(x)), int(round(y))
        yy, xx = np.mgrid[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]
        image[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1] += norm * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        )
    return image, pts


def gaussian_halo(shape, center, width, amplitude, tilt=0.0):
    """
    Broad gaussian glow plus a linear gradient in x, like scattered light around a WFS pupil.
    """
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    r2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return amplitude * np.exp(-r2 / (2.0 * width**2)) + tilt * xx
