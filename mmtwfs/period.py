# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Measure the period of a Shack-Hartmann spot grid from the power spectrum of the pupil image. This works when the
spots are too blurred to detect and centroid individually. Defocus rescales the grid, so the scale relative to the
reference grid gives a focus correction.
"""

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt

__all__ = ["annular_window", "measure_grid_period", "grid_scale", "plot_periodicity"]


def annular_window(n, radius, inner=0.0, taper=10.0):
    """
    n x n window that is 1 inside an annulus and rolls off to 0 with raised-cosine edges of width taper.
    """
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.0
    r = np.hypot(xx, yy)
    w = np.ones_like(r)
    t0 = radius - taper
    w[r >= radius] = 0.0
    edge = (r > t0) & (r < radius)
    w[edge] = 0.5 * (1.0 + np.cos(np.pi * (r[edge] - t0) / taper))
    if inner > 0.0:
        w[r <= inner] = 0.0
        edge = (r > inner) & (r < inner + taper)
        w[edge] *= 0.5 * (1.0 - np.cos(np.pi * (r[edge] - inner) / taper))
    return w


def _cutout(data, center, half):
    """
    2*half square cutout centered on center, zero-filled where it hangs off the image.
    """
    n = 2 * half
    x0 = int(round(center[0])) - half
    y0 = int(round(center[1])) - half
    sub = np.zeros((n, n))
    ys = slice(max(y0, 0), min(y0 + n, data.shape[0]))
    xs = slice(max(x0, 0), min(x0 + n, data.shape[1]))
    if ys.start < ys.stop and xs.start < xs.stop:
        sub[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = data[ys, xs]
    return sub


def _refine_peak(power, n, iy, ix, noise):
    """
    Sub-bin peak position from 3-point log-parabola (i.e. gaussian) interpolation along each axis. The power of a
    signal peak plus noise has variance of about 2 * P_signal * P_noise, so ln(P) has sigma sqrt(2 P_noise / P);
    that is propagated through the interpolation formula.

    Returns (fx, fy, radial_sigma) in cycles/pixel, or None if this isn't a local maximum.
    """
    patch = np.maximum(power[iy - 1:iy + 2, ix - 1:ix + 2], np.finfo(float).tiny)
    lnp = np.log(patch)
    slnp = np.sqrt(np.minimum(2.0 * noise / patch, 1.0))
    offsets, variances = [], []
    for (lm, l0, lp), (sm, s0, sp) in (
        ((lnp[1, 0], lnp[1, 1], lnp[1, 2]), (slnp[1, 0], slnp[1, 1], slnp[1, 2])),
        ((lnp[0, 1], lnp[1, 1], lnp[2, 1]), (slnp[0, 1], slnp[1, 1], slnp[2, 1])),
    ):
        d = lm - 2.0 * l0 + lp
        if d >= 0.0:
            return None
        num = lm - lp
        offsets.append(num / (2.0 * d))
        variances.append(
            ((d - num) / (2.0 * d**2) * sm) ** 2
            + ((-d - num) / (2.0 * d**2) * sp) ** 2
            + (num / d**2 * s0) ** 2
        )
    fx = (ix - n // 2 + offsets[0]) / n
    fy = (iy - n // 2 + offsets[1]) / n
    sfx = np.sqrt(variances[0]) / n
    sfy = np.sqrt(variances[1]) / n
    sigma = np.hypot(fx * sfx, fy * sfy) / np.hypot(fx, fy)
    return fx, fy, sigma


def measure_grid_period(
    data, center, radius, ref_spacing, inner=0.0, search=0.2, pad=4, snr_thresh=20.0, min_sep=60.0
):
    """
    Measure the two fundamental grid frequencies of a WFS spot pattern from the power spectrum of the pupil.

    Parameters
    ----------
    data : 2D np.ndarray
        WFS image
    center : list-like
        Pupil center (x, y) in pixels
    radius : float
        Pupil radius in pixels
    ref_spacing : float or list-like
        Expected spot spacing in pixels (mean is used); sets the search annulus
    inner : float
        Central obscuration radius in pixels
    search : float
        Half-width of the search annulus as a fraction of the expected frequency
    pad : int
        Zero-padding factor for the FFT
    snr_thresh : float
        Minimum peak power / noise power for both peaks; below this, return None
    min_sep : float
        Minimum angle in degrees between the two grid vectors (60 handles hexagonal and square grids)

    Returns
    -------
    dict or None
    """
    spacing = float(np.mean(ref_spacing))
    half = int(np.ceil(radius))
    sub = _cutout(np.asarray(data, dtype=float), center, half)

    # high-pass: anything smoother than the grid (halo, pedestal) would otherwise leak through the window
    # into the search annulus and bias the peaks
    sub = sub - ndimage.gaussian_filter(sub, spacing)
    w = annular_window(2 * half, radius, inner=inner, taper=spacing)
    sub = (sub - np.sum(sub * w) / np.sum(w)) * w

    n = pad * 2 * half
    power = np.abs(np.fft.fftshift(np.fft.fft2(sub, s=(n, n)))) ** 2
    f = (np.arange(n) - n // 2) / n
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    f0 = 1.0 / spacing
    fmin, fmax = f0 * (1.0 - search), f0 * (1.0 + search)
    annulus = (fr > fmin) & (fr < fmax)
    upper = (fy > 0) | ((fy == 0) & (fx > 0))  # one of each +/- conjugate pair
    candidates = annulus & upper

    # noise power is exponentially distributed, so median = ln(2) * mean
    noise = np.median(power[annulus]) / np.log(2.0)
    if not noise > 0.0:
        return None

    angle = np.degrees(np.arctan2(fy, fx)) % 180.0
    first = np.unravel_index(np.argmax(np.where(candidates, power, -1.0)), power.shape)
    dang = np.abs(angle - angle[first])
    dang = np.minimum(dang, 180.0 - dang)
    second_ok = candidates & (dang >= min_sep)
    if not second_ok.any():
        return None
    second = np.unravel_index(np.argmax(np.where(second_ok, power, -1.0)), power.shape)

    freqs, ferrs, snrs = [], [], []
    for iy, ix in (first, second):
        snr = power[iy, ix] / noise
        refined = _refine_peak(power, n, iy, ix, noise)
        if snr < snr_thresh or refined is None:
            return None
        freqs.append(refined[:2])
        ferrs.append(refined[2])
        snrs.append(snr)

    freqs = np.array(freqs)
    return {
        "freqs": freqs,
        "freq_errs": np.array(ferrs),
        "snr": np.array(snrs),
        "spacing": 1.0 / np.hypot(freqs[:, 0], freqs[:, 1]),
        "angle": np.degrees(np.arctan2(freqs[:, 1], freqs[:, 0])) % 180.0,
        "power": power,
        "freq_axis": f,
        "annulus": (fmin, fmax),
    }


def grid_scale(meas, ref):
    """
    Compare measured grid frequencies with the reference grid's. Scale is measured spacing / reference spacing, the
    same convention as the grid fit in `~mmtwfs.wfs.get_slopes`. It comes from the ratio of the areas of the
    frequency cells spanned by the two vectors, which is unchanged to first order by astigmatism (a traceless
    stretch) on any grid; the mean of the two vector lengths is not on a hexagonal grid. Any two of a hex grid's
    three fundamentals span the same area, so it also doesn't matter which pair each image picked.

    The per-vector scales and the rotation come from matching each measured vector to the reference vector closest
    in angle and are diagnostics only.

    Returns
    -------
    dict with scale, scale_err_fit, scales (per vector), rotation (deg)
    """
    fm = np.hypot(meas["freqs"][:, 0], meas["freqs"][:, 1])
    # a hexagonal grid has three equally strong fundamentals, f1, f2 and f1 - f2 (or f1 + f2), and the two picked
    # in each image can differ. add the third to the reference candidates; for a square grid it is sqrt(2) longer
    # and drops out of the length cut below.
    f1, f2 = ref["freqs"]
    rfreqs = np.array([f1, f2, f1 - f2, f1 + f2])
    fr = np.hypot(rfreqs[:, 0], rfreqs[:, 1])
    rangle = np.degrees(np.arctan2(rfreqs[:, 1], rfreqs[:, 0])) % 180.0
    fundamental = fr < 1.2 * fr[:2].max()
    scales, rots = [], []
    for i in range(len(fm)):
        d = np.abs(rangle - meas["angle"][i])
        d = np.where(fundamental, np.minimum(d, 180.0 - d), np.inf)
        j = int(np.argmin(d))
        s = fr[j] / fm[i]
        scales.append(s)
        rots.append((meas["angle"][i] - rangle[j] + 90.0) % 180.0 - 90.0)
    scales = np.array(scales)

    area_ref = np.abs(np.linalg.det(ref["freqs"]))
    area_meas = np.abs(np.linalg.det(meas["freqs"]))
    scale = np.sqrt(area_ref / area_meas)
    # the cell area is the product of the vector lengths times the sine of the angle between them; only the radial
    # errors are propagated, so this is the same error as the mean of the per-vector scales
    rel = np.concatenate([meas["freq_errs"] / fm, ref["freq_errs"] / fr[:2]])
    return {
        "scale": float(scale),
        "scale_err_fit": float(0.5 * scale * np.sqrt(np.sum(rel**2))),
        "scales": scales,
        "rotation": float(np.mean(rots)),
    }


def plot_periodicity(meas):
    """
    Show the power spectrum around the search annulus with the detected grid peaks.
    """
    f = meas["freq_axis"]
    fmin, fmax = meas["annulus"]
    lim = 1.3 * fmax
    keep = np.abs(f) <= lim
    fig, ax = plt.subplots()
    fig.set_label("Grid Periodicity")
    logp = np.log10(meas["power"][np.ix_(keep, keep)] + np.finfo(float).tiny)
    # stretch from the noise floor to the strongest peak. the full range reaches down to near-empty bins many decades
    # below the noise and leaves everything that matters a flat, dark grey.
    ax.imshow(
        logp,
        origin="lower",
        cmap="Greys",
        vmin=np.median(logp),
        vmax=logp.max(),
        extent=(f[keep][0], f[keep][-1], f[keep][0], f[keep][-1]),
    )
    t = np.linspace(0, 2 * np.pi, 361)
    for rad in (fmin, fmax):
        ax.plot(rad * np.cos(t), rad * np.sin(t), color="blue", lw=0.8)
    for (fx, fy), snr, sp in zip(meas["freqs"], meas["snr"], meas["spacing"]):
        ax.scatter([fx, -fx], [fy, -fy], facecolors="none", edgecolors="red", s=80)
        ax.annotate(
            f"{sp:.3f} px (SNR {snr:.0f})", (fx, fy), xytext=(8, 8), textcoords="offset points", color="red", fontsize=8
        )
    ax.set_xlabel("f$_x$ (cycles/pixel)")
    ax.set_ylabel("f$_y$ (cycles/pixel)")
    ax.set_title("Grid period (focus-only fallback)")
    return fig
