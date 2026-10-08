# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Broadband PSFs from a measured wavefront, with and without atmospheric seeing.

The optics-only PSF is the diffraction pattern of the telescope pupil with the wavefront
aberrations described by a `~mmtwfs.zernike.ZernikeVector`, averaged over a passband. The
delivered PSF convolves each monochromatic optics PSF with the Kolmogorov long-exposure seeing
transfer function, T(f) = exp[-3.44(lambda*f/r0)^5/3], the same model that
`~mmtwfs.wfs.WFS.vlt_seeing` fits to measure the seeing. Seeing is specified as the FWHM at
500 nm, the wavelength the WFS seeing values are referenced to, and r0 is scaled to each
wavelength as lambda^6/5.
"""

import logging
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.visualization import ImageNormalize, SqrtStretch

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mmtwfs.custom_exceptions import WFSConfigException

log = logging.getLogger("PSF")
log.setLevel(logging.INFO)

__all__ = [
    "PSF_BANDS",
    "PSFResult",
    "band_wavelengths",
    "pupil_model",
    "area_fwhm",
    "ee50_diameter",
    "broadband_psf",
    "plot_psf",
]

# passbands as (center wavelength, fractional half-width). seeing is referenced to 500 nm so that is the default.
PSF_BANDS = {
    "500nm": (500 * u.nm, 0.1),
    "R": (650 * u.nm, 0.1),
    "I": (800 * u.nm, 0.1),
    "z": (900 * u.nm, 0.1),
    "J": (1.25 * u.um, 0.1),
    "H": (1.65 * u.um, 0.1),
    "K": (2.2 * u.um, 0.1),
}

SEEING_REF_WAVE = 500 * u.nm
RAD2ASEC = u.rad.to(u.arcsec)

# limits on the sizes of the pupil and image grids to keep the calculation interactive
MIN_PUPIL_NPIX = 256
MAX_PUPIL_NPIX = 1024
OPTICS_NPIX = 400
MAX_OPTICS_NPIX = 1024
# number of delivered-PSF pixels across the seeing FWHM and width of the delivered field in seeing FWHMs
DELIVERED_PIX_PER_FWHM = 20
DELIVERED_FOV_FWHM = 8


@dataclass
class PSFResult:
    """
    Optics-only and delivered PSF images. Images give the fraction of the total flux in each pixel and
    are centered on the middle pixel. The delivered values are None if no seeing was given.
    """

    band: object
    wavelengths: u.Quantity
    optics: np.ndarray
    optics_pixel_scale: u.Quantity
    optics_fwhm: u.Quantity
    optics_ee50: u.Quantity
    seeing: u.Quantity | None = None
    seeing_fwhm: u.Quantity | None = None
    delivered: np.ndarray | None = None
    delivered_pixel_scale: u.Quantity | None = None
    delivered_fwhm: u.Quantity | None = None
    delivered_ee50: u.Quantity | None = None

    @property
    def optics_fov(self):
        return self.optics.shape[0] * self.optics_pixel_scale

    @property
    def delivered_fov(self):
        if self.delivered is None:
            return None
        return self.delivered.shape[0] * self.delivered_pixel_scale


def band_wavelengths(band="500nm", nwave=11):
    """
    Return the wavelengths sampling **band**, either a key of `PSF_BANDS` or a (center, fractional half-width) tuple.
    """
    if isinstance(band, str):
        if band not in PSF_BANDS:
            raise WFSConfigException(value=f"Unknown PSF band, {band}. Valid bands are {list(PSF_BANDS)}.")
        center, frac = PSF_BANDS[band]
    else:
        center, frac = band
    center = u.Quantity(center, u.nm)
    if frac == 0 or nwave == 1:
        return u.Quantity([center])
    return np.linspace(center * (1 - frac), center * (1 + frac), nwave)


def _pupil_coords(npix, diameter):
    dp = diameter.to(u.m).value / npix
    x = (np.arange(npix) - (npix - 1) / 2) * dp
    return x, dp


def pupil_model(npix, diameter, obscuration_diameter, n_supports=0, support_width=0 * u.m, support_offset=0 * u.deg):
    """
    Model a pupil with a central obscuration and radial support vanes on an **npix** x **npix** grid that
    spans the primary. Returns the pupil transmission array and the grid spacing.
    """
    x, dp = _pupil_coords(npix, diameter)
    xx, yy = np.meshgrid(x, x)
    r = np.hypot(xx, yy)
    pupil = (r <= diameter.to(u.m).value / 2) & (r >= obscuration_diameter.to(u.m).value / 2)
    half_width = support_width.to(u.m).value / 2
    for k in range(n_supports):
        theta = support_offset.to(u.rad).value + 2 * np.pi * k / n_supports
        along = xx * np.cos(theta) + yy * np.sin(theta)
        across = -xx * np.sin(theta) + yy * np.cos(theta)
        pupil &= ~((along > 0) & (np.abs(across) < half_width))
    return pupil.astype(float), dp * u.m


def area_fwhm(image, pixel_scale):
    """
    FWHM of the circle with the same area as the part of **image** above half its maximum. Unlike a profile fit,
    this stays meaningful for the lumpy, asymmetric PSFs that aberrations produce.
    """
    npix = np.count_nonzero(image >= image.max() / 2)
    return 2 * np.sqrt(npix / np.pi) * u.Quantity(pixel_scale, u.arcsec)


def ee50_diameter(image, pixel_scale):
    """
    Diameter of the circle about the flux centroid of **image** that encloses half of its flux. This follows the
    overall size of aberrated PSFs, where the half-max area only measures the brightest speckle.
    """
    n = image.shape[0]
    c = np.arange(n) - (n - 1) / 2
    xx, yy = np.meshgrid(c, c)
    frac = image / image.sum()
    r = np.hypot(xx - (frac * xx).sum(), yy - (frac * yy).sum()).ravel()
    order = np.argsort(r)
    enclosed = np.cumsum(frac.ravel()[order])
    return 2 * r[order][np.searchsorted(enclosed, 0.5)] * u.Quantity(pixel_scale, u.arcsec)


def _odd(n):
    n = int(np.ceil(n))
    return n if n % 2 else n + 1


def _without_tilt(zv):
    """
    Copy of **zv** in meters without piston and tilt, which only shift the image.
    """
    zv = zv.copy()
    for k in ("Z01", "Z02", "Z03"):
        del zv[k]
    zv.units = u.m
    return zv


def _wavefront_rms(zv):
    return zv.rms.to(u.m).value if len(zv.coeffs) > 0 else 0.0


def _wavefront(zv, rho, phi):
    """
    Wavefront in meters at pupil coordinates (**rho**, **phi**).
    """
    if len(zv.coeffs) == 0:
        return np.zeros_like(rho)
    return u.Quantity(zv.total_phase(rho, phi), u.m).value


def broadband_psf(
    zv,
    diameter,
    obscuration_diameter,
    n_supports=0,
    support_width=0 * u.m,
    support_offset=0 * u.deg,
    band="500nm",
    seeing=None,
    nwave=11,
    pixel_scale=None,
    fov=None,
):
    """
    Calculate the optics-only PSF for the wavefront **zv** averaged over **band** and, if **seeing**
    (FWHM at 500 nm, arcsec if unitless) is given, the delivered PSF including atmospheric seeing.

    The default optics field is sized to hold the aberrated PSF and sampled at least at Nyquist at the
    shortest wavelength. **pixel_scale** and **fov** override them. The delivered PSF is sampled with
    about 20 pixels across the seeing FWHM over a field 8 seeing FWHMs wide.
    """
    waves = band_wavelengths(band, nwave).to(u.m).value
    lmin, lmax = waves.min(), waves.max()
    d = diameter.to(u.m).value
    zv = _without_tilt(zv)

    # optics grid. the geometric blur diameter of defocus is 16 x Z04 / D, or about 28 x wavefront RMS / D, so
    # make the field several times that, but no smaller than 32 lambda/D.
    if fov is None:
        fov = max(32 * lmax / d, 3 * 28 * _wavefront_rms(zv) / d) * RAD2ASEC
    fov = u.Quantity(fov, u.arcsec).value
    if pixel_scale is None:
        pixel_scale = min(fov / OPTICS_NPIX, lmin / (2 * d) * RAD2ASEC)
    pix = u.Quantity(pixel_scale, u.arcsec).value
    nopt = int(np.ceil(fov / pix))
    if nopt > MAX_OPTICS_NPIX:
        nopt = MAX_OPTICS_NPIX
        pix = fov / nopt
        log.info(f"Optics PSF field of {fov:.2f} arcsec needs coarser sampling than Nyquist, {pix:.4f} arcsec/pixel.")

    # the delivered PSF is calculated by binning the optics PSF by an odd factor so the centers stay aligned
    bin_factor = 1
    if seeing is not None:
        seeing = u.Quantity(seeing, u.arcsec)
        seeing_at = (seeing * (waves * u.m / SEEING_REF_WAVE).decompose() ** -0.2).to(u.arcsec).value
        bin_factor = max(1, int(seeing_at.min() / DELIVERED_PIX_PER_FWHM / pix))
        if bin_factor % 2 == 0:
            bin_factor -= 1
    nbin = _odd(nopt / bin_factor)
    nopt = nbin * bin_factor
    fov = nopt * pix

    # the transform of the sampled pupil repeats every N lambda / D so make the pupil grid fine enough to keep
    # the repeats out of the field
    npup = min(MAX_PUPIL_NPIX, max(MIN_PUPIL_NPIX, int(np.ceil(1.25 * fov / RAD2ASEC * d / lmin))))
    pupil, dp = pupil_model(npup, diameter, obscuration_diameter, n_supports, support_width, support_offset)
    dp = dp.value
    x, _ = _pupil_coords(npup, diameter)
    xx, yy = np.meshgrid(x, x)
    inpupil = pupil > 0
    wf = np.zeros_like(pupil)
    wf[inpupil] = _wavefront(zv, np.hypot(xx, yy)[inpupil] / (d / 2), np.arctan2(yy, xx)[inpupil])

    alpha = (np.arange(nopt) - (nopt - 1) / 2) * pix / RAD2ASEC
    norm = (dp * pix / RAD2ASEC) ** 2 / np.sum(pupil**2)
    mono = []
    for wave in waves:
        field = pupil * np.exp(2j * np.pi * wf / wave)
        # matrix Fourier transform gives the same angular sampling at every wavelength
        ft = np.exp(-2j * np.pi * np.outer(alpha, x) / wave)
        amp = ft @ field @ ft.T
        mono.append(np.abs(amp) ** 2 * norm / wave**2)
    optics = np.mean(mono, axis=0)

    result = PSFResult(
        band=band,
        wavelengths=(waves * u.m).to(u.nm),
        optics=optics,
        optics_pixel_scale=pix * u.arcsec,
        optics_fwhm=area_fwhm(optics, pix),
        optics_ee50=ee50_diameter(optics, pix),
    )

    if seeing is None:
        return result

    center = np.mean(waves) * u.m
    result.seeing = seeing
    result.seeing_fwhm = (seeing * (center / SEEING_REF_WAVE).decompose() ** -0.2).to(u.arcsec)

    dpix = pix * bin_factor
    ndel = max(nbin, _odd(DELIVERED_FOV_FWHM * seeing_at.max() / dpix))
    off = (ndel - nbin) // 2
    f = np.fft.fftfreq(ndel, d=dpix / RAD2ASEC)
    freq = np.hypot(f[np.newaxis, :], f[:, np.newaxis])
    r0_ref = 0.976 * SEEING_REF_WAVE.to(u.m).value / (seeing.to(u.arcsec).value / RAD2ASEC)
    delivered = np.zeros((ndel, ndel))
    for wave, psf in zip(waves, mono):
        binned = psf.reshape(nbin, bin_factor, nbin, bin_factor).sum(axis=(1, 3))
        padded = np.zeros((ndel, ndel))
        padded[off:off + nbin, off:off + nbin] = binned
        r0 = r0_ref * (wave / SEEING_REF_WAVE.to(u.m).value) ** 1.2
        otf = np.exp(-3.44 * (wave * freq / r0) ** (5 / 3))
        delivered += np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(padded)) * otf).real)
    delivered /= len(waves)

    result.delivered = delivered
    result.delivered_pixel_scale = dpix * u.arcsec
    result.delivered_fwhm = area_fwhm(delivered, dpix)
    result.delivered_ee50 = ee50_diameter(delivered, dpix)
    return result


def _display_halfwidth(im, pixel_scale):
    """
    Half-width of the centered square that holds 90% of the flux in **im**, with some margin.
    """
    n = im.shape[0]
    c = np.abs(np.arange(n) - (n - 1) / 2)
    dist = np.maximum(c[np.newaxis, :], c[:, np.newaxis]).ravel()
    order = np.argsort(dist)
    enclosed = np.cumsum(im.ravel()[order])
    r90 = dist[order][np.searchsorted(enclosed, 0.9 * enclosed[-1])]
    return min(n / 2, 1.25 * (r90 + 1)) * pixel_scale


def _show(ax, im, pixel_scale, title):
    pixel_scale = pixel_scale.to(u.arcsec).value
    fov = im.shape[0] * pixel_scale
    norm = ImageNormalize(im, stretch=SqrtStretch())
    ims = ax.imshow(im, extent=[-fov / 2, fov / 2, -fov / 2, fov / 2], origin="lower", cmap=cm.magma, norm=norm)
    half = _display_halfwidth(im, pixel_scale)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_title(title)
    cb = ax.figure.colorbar(ims, ax=ax)
    cb.set_label("Fraction of Total Flux per Pixel")


def plot_psf(result):
    """
    Plot the optics-only PSF and, if calculated, the delivered PSF beside it.
    """
    npanels = 1 if result.delivered is None else 2
    fig, axes = plt.subplots(1, npanels, figsize=(6 * npanels, 5), squeeze=False, layout="constrained")
    waves = result.wavelengths
    band = result.band if isinstance(result.band, str) else f"{waves.mean().to(u.nm).value:.0f} nm"
    fig.suptitle(f"PSF at {band} ({waves.min().value:.0f}-{waves.max().value:.0f} nm)")
    _show(
        axes[0, 0],
        result.optics,
        result.optics_pixel_scale,
        f"Optics: EE50 diameter {result.optics_ee50.value:.3f}\"",
    )
    if npanels == 2:
        _show(
            axes[0, 1],
            result.delivered,
            result.delivered_pixel_scale,
            f"Delivered: FWHM {result.delivered_fwhm.value:.2f}\" (seeing {result.seeing_fwhm.value:.2f}\")",
        )
    fig.set_label("PSF")
    return fig
