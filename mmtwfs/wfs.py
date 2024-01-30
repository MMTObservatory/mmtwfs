# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Classes and utilities for operating the wavefront sensors of the MMTO and analyzing the data they produce
"""

import warnings

import pathlib

import numpy as np
import photutils

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import feature
from scipy import ndimage, optimize
from scipy.ndimage import rotate
from scipy.spatial import cKDTree

import lmfit

import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
from astropy import stats, visualization, timeseries
from astropy.modeling.models import Gaussian2D, Polynomial2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import conf as table_conf
from astroscrappy import detect_cosmics

from ccdproc.utils.slices import slice_from_string

from mmtwfs.config import recursive_subclasses, merge_config, mmtwfs_config
from mmtwfs.telescope import TelescopeFactory
from mmtwfs.f9topbox import CompMirror
from mmtwfs.zernike import ZernikeVector, zernike_slopes, cart2pol, pol2cart
from mmtwfs.photometry import make_spot_mask
from mmtwfs.custom_exceptions import WFSConfigException, WFSAnalysisFailed, WFSCommandException

import logging
import logging.handlers
log = logging.getLogger("WFS")
log.setLevel(logging.INFO)

warnings.simplefilter(action="ignore", category=FutureWarning)
table_conf.replace_warnings = ['attributes']


__all__ = ['SH_Reference', 'WFS', 'F9', 'NewF9', 'F5', 'Binospec', 'MMIRS', 'WFSFactory', 'wfs_norm', 'check_wfsdata',
           'wfsfind', 'grid_spacing', 'center_pupil', 'get_apertures', 'match_apertures', 'aperture_distance', 'fit_apertures',
           'get_slopes', 'make_init_pars', 'slope_diff', 'mk_wfs_mask']


def wfs_norm(data, interval=visualization.ZScaleInterval(contrast=0.05), stretch=visualization.LinearStretch()):
    """
    Define default image normalization to use for WFS images
    """
    norm = visualization.mpl_normalize.ImageNormalize(
        data,
        interval=interval,
        stretch=stretch
    )
    return norm


def check_wfsdata(data, header=False):
    """
    Utility to validate WFS data

    Parameters
    ----------
    data : FITS filename or 2D ndarray
        WFS image

    Returns
    -------
    data : 2D np.ndarray
        Validated 2D WFS image
    """
    hdr = None
    if isinstance(data, (str, pathlib.PosixPath)):
        # we're a fits file (hopefully)
        try:
            with fits.open(data, verify='fix+ignore', ignore_missing_simple=True) as h:
                data = h[-1].data  # binospec images put the image data into separate extension so always grab last available.
                if header:
                    hdr = h[-1].header
        except Exception as e:
            msg = "Error reading FITS file, %s (%s)" % (data, repr(e))
            raise WFSConfigException(value=msg)
    if not isinstance(data, np.ndarray):
        msg = "WFS image data in improper format, %s" % type(data)
        raise WFSConfigException(value=msg)
    if len(data.shape) != 2:
        msg = "WFS image data has improper shape, %dD. Must be 2D image." % len(data.shape)
        raise WFSConfigException(value=msg)

    if header and hdr is not None:
        return data, hdr
    else:
        return data


def mk_wfs_mask(data, thresh_factor=50., outfile="wfs_mask.fits"):
    """
    Take a WFS image and mask/scale it so that it can be used as a reference for pupil centering

    Parameters
    ----------
    data : FITS filename or 2D ndarray
        WFS image
    thresh_factor : float (default: 50.)
        Fraction of maximum value below which will be masked to 0.
    outfile : string (default: wfs_mask.fits)
        Output FITS file to write the resulting image to.

    Returns
    -------
    scaled : 2D ndarray
        Scaled and masked WFS image
    """
    data = check_wfsdata(data)
    mx = data.max()
    thresh = mx / thresh_factor
    data[data < thresh] = 0.
    scaled = data / mx
    if outfile is not None:
        fits.writeto(outfile, scaled)
    return scaled


def wfsfind(data, fwhm=7.0, threshold=5.0, plot=True, ap_radius=5.0, std=None):
    """
    Use photutils.detection.DAOStarFinder() to find and centroid spots in a Shack-Hartmann WFS image.

    Parameters
    ----------
    data : FITS filename or 2D ndarray
        WFS image
    fwhm : float (default: 5.)
        FWHM in pixels of DAOfind convolution kernel
    threshold : float
        DAOfind threshold in units of the standard deviation of the image
    plot:  bool
        Toggle plotting of the reference image and overlayed apertures
    ap_radius : float
        Radius of plotted apertures
    """
    # data should be background subtracted first...
    data = check_wfsdata(data)
    if std is None:
        mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    daofind = photutils.detection.DAOStarFinder(fwhm=fwhm, threshold=threshold*std, sharphi=0.95)
    sources = daofind(data)

    if sources is None:
        msg = "WFS spot detection failed or no spots detected."
        raise WFSAnalysisFailed(value=msg)

    # this may be redundant given the above check...
    nsrcs = len(sources)
    if nsrcs == 0:
        msg = "No WFS spots detected."
        raise WFSAnalysisFailed(value=msg)

    # only keep spots more than 1/4 as bright as the max. need this for f/9 especially.
    sources = sources[sources['flux'] > sources['flux'].max()/4.]

    fig = None
    if plot:
        fig, ax = plt.subplots()
        fig.set_label("WFSfind")
        positions = list(zip(sources['xcentroid'], sources['ycentroid']))
        apertures = photutils.aperture.CircularAperture(positions, r=ap_radius)
        norm = wfs_norm(data)
        ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
        apertures.plot(color='red', lw=1.5, alpha=0.5, ax=ax)
    return sources, fig


def grid_spacing(data, apertures):
    """
    Measure the WFS grid spacing which changes with telescope focus.

    Parameters
    ----------
    data : WFS image (FITS or np.ndarray)
    apertures : `~astropy.table.Table`
        WFS aperture data to analyze

    Returns
    -------
    xspacing, yspacing : float, float
        Average grid spacing in X and Y axes
    """
    data = check_wfsdata(data)
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    bx = np.arange(data.shape[1]+1)
    by = np.arange(data.shape[0]+1)

    # bin the spot positions along the axes and use Lomb-Scargle to measure the grid spacing in each direction
    xsum = np.histogram(apertures['xcentroid'], bins=bx)
    ysum = np.histogram(apertures['ycentroid'], bins=by)

    k = np.linspace(10.0, 50., 1000)  # look for spacings from 10 to 50 pixels (plenty of range, but not too small to alias)
    f = 1.0 / k  # convert spacing to frequency
    xp = timeseries.LombScargle(x, xsum[0]).power(f)
    yp = timeseries.LombScargle(y, ysum[0]).power(f)

    # the peak of the power spectrum will coincide with the average spacing
    xspacing = k[xp.argmax()]
    yspacing = k[yp.argmax()]

    return xspacing, yspacing


def center_pupil(input_data, pup_mask, threshold=0.8, sigma=10., plot=True):
    """
    Find the center of the pupil in a WFS image using skimage.feature.match_template(). This generates
    a correlation image and we centroid the peak of the correlation to determine the center.

    Parameters
    ----------
    data : str or 2D ndarray
        WFS image to analyze, either FITS file or ndarray image data
    pup_mask : str or 2D ndarray
        Pupil model to use in the template matching
    threshold : float (default: 0.0)
        Sets image to 0 where it's below threshold * image.max()
    sigma : float (default: 20.)
        Sigma of gaussian smoothing kernel
    plot : bool
        Toggle plotting of the correlation image

    Returns
    -------
    cen : tuple (float, float)
        X and Y pixel coordinates of the pupil center
    """
    data = np.copy(check_wfsdata(input_data))
    pup_mask = check_wfsdata(pup_mask).astype(np.float64)  # need to force float64 here to make scipy >= 1.4 happy...

    # smooth the image to increae the S/N.
    smo = ndimage.gaussian_filter(data, sigma)

    # use skimage.feature.match_template() to do a fast cross-correlation between the WFS image and the pupil model.
    # the location of the peak of the correlation will be the center of the WFS pattern.
    match = feature.match_template(smo, pup_mask, pad_input=True)
    find_thresh = threshold * match.max()
    t = photutils.detection.find_peaks(match, find_thresh, box_size=5, centroid_func=photutils.centroids.centroid_com)

    if t is None:
        msg = "No valid pupil or spot pattern detected."
        raise WFSAnalysisFailed(value=msg)

    peak = t['peak_value'].max()
    xps = []
    yps = []
    # if there are peaks that are very nearly correlated, average their positions
    for p in t:
        if p['peak_value'] >= 0.95*peak:
            xps.append(p['x_centroid'])
            yps.append(p['y_centroid'])
    xp = np.mean(xps)
    yp = np.mean(yps)
    fig = None
    if plot:
        fig, ax = plt.subplots()
        fig.set_label("Pupil Correlation Image (masked)")
        ax.imshow(match, interpolation=None, cmap=cm.magma, origin='lower')
        ax.scatter(xp, yp, marker="+", color="green")
    return xp, yp, fig


def get_apertures(data, apsize, fwhm=5.0, thresh=7.0, plot=True, cen=None):
    """
    Use wfsfind to locate and centroid spots.  Measure their S/N ratios and the sigma of a 2D gaussian fit to
    the co-added spot.

    Parameters
    ----------
    data : str or 2D ndarray
        WFS image to analyze, either FITS file or ndarray image data
    apsize : float
        Diameter/width of the SH apertures

    Returns
    -------
    srcs : astropy.table.Table
        Detected WFS spot positions and properties
    masks : list of photutils.ApertureMask objects
        Masks used for aperture centroiding
    snrs : 1D np.ndarray
        S/N for each located spot
    sigma : float
    """
    data = check_wfsdata(data)

    # set maxiters to None to let this clip all the way to convergence
    if cen is None:
        mean, median, stddev = stats.sigma_clipped_stats(data, sigma=3.0, maxiters=None)
    else:
        xcen, ycen = int(cen[0]), int(cen[1])
        mean, median, stddev = stats.sigma_clipped_stats(data[ycen-50:ycen+50, xcen-50:ycen+50], sigma=3.0, maxiters=None)

    # use wfsfind() and pass it the clipped stddev from here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        srcs, wfsfind_fig = wfsfind(data, fwhm=fwhm, threshold=thresh, std=stddev, plot=plot)

    # we use circular apertures here because they generate square masks of the appropriate size.
    # rectangular apertures produced masks that were sqrt(2) too large.
    # see https://github.com/astropy/photutils/issues/499 for details.
    apers = photutils.aperture.CircularAperture(
        list(zip(srcs['xcentroid'], srcs['ycentroid'])),
        r=apsize/2.
    )
    masks = apers.to_mask(method='subpixel')
    sigma = 0.0
    snrs = []
    if len(masks) >= 1:
        spot = np.zeros(masks[0].shape)
        for m in masks:
            subim = m.cutout(data)

            # make co-added spot image for use in calculating the seeing
            if subim.shape == spot.shape:
                spot += subim

            signal = subim.sum()
            noise = np.sqrt(stddev**2 * subim.shape[0] * subim.shape[1])
            snr = signal / noise
            snrs.append(snr)

        snrs = np.array(snrs)

        # set up 2D gaussian model plus constant background to fit to the coadded spot
        with warnings.catch_warnings():
            # ignore astropy warnings about issues with the fit...
            warnings.simplefilter("ignore")
            g2d = Gaussian2D(amplitude=spot.max(), x_mean=spot.shape[1]/2, y_mean=spot.shape[0]/2)
            p2d = Polynomial2D(degree=0)
            model = g2d + p2d
            fitter = LevMarLSQFitter()
            y, x = np.mgrid[:spot.shape[0], :spot.shape[1]]
            fit = fitter(model, x, y, spot)

            sigma = 0.5 * (fit.x_stddev_0.value + fit.y_stddev_0.value)

    return srcs, masks, snrs, sigma, wfsfind_fig


def match_apertures(refx, refy, spotx, spoty, max_dist=25.):
    """
    Given reference aperture and spot X/Y positions, loop through reference apertures and find closest spot. Use
    max_dist to exclude matches that are too far from reference position.  Return masks to use to denote validly
    matched apertures.
    """
    refs = np.array([refx, refy])
    spots = np.array([spotx, spoty])
    match = np.nan * np.ones(len(refx))
    matched = []
    for i in np.arange(len(refx)):
        dists = np.sqrt((spots[0]-refs[0][i])**2 + (spots[1]-refs[1][i])**2)
        min_i = np.argmin(dists)
        if np.min(dists) < max_dist:
            if min_i not in matched:
                match[i] = min_i
                matched.append(min_i)
        else:
            if min_i not in matched:
                match[i] = np.nan
    ref_mask = ~np.isnan(match)
    src_mask = match[ref_mask]
    return ref_mask, src_mask.astype(int)


def aperture_distance(refx, refy, spotx, spoty):
    """
    Calculate the sum of the distances between each reference aperture and the closest measured spot position.
    This total distance is the statistic to minimize when fitting the reference aperture grid to the data.
    """
    refs = np.array([refx, refy]).transpose()
    spots = np.array([spotx, spoty]).transpose()
    tree = cKDTree(refs)
    mindist, _ = tree.query(spots)
    tot_dist = mindist.sum()
    return np.log(tot_dist)


def fit_apertures(pars, ref, spots):
    """
    Scale the reference positions by the fit parameters and calculate the total distance between the matches.
    The parameters of the fit are:

        ``xc, yc = center positions``

        ``scale = magnification of the grid (focus)``

        ``xcoma, ycoma = linear change in magnification as a function of x/y (coma)``

    'ref' and 'spots' are assumed to be dict-like and must have the keys 'xcentroid' and 'ycentroid'.

    Parameters
    ----------
    pars : list-like
        The fit parameters passed in as a 5 element list: (xc, yc, scale, xcoma, ycoma)

    ref : dict-like
        Dict containing ``xcentroid`` and ``ycentroid`` keys that contain the reference X and Y
        positions of the apertures.

    spots : dict-like
        Dict containing ``xcentroid`` and ``ycentroid`` keys that contain the measured X and Y
        positions of the apertures.

    Returns
    -------
    dist : float
        The cumulative distance between the matched reference and measured aperture positions.
    """
    xc = pars[0]
    yc = pars[1]
    scale = pars[2]
    xcoma = pars[3]
    ycoma = pars[4]
    refx = ref['xcentroid'] * (scale + ref['xcentroid'] * xcoma) + xc
    refy = ref['ycentroid'] * (scale + ref['ycentroid'] * ycoma) + yc
    spotx = spots['xcentroid']
    spoty = spots['ycentroid']
    dist = aperture_distance(refx, refy, spotx, spoty)
    return dist


def get_slopes(data, ref, pup_mask, fwhm=7., thresh=5., cen=[255, 255],
               cen_thresh=0.8, cen_sigma=10., cen_tol=50., spot_snr_thresh=3.0, plot=True):
    """
    Analyze a WFS image and produce pixel offsets between reference and observed spot positions.

    Parameters
    ----------
    data : str or 2D np.ndarray
        FITS file or np.ndarray containing WFS observation
    ref : `~astropy.table.Table`
        Table of reference apertures
    pup_mask : str or 2D np.ndarray
        FITS file or np.ndarray containing mask used to register WFS spot pattern via cross-correlation
    fwhm : float (default: 7.0)
        FWHM of convolution kernel applied to image by the spot finding algorithm
    thresh : float (default: 5.0)
        Number of sigma above background for a spot to be considered detected
    cen : list-like with 2 elements (default: [255, 255])
        Expected position of the center of the WFS spot pattern in form [X_cen, Y_cen]
    cen_thresh : float (default: 0.8)
        Masking threshold as fraction of peak value used in `~photutils.detection.find_peaks`
    cen_sigma : float (default: 10.0)
        Width of gaussian filter applied to image by `~mmtwfs.wfs.center_pupil`
    cen_tol : float (default: 50.0)
        Tolerance for difference between expected and measureed pupil center
    spot_snr_thresh : float (default: 3.0)
        S/N tolerance for a WFS spot to be considered valid for analysis
    plot : bool
        Toggle plotting of image with aperture overlays

    Returns
    -------
    results : dict
        Results of the wavefront slopes measurement packaged into a dict with the following keys:
            slopes - mask np.ndarry containing the slope values in pixel units
            pup_coords - pupil coordinates for the position for each slope value
            spots - `~astropy.table.Table` as returned by photutils star finder routines
            src_aps - `~photutils.aperture.CircularAperture` for each detected spot
            spacing - list-like of form (xspacing, yspacing) containing the mean spacing between rows and columns of spots
            center - list-like of form (xcen, ycen) containing the center of the spot pattern
            ref_mask - np.ndarray of matched spots in reference image
            src_mask - np.ndarray of matched spots in the data image
            spot_sigma - sigma of a gaussian fit to a co-addition of detected spots
            figures - dict of figures that are optionally produced
            grid_fit - dict of best-fit parameters of grid fit used to do fine registration between source and reference spots
    """
    data = check_wfsdata(data)
    pup_mask = check_wfsdata(pup_mask)

    if ref.pup_outer is None:
        raise WFSConfigException("No pupil information applied to SH reference.")

    pup_outer = ref.pup_outer
    pup_inner = ref.pup_inner

    # input data should be background subtracted for best results. this initial guess of the center positions
    # will be good enough to get the central obscuration, but will need to be fine-tuned for aperture association.
    xcen, ycen, pupcen_fig = center_pupil(data, pup_mask, threshold=cen_thresh, sigma=cen_sigma, plot=plot)

    if np.hypot(xcen-cen[0], ycen-cen[1]) > cen_tol:
        msg = f"Measured pupil center [{round(xcen)}, {round(ycen)}] more than {cen_tol} pixels from {cen}."
        raise WFSAnalysisFailed(value=msg)

    # using the mean spacing is straightforward for square apertures and a reasonable underestimate for hexagonal ones
    ref_spacing = np.mean([ref.xspacing, ref.yspacing])
    apsize = ref_spacing

    srcs, masks, snrs, sigma, wfsfind_fig = get_apertures(data, apsize, fwhm=fwhm, thresh=thresh, cen=(xcen, ycen))

    # ignore low S/N spots
    srcs = srcs[snrs > spot_snr_thresh]

    # get grid spacing of the data
    xspacing, yspacing = grid_spacing(data, srcs)

    # find the scale difference between data and ref and use as init
    init_scale = (xspacing/ref.xspacing + yspacing/ref.yspacing) / 2.

    # apply masking to detected sources to avoid partially illuminated apertures at the edges
    srcs['dist'] = np.sqrt((srcs['xcentroid'] - xcen)**2 + (srcs['ycentroid'] - ycen)**2)
    srcs = srcs[(srcs['dist'] > pup_inner*init_scale) & (srcs['dist'] < pup_outer*init_scale)]

    # if we don't detect spots in at least half of the reference apertures, we can't usually get a good wavefront measurement
    if len(srcs) < 0.5 * len(ref.masked_apertures['xcentroid']):
        msg = "Only %d spots detected out of %d apertures." % (len(srcs), len(ref.masked_apertures['xcentroid']))
        raise WFSAnalysisFailed(value=msg)

    src_aps = photutils.aperture.CircularAperture(
        list(zip(srcs['xcentroid'], srcs['ycentroid'])),
        r=apsize/2.
    )

    # set up to do a fit of the reference apertures to the spot positions with the center, scaling, and position-dependent
    # scaling (coma) as free parameters
    args = (ref.masked_apertures, srcs)
    par_keys = ('xcen', 'ycen', 'scale', 'xcoma', 'ycoma')
    pars = (xcen, ycen, init_scale, 0.0, 0.0)
    coma_bound = 1e-4  # keep coma constrained by now since it can cause trouble
    # scipy.optimize.minimize can do bounded minimization so leverage that to keep the solution within a reasonable range.
    bounds = (
        (xcen-15, xcen+15),  # hopefully we're not too far off from true center...
        (ycen-15, ycen+15),
        (init_scale-0.05, init_scale+0.05),  # reasonable range of expected focus difference...
        (-coma_bound, coma_bound),
        (-coma_bound, coma_bound)
    )
    try:
        min_results = optimize.minimize(fit_apertures, pars, args=args, bounds=bounds, options={'ftol': 1e-13, 'gtol': 1e-7})
    except Exception as e:
        msg = f"Aperture grid matching failed: {e}"
        raise WFSAnalysisFailed(value=msg)

    fit_results = {}
    for i, k in enumerate(par_keys):
        fit_results[k] = min_results['x'][i]

    # this is more reliably the center of the actual pupil image whereas fit_results shifts a bit depending on detected spots.
    # the lenslet pattern can move around a bit on the pupil, but we need the center of the pupil to calculate their pupil
    # coordinates.
    pup_center = [xcen, ycen]

    scale = fit_results['scale']
    xcoma, ycoma = fit_results['xcoma'], fit_results['ycoma']

    refx = ref.masked_apertures['xcentroid'] * (scale + ref.masked_apertures['xcentroid'] * xcoma) + fit_results['xcen']
    refy = ref.masked_apertures['ycentroid'] * (scale + ref.masked_apertures['ycentroid'] * ycoma) + fit_results['ycen']

    xspacing = scale * ref.xspacing
    yspacing = scale * ref.yspacing

    # coarse match reference apertures to spots
    spacing = np.max([xspacing, yspacing])
    ref_mask, src_mask = match_apertures(refx, refy, srcs['xcentroid'], srcs['ycentroid'], max_dist=spacing/2.)

    # these are unscaled so that the slope includes defocus
    trim_refx = ref.masked_apertures['xcentroid'][ref_mask] + fit_results['xcen']
    trim_refy = ref.masked_apertures['ycentroid'][ref_mask] + fit_results['ycen']

    ref_aps = photutils.aperture.CircularAperture(
        list(zip(trim_refx, trim_refy)),
        r=ref_spacing/2.
    )

    slope_x = srcs['xcentroid'][src_mask] - trim_refx
    slope_y = srcs['ycentroid'][src_mask] - trim_refy

    pup_coords = (ref_aps.positions - pup_center) / [pup_outer, pup_outer]

    aps_fig = None
    if plot:
        norm = wfs_norm(data)
        aps_fig, ax = plt.subplots()
        aps_fig.set_label("Aperture Positions")
        ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
        ax.scatter(pup_center[0], pup_center[1])
        src_aps.plot(color='blue', ax=ax)

    # need full slopes array the size of the complete set of reference apertures and pre-filled with np.nan for masking
    slopes = np.nan * np.ones((2, len(ref.masked_apertures['xcentroid'])))

    slopes[0][ref_mask] = slope_x
    slopes[1][ref_mask] = slope_y

    figures = {}
    figures['pupil_center'] = pupcen_fig
    figures['slopes'] = aps_fig
    results = {
        "slopes": np.ma.masked_invalid(slopes),
        "pup_coords": pup_coords.transpose(),
        "spots": srcs,
        "src_aps": src_aps,
        "spacing": (xspacing, yspacing),
        "center": pup_center,
        "ref_mask": ref_mask,
        "src_mask": src_mask,
        "spot_sigma": sigma,
        "figures": figures,
        "grid_fit": fit_results
    }
    return results


def make_init_pars(nmodes=21, modestart=2, init_zv=None):
    """
    Make a set of initial parameters that can be used with `~lmfit.minimize` to make a wavefront fit with
    parameter names that are compatible with ZernikeVectors.

    Parameters
    ----------
    nmodes: int (default: 21)
        Number of Zernike modes to fit.
    modestart: int (default: 2)
        First Zernike mode to be used.
    init_zv: ZernikeVector (default: None)
        ZernikeVector containing initial values for the fit.

    Returns
    -------
    params: `~lmfit.Parameters` instance
        Initial parameters in form that can be passed to `~lmfit.minimize`.
    """
    pars = []
    for i in range(modestart, modestart+nmodes, 1):
        key = "Z{:02d}".format(i)
        if init_zv is not None:
            val = init_zv[key].value
            if val < 2. * np.finfo(float).eps:
                val = 0.0
        else:
            val = 0.0
        zpar = (key, val)
        pars.append(zpar)
    params = lmfit.Parameters()
    params.add_many(*pars)
    return params


def slope_diff(pars, coords, slopes, norm=False):
    """
    For a given set of wavefront fit parameters, calculate the "distance" between the predicted and measured wavefront
    slopes. This function is used by `~lmfit.minimize` which expects the sqrt to be applied rather than a chi-squared,
    """
    parsdict = pars.valuesdict()
    rho, phi = cart2pol(coords)
    xslope = slopes[0]
    yslope = slopes[1]
    pred_xslope, pred_yslope = zernike_slopes(parsdict, rho, phi, norm=norm)
    dist = np.sqrt((xslope - pred_xslope)**2 + (yslope - pred_yslope)**2)
    return dist


class SH_Reference(object):
    """
    Class to handle Shack-Hartmann reference data
    """
    def __init__(self, data, fwhm=4.5, threshold=20.0, plot=True):
        """
        Read WFS reference image and generate reference magnifications (i.e. grid spacing) and
        aperture positions.

        Parameters
        ----------
        data : FITS filename or 2D ndarray
            WFS reference image
        fwhm : float
            FWHM in pixels of DAOfind convolution kernel
        threshold : float
            DAOfind threshold in units of the standard deviation of the image
        plot : bool
            Toggle plotting of the reference image and overlayed apertures
        """
        self.data = check_wfsdata(data)
        data = data - np.median(data)
        self.apertures, self.figure = wfsfind(data, fwhm=fwhm, threshold=threshold, plot=plot)
        if plot:
            self.figure.set_label("Reference Image")

        self.xcen = self.apertures['xcentroid'].mean()
        self.ycen = self.apertures['ycentroid'].mean()
        self.xspacing, self.yspacing = grid_spacing(data, self.apertures)

        # make masks for each reference spot and fit a 2D gaussian to get its FWHM. the reference FWHM is subtracted in
        # quadrature from the observed FWHM when calculating the seeing.
        apsize = np.mean([self.xspacing, self.yspacing])
        apers = photutils.aperture.CircularAperture(
            list(zip(self.apertures['xcentroid'], self.apertures['ycentroid'])),
            r=apsize/2.
        )
        masks = apers.to_mask(method='subpixel')
        self.photapers = apers
        self.spot = np.zeros(masks[0].shape)
        for m in masks:
            subim = m.cutout(data)
            # make co-added spot image for use in calculating the seeing
            if subim.shape == self.spot.shape:
                self.spot += subim

        self.apertures['xcentroid'] -= self.xcen
        self.apertures['ycentroid'] -= self.ycen
        self.apertures['dist'] = np.sqrt(self.apertures['xcentroid']**2 + self.apertures['ycentroid']**2)
        self.masked_apertures = self.apertures

        self.pup_inner = None
        self.pup_outer = None

    def adjust_center(self, x, y):
        """
        Adjust reference center to new x, y position.
        """
        self.apertures['xcentroid'] += self.xcen
        self.apertures['ycentroid'] += self.ycen
        self.apertures['xcentroid'] -= x
        self.apertures['ycentroid'] -= y
        self.apertures['dist'] = np.sqrt(self.apertures['xcentroid']**2 + self.apertures['ycentroid']**2)
        self.xcen = x
        self.ycen = y
        self.apply_pupil(self.pup_inner, self.pup_outer)

    def apply_pupil(self, pup_inner, pup_outer):
        """
        Apply a pupil mask to the reference apertures
        """
        if pup_inner is not None and pup_outer is not None:
            self.masked_apertures = self.apertures[(self.apertures['dist'] > pup_inner) & (self.apertures['dist'] < pup_outer)]
            self.pup_inner = pup_inner
            self.pup_outer = pup_outer

    def pup_coords(self, pup_outer):
        """
        Take outer radius of pupil and calculate pupil coordinates for the masked apertures
        """
        coords = (self.masked_apertures['xcentroid']/pup_outer, self.masked_apertures['ycentroid']/pup_outer)
        return coords


def WFSFactory(wfs="f5", config={}, **kwargs):
    """
    Build and return proper WFS sub-class instance based on the value of 'wfs'.
    """
    config = merge_config(config, dict(**kwargs))
    wfs = wfs.lower()

    types = recursive_subclasses(WFS)
    wfses = [t.__name__.lower() for t in types]
    wfs_map = dict(list(zip(wfses, types)))

    if wfs not in wfses:
        raise WFSConfigException(value="Specified WFS, %s, not valid or not implemented." % wfs)

    if 'plot' in config:
        plot = config['plot']
    else:
        plot = True

    wfs_cls = wfs_map[wfs](config=config, plot=plot)
    return wfs_cls


class WFS(object):
    """
    Defines configuration pattern and methods common to all WFS systems
    """
    def __init__(self, config={}, plot=True, **kwargs):
        key = self.__class__.__name__.lower()
        self.__dict__.update(merge_config(mmtwfs_config['wfs'][key], config))
        self.telescope = TelescopeFactory(telescope=self.telescope, secondary=self.secondary)
        self.secondary = self.telescope.secondary
        self.plot = plot
        self.connected = False
        self.ref_fwhm = self.ref_spot_fwhm()

        # this factor calibrates spot motion in pixels to nm of wavefront error
        self.tiltfactor = self.telescope.nmperasec * (self.pix_size.to(u.arcsec).value)

        # if this is the same for all modes, load it once here
        if hasattr(self, "reference_file"):
            refdata, hdr = check_wfsdata(self.reference_file, header=True)
            refdata = self.trim_overscan(refdata, hdr)
            reference = SH_Reference(refdata, plot=self.plot)

        # now assign 'reference' for each mode so that it can be accessed consistently in all cases
        for mode in self.modes:
            if 'reference_file' in self.modes[mode]:
                refdata, hdr = check_wfsdata(self.modes[mode]['reference_file'], header=True)
                refdata = self.trim_overscan(refdata, hdr)
                self.modes[mode]['reference'] = SH_Reference(
                    refdata,
                    plot=self.plot
                )
            else:
                self.modes[mode]['reference'] = reference

    def ref_spot_fwhm(self):
        """
        Calculate the Airy FWHM in pixels of a perfect WFS spot from the optical prescription and detector pixel size
        """
        theta_fwhm = 1.028 * self.eff_wave / self.lenslet_pitch
        det_fwhm = np.arctan(theta_fwhm).value * self.lenslet_fl
        det_fwhm_pix = det_fwhm.to(u.um).value / self.pix_um.to(u.um).value
        return det_fwhm_pix

    def get_flipud(self, mode=None):
        """
        Determine if the WFS image needs to be flipped up/down
        """
        return False

    def get_fliplr(self, mode=None):
        """
        Determine if the WFS image needs to be flipped left/right
        """
        return False

    def ref_pupil_location(self, mode, hdr=None):
        """
        Get the center of the pupil on the reference image
        """
        ref = self.modes[mode]['reference']
        x = ref.xcen
        y = ref.ycen
        return x, y

    def seeing(self, mode, sigma, airmass=None):
        """
        Given a sigma derived from a gaussian fit to a WFS spot, deconvolve the systematic width from the reference image
        and relate the remainder to r_0 and thus a seeing FWHM.
        """
        # the effective wavelength of the WFS imagers is about 600-700 nm. mmirs and the oldf9 system use blue-blocking filters
        wave = self.eff_wave
        wave = wave.to(u.m).value  # r_0 equation expects meters so convert

        refwave = 500 * u.nm  # standard wavelength that seeing values are referenced to
        refwave = refwave.to(u.m).value

        # calculate the physical size of each aperture.
        ref = self.modes[mode]['reference']
        apsize_pix = np.max((ref.xspacing, ref.yspacing))
        d = self.telescope.diameter * apsize_pix / self.pup_size
        d = d.to(u.m).value  # r_0 equation expects meters so convert

        # we need to deconvolve the instrumental spot width from the measured one to get the portion of the width that
        # is due to spot motion
        ref_sigma = stats.funcs.gaussian_fwhm_to_sigma * self.ref_fwhm
        if sigma > ref_sigma:
            corr_sigma = np.sqrt(sigma**2 - ref_sigma**2)
        else:
            return 0.0 * u.arcsec, 0.0 * u.arcsec

        corr_sigma *= self.pix_size.to(u.rad).value  # r_0 equation expects radians so convert

        # this equation relates the motion within a single aperture to the characteristic scale size of the
        # turbulence, r_0.
        r_0 = (0.179 * (wave**2) * (d**(-1/3))/corr_sigma**2)**0.6

        # this equation relates the turbulence scale size to an expected image FWHM at the given wavelength.
        raw_seeing = u.Quantity(u.rad * 0.98 * wave / r_0, u.arcsec)

        # seeing scales as lambda^-1/5 so calculate factor to scale to reference lambda
        wave_corr = refwave**-0.2 / wave**-0.2

        raw_seeing *= wave_corr

        # correct seeing to zenith
        if airmass is not None:
            seeing = raw_seeing / airmass**0.6
        else:
            seeing = raw_seeing

        return seeing, raw_seeing

    def pupil_mask(self, hdr=None):
        """
        Load and return the WFS spot mask used to locate and register the pupil
        """
        pup_mask = check_wfsdata(self.wfs_mask)
        return pup_mask

    def reference_aberrations(self, mode, **kwargs):
        """
        Create reference ZernikeVector for 'mode'.
        """
        z = ZernikeVector(**self.modes[mode]['ref_zern'])
        return z

    def get_mode(self, hdr):
        """
        If mode is not specified, either set it to the default mode or figure out the mode from the header.
        """
        mode = self.default_mode
        return mode

    def process_image(self, fitsfile):
        """
        Process the image to make it suitable for accurate wavefront analysis.  Steps include nuking cosmic rays,
        subtracting background, handling overscan regions, etc.
        """
        rawdata, hdr = check_wfsdata(fitsfile, header=True)

        trimdata = self.trim_overscan(rawdata, hdr=hdr)

        # MMIRS gets a lot of hot pixels/CRs so make a quick pass to nuke them
        cr_mask, data = detect_cosmics(trimdata, sigclip=5., niter=5, cleantype='medmask', psffwhm=5.)

        # calculate the background and subtract it
        bkg_estimator = photutils.background.ModeEstimatorBackground()
        mask = make_spot_mask(data, nsigma=2, npixels=5, dilate_size=11)
        bkg = photutils.background.Background2D(data, (10, 10), filter_size=(5, 5), bkg_estimator=bkg_estimator, mask=mask)
        data -= bkg.background

        return data, hdr

    def trim_overscan(self, data, hdr=None):
        """
        Use the DATASEC in the header to determine the region to trim out. If no header provided or if the header
        doesn't contain DATASEC, return data unchanged.
        """
        if hdr is None:
            return data

        if 'DATASEC' not in hdr:
            # if no DATASEC in header, punt and return unchanged
            return data

        datasec = slice_from_string(hdr['DATASEC'], fits_convention=True)
        return data[datasec]

    def measure_slopes(self, fitsfile, mode=None, plot=True, flipud=False, fliplr=False):
        """
        Take a WFS image in FITS format, perform background subtration, pupil centration, and then use get_slopes()
        to perform the aperture placement and spot centroiding.
        """
        data, hdr = self.process_image(fitsfile)
        plot = plot and self.plot

        # flip data up/down if we need to. only binospec needs to currently.
        if flipud or self.get_flipud(mode=mode):
            data = np.flipud(data)

        # flip left/right if we need to. no mode currently does, but who knows what the future holds.
        if fliplr or self.get_fliplr(mode=mode):
            data = np.fliplr(data)

        if mode is None:
            mode = self.get_mode(hdr)

        if mode not in self.modes:
            msg = "Invalid mode, %s, for WFS system, %s." % (mode, self.__class__.__name__)
            raise WFSConfigException(value=msg)

        # if available, get the rotator angle out of the header
        if 'ROT' in hdr:
            rotator = hdr['ROT'] * u.deg
        else:
            rotator = 0.0 * u.deg

        # if there's a ROTOFF in the image header, grab it and adjust the rotator angle accordingly
        if 'ROTOFF' in hdr:
            rotator -= hdr['ROTOFF'] * u.deg

        # make mask for finding wfs spot pattern
        pup_mask = self.pupil_mask(hdr=hdr)

        # get adjusted reference center position and update the reference
        xcen, ycen = self.ref_pupil_location(mode, hdr=hdr)
        self.modes[mode]['reference'].adjust_center(xcen, ycen)

        # apply pupil to the reference
        self.modes[mode]['reference'].apply_pupil(self.pup_inner, self.pup_size/2.)

        ref_zv = self.reference_aberrations(mode, hdr=hdr)

        zref = ref_zv.array
        if len(zref) < self.nzern:
            pad = np.zeros(self.nzern - len(zref))
            zref = np.hstack((zref, pad))

        try:
            slope_results = get_slopes(
                data,
                self.modes[mode]['reference'],
                pup_mask,
                fwhm=self.find_fwhm,
                thresh=self.find_thresh,
                cen=self.cor_coords,
                cen_thresh=self.cen_thresh,
                cen_sigma=self.cen_sigma,
                cen_tol=self.cen_tol,
                plot=plot
            )
            slopes = slope_results['slopes']
            coords = slope_results['pup_coords']
            ref_pup_coords = self.modes[mode]['reference'].pup_coords(self.pup_size/2.)

            rho, phi = cart2pol(ref_pup_coords)
            ref_slopes = -(1. / self.tiltfactor) * np.array(zernike_slopes(ref_zv, rho, phi))
            aps = slope_results['src_aps']
            ref_mask = slope_results['ref_mask']
            src_mask = slope_results['src_mask']
            figures = slope_results['figures']
        except WFSAnalysisFailed as e:
            log.warning(f"Wavefront slope measurement failed: {e}")
            slope_fig = None
            if plot:
                slope_fig, ax = plt.subplots()
                slope_fig.set_label("WFS Image")
                norm = wfs_norm(data)
                ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
            results = {}
            results['slopes'] = None
            results['figures'] = {}
            results['mode'] = mode
            results['figures']['slopes'] = slope_fig
            return results
        except Exception as e:
            raise WFSAnalysisFailed(value=str(e))

        # use the average width of the spots to estimate the seeing and use the airmass to extrapolate to zenith seeing
        if 'AIRMASS' in hdr:
            airmass = hdr['AIRMASS']
        else:
            airmass = None
        seeing, raw_seeing = self.seeing(mode=mode, sigma=slope_results['spot_sigma'], airmass=airmass)

        if plot:
            sub_slopes = slopes - ref_slopes
            x = aps.positions.transpose()[0][src_mask]
            y = aps.positions.transpose()[1][src_mask]
            uu = sub_slopes[0][ref_mask]
            vv = sub_slopes[1][ref_mask]
            norm = wfs_norm(data)
            figures['slopes'].set_label("Aperture Positions and Spot Movement")
            ax = figures['slopes'].axes[0]
            ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
            aps.plot(color='blue', ax=ax)
            ax.quiver(x, y, uu, vv, scale_units='xy', scale=0.2, pivot='tip', color='red')
            xl = [0.1*data.shape[1]]
            yl = [0.95*data.shape[0]]
            ul = [1.0/self.pix_size.value]
            vl = [0.0]
            ax.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.2, pivot='tip', color='red')
            ax.scatter([slope_results['center'][0]], [slope_results['center'][1]])
            ax.text(0.12*data.shape[1], 0.95*data.shape[0], "1{0:unicode}".format(u.arcsec), verticalalignment='center')
            ax.set_title("Seeing: %.2f\" (%.2f\" @ zenith)" % (raw_seeing.value, seeing.value))

        results = {}
        results['seeing'] = seeing
        results['raw_seeing'] = raw_seeing
        results['slopes'] = slopes
        results['ref_slopes'] = ref_slopes
        results['ref_zv'] = ref_zv
        results['spots'] = slope_results['spots']
        results['pup_coords'] = coords
        results['ref_pup_coords'] = ref_pup_coords
        results['apertures'] = aps
        results['xspacing'] = slope_results['spacing'][0]
        results['yspacing'] = slope_results['spacing'][1]
        results['xcen'] = slope_results['center'][0]
        results['ycen'] = slope_results['center'][1]
        results['pup_mask'] = pup_mask
        results['data'] = data
        results['header'] = hdr
        results['rotator'] = rotator
        results['mode'] = mode
        results['ref_mask'] = ref_mask
        results['src_mask'] = src_mask
        results['fwhm'] = stats.funcs.gaussian_sigma_to_fwhm * slope_results['spot_sigma']
        results['figures'] = figures
        results['grid_fit'] = slope_results['grid_fit']

        return results

    def fit_wavefront(self, slope_results, plot=True):
        """
        Use results from self.measure_slopes() to fit a set of zernike polynomials to the wavefront shape.
        """
        plot = plot and self.plot
        if slope_results['slopes'] is not None:
            results = {}
            slopes = -self.tiltfactor * slope_results['slopes']
            coords = slope_results['ref_pup_coords']
            rho, phi = cart2pol(coords)

            zref = slope_results['ref_zv']
            params = make_init_pars(nmodes=self.nzern, init_zv=zref)
            results['fit_report'] = lmfit.minimize(slope_diff, params, args=(coords, slopes))
            zfit = ZernikeVector(coeffs=results['fit_report'])

            results['raw_zernike'] = zfit

            # derotate the zernike solution to match the primary mirror coordinate system
            total_rotation = self.rotation - slope_results['rotator']
            zv_rot = ZernikeVector(coeffs=results['fit_report'])
            zv_rot.rotate(angle=-total_rotation)
            results['rot_zernike'] = zv_rot

            # subtract the reference aberrations
            zsub = zv_rot - zref
            results['ref_zernike'] = zref
            results['zernike'] = zsub

            pred_slopes = np.array(zernike_slopes(zfit, rho, phi))
            diff = slopes - pred_slopes
            diff_pix = diff / self.tiltfactor
            rms = np.sqrt((diff[0]**2 + diff[1]**2).mean())
            results['residual_rms_asec'] = rms / self.telescope.nmperasec * u.arcsec
            results['residual_rms'] = rms * zsub.units
            results['zernike_rms'] = zsub.rms
            results['zernike_p2v'] = zsub.peak2valley

            fig = None
            if plot:
                ref_mask = slope_results['ref_mask']
                src_mask = slope_results['src_mask']
                im = slope_results['data']
                gnorm = wfs_norm(im)
                fig, ax = plt.subplots()
                fig.set_label("Zernike Fit Residuals")
                ax.imshow(im, cmap='Greys', origin='lower', norm=gnorm, interpolation='None')
                x = slope_results['apertures'].positions.transpose()[0][src_mask]
                y = slope_results['apertures'].positions.transpose()[1][src_mask]
                ax.quiver(x, y, diff_pix[0][ref_mask], diff_pix[1][ref_mask], scale_units='xy',
                          scale=0.05, pivot='tip', color='red')
                xl = [0.1*im.shape[1]]
                yl = [0.95*im.shape[0]]
                ul = [0.2/self.pix_size.value]
                vl = [0.0]
                ax.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.05, pivot='tip', color='red')
                ax.text(0.12*im.shape[1], 0.95*im.shape[0], "0.2{0:unicode}".format(u.arcsec), verticalalignment='center')
                ax.text(
                    0.95*im.shape[1],
                    0.95*im.shape[0],
                    "Residual RMS: {0.value:0.2f}{0.unit:unicode}".format(results['residual_rms_asec']),
                    verticalalignment='center',
                    horizontalalignment='right'
                )
                iq = np.sqrt(results['residual_rms_asec']**2 +
                             (results['zernike_rms'].value / self.telescope.nmperasec * u.arcsec)**2)
                ax.set_title("Image Quality: {0.value:0.2f}{0.unit:unicode}".format(iq))

            results['resid_plot'] = fig
        else:
            results = None
        return results

    def calculate_primary(self, zv, threshold=0.0 * u.nm, mask=[]):
        """
        Calculate force corrections to primary mirror and any required focus offsets. Use threshold to determine which
        terms in 'zv' to use in the force calculations. Any terms with normalized amplitude less than threshold will
        not be used in the force calculation. In addition, individual terms can be forced to be masked.
        """
        zv.denormalize()
        zv_masked = ZernikeVector()
        zv_norm = zv.copy()
        zv_norm.normalize()

        log.debug(f"thresh: {threshold} mask {mask}")

        for z in zv:
            if abs(zv_norm[z]) >= threshold:
                zv_masked[z] = zv[z]
                log.debug(f"{z}: Good")
            else:
                log.debug(f"{z}: Bad")
        zv_masked.denormalize()  # need to assure we're using fringe coeffs
        log.debug(f"\nInput masked: {zv_masked}")

        # use any available error bars to mask down to 1 sigma below amplitude or 0 if error bars are larger than amplitude.
        for z in zv_masked:
            frac_err = 1. - min(zv_masked.frac_error(key=z), 1.)
            zv_masked[z] *= frac_err
        log.debug(f"\nErrorbar masked: {zv_masked}")
        forces, m1focus, zv_allmasked = self.telescope.calculate_primary_corrections(
            zv=zv_masked,
            mask=mask,
            gain=self.m1_gain
        )
        log.debug(f"\nAll masked: {zv_allmasked}")
        return forces, m1focus, zv_allmasked

    def calculate_focus(self, zv):
        """
        Convert Zernike defocus to um of secondary offset.
        """
        z_denorm = zv.copy()
        z_denorm.denormalize()  # need to assure we're using fringe coeffs
        frac_err = 1. - min(z_denorm.frac_error(key='Z04'), 1.)
        foc_corr = -self.m2_gain * frac_err * z_denorm['Z04'] / self.secondary.focus_trans

        return foc_corr.round(2)

    def calculate_cc(self, zv):
        """
        Convert Zernike coma (Z07 and Z08) into arcsec of secondary center-of-curvature tilts.
        """
        z_denorm = zv.copy()
        z_denorm.denormalize()  # need to assure we're using fringe coeffs

        # fix coma using tilts around the M2 center of curvature.
        y_frac_err = 1. - min(z_denorm.frac_error(key='Z07'), 1.)
        x_frac_err = 1. - min(z_denorm.frac_error(key='Z08'), 1.)
        cc_y_corr = -self.m2_gain * y_frac_err * z_denorm['Z07'] / self.secondary.theta_cc
        cc_x_corr = -self.m2_gain * x_frac_err * z_denorm['Z08'] / self.secondary.theta_cc

        return cc_x_corr.round(3), cc_y_corr.round(3)

    def calculate_recenter(self, fit_results, defoc=1.0):
        """
        Perform zero-coma hexapod tilts to align the pupil center to the center-of-rotation.
        The location of the CoR is configured to be at self.cor_coords.
        """
        xc = fit_results['xcen']
        yc = fit_results['ycen']
        xref = self.cor_coords[0]
        yref = self.cor_coords[1]
        dx = xc - xref
        dy = yc - yref

        total_rotation = u.Quantity(self.rotation - fit_results['rotator'], u.rad).value

        dr, phi = cart2pol([dx, dy])

        derot_phi = phi + total_rotation

        az, el = pol2cart([dr, derot_phi])

        az *= self.az_parity * self.pix_size * defoc  # pix size scales with the pupil size as focus changes.
        el *= self.el_parity * self.pix_size * defoc

        return az.round(3), el.round(3)

    def clear_m1_corrections(self):
        """
        Clear corrections applied to the primary mirror. This includes the 'm1spherical' offsets sent to the secondary.
        """
        log.info("Clearing WFS corrections from M1 and m1spherical offsets from M2.")
        clear_forces, clear_m1focus = self.telescope.clear_forces()
        return clear_forces, clear_m1focus

    def clear_m2_corrections(self):
        """
        Clear corrections sent to the secondary mirror, specifically the 'wfs' offsets.
        """
        log.info("Clearing WFS offsets from M2's hexapod.")
        cmds = self.secondary.clear_wfs()
        return cmds

    def clear_corrections(self):
        """
        Clear all applied WFS corrections
        """
        forces, m1focus = self.clear_m1_corrections()
        cmds = self.clear_m2_corrections()
        return forces, m1focus, cmds

    def connect(self):
        """
        Set state to connected
        """
        self.telescope.connect()
        self.secondary.connect()

        if self.telescope.connected and self.secondary.connected:
            self.connected = True
        else:
            self.connected = False

    def disconnect(self):
        """
        Set state to disconnected
        """
        self.telescope.disconnect()
        self.secondary.disconnect()
        self.connected = False


class F9(WFS):
    """
    Defines configuration and methods specific to the F/9 WFS system
    """
    def __init__(self, config={}, plot=True):
        super(F9, self).__init__(config=config, plot=plot)

        self.connected = False

        # set up CompMirror object
        self.compmirror = CompMirror()

    def connect(self):
        """
        Run parent connect() method and then connect to the topbox if we can connect to the rest.
        """
        super(F9, self).connect()
        if self.connected:
            self.compmirror.connect()

    def disconnect(self):
        """
        Run parent disconnect() method and then disconnect the topbox
        """
        super(F9, self).disconnect()
        self.compmirror.disconnect()


class NewF9(F9):
    """
    Defines configuration and methods specific to the F/9 WFS system with the new SBIG CCD
    """
    def process_image(self, fitsfile):
        """
        Process the image to make it suitable for accurate wavefront analysis.  Steps include nuking cosmic rays,
        subtracting background, handling overscan regions, etc.
        """
        rawdata, hdr = check_wfsdata(fitsfile, header=True)

        cr_mask, data = detect_cosmics(rawdata, sigclip=15., niter=5, cleantype='medmask', psffwhm=10.)

        # calculate the background and subtract it
        bkg_estimator = photutils.background.ModeEstimatorBackground()
        mask = make_spot_mask(data, nsigma=2, npixels=7, dilate_size=13)
        bkg = photutils.background.Background2D(data, (50, 50), filter_size=(15, 15), bkg_estimator=bkg_estimator, mask=mask)
        data -= bkg.background

        return data, hdr


class F5(WFS):
    """
    Defines configuration and methods specific to the F/5 WFS systems
    """
    def __init__(self, config={}, plot=True):
        super(F5, self).__init__(config=config, plot=plot)

        self.connected = False
        self.sock = None

        # load lookup table for off-axis aberrations
        self.aberr_table = ascii.read(self.aberr_table_file)

    def process_image(self, fitsfile):
        """
        Process the image to make it suitable for accurate wavefront analysis.  Steps include nuking cosmic rays,
        subtracting background, handling overscan regions, etc.
        """
        rawdata, hdr = check_wfsdata(fitsfile, header=True)

        trimdata = self.trim_overscan(rawdata, hdr=hdr)

        cr_mask, data = detect_cosmics(trimdata, sigclip=15., niter=5, cleantype='medmask', psffwhm=10.)

        # calculate the background and subtract it
        bkg_estimator = photutils.background.ModeEstimatorBackground()
        mask = make_spot_mask(data, nsigma=2, npixels=5, dilate_size=11)
        bkg = photutils.background.Background2D(data, (20, 20), filter_size=(11, 11), bkg_estimator=bkg_estimator, mask=mask)
        data -= bkg.background

        return data, hdr

    def ref_pupil_location(self, mode, hdr=None):
        """
        For now we set the F/5 wfs center by hand based on engineering data. Should determine this more carefully.
        """
        x = 262.0
        y = 259.0
        return x, y

    def focal_plane_position(self, hdr):
        """
        Need to fill this in for the hecto f/5 WFS system. For now will assume it's always on-axis.
        """
        return 0.0 * u.deg, 0.0 * u.deg

    def calculate_recenter(self, fit_results, defoc=1.0):
        """
        Perform zero-coma hexapod tilts to align the pupil center to the center-of-rotation.
        The location of the CoR is configured to be at self.cor_coords.
        """
        xc = fit_results['xcen']
        yc = fit_results['ycen']
        xref = self.cor_coords[0]
        yref = self.cor_coords[1]
        dx = xc - xref
        dy = yc - yref

        cam_rotation = self.rotation - 90 * u.deg  # pickoff plus fold mirror makes a 90 deg rotation
        total_rotation = u.Quantity(cam_rotation - fit_results['rotator'], u.rad).value

        dr, phi = cart2pol([dx, -dy])  # F/5 camera needs an up/down flip

        derot_phi = phi + total_rotation

        az, el = pol2cart([dr, derot_phi])

        az *= self.az_parity * self.pix_size * defoc  # pix size scales with the pupil size as focus changes.
        el *= self.el_parity * self.pix_size * defoc

        return az.round(3), el.round(3)

    def reference_aberrations(self, mode, hdr=None):
        """
        Create reference ZernikeVector for 'mode'.  Pass 'hdr' to self.focal_plane_position() to get position of
        the WFS when the data was acquired.
        """
        # for most cases, this gets the reference focus
        z_default = ZernikeVector(**self.modes[mode]['ref_zern'])

        # now get the off-axis aberrations
        z_offaxis = ZernikeVector()
        if hdr is None:
            log.warning("Missing WFS header. Assuming data is acquired on-axis.")
            field_r = 0.0 * u.deg
            field_phi = 0.0 * u.deg
        else:
            field_r, field_phi = self.focal_plane_position(hdr)

        # ignore piston and x/y tilts
        for i in range(4, 12):
            k = "Z%02d" % i
            z_offaxis[k] = np.interp(field_r.to(u.deg).value, self.aberr_table['field_r'], self.aberr_table[k]) * u.um

        # remove the 90 degree offset between the MMT and zernike conventions and then rotate the offaxis aberrations
        z_offaxis.rotate(angle=field_phi - 90. * u.deg)

        z = z_default + z_offaxis

        return z


class Binospec(F5):
    """
    Defines configuration and methods specific to the Binospec WFS system. Binospec uses the same aberration table
    as the F5 system so we inherit from that.
    """
    def get_flipud(self, mode):
        """
        Method to determine if the WFS image needs to be flipped up/down

        During the first binospec commissioning run the images were flipped u/d as they came in. Since then, they are
        left as-is and get flipped internally based on this flag. The reference file is already flipped.
        """
        return True

    def ref_pupil_location(self, mode, hdr=None):
        """
        If a header is passed in, use Jan Kansky's linear relations to get the pupil center on the reference image.
        Otherwise, use the default method.
        """
        if hdr is None:
            ref = self.modes[mode]['reference']
            x = ref.xcen
            y = ref.ycen
        else:
            for k in ['STARXMM', 'STARYMM']:
                if k not in hdr:
                    # we'll be lenient for now with missing header info. if not provided, assume we're on-axis.
                    msg = f"Missing value, {k}, that is required to transform Binospec guider coordinates. Defaulting to 0.0."
                    log.warning(msg)
                    hdr[k] = 0.0
            y = 232.771 + 0.17544 * hdr['STARXMM']
            x = 265.438 + -0.20406 * hdr['STARYMM'] + 12.0
        return x, y

    def focal_plane_position(self, hdr):
        """
        Transform from the Binospec guider coordinate system to MMTO focal plane coordinates.
        """
        for k in ['ROT', 'STARXMM', 'STARYMM']:
            if k not in hdr:
                # we'll be lenient for now with missing header info. if not provided, assume we're on-axis.
                msg = f"Missing value, {k}, that is required to transform Binospec guider coordinates. Defaulting to 0.0."
                log.warning(msg)
                hdr[k] = 0.0

        guide_x = hdr['STARXMM']
        guide_y = hdr['STARYMM']
        rot = hdr['ROT']

        guide_r = np.sqrt(guide_x**2 + guide_y**2) * u.mm
        rot = u.Quantity(rot, u.deg)  # make sure rotation is cast to degrees

        # the MMTO focal plane coordinate convention has phi=0 aligned with +Y instead of +X
        if guide_y != 0.0:
            guide_phi = np.arctan2(guide_x, guide_y) * u.rad
        else:
            guide_phi = 90. * u.deg

        # transform radius in guider coords to degrees in focal plane
        focal_r = (guide_r / self.secondary.plate_scale).to(u.deg)
        focal_phi = guide_phi + rot + self.rotation

        log.debug(f"guide_phi: {guide_phi.to(u.rad)} rot: {rot}")

        return focal_r, focal_phi

    def in_wfs_region(self, xw, yw, x, y):
        """
        Determine if a position is within the region available to Binospec's WFS
        """
        return True  # placekeeper until the optical prescription is implemented

    def pupil_mask(self, hdr, npts=14):
        """
        Generate a synthetic pupil mask
        """
        if hdr is not None:
            x_wfs = hdr.get('STARXMM', 150.0)
            y_wfs = hdr.get('STARYMM', 0.0)
        else:
            x_wfs = 150.0
            y_wfs = 0.0
            log.warning("Header information not available for Binospec pupil mask. Assuming default position.")

        good = []
        center = self.pup_size / 2.
        obsc = self.telescope.obscuration.value
        spacing = 2.0 / npts
        for x in np.arange(-1, 1, spacing):
            for y in np.arange(-1, 1, spacing):
                r = np.hypot(x, y)
                if (r < 1 and np.hypot(x, y) >= obsc):
                    if self.in_wfs_region(x_wfs, y_wfs, x, y):
                        x_impos = center * (x + 1.)
                        y_impos = center * (y + 1.)
                        amp = 1.
                        # this is kind of a hacky way to dim spots near the edge, but easier than doing full calc
                        # of the aperture intersection with pupil. it also doesn't need to be that accurate for the
                        # purposes of the cross-correlation used to register the pupil.
                        if r > 1. - spacing:
                            amp = 1. - (r - (1. - spacing)) / spacing
                        if r - obsc < spacing:
                            amp = (r - obsc) / spacing
                        good.append((amp, x_impos, y_impos))

        yi, xi = np.mgrid[0:self.pup_size, 0:self.pup_size]
        im = np.zeros((self.pup_size, self.pup_size))
        sigma = 3.
        for g in good:
            im += Gaussian2D(g[0], g[1], g[2], sigma, sigma)(xi, yi)

        # Measured by hand from reference LED image
        cam_rot = 0.595

        im_rot = rotate(im, cam_rot, reshape=False)
        im_rot[im_rot < 1e-2] = 0.0

        return im_rot


class MMIRS(F5):
    """
    Defines configuration and methods specific to the MMIRS WFS system
    """
    def __init__(self, config={}, plot=True):
        super(MMIRS, self).__init__(config=config, plot=plot)

        # Parameters describing MMIRS pickoff mirror geometry
        # Location and diameter of exit pupil
        # Determined by tracing chief ray at 7.2' field angle with mmirs_asbuiltoptics_20110107_corronly.zmx
        self.zp = 71.749 / 0.02714
        self.dp = self.zp / 5.18661  # Working f/# from Zemax file

        # Location of fold mirror
        self.zm = 114.8

        # Angle of fold mirror
        self.am = 42 * u.deg

        # Following dimensions from drawing MMIRS-1233_Rev1.pdf
        # Diameter of pickoff mirror
        self.pickoff_diam = (6.3 * u.imperial.inch).to(u.mm).value

        # X size of opening in pickoff mirror
        self.pickoff_xsize = (3.29 * u.imperial.inch).to(u.mm).value

        # Y size of opening in pickoff mirror
        self.pickoff_ysize = (3.53 * u.imperial.inch).to(u.mm).value

        # radius of corner  in pickoff mirror
        self.pickoff_rcirc = (0.4 * u.imperial.inch).to(u.mm).value

    def mirrorpoint(self, x0, y0, x, y):
        """
        Compute intersection of ray with pickoff mirror.
        The ray leaves the exit pupil at position x,y and hits the focal surface at x0,y0.
        Math comes from http://geomalgorithms.com/a05-_intersect-1.html
        """
        # Point in focal plane
        P0 = np.array([x0, y0, 0])

        # Point in exit pupil
        P1 = np.array([x * self.dp / 2, y * self.dp / 2, self.zp])

        # Pickoff mirror intesection with optical axis
        V0 = np.array([0, 0, self.zm])

        # normal to mirror
        if (x0 < 0):
            n = np.array([-np.sin(self.am), 0, np.cos(self.am)])
        else:
            n = np.array([np.sin(self.am), 0, np.cos(self.am)])

        w = P0 - V0

        # Vector connecting P0 to P1
        u = P1 - P0

        # Distance from P0 to intersection as a fraction of abs(u)
        s = -n.dot(w) / n.dot(u)

        # Intersection point on mirror
        P = P0 + s * u

        return (P[0], P[1])

    def onmirror(self, x, y, side):
        """
        Determine if a point is on the pickoff mirror surface:
            x,y = coordinates of ray
            side=1 means right face of the pickoff mirror, -1=left face
        """
        if np.hypot(x, y) > self.pickoff_diam / 2.:
            return False
        if x * side < 0:
            return False
        x = abs(x)
        y = abs(y)
        if ((x > self.pickoff_xsize/2) or (y > self.pickoff_ysize/2)
            or (x > self.pickoff_xsize/2 - self.pickoff_rcirc and y > self.pickoff_ysize/2 - self.pickoff_rcirc
                and np.hypot(x - (self.pickoff_xsize/2 - self.pickoff_rcirc),
                             y - (self.pickoff_ysize/2 - self.pickoff_rcirc)) > self.pickoff_rcirc)):
            return True
        else:
            return False

    def drawoutline(self, ax):
        """
        Draw outline of MMIRS pickoff mirror onto matplotlib axis, ax
        """
        circ = np.arange(360) * u.deg
        ax.plot(np.cos(circ) * self.pickoff_diam/2, np.sin(circ) * self.pickoff_diam/2, "b")
        ax.set_aspect('equal', 'datalim')
        ax.plot(
            [-(self.pickoff_xsize/2 - self.pickoff_rcirc), (self.pickoff_xsize/2 - self.pickoff_rcirc)],
            [self.pickoff_ysize/2, self.pickoff_ysize/2],
            "b"
        )
        ax.plot(
            [-(self.pickoff_xsize/2 - self.pickoff_rcirc), (self.pickoff_xsize/2 - self.pickoff_rcirc)],
            [-self.pickoff_ysize/2, -self.pickoff_ysize/2],
            "b"
        )
        ax.plot(
            [-(self.pickoff_xsize/2), -(self.pickoff_xsize/2)],
            [self.pickoff_ysize/2 - self.pickoff_rcirc, -(self.pickoff_ysize/2 - self.pickoff_rcirc)],
            "b"
        )
        ax.plot(
            [(self.pickoff_xsize/2), (self.pickoff_xsize/2)],
            [self.pickoff_ysize/2 - self.pickoff_rcirc, -(self.pickoff_ysize/2 - self.pickoff_rcirc)],
            "b"
        )
        ax.plot(
            np.cos(circ[0:90]) * self.pickoff_rcirc + self.pickoff_xsize/2 - self.pickoff_rcirc,
            np.sin(circ[0:90]) * self.pickoff_rcirc + self.pickoff_ysize/2 - self.pickoff_rcirc,
            "b"
        )
        ax.plot(
            np.cos(circ[90:180]) * self.pickoff_rcirc - self.pickoff_xsize/2 + self.pickoff_rcirc,
            np.sin(circ[90:180]) * self.pickoff_rcirc + self.pickoff_ysize/2 - self.pickoff_rcirc,
            "b"
        )
        ax.plot(
            np.cos(circ[180:270]) * self.pickoff_rcirc - self.pickoff_xsize/2 + self.pickoff_rcirc,
            np.sin(circ[180:270]) * self.pickoff_rcirc - self.pickoff_ysize/2 + self.pickoff_rcirc,
            "b"
        )
        ax.plot(
            np.cos(circ[270:360]) * self.pickoff_rcirc + self.pickoff_xsize/2 - self.pickoff_rcirc,
            np.sin(circ[270:360]) * self.pickoff_rcirc - self.pickoff_ysize/2 + self.pickoff_rcirc,
            "b"
        )
        ax.plot([0, 0], [self.pickoff_ysize/2, self.pickoff_diam/2], "b")
        ax.plot([0, 0], [-self.pickoff_ysize/2, -self.pickoff_diam/2], "b")

    def plotgrid(self, x0, y0, ax, npts=15):
        """
        Plot a grid of points representing Shack-Hartmann apertures corresponding to wavefront sensor positioned at
        a focal plane position of x0, y0 mm. This position is written in the FITS header keywords GUIDERX and GUIDERY.
        """
        ngood = 0
        for x in np.arange(-1, 1, 2.0 / npts):
            for y in np.arange(-1, 1, 2.0 / npts):
                if (np.hypot(x, y) < 1 and np.hypot(x, y) >= self.telescope.obscuration):  # Only plot points w/in the pupil
                    xm, ym = self.mirrorpoint(x0, y0, x, y)  # Get intersection with pickoff
                    if self.onmirror(xm, ym, x0/abs(x0)):  # Find out if point is on the mirror surface
                        ax.scatter(xm, ym, 1, "g")
                        ngood += 1
                    else:
                        ax.scatter(xm, ym, 1, "r")
        return ngood

    def plotgrid_hdr(self, hdr, ax, npts=15):
        """
        Wrap self.plotgrid() and get x0, y0 values from hdr.
        """
        if 'GUIDERX' not in hdr or 'GUIDERY' not in hdr:
            msg = "No MMIRS WFS position available in header."
            raise WFSCommandException(value=msg)
        x0 = hdr['GUIDERX']
        y0 = hdr['GUIDERY']
        ngood = self.plotgrid(x0, y0, ax=ax, npts=npts)
        return ngood

    def pupil_mask(self, hdr, npts=15):
        """
        Use MMIRS pickoff mirror geometry to calculate the pupil mask
        """
        if 'GUIDERX' not in hdr or 'GUIDERY' not in hdr:
            msg = "No MMIRS WFS position available in header."
            raise WFSCommandException(value=msg)
        if 'CA' not in hdr:
            msg = "No camera rotation angle available in header."
            raise WFSCommandException(value=msg)
        cam_rot = hdr['CA']
        x0 = hdr['GUIDERX']
        y0 = hdr['GUIDERY']

        good = []
        center = self.pup_size / 2.
        obsc = self.telescope.obscuration.value
        spacing = 2.0 / npts
        for x in np.arange(-1, 1, spacing):
            for y in np.arange(-1, 1, spacing):
                r = np.hypot(x, y)
                if (r < 1 and np.hypot(x, y) >= obsc):
                    xm, ym = self.mirrorpoint(x0, y0, x, y)
                    if self.onmirror(xm, ym, x0/abs(x0)):
                        x_impos = center * (x + 1.)
                        y_impos = center * (y + 1.)
                        amp = 1.
                        # this is kind of a hacky way to dim spots near the edge, but easier than doing full calc
                        # of the aperture intersection with pupil. it also doesn't need to be that accurate for the
                        # purposes of the cross-correlation used to register the pupil.
                        if r > 1. - spacing:
                            amp = 1. - (r - (1. - spacing)) / spacing
                        if r - obsc < spacing:
                            amp = (r - obsc) / spacing
                        good.append((amp, x_impos, y_impos))

        yi, xi = np.mgrid[0:self.pup_size, 0:self.pup_size]
        im = np.zeros((self.pup_size, self.pup_size))
        sigma = 3.
        for g in good:
            im += Gaussian2D(g[0], g[1], g[2], sigma, sigma)(xi, yi)

        # camera 2's lenslet array is rotated -1.12 deg w.r.t. the camera.
        if hdr['CAMERA'] == 1:
            cam_rot -= 1.12

        im_rot = rotate(im, cam_rot, reshape=False)
        im_rot[im_rot < 1e-2] = 0.0

        return im_rot

    def get_mode(self, hdr):
        """
        For MMIRS we figure out the mode from which camera the image is taken with.
        """
        cam = hdr['CAMERA']
        mode = f"mmirs{cam}"
        return mode

    def trim_overscan(self, data, hdr=None):
        """
        MMIRS leaves the overscan in, but doesn't give any header information. So gotta trim by hand...
        """
        return data[5:, 12:]

    def process_image(self, fitsfile):
        """
        Process the image to make it suitable for accurate wavefront analysis.  Steps include nuking cosmic rays,
        subtracting background, handling overscan regions, etc.
        """
        rawdata, hdr = check_wfsdata(fitsfile, header=True)

        trimdata = self.trim_overscan(rawdata, hdr=hdr)

        # MMIRS gets a lot of hot pixels/CRs so make a quick pass to nuke them
        cr_mask, data = detect_cosmics(trimdata, sigclip=5., niter=5, cleantype='medmask', psffwhm=5.)

        # calculate the background and subtract it
        bkg_estimator = photutils.background.ModeEstimatorBackground()
        mask = make_spot_mask(data, nsigma=2, npixels=5, dilate_size=11)
        bkg = photutils.background.Background2D(data, (20, 20), filter_size=(7, 7), bkg_estimator=bkg_estimator, mask=mask)
        data -= bkg.background

        return data, hdr

    def focal_plane_position(self, hdr):
        """
        Transform from the MMIRS guider coordinate system to MMTO focal plane coordinates.
        """
        for k in ['ROT', 'GUIDERX', 'GUIDERY']:
            if k not in hdr:
                msg = f"Missing value, {k}, that is required to transform MMIRS guider coordinates."
                raise WFSConfigException(value=msg)

        guide_x = hdr['GUIDERX']
        guide_y = hdr['GUIDERY']
        rot = hdr['ROT']

        guide_r = np.sqrt(guide_x**2 + guide_y**2)
        rot = u.Quantity(rot, u.deg)  # make sure rotation is cast to degrees

        # the MMTO focal plane coordinate convention has phi=0 aligned with +Y instead of +X
        if guide_y != 0.0:
            guide_phi = np.arctan2(guide_x, guide_y) * u.rad
        else:
            guide_phi = 90. * u.deg

        # transform radius in guider coords to degrees in focal plane
        focal_r = (0.0016922 * guide_r - 4.60789e-9 * guide_r**3 - 8.111307e-14 * guide_r**5) * u.deg
        focal_phi = guide_phi + rot + self.rotation

        return focal_r, focal_phi


class FLWO12(WFS):
    """
    Defines configuration and methods for the WFS on the FLWO 1.2-meter
    """
    def trim_overscan(self, data, hdr=None):
        # remove last column that is always set to 0
        return data[:, :510]


class FLWO15(FLWO12):
    """
    Defines configuration and methods for the WFS on the FLWO 1.5-meter
    """
    pass
