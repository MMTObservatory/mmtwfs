# Licensed under GPL3
# coding=utf-8

"""
Classes and utilities for operating the wavefront sensors of the MMTO and analyzing the data they produce
"""

import warnings
import socket

import logging
import logging.handlers
log = logging.getLogger("WFS")
log.setLevel(logging.INFO)

import numpy as np
import photutils

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import feature
from skimage.morphology import reconstruction
from skimage.transform import rotate as imrotate
from scipy import stats, ndimage, optimize

import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
from astropy import stats, visualization
from astropy.modeling.models import Gaussian2D, Polynomial2D
from astropy.modeling.fitting import LevMarLSQFitter

from astroscrappy import detect_cosmics

from ccdproc.utils.slices import slice_from_string

from .utils import srvlookup
from .config import recursive_subclasses, merge_config, mmt_config
from .telescope import MMT
from .zernike import zernike_influence_matrix, ZernikeVector, cart2pol, pol2cart
from .custom_exceptions import WFSConfigException, WFSAnalysisFailed


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

    Arguments
    ---------
    data: FITS filename or 2D ndarray
        WFS image

    Returns
    -------
    data: 2D np.ndarray
        Validated 2D WFS image
    """
    hdr = None
    if isinstance(data, str):
        # we're a fits file (hopefully)
        try:
            with fits.open(data) as h:
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


def wfsfind(data, fwhm=7.0, threshold=5.0, plot=True, ap_radius=5.0, std=None):
    """
    Use photutils.DAOStarFinder() to find and centroid spots in a Shack-Hartmann WFS image.

    Arguments
    ---------
    data: FITS filename or 2D ndarray
        WFS image
    fwhm: float (default: 5.)
        FWHM in pixels of DAOfind convolution kernel
    threshold: float
        DAOfind threshold in units of the standard deviation of the image
    plot: bool
        Toggle plotting of the reference image and overlayed apertures
    ap_radius: float
        Radius of plotted apertures
    """
    # data should be background subtracted first...
    data = check_wfsdata(data)
    if std is None:
        mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, iters=5)
    daofind = photutils.DAOStarFinder(fwhm=fwhm, threshold=threshold*std, sharphi=0.9)
    sources = daofind(data)

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
        positions = (sources['xcentroid'], sources['ycentroid'])
        apertures = photutils.CircularAperture(positions, r=ap_radius)
        norm = wfs_norm(data)
        ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
        apertures.plot(color='red', lw=1.5, alpha=0.5, ax=ax)
    return sources, fig


def mk_reference(data, xoffset=0, yoffset=0, pup_inner=45., pup_outer=175., fwhm=4.0, threshold=30.0, init_scale=1.0, plot=True):
    """
    Read WFS reference image and generate reference magnifications (i.e. grid spacing) and
    aperture positions.

    Arguments
    ---------
    data: FITS filename or 2D ndarray
        WFS reference image
    xoffset, yoffset: float
        Offsets in units of aperture spacing between the center of the reference aperture grid
        and the center of the pupil projected onto the grid.
    pup_inner: float
        Reference radius in pixels of the central hole of the pupil
    pup_outer: float
        Reference radius in pixels of the outer extent of the pupil
    fwhm: float
        FWHM in pixels of DAOfind convolution kernel
    threshold: float
        DAOfind threshold in units of the standard deviation of the image
    init_scale: float
        Initial guess for scale factor between reference aperture grid and an observed one
    plot: bool
        Toggle plotting of the reference image and overlayed apertures

    Returns
    -------
    ref: dict
        Keys -
            xspacing: float
                Mean grid spacing along X axis (pixels)
            yspacing: float
                Mean grid spacing along Y axis (pixels)
            apertures: astropy.Table
                Reference apertures within pup_inner and pup_outer
            pup_coords: tuple (1D ndarray, 1D ndarray)
                X and Y positions of apertures in pupil coordinates
            init_scale: float
                Initial guess for scale factor between reference aperture grid and an observed one
    """
    data = check_wfsdata(data)
    spots, wfsfind_fig = wfsfind(data, fwhm=fwhm, threshold=threshold, plot=plot)
    wfsfind_fig.set_label("Reference Image")
    xcen = spots['xcentroid'].mean()
    ycen = spots['ycentroid'].mean()
    spacing = grid_spacing(data, spots)

    # make masks for each reference spot and fit a 2D gaussian to get its FWHM. the reference FWHM is subtracted in
    # quadrature from the observed FWHM when calculating the seeing.
    apsize = np.mean(spacing)
    apers = photutils.CircularAperture(
        (spots['xcentroid'], spots['ycentroid']),
        r=apsize/2.
    )
    masks = apers.to_mask(method='subpixel')
    sigmas = []
    spot = np.zeros(masks[0].shape)
    for m in masks:
        subim = m.cutout(data)
        # make co-added spot image for use in calculating the seeing
        if subim.shape == spot.shape:
            spot += subim

    # just using the mean will be offset from the true center due to missing spots at edges.
    # find the spot closest to the mean and make it the center position of the pattern.
    dist = ((spots['xcentroid'] - xcen)**2 + (spots['ycentroid'] - ycen)**2)
    closest = np.argmin(dist)
    xoff = spots['xcentroid'][closest] - xcen
    yoff = spots['ycentroid'][closest] - ycen
    xcen += xoff
    ycen += yoff
    xcen += xoffset*spacing[0]
    ycen += yoffset*spacing[1]
    spots['xcentroid'] -= xcen
    spots['ycentroid'] -= ycen
    spots['dist'] = np.sqrt(spots['xcentroid']**2 + spots['ycentroid']**2)
    ref = {}
    ref['xspacing'] = spacing[0]
    ref['yspacing'] = spacing[1]
    spacing = max(spacing[0], spacing[1])
    # we set the limit to half an aperture spacing in from the outer edge to make sure all of the points
    # lie within the zernike radius
    ref['apertures'] = spots[(spots['dist'] > pup_inner) & (spots['dist'] < pup_outer)]
    ref['pup_coords'] = (ref['apertures']['xcentroid']/pup_outer, ref['apertures']['ycentroid']/pup_outer)
    ref['pup_inner'] = pup_inner
    ref['pup_outer'] = pup_outer
    ref['xcen'] = xcen
    ref['ycen'] = ycen
    ref['figure'] = wfsfind_fig
    ref['init_scale'] = init_scale

    # set up 2D gaussian model plus constant background to fit to the coadded spot.  tested this compared to fitting each
    # spot individually and they give the same result with this method being faster.
    with warnings.catch_warnings():
        # ignore astropy warnings about issues with the fit...
        warnings.simplefilter("ignore")
        model = Gaussian2D(amplitude=spot.max(), x_mean=spot.shape[1]/2, y_mean=spot.shape[0]/2) + Polynomial2D(degree=0)
        fitter = LevMarLSQFitter()
        y, x = np.mgrid[:spot.shape[0], :spot.shape[1]]
        fit = fitter(model, x, y, spot)

    sigma = 0.5 * (fit.x_stddev_0.value + fit.y_stddev_0.value)
    fwhm = stats.funcs.gaussian_sigma_to_fwhm * sigma
    ref['fwhm'] = fwhm
    ref['sigma'] = sigma
    ref['spot'] = spot

    return ref


def grid_spacing(data, apertures):
    """
    Measure the WFS grid spacing which changes with telescope focus.

    Arguments
    ---------
    data: WFS image (FITS or np.ndarray)
    apertures: astropy.table.Table
        WFS aperture data to analyze

    Returns
    -------
    xspacing, yspacing: float, float
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

    k = np.linspace(5.0, 50., 1000)  # look for spacings from 5 to 50 pixels (plenty of range)
    f = 1.0 / k  # convert spacing to frequency
    xp = stats.LombScargle(x, xsum[0]).power(f)
    yp = stats.LombScargle(y, ysum[0]).power(f)

    # the peak of the power spectrum will coincide with the average spacing
    xspacing = k[xp.argmax()]
    yspacing = k[yp.argmax()]

    return xspacing, yspacing


def center_pupil(data, pup_mask, threshold=0.5, sigma=20., plot=True):
    """
    Find the center of the pupil in a WFS image using skimage.feature.match_template(). This generates
    a correlation image and we centroid the peak of the correlation to determine the center.

    Arguments
    ---------
    data: str or 2D ndarray
        WFS image to analyze, either FITS file or ndarray image data
    pup_mask: str or 2D ndarray
        Pupil model to use in the template matching
    threshold: float (default: 0.0)
        Sets image to 0 where it's below threshold * image.max()
    sigma: float (default: 20.)
        Sigma of gaussian smoothing kernel
    plot: bool
        Toggle plotting of the correlation image

    Returns
    -------
    cen: tuple (float, float)
        X and Y pixel coordinates of the pupil center
    """
    data = check_wfsdata(data)
    pup_mask = check_wfsdata(pup_mask)

    # we smooth the image heavily to reduce the aliasing from the SH spots.
    smo = ndimage.gaussian_filter(data, sigma)

    # use skimage.feature.match_template() to do a fast cross-correlation between the WFS image and the pupil model.
    # the location of the peak of the correlation will be the center of the WFS pattern.
    match = feature.match_template(smo, pup_mask, pad_input=True)
    match[match < threshold * match.max()] = 0
    cen = photutils.centroids.centroid_com(match)
    fig = None
    if plot:
        fig, ax = plt.subplots()
        fig.set_label("Pupil Correlation Image (masked)")
        ax.imshow(match, interpolation=None, cmap=cm.magma, origin='lower')
    return cen[0], cen[1], fig


def get_apertures(data, apsize, fwhm=5.0, thresh=7.0, plot=True):
    """
    Use wfsfind to locate and centroid spots.  Measure their S/N ratios and the sigma of a 2D gaussian fit to
    the co-added spot.

    Arguments
    ---------
    data: str or 2D ndarray
        WFS image to analyze, either FITS file or ndarray image data
    apsize: float
        Diameter/width of the SH apertures

    Returns
    -------
    srcs: astropy.table.Table
        Detected WFS spot positions and properties
    masks: list of photutils.ApertureMask objects
        Masks used for aperture centroiding
    snrs: 1D np.ndarray
        S/N for each located spot
    sigma: float
    """
    data = check_wfsdata(data)

    # set iters to None to let this clip all the way to convergence
    mean, median, stddev = stats.sigma_clipped_stats(data, sigma=3.0, iters=None)

    # use wfsfind() and pass it the clipped stddev from here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        srcs, wfsfind_fig = wfsfind(data, fwhm=fwhm, threshold=thresh, std=stddev, plot=plot)

    # we use circular apertures here because they generate square masks of the appropriate size.
    # rectangular apertures produced masks that were sqrt(2) too large.
    # see https://github.com/astropy/photutils/issues/499 for details.
    apers = photutils.CircularAperture(
        (srcs['xcentroid'], srcs['ycentroid']),
        r=apsize/2.
    )
    masks = apers.to_mask(method='subpixel')
    snrs = []
    spot = np.zeros(masks[0].shape)
    for m in masks:
        subim = m.cutout(data)

        # make co-added spot image for use in calculating the seeing
        if subim.shape == spot.shape:
            spot += subim

        err = stddev * np.ones_like(subim)
        signal = subim.sum()
        noise = np.sqrt(stddev**2 / (subim.shape[0] * subim.shape[1]))
        snr = signal / noise
        snrs.append(snr)

    snrs = np.array(snrs)

    # set up 2D gaussian model plus constant background to fit to the coadded spot
    with warnings.catch_warnings():
        # ignore astropy warnings about issues with the fit...
        warnings.simplefilter("ignore")
        model = Gaussian2D(amplitude=spot.max(), x_mean=spot.shape[1]/2, y_mean=spot.shape[0]/2) + Polynomial2D(degree=0)
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
        dists = np.sqrt( (spots[0]-refs[0][i])**2 + (spots[1]-refs[1][i])**2 )
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
    tot_dist = 0.0
    refs = np.array([refx, refy])
    spots = np.array([spotx, spoty])
    for i in np.arange(len(refx)):
        dists = np.sqrt( (spots[0]-refs[0][i])**2 + (spots[1]-refs[1][i])**2 )
        tot_dist += np.min(dists)
    return np.log(tot_dist)


def fit_apertures(pars, ref, spots):
    """
    Scale the reference positions by the fit parameters and calculate the total distance between the matches.
    The parameters are:
        xc, yc = center positions
        scale = magnification of the grid (focus)
        xcoma, ycoma = linear change in magnification as a function of x/y (coma)

    'ref' and 'spots' are assumed to be dict-like and must have the keys 'xcentroid' and 'ycentroid'.
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


def get_slopes(data, ref, pup_mask, fwhm=7.0, thresh=5.0, plot=True):
    """
    Analyze a WFS image and produce pixel offsets between reference and observed spot positions.

    Arguments
    ---------
    data: str or 2D np.ndarray
        FITS file or np.ndarray containing WFS observation
    ref: astropy.Table
        Table of reference apertures
    plot: bool
        Toggle plotting of image with aperture overlays

    Returns
    -------
    slopes: list of tuples
        X/Y pixel offsets between measured and reference aperture positions.
    final_aps: astropy.Table
        Centroided observed apertures
    xspacing, yspacing: float, float
        Observed X and Y grid spacing
    xcen, ycen: float, float
        Center of pupil image
    idx: list
        Index of reference apertures that have detected spots
    sigma: float
        Sigma of gaussian fit to co-added WFS spot
    """
    data = check_wfsdata(data)
    pup_mask = check_wfsdata(pup_mask)

    # input data should be background subtracted for best results. this initial guess of the center positions
    # will be good enough to get the central obscuration, but will need to be fine-tuned for aperture association.
    xcen, ycen, pupcen_fig = center_pupil(data, pup_mask, plot=plot)
    pup_center = [xcen, ycen]

    # using the mean spacing is straightforward for square apertures and a reasonable underestimate for hexagonal ones (e.g. f/9)
    ref_spacing = np.mean([ref['xspacing'], ref['yspacing']])
    apsize = ref_spacing

    srcs, masks, snrs, sigma, wfsfind_fig = get_apertures(data, apsize, fwhm=fwhm, thresh=thresh)

    # if we don't detect spots in at least half of the reference apertures, we can't usually get a good wavefront measurement
    if len(srcs) < 0.5 * len(ref['apertures']['xcentroid']):
        msg = "Only %d spots detected out of %d apertures." % (len(srcs), len(ref['apertures']['xcentroid']))
        raise WFSAnalysisFailed(value=msg)

    src_aps = photutils.CircularAperture(
        (srcs['xcentroid'], srcs['ycentroid']),
        r=apsize/2.
    )

    # get grid spacing of the data
    xspacing, yspacing = grid_spacing(data, srcs)

    # find the scale difference between data and ref and use as init
    init_scale = (xspacing/ref['xspacing'] + yspacing/ref['yspacing']) / 2.

    # set up to do a fit of the reference apertures to the spot positions with the center, scaling, and position-dependent
    # scaling (coma) as free parameters
    args = (ref['apertures'], srcs)
    par_keys = ('xcen', 'ycen', 'scale', 'xcoma', 'ycoma')
    pars = (xcen, ycen, init_scale, 0.0, 0.0)
    coma_bound = 1e-7
    # scipy.optimize.minimize can do bounded minimization so leverage that to keep the solution within a reasonable range.
    bounds = (
        (xcen-20, xcen+20),  # hopefully we're not too far off from true center...
        (ycen-20, ycen+20),
        (init_scale-0.05, init_scale+0.05),  # reasonable range of expected focus difference...
        (-coma_bound, coma_bound),  # this should be way more than enough to account for any reasonable amount of coma we'll encounter.
        (-coma_bound, coma_bound)   # however, the larger range range appears to help minimize avoid local minima
    )
    min_results = optimize.minimize(fit_apertures, pars, args=args, bounds=bounds, options={'ftol': 1e-13, 'gtol': 1e-7})

    fit_results = {}
    for i, k in enumerate(par_keys):
        fit_results[k] = min_results['x'][i]

    xcen, ycen = fit_results['xcen'], fit_results['ycen']
    scale = fit_results['scale']
    xcoma, ycoma = fit_results['xcoma'], fit_results['ycoma']

    refx = ref['apertures']['xcentroid'] * (scale + ref['apertures']['xcentroid'] * xcoma) + fit_results['xcen']
    refy = ref['apertures']['ycentroid'] * (scale + ref['apertures']['ycentroid'] * ycoma) + fit_results['ycen']
    xspacing = scale * ref['xspacing']
    yspacing = scale * ref['yspacing']

    # match reference apertures to spots
    spacing = np.max([xspacing, yspacing])
    ref_mask, src_mask = match_apertures(refx, refy, srcs['xcentroid'], srcs['ycentroid'], max_dist=spacing/2.)

    # these are unscaled so that the slope includes defocus
    trim_refx = ref['apertures']['xcentroid'][ref_mask] + xcen
    trim_refy = ref['apertures']['ycentroid'][ref_mask] + ycen
    ref_aps = photutils.CircularAperture(
        (trim_refx, trim_refy),
        r=ref_spacing/2.
    )

    slope_x = srcs['xcentroid'][src_mask] - trim_refx
    slope_y = srcs['ycentroid'][src_mask] - trim_refy

    # use the pupil center found by correlation because the pupil can shift around on the lenslet array so
    # using that center can affect determination of pupil coordinates
    pup_size = ref['pup_outer']
    pup_coords = (ref_aps.positions - pup_center) / [pup_size, pup_size]

    aps_fig = None
    if plot:
        norm = wfs_norm(data)
        aps_fig, ax = plt.subplots()
        aps_fig.set_label("Aperture Positions")
        ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
        #apers.plot(color='red')
        ax.scatter(pup_center[0], pup_center[1])
        src_aps.plot(color='blue', ax=ax)

    # need full slopes array the size of the complete set of reference apertures and pre-filled with np.nan for masking
    slopes = np.nan * np.ones((2, len(ref['apertures']['xcentroid'])))

    slopes[0][ref_mask] = slope_x
    slopes[1][ref_mask] = slope_y

    figures = {}
    figures['pupil_center'] = pupcen_fig
    figures['slopes'] = aps_fig
    results = {
        "slopes": np.ma.masked_invalid(slopes),
        "pup_coords": pup_coords.transpose(),
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

    wfs_cls = wfs_map[wfs](config=config)
    return wfs_cls


class WFS(object):
    """
    Defines configuration pattern and methods common to all WFS systems
    """
    def __init__(self, config={}):
        key = self.__class__.__name__.lower()
        self.__dict__.update(merge_config(mmt_config['wfs'][key], config))
        self.telescope = MMT(secondary=self.secondary)
        self.secondary = self.telescope.secondary

        self.connected = False

        # this factor calibrates spot motion in pixels to nm of wavefront error
        self.tiltfactor = self.telescope.nmperasec * (self.pix_size.to(u.arcsec).value)

        # if this is the same for all modes, load it once here
        if hasattr(self, "reference_file"):
            refdata, hdr = check_wfsdata(self.reference_file, header=True)
            refdata = self.trim_overscan(refdata, hdr)
            reference = mk_reference(
                refdata,
                xoffset=self.pup_offset[0],
                yoffset=self.pup_offset[1],
                pup_inner=self.pup_inner,
                pup_outer=self.pup_size / 2.,
                init_scale=self.init_scale,
                plot=True
            )

        # now assign 'reference' for each mode so that it can be accessed consistently in all cases
        for mode in self.modes:
            if 'pup_offset' in self.modes[mode]:
                pup_off = self.modes[mode]['pup_offset']
            else:
                pup_off = self.pup_offset

            if 'reference_file' in self.modes[mode]:
                refdata, hdr = check_wfsdata(self.modes[mode]['reference_file'], header=True)
                refdata = self.trim_overscan(refdata, hdr)
                self.modes[mode]['reference'] = mk_reference(
                    refdata,
                    xoffset=pup_off[0],
                    yoffset=pup_off[1],
                    pup_inner=self.pup_inner,
                    pup_outer=self.pup_size / 2.,
                    init_scale=self.init_scale,
                    plot=True
                )
            else:
                self.modes[mode]['reference'] = reference

            self.modes[mode]['zernike_matrix'] = zernike_influence_matrix(
                pup_coords=self.modes[mode]['reference']['pup_coords'],
                nmodes=self.nzern,
                modestart=2  # ignore the piston term
            )

    def connect(self):
        """
        Set state to connected so that calculated corrections get passed through to appropriate systems.
        """
        self.connected = True
        self.telescope.connect()
        self.secondary.connect()

    def disconnect(self):
        """
        Set state to disconnected for testing/development
        """
        self.connected = False
        self.telescope.disconnect()
        self.secondary.disconnect()

    def get_flipud(self, mode=None):
        """
        Method to determine if the WFS image needs to be flipped up/down
        """
        return False

    def get_fliplr(self, mode=None):
        """
        Method to determine if the WFS image needs to be flipped left/right
        """
        return False

    def seeing(self, mode, sigma, airmass=None):
        """
        Given a sigma derived from a gaussian fit to a WFS spot, deconvolve the systematic width from the reference image
        and relate the remainder to r_0 and thus a seeing FWHM.
        """
        # the effective wavelength of the WFS imagers is about 600-650 nm. we use 650 nm to maintain consistency
        # with the value used by the old SHWFS system.
        wave = 650 * u.nm
        wave = wave.to(u.m).value  # r_0 equation expects meters so convert

        # calculate the physical size of each aperture.
        ref = self.modes[mode]['reference']
        apsize_pix = np.max((ref['xspacing'], ref['yspacing']))
        d = self.telescope.diameter * apsize_pix / self.pup_size
        d = d.to(u.m).value  # r_0 equation expects meters so convert

        # we need to deconvolve the instrumental spot width from the measured one to get the portion of the width that
        # is due to spot motion
        ref_sigma = ref['sigma']
        if sigma > ref_sigma:
            corr_sigma = np.sqrt(sigma**2 - ref_sigma**2)
        else:
            return 0.0 * u.arcsec, 0.0 * u.arcsec

        corr_sigma *= self.pix_size.to(u.rad).value  # r_0 equation expects radians so convert

        # this equation relates the motion within a single aperture to the characteristic scale size of the
        # turbulence, r_0.
        r_0 = ( 0.179 * (wave**2) * (d**(-1/3))/corr_sigma**2 )**0.6

        # this equation relates the turbulence scale size to an expected image FWHM at the given wavelength.
        raw_seeing = u.Quantity(u.rad * 0.98 * wave / r_0, u.arcsec)

        # correct seeing to zenith
        if airmass is not None:
            seeing = raw_seeing / airmass**0.6
        else:
            seeing = raw_seeing

        return seeing, raw_seeing

    def pupil_mask(self, rotator=0.0):
        """
        Wrap the Telescope.pupil_mask() method to include both WFS and instrument rotator rotation angles
        """
        rotator = u.Quantity(rotator, u.deg)
        pup = self.telescope.pupil_mask(rotation=self.rotation+rotator, size=self.pup_size)
        return pup

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
        bkg_estimator = photutils.MedianBackground()
        bkg = photutils.Background2D(data, (10, 10), filter_size=(5, 5), bkg_estimator=bkg_estimator)
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

        # make rotated pupil mask
        pup_mask = self.pupil_mask(rotator=rotator)

        try:
            slope_results = get_slopes(
                data,
                self.modes[mode]['reference'],
                pup_mask,
                fwhm=self.find_fwhm,
                thresh=self.find_thresh,
                plot=plot
            )
            slopes = slope_results['slopes']
            coords = slope_results['pup_coords']
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
            x = aps.positions.transpose()[0][src_mask]
            y = aps.positions.transpose()[1][src_mask]
            uu = slopes[0][ref_mask]
            vv = slopes[1][ref_mask]
            norm = wfs_norm(data)
            figures['slopes'].set_label("Aperture Positions and Spot Movement")
            ax = figures['slopes'].axes[0]
            ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
            aps.plot(color='blue', ax=ax)
            ax.quiver(x, y, uu, vv, scale_units='xy', scale=0.2, pivot='tip', color='red')
            xl = [60.0]
            yl = [data.shape[0]-30]
            ul = [1.0/self.pix_size.value]
            vl = [0.0]
            ax.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.2, pivot='tip', color='red')
            ax.scatter([slope_results['center'][0]], [slope_results['center'][1]])
            ax.text(60, data.shape[0]-30, "1\"", verticalalignment='center')
            ax.set_title("Seeing: %.2f\" (%.2f\" @ zenith)" % (raw_seeing.value, seeing.value))

        results = {}
        results['seeing'] = seeing
        results['raw_seeing'] = raw_seeing
        results['slopes'] = slopes
        results['pup_coords'] = coords
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
        if slope_results['slopes'] is not None:
            results = {}
            mode = slope_results['mode']
            infmat = self.modes[mode]['zernike_matrix'][0]
            inverse_infmat = self.modes[mode]['zernike_matrix'][1]
            slopes = slope_results['slopes']
            slope_vec = -self.tiltfactor * slopes.ravel()  # convert arcsec to radians
            zfit = np.dot(slope_vec, infmat)

            results['raw_zernike'] = ZernikeVector(coeffs=zfit)

            # derotate the zernike solution to match the primary mirror coordinate system
            total_rotation = self.rotation - slope_results['rotator']
            zv_rot = ZernikeVector(coeffs=zfit)
            zv_rot.rotate(angle=-total_rotation)
            results['rot_zernike'] = zv_rot

            # subtract the reference aberrations
            zref = self.reference_aberrations(mode, hdr=slope_results['header'])
            zsub = zv_rot - zref
            results['ref_zernike'] = zref
            results['zernike'] = zsub

            pred = np.dot(zfit, inverse_infmat)
            pred_slopes = -(1. / self.tiltfactor) * pred.reshape(2, slopes.shape[1])
            diff = slopes - pred_slopes
            rms = self.pix_size * np.sqrt((diff[0]**2 + diff[1]**2).mean())
            results['residual_rms_asec'] = rms.to(u.arcsec)
            results['residual_rms'] = rms.to(u.arcsec).value * self.tiltfactor * zsub.units
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
                ax.quiver(x, y, diff[0][ref_mask], diff[1][ref_mask], scale_units='xy', scale=0.05, pivot='tip', color='red')
                xl = [50.0]
                yl = [im.shape[0]-30]
                ul = [0.2/self.pix_size.value]
                vl = [0.0]
                ax.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.05, pivot='tip', color='red')
                ax.text(60, im.shape[0]-30, "0.2\"", verticalalignment='center')
                ax.set_title("Residual RMS: {0:0.4g} ({1:0.2f})".format(results['residual_rms'], results['residual_rms_asec']))

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
        zv.normalize()
        zv_masked = ZernikeVector()

        log.info(f"zv {zv} thresh: {threshold} mask {mask}")

        for z in zv:
            if abs(zv[z]) >= threshold:
                zv_masked[z] = zv[z]
                log.info(f"{z}: Good")
            else:
                log.info(f"{z}: Bad")
        zv_masked.denormalize()  # need to assure we're using fringe coeffs
        forces, m1focus, zv_allmasked = self.telescope.calculate_primary_corrections(zv=zv_masked, mask=mask, gain=self.m1_gain)
        log.info(f"All masked: {zv_allmasked}")
        return forces, m1focus, zv_allmasked

    def calculate_focus(self, zv):
        """
        Convert Zernike defocus to um of secondary offset.
        """
        z_denorm = zv.copy()
        z_denorm.denormalize()  # need to assure we're using fringe coeffs
        foc_corr = -self.m2_gain * z_denorm['Z04'] / self.secondary.focus_trans

        return foc_corr.round(2)

    def calculate_cc(self, zv):
        """
        Convert Zernike coma (Z07 and Z08) into arcsec of secondary center-of-curvature tilts.
        """
        z_denorm = zv.copy()
        z_denorm.denormalize()  # need to assure we're using fringe coeffs

        # fix coma using tilts around the M2 center of curvature.
        cc_y_corr = -self.m2_gain * z_denorm['Z07'] / self.secondary.theta_cc
        cc_x_corr = -self.m2_gain * z_denorm['Z08'] / self.secondary.theta_cc

        return cc_x_corr.round(3), cc_y_corr.round(3)

    def calculate_recenter(self, fit_results, defoc=1.0):
        """
        Perform zero-coma hexapod tilts to align the pupil center to the center-of-rotation. The location of the CoR is configured
        to be at self.cor_coords
        """
        xc = fit_results['xcen']
        yc = fit_results['ycen']
        xref = self.cor_coords[0]
        yref = self.cor_coords[1]
        dx = -(xc - xref)  # flip the sign here because E to the right in derot pupil so +AZ to the left
        dy = yc - yref

        total_rotation = u.Quantity(fit_results['rotator'] + self.rotation, u.rad).value

        dr, phi = cart2pol([dx, dy])

        derot_phi = phi - total_rotation

        az, el = pol2cart([dr, derot_phi])

        az *= -1 * self.pix_size * defoc  # pix size scales with the pupil size as focus changes.
        el *= -1 * self.pix_size * defoc

        return az.round(3), el.round(3)

    def clear_corrections(self):
        """
        Clear all applied WFS corrections
        """
        log.info("Clearing WFS corrections from primary and secondary...")
        clear_forces, clear_m1focus = self.telescope.clear_forces()
        cmds = self.secondary.clear_wfs()
        return clear_forces, clear_m1focus

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
        if self.sock:
            try:
                self.sock.close()
                self.sock = None
            except Exception as e:
                log.error(f"Error closing connection to topbox server: {e}")


class F9(WFS):
    """
    Defines configuration and methods specific to the F/9 WFS system
    """
    def __init__(self, config={}):
        super(F9, self).__init__(config=config)

        self.connected = False
        self.sock = None

        # get host/port to use for topbox communication
        self.host, self.port = srvlookup(self.lampsrv)


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
        bkg_estimator = photutils.MedianBackground()
        bkg = photutils.Background2D(data, (50, 50), filter_size=(15, 15), bkg_estimator=bkg_estimator)
        data -= bkg.background

        return data, hdr


class F5(WFS):
    """
    Defines configuration and methods specific to the F/5 WFS systems
    """
    def __init__(self, config={}):
        super(F5, self).__init__(config=config)

        self.connected = False
        self.sock = None

        # load lookup table for off-axis aberrations
        self.aberr_table = ascii.read(self.aberr_table_file)

    def focal_plane_position(self, hdr):
        """
        Need to fill this in for the hecto f/5 WFS system. For now will assume it's always on-axis.
        """
        return 0.0 * u.deg, 0.0 * u.deg

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

        # now rotate the off-axis aberrations
        z_offaxis.rotate(angle=field_phi)

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
        if mode == "old_binospec":
            return False
        else:
            return True

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
        focal_phi = guide_phi + rot + 180. * u.deg

        log.info(f"guide_phi: {guide_phi.to(u.rad)} rot: {rot}")

        return focal_r, focal_phi


class MMIRS(F5):
    """
    Defines configuration and methods specific to the MMIRS WFS system
    """
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
        focal_phi = guide_phi + rot

        return focal_r, focal_phi
