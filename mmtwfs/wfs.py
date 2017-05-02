# Licensed under GPL3
# coding=utf-8

"""
Classes and utilities for operating the wavefront sensors of the MMTO and analyzing the data they produce
"""

import warnings

import numpy as np
import photutils

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import feature
from skimage.morphology import reconstruction
from scipy import stats, ndimage
from scipy.misc import imrotate

import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
from astropy import stats, visualization
from astropy.modeling.models import Gaussian2D, Polynomial2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.coordinates import SkyCoord, match_coordinates_3d

from astroscrappy import detect_cosmics

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


def check_wfsdata(data):
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
    if isinstance(data, str):
        # we're a fits file (hopefully)
        try:
            data = fits.open(data)[0].data
        except Exception as e:
            msg = "Error reading FITS file, %s (%s)" % (data, repr(e))
            raise WFSConfigException(value=msg)
    if not isinstance(data, np.ndarray):
        msg = "WFS image data in improper format, %s" % type(data)
        raise WFSConfigException(value=msg)
    if len(data.shape) != 2:
        msg = "WFS image data has improper shape, %s. Must be 2D image." % data.shape
        raise WFSConfigException(value=msg)
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
        positions = (sources['xcentroid'], sources['ycentroid'])
        apertures = photutils.CircularAperture(positions, r=ap_radius)
        norm = wfs_norm(data)
        ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
        apertures.plot(color='red', lw=1.5, alpha=0.5, ax=ax)
    return sources, fig


def mk_reference(data, xoffset=0, yoffset=0, pup_inner=45., pup_outer=175., fwhm=4.0, threshold=30.0, plot=True):
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
    """
    data = check_wfsdata(data)
    spots, wfsfind_fig = wfsfind(data, fwhm=fwhm, threshold=threshold, plot=plot)
    xcen = spots['xcentroid'].mean()
    ycen = spots['ycentroid'].mean()
    spacing = grid_spacing(data)

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


def grid_spacing(data):
    """
    Measure the WFS grid spacing which changes with telescope focus.

    Arguments
    ---------
    data: str or 2D ndarray
        WFS data to analyze

    Returns
    -------
    xspacing, yspacing: float, float
        Average grid spacing in X and Y axes
    """
    data = check_wfsdata(data)
    # sum along the axes and use Lomb-Scargle to measure the grid spacing in each direction
    xsum = np.sum(data, axis=0)
    ysum = np.sum(data, axis=1)
    x = np.arange(len(xsum))
    y = np.arange(len(ysum))
    k = np.linspace(5.0, 50., 1000)  # look for spacings from 5 to 50 pixels (plenty of range)
    f = 1.0 / k  # convert spacing to frequency
    xp = stats.LombScargle(x, xsum).power(f)
    yp = stats.LombScargle(y, ysum).power(f)
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
        ax.imshow(match, interpolation=None, cmap=cm.magma, origin='lower')
    return cen[0], cen[1], fig


def get_apertures(data, apsize, plot=True):
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
        srcs, wfsfind_fig = wfsfind(data, std=stddev, plot=plot)

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


def get_slopes(data, ref, pup_mask, plot=True):
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
    # will be good enough to get the central obscuration, but will need to be fine-tuned.
    xcen, ycen, pupcen_fig = center_pupil(data, pup_mask, plot=plot)
    xspacing, yspacing = grid_spacing(data)

    # using the mean spacing is straightforward for square apertures and a reasonable underestimate for hexagonal ones (e.g. f/9)
    apsize = np.mean([xspacing, yspacing])
    ref_spacing = np.mean([ref['xspacing'], ref['yspacing']])

    srcs, masks, snrs, sigma, wfsfind_fig = get_apertures(data, apsize)

    # if we don't detect spots in at least half of the reference apertures, we can't usually get a good wavefront measurement
    if len(srcs) < 0.5 * len(ref['apertures']['xcentroid']):
        msg = "Only %d spots detected out of %d apertures." % (len(srcs), len(ref['apertures']['xcentroid']))
        raise WFSAnalysisFailed(value=msg)

    src_aps = photutils.CircularAperture(
        (srcs['xcentroid'], srcs['ycentroid']),
        r=apsize/2.
    )

    # should make this a config var, but need to play with it more...
    snr_mask = np.where(snrs < 0.1*snrs.max())

    # the first step to fine-tuning the WFS pattern center is compare the marginal sums for the whole image to the ones
    # for the part centered on the initial guess for the center position.
    xl = int(xcen - 2*xspacing)
    xu = int(xcen + 2*xspacing)
    yl = int(ycen - 2*yspacing)
    yu = int(ycen + 2*yspacing)

    # normalize the sums to their maximums so they can be compared more directly
    xsum = np.sum(data[yl:yu, :], axis=0)
    xsum /= xsum.max()
    xtot = np.sum(data, axis=0)
    xtot /= xtot.max()
    ysum = np.sum(data[:, xl:xu], axis=1)
    ysum /= ysum.max()
    ytot = np.sum(data, axis=1)
    ytot /= ytot.max()
    xdiff = xtot - xsum

    # set high enough to discriminate where the obscuration is, but low enough to get better centroid
    xdiff[xdiff < 0.25] = 0.0
    ydiff = ytot - ysum
    ydiff[ydiff < 0.25] = 0.0

    xcen = ndimage.measurements.center_of_mass(xdiff)[0]
    ycen = ndimage.measurements.center_of_mass(ydiff)[0]

    # use the ratio of spacings to magnify the grid and the updated pupil center guess to offset it
    refx = (xspacing / ref['xspacing']) * ref['apertures']['xcentroid'] + xcen
    refy = (yspacing / ref['yspacing']) * ref['apertures']['ycentroid'] + ycen

    # set up cartesian coordinates for the measured and reference spot positions
    src_coord = SkyCoord(x=srcs['xcentroid'], y=srcs['ycentroid'], z=0.0, representation='cartesian')
    ref_coord = SkyCoord(x=refx, y=refy, z=0.0, representation='cartesian')

    # perform the matching...
    idx, sep, dist = match_coordinates_3d(src_coord, ref_coord)

    # sometimes the initial center will be up to ~half aperture off so some apertures will go one way, while the rest go another.
    # fix that by finding the mean offset from the first match, apply it to the reference coords, and then re-match.
    xoff = np.mean(srcs['xcentroid'] - refx[idx])
    yoff = np.mean(srcs['ycentroid'] - refy[idx])
    refx += xoff
    refy += yoff
    xcen += xoff
    ycen += yoff

    # now do the re-matching with better center position...
    ref_coord = SkyCoord(x=refx, y=refy, z=0.0, representation='cartesian')
    idx, sep, dist = match_coordinates_3d(src_coord, ref_coord)
    xoff = np.mean(srcs['xcentroid'] - refx[idx])
    yoff = np.mean(srcs['ycentroid'] - refy[idx])
    xcen += xoff
    ycen += yoff

    # these are unscaled so that the slope includes defocus
    trim_refx = ref['apertures']['xcentroid'][idx] + xcen
    trim_refy = ref['apertures']['ycentroid'][idx] + ycen
    ref_aps = photutils.CircularAperture(
        (trim_refx, trim_refy),
        r=ref_spacing/2.
    )

    slope_x = srcs['xcentroid'] - trim_refx
    slope_y = srcs['ycentroid'] - trim_refy

    pup_size = ref['pup_outer']
    pup_coords = (ref_aps.positions - [xcen, ycen]) / [pup_size, pup_size]

    aps_fig = None
    if plot:
        norm = wfs_norm(data)
        aps_fig, ax = plt.subplots()
        ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
        #apers.plot(color='red')
        ax.scatter(xcen, ycen)
        src_aps.plot(color='blue', ax=ax)

    # need full slopes array the size of the complete set of reference apertures and pre-filled with np.nan for masking
    slopes = np.nan * np.ones((2, len(ref['apertures']['xcentroid'])))

    # check mis-IDed spots
    spacing = np.max([xspacing, yspacing])
    diffx = srcs['xcentroid'] - refx[idx]
    diffy = srcs['ycentroid'] - refy[idx]
    dist = np.sqrt(diffx**2 + diffy**2)
    slope_x[dist > spacing/2.] = np.nan
    slope_y[dist > spacing/2.] = np.nan

    # apply SNR mask
    slope_x[snr_mask] = np.nan
    slope_y[snr_mask] = np.nan

    slopes[0][idx] = slope_x
    slopes[1][idx] = slope_y
    figures = {}
    figures['pupil_center'] = pupcen_fig
    figures['slopes'] = aps_fig
    return np.ma.masked_invalid(slopes), pup_coords.transpose(), src_aps, (xspacing, yspacing), (xcen, ycen), idx, sigma, figures


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
            reference = mk_reference(
                self.reference_file,
                xoffset=self.pup_offset[0],
                yoffset=self.pup_offset[1],
                pup_inner=self.pup_inner,
                pup_outer=self.pup_size / 2.,
                plot=True
            )

        # now assign 'reference' for each mode so that it can be accessed consistently in all cases
        for mode in self.modes:
            if 'pup_offset' in self.modes[mode]:
                pup_off = self.modes[mode]['pup_offset']
            else:
                pup_off = self.pup_offset

            if 'reference_file' in self.modes[mode]:
                self.modes[mode]['reference'] = mk_reference(
                    self.modes[mode]['reference_file'],
                    xoffset=pup_off[0],
                    yoffset=pup_off[1],
                    pup_inner=self.pup_inner,
                    pup_outer=self.pup_size / 2.,
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
            corr_sigma = 0.0
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
        pup = self.telescope.pupil_mask(rotation=self.rotation+rotator)
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
        # we're a fits file (hopefully)
        try:
            fitsdata = fits.open(fitsfile)[0]
            rawdata = fitsdata.data
            hdr = fitsdata.header
        except Exception as e:
            msg = "Error reading FITS file, %s (%s)" % (fitsfile, repr(e))
            raise WFSConfigException(value=msg)
        rawdata = check_wfsdata(rawdata)

        # MMIRS gets a lot of hot pixels/CRs so make a quick pass to nuke them
        cr_mask, data = detect_cosmics(rawdata, sigclip=4., niter=10, cleantype='medmask', psffwhm=5.)

        # calculate the background and subtract it
        bkg_estimator = photutils.MedianBackground()
        bkg = photutils.Background2D(data, (10, 10), filter_size=(5, 5), bkg_estimator=bkg_estimator)
        data -= bkg.background

        # trim overscan (this is needed for MMIRS, but ok for rest)
        data[:5, :] = 0.0
        data[:, :12] = 0.0

        return data, hdr

    def measure_slopes(self, fitsfile, mode=None, plot=True):
        """
        Take a WFS image in FITS format, perform background subtration, pupil centration, and then use get_slopes()
        to perform the aperture placement and spot centroiding.
        """
        data, hdr = self.process_image(fitsfile)

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

        # make rotated pupil mask
        pup_mask = self.pupil_mask(rotator=rotator)

        try:
            slopes, coords, aps, spacing, cen, mask, sigma, figures = get_slopes(
                data,
                self.modes[mode]['reference'],
                pup_mask,
                plot=plot
            )
        except WFSAnalysisFailed as e:
            print("Wavefront slope measurement failed: %s" % e.args[1])
            slope_fig = None
            if plot:
                slope_fig, ax = plt.subplots()
                norm = wfs_norm(data)
                ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
            results = {}
            results['slopes'] = None
            results['figures'] = {}
            results['figures']['slopes'] = slope_fig
            return results
        except Exception as e:
            raise WFSAnalysisFailed(value=str(e))

        # use the average width of the spots to estimate the seeing and use the airmass to extrapolate to zenith seeing
        if 'AIRMASS' in hdr:
            airmass = hdr['AIRMASS']
        else:
            airmass = None
        seeing, raw_seeing = self.seeing(mode=mode, sigma=sigma, airmass=airmass)

        if plot:
            x = aps.positions.transpose()[0]
            y = aps.positions.transpose()[1]
            uu = slopes[0][mask]
            vv = slopes[1][mask]
            norm = wfs_norm(data)
            ax = figures['slopes'].axes[0]
            ax.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
            aps.plot(color='blue', ax=ax)
            ax.quiver(x, y, uu, vv, scale_units='xy', scale=0.2, pivot='tip', color='red')
            xl = [50.0]
            yl = [data.shape[0]-30]
            ul = [1.0/self.pix_size.value]
            vl = [0.0]
            ax.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.2, pivot='tip', color='red')
            ax.scatter([cen[0]], [cen[1]])
            ax.text(60, 480, "1\"", verticalalignment='center')

        results = {}
        results['seeing'] = seeing
        results['raw_seeing'] = raw_seeing
        results['slopes'] = slopes
        results['pup_coords'] = coords
        results['apertures'] = aps
        results['xspacing'] = spacing[0]
        results['yspacing'] = spacing[1]
        results['xcen'] = cen[0]
        results['ycen'] = cen[1]
        results['pup_mask'] = pup_mask
        results['data'] = data
        results['header'] = hdr
        results['rotator'] = rotator
        results['mode'] = mode
        results['ref_mask'] = mask
        results['fwhm'] = stats.funcs.gaussian_sigma_to_fwhm * sigma
        results['figures'] = figures
        return results

    def fit_wavefront(self, slope_results, plot=True):
        """
        Use results from self.measure_slopes() to fit a set of zernike polynomials to the wavefront shape.
        """
        results = {}
        mode = slope_results['mode']
        infmat = self.modes[mode]['zernike_matrix'][0]
        inverse_infmat = self.modes[mode]['zernike_matrix'][1]
        slopes = slope_results['slopes']
        slope_vec = -self.tiltfactor * slopes.ravel()  # convert arcsec to radians
        zfit = np.dot(slope_vec, infmat)

        results['raw_zernike'] = ZernikeVector(coeffs=zfit)

        # derotate the zernike solution to match the primary mirror coordinate system
        total_rotation = slope_results['rotator'] + self.rotation
        zv_rot = ZernikeVector(coeffs=zfit)
        zv_rot.rotate(angle=-total_rotation)
        results['rot_zernike'] = zv_rot

        # subtract the reference aberrations
        zref = self.reference_aberrations(mode, hdr=slope_results['header'])
        zsub = zv_rot - zref
        results['zernike'] = zsub

        pred = np.dot(zfit, inverse_infmat)
        pred_slopes = -(1. / self.tiltfactor) * pred.reshape(2, slopes.shape[1])
        diff = slopes - pred_slopes
        rms = self.pix_size * np.sqrt((diff[0]**2 + diff[1]**2).mean())
        results['residual_rms'] = rms.to(u.arcsec).value * self.tiltfactor * zsub.units
        results['zernike_rms'] = zsub.rms
        results['zernike_p2v'] = zsub.rms

        fig = None
        if plot:
            ref_mask = slope_results['ref_mask']
            im = slope_results['data']
            gnorm = wfs_norm(im)
            fig, ax = plt.subplots()
            ax.imshow(im, cmap='Greys', origin='lower', norm=gnorm, interpolation='None')
            x = slope_results['apertures'].positions.transpose()[0]
            y = slope_results['apertures'].positions.transpose()[1]
            ax.quiver(x, y, diff[0][ref_mask], diff[1][ref_mask], scale_units='xy', scale=0.05, pivot='tip', color='red')
            xl = [50.0]
            yl = [im.shape[0]-30]
            ul = [0.2/self.pix_size.value]
            vl = [0.0]
            ax.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.05, pivot='tip', color='red')
            ax.text(60, 480, "0.2\"", verticalalignment='center')

        results['resid_plot'] = fig
        return results

    def correct_primary(self, zv, forcefile="zfile.txt", mask=[]):
        """
        Apply force corrections to primary mirror. Use 'mask' to determine which terms in 'zv' to use in the force
        calculations.
        """
        z_denorm = zv.copy()
        z_denorm.denormalize()  # need to assure we're using fringe coeffs
        forces, m1focus = self.telescope.correct_primary(zv=z_denorm, mask=mask, filename=forcefile, gain=self.m1_gain)
        return forces, m1focus

    def correct_focus(self, zv):
        """
        Convert Zernike defocus to um of secondary offset and apply offsets if connected.
        """
        z_denorm = zv.copy()
        z_denorm.denormalize()  # need to assure we're using fringe coeffs
        foc_corr = -self.m2_gain * z_denorm['Z04'] / self.secondary.focus_trans
        print("Correcting focus by moving secondary {0:0.03f}...".format(foc_corr))
        if self.connected:
            self.secondary.focus(foc_corr)
        return foc_corr

    def correct_coma(self, zv):
        """
        Convert Zernike coma (Z07 and Z08) into arcsec of secondary center-of-curvature tilts.
        """
        z_denorm = zv.copy()
        z_denorm.denormalize()  # need to assure we're using fringe coeffs

        #
        # Y coma is caused by a rotation around the X axis and X coma by a rotation around the Y axis.
        #
        # the zernike convention has Y coma as a positive tilt towards the +Y direction. however,
        # the hexapod control obeys the right-hand rule so a positive tilt around the X axis tilts
        # the wavefront towards -Y. X coma moves in the same sense as Y tilts, though. hence the difference
        # in signs here.
        cc_x_corr = self.m2_gain * z_denorm['Z07'] / self.secondary.theta_cc
        cc_y_corr = -self.m2_gain * z_denorm['Z08'] / self.secondary.theta_cc

        print("Correcting {0:0.03f} Y coma with {1:0.03f} of CC tilt in X...".format(zv['Z07'], cc_x_corr))
        print("Correcting {0:0.03f} X coma with {1:0.03f} of CC tilt in Y...".format(zv['Z08'], cc_y_corr))
        if self.connected:
            self.secondary.cc('x', cc_x_corr)
            self.secondary.cc('y', cc_y_corr)
        return cc_x_corr, cc_y_corr

    def recenter(self, fit_results):
        """
        Perform zero-coma hexapod tilts to align the pupil center to the center-of-rotation. The location of the CoR is configured
        to be at self.cor_coords
        """
        xc = fit_results['xcen']
        yc = fit_results['ycen']
        xref = self.cor_coords[0]
        yref = self.cor_coords[1]
        dx = xc - xref
        dy = yc - yref

        total_rotation = u.Quantity(fit_results['rotator'] + self.rotation, u.rad).value

        dr, phi = cart2pol([dx, dy])

        derot_phi = phi - total_rotation

        az, el = pol2cart([dr, derot_phi])

        az *= self.pix_size
        el *= self.pix_size

        print("Offsetting hexapod {0:0.03f} in AZ and {1:0.03f} in EL...".format(az, el))

        if self.connected:
            self.secondary.zc('x', el)
            self.secondary.zc('y', az)

        return az, el

    def clear_corrections(self):
        """
        Clear all applied WFS corrections
        """
        print("Clearing WFS corrections from primary and secondary...")
        clear_forces, clear_m1focus = self.telescope.clear_forces()
        cmds = self.secondary.clear_wfs()
        return clear_forces, clear_m1focus


class F9(WFS):
    """
    Defines configuration and methods specific to the F/9 WFS system
    """
    pass


class F5(WFS):
    """
    Defines configuration and methods specific to the F/5 WFS system
    """
    def __init__(self, config={}):
        super(F5, self).__init__(config=config)

        # load lookup table for off-axis aberrations
        self.aberr_table = ascii.read(self.aberr_table_file)


class MMIRS(WFS):
    """
    Defines configuration and methods specific to the MMIRS WFS system
    """
    def __init__(self, config={}):
        super(MMIRS, self).__init__(config=config)

        # load lookup table for off-axis aberrations
        self.aberr_table = ascii.read(self.aberr_table_file)

    def get_mode(self, hdr):
        """
        For MMIRS we figure out the mode from which camera the image is taken with.
        """
        cam = hdr['CAMERA']
        mode = "mmirs%d" % cam
        return mode

    def reference_aberrations(self, mode, hdr=None):
        """
        Create reference ZernikeVector for 'mode'.  For MMIRS, also need to get the WFS probe positions from 'hdr' to
        get the known off-axis aberrations at that position.
        """
        if hdr is None:
            msg = "MMIRS requires valid FITS header to determine off-axis aberrations."
            raise WFSConfigException(value=msg)

        for k in ['ROT', 'GUIDERX', 'GUIDERY']:
            if k not in hdr:
                msg = "Missing value, %s, that is required to transform MMIRS guider coordinates."
                raise WFSConfigException(value=msg)

        # for MMIRS, this gets the reference focus
        z_default = ZernikeVector(**self.modes[mode]['ref_zern'])

        # now get the off-axis aberrations
        z_offaxis = ZernikeVector()
        field_r, field_phi = self.guider_to_focal_plane(hdr['GUIDERX'], hdr['GUIDERY'], hdr['ROT'])

        # ignore piston and x/y tilts
        for i in range(4, 12):
            k = "Z%02d" % i
            z_offaxis[k] = np.interp(field_r.to(u.deg).value, self.aberr_table['field_r'], self.aberr_table[k]) * u.um

        # now rotate the off-axis aberrations
        z_offaxis.rotate(angle=field_phi)

        z = z_default + z_offaxis

        return z

    def guider_to_focal_plane(self, guide_x, guide_y, rot):
        """
        Transform from the MMIRS guider coordinate system to MMTO focal plane coordinates.
        """
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
