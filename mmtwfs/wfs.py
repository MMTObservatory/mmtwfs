# Licensed under GPL3
# coding=utf-8

"""
Classes and utilities for operating the wavefront sensors of the MMTO and analyzing the data they produce
"""

import numpy as np
import photutils

import matplotlib.pyplot as plt

from skimage import feature
from skimage.morphology import reconstruction
from scipy import stats, ndimage
from scipy.misc import imrotate

import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
from astropy import stats, visualization

from astroscrappy import detect_cosmics

from .config import recursive_subclasses, merge_config, mmt_config
from .telescope import MMT
from .zernike import zernike_influence_matrix, ZernikeVector, cart2pol, pol2cart
from .custom_exceptions import WFSConfigException


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


def wfsfind(data, fwhm=5.0, threshold=7.0, plot=False, ap_radius=5.0):
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
    data = check_wfsdata(data)
    mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, iters=5)
    daofind = photutils.DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    sources = daofind(data - median)
    if plot:
        positions = (sources['xcentroid'], sources['ycentroid'])
        apertures = photutils.CircularAperture(positions, r=ap_radius)
        norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.AsinhStretch())
        plt.imshow(data, cmap='gray', origin='lower', norm=norm, interpolation='None')
        apertures.plot(color='red', lw=1.5, alpha=0.5)
    return sources


def mk_reference(data, xoffset=0, yoffset=0, pup_inner=45., pup_outer=175., fwhm=4.0, threshold=30.0, plot=False):
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
    spots = wfsfind(data, fwhm=fwhm, threshold=threshold, plot=plot)
    xcen = spots['xcentroid'].mean()
    ycen = spots['ycentroid'].mean()
    spacing = grid_spacing(data)
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


def background(data, h=0.9):
    """
    Use skimage.morphology.reconstruction to filter low spatial order background from WFS images.
    See http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html for details on process.

    Arguments
    ---------
    data: str or 2D ndarray
        WFS data to analyze
    h: float (default: 0.7)
        Scale factor used to create seed image. It is used to scale the max of the image.

    Returns
    -------
    data: 2D ndarray
        Reconstructed background of the WFS image
    """
    data = check_wfsdata(data)
    seed = data - h * data.max()
    dilated = reconstruction(seed, data, method='dilation')
    return dilated


def center_pupil(data, pup_mask, threshold=0.5, sigma=20., plot=False):
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
    if plot:
        plt.imshow(match, interpolation=None, origin='lower')
    return cen


def get_apertures(data, ref, xcen, ycen, xspacing, yspacing, offset=[0.0, 0.0]):
    """
    Use the X/Y center positions and grid spacings to place the reference apertures onto the WFS
    frame.  Perform center-of-mass centroiding within each aperture.

    Arguments
    ---------
    data: str or 2D ndarray
        WFS image to analyze, either FITS file or ndarray image data
    ref: astropy.Table
        Table of reference aperture positions
    xcen, ycen: float, float
        X and Y positions of the pupil center
    xspacing, yspacing: float, float
        Aperture grid spacing along X and Y axes

    Returns
    -------
    apers: photutils.CircularAperture
        WFS apertures scaled and placed onto image
    masks: list of photutils.ApertureMask objects
        Masks used for aperture centroiding
    offsets: list of tuples
        X/Y offsets of spot centroids from aperture centers
    """
    data = check_wfsdata(data)

    # mask out the inner part of the image centered on the WFS spots and find the per-pixel standard deviation
    # from the outer parts.
    outer = data.copy()
    n = outer.shape[0]
    r = 0.5 * n  # this is conservative to get well away from pupil light
    y, x = np.ogrid[-ycen:n-ycen, -xcen:n-xcen]
    mask = x**2 + y**2 < r**2
    outer[mask] = np.nan
    stddev = np.nanstd(outer)
    sky = np.nanmean(outer)

    # we use circular apertures here because they generate square masks of the appropriate size.
    # rectangular apertures produced masks that were sqrt(2) too large.
    # see https://github.com/astropy/photutils/issues/499 for details.
    spacing = np.mean([xspacing, yspacing])  # workaround to support hexagonal grid for f/9
    apers = photutils.CircularAperture(
        ((xspacing/ref['xspacing'])*ref['apertures']['xcentroid']+xcen+offset[0],
        (yspacing/ref['yspacing'])*ref['apertures']['ycentroid']+ycen+offset[1]),
        r=spacing/2.
    )
    masks = apers.to_mask(method='subpixel')
    offsets = []
    snr = []
    for m in masks:
        subim = m.cutout(data)
        # center-of-mass centroiding is the fastest, most reliable method, especially for faint or elongated spots
        spotx, spoty = photutils.centroid_com(subim)
        msky = sky * subim.shape[0] * subim.shape[1]
        signal = subim.sum() - msky
        noise = np.sqrt(signal + msky + stddev**2 / (subim.shape[0] * subim.shape[1]))
        snr.append(signal / noise)
        offsets.append((spotx-m.shape[1]/2, spoty-m.shape[0]/2))
    offsets = np.array(offsets)
    snr = np.array(snr)
    return apers, masks, offsets, snr


def get_slopes(data, back, ref, pup_mask, plot=False):
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
    """
    data = check_wfsdata(data)
    back = check_wfsdata(back)
    pup_mask = check_wfsdata(pup_mask)

    subt = data - back

    # input data should be background subtracted for best results
    xcen, ycen = center_pupil(subt, pup_mask, plot=False)
    xspacing, yspacing = grid_spacing(data)
    print(xspacing, yspacing)
    # use the min spacing to support f/9 hexagonal geometry.
    ap_size = np.mean([xspacing, yspacing])
    ref_spacing = np.mean([ref['xspacing'], ref['yspacing']])

    apers, masks, ipos, snr = get_apertures(subt, ref, xcen, ycen, xspacing, yspacing)

    # feed initial offsets back in for a 2nd iteration to account for biasing from spots near edges of apertures
    apers, masks, pos, snr = get_apertures(subt, ref, xcen, ycen, xspacing, yspacing, offset=ipos.transpose())

    # should make this a config var, but need to play with it more...
    snr_mask = np.where(snr < 100.)

    meas_pos = apers.positions + pos
    final_aps = photutils.CircularAperture(
            meas_pos,
            r=ap_size/2.
    )

    ref_x = ref['apertures']['xcentroid']
    ref_y = ref['apertures']['ycentroid']
    ref_aps = photutils.CircularAperture(
        (ref_x+xcen, ref_y+ycen),
        r=ref_spacing/2.
    )

    pup_size = ref['pup_outer']
    slopes = final_aps.positions - ref_aps.positions
    pup_coords = (ref_aps.positions - [xcen, ycen]) / [pup_size, pup_size]

    if plot:
        norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.AsinhStretch())
        plt.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='None')
        #apers.plot(color='red')
        plt.scatter(xcen, ycen)
        final_aps.plot(color='blue')

    slopes = slopes.transpose()
    slopes[0][snr_mask] = np.nan
    slopes[1][snr_mask] = np.nan
    return np.ma.masked_invalid(slopes), pup_coords.transpose(), final_aps, (xspacing, yspacing), (xcen, ycen)


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
                plot=False
            )

        # now assign 'reference' for each mode so that it can be accessed consistently in all cases
        for mode in self.modes:
            if 'reference_file' in self.modes[mode]:
                self.modes[mode]['reference'] = mk_reference(
                    self.modes[mode]['reference_file'],
                    xoffset=self.pup_offset[0],
                    yoffset=self.pup_offset[1],
                    pup_inner=self.pup_inner,
                    pup_outer=self.pup_size / 2.,
                    plot=False
                )
            else:
                self.modes[mode]['reference'] = reference

            self.modes[mode]['zernike_matrix'] = zernike_influence_matrix(
                pup_coords=self.modes[mode]['reference']['pup_coords'],
                nmodes=self.nzern,
                modestart=2  # ignore the piston term
            )

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

    def measure_slopes(self, fitsfile, mode, plot=False):
        """
        Take a WFS image in FITS format, perform background subtration, pupil centration, and then use get_slopes()
        to perform the aperture placement and spot centroiding.
        """
        if mode not in self.modes:
            msg = "Invalid mode, %s, for WFS system, %s." % (mode, self.__class__.__name__)
            raise WFSConfigException(value=msg)

        # we're a fits file (hopefully)
        try:
            fitsdata = fits.open(fitsfile)[0]
            rawdata = fitsdata.data
            hdr = fitsdata.header
        except Exception as e:
            msg = "Error reading FITS file, %s (%s)" % (fitsfile, repr(e))
            raise WFSConfigException(value=msg)

        # MMIRS gets a lot of hot pixels/CRs so make a quick pass to nuke them
        cr_mask, data = detect_cosmics(rawdata)

        # if available, get the rotator angle out of the header
        if 'ROT' in hdr:
            rotator = hdr['ROT'] * u.deg
        else:
            rotator = 0.0 * u.deg

        # make rotated pupil mask
        pup_mask = self.pupil_mask(rotator=rotator)

        # calculate the background and subtract it
        back = background(data, h=self.back_h)
        subt = data - back

        slopes, coords, aps, spacing, cen = get_slopes(data, back, self.modes[mode]['reference'], pup_mask, plot=plot)

        if plot:
            x = aps.positions.transpose()[0]
            y = aps.positions.transpose()[1]
            uu = slopes[0]
            vv = slopes[1]
            norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.AsinhStretch())
            plt.imshow(subt, cmap='Greys', origin='lower', norm=norm, interpolation='None')
            plt.quiver(x, y, uu, vv, scale_units='xy', scale=0.2, pivot='tip', color='red')
            xl = [50.0]
            yl = [480.0]
            ul = [1.0/self.pix_size.value]
            vl = [0.0]
            plt.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.2, pivot='tip', color='red')
            plt.scatter([cen[0]], [cen[1]])
            plt.text(60, 480, "1\"", verticalalignment='center')

        results = {}
        results['slopes'] = slopes
        results['pup_coords'] = coords
        results['apertures'] = aps
        results['xspacing'] = spacing[0]
        results['yspacing'] = spacing[1]
        results['xcen'] = cen[0]
        results['ycen'] = cen[1]
        results['pup_mask'] = pup_mask
        results['background'] = back
        results['data'] = data
        results['header'] = hdr
        results['rotator'] = rotator
        results['mode'] = mode
        return results

    def fit_wavefront(self, slope_results, plot=False):
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
        results['residual_rms'] = rms
        results['zernike_rms'] = zsub.rms
        results['zernike_p2v'] = zsub.rms

        if plot:
            gnorm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.AsinhStretch())
            im = slope_results['data'] - slope_results['background']
            plt.imshow(im, cmap='Greys', origin='lower', norm=gnorm, interpolation='None')
            x = slope_results['apertures'].positions.transpose()[0]
            y = slope_results['apertures'].positions.transpose()[1]
            plt.quiver(x, y, diff[0], diff[1], scale_units='xy', scale=0.05, pivot='tip', color='red')
            xl = [50.0]
            yl = [480.0]
            ul = [0.2/self.pix_size.value]
            vl = [0.0]
            plt.quiver(xl, yl, ul, vl, scale_units='xy', scale=0.05, pivot='tip', color='red')
            plt.text(60, 480, "0.2\"", verticalalignment='center')

        return results


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
