# Licensed under GPL3 (see LICENSE)
# coding=utf-8

"""
zernike.py -- A collection of functions and classes for performing wavefront analysis using Zernike polynomials.
Several of these routines were adapted from https://github.com/tvwerkhoven/libtim-py. They have been updated to make them
more applicable for MMTO usage and comments added to clarify what they do and how.
"""

import re

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import astropy.units as u

from collections import MutableMapping
from scipy.misc import factorial as fac

from .custom_exceptions import ZernikeException


def cart2pol(arr):
    """
    convert array of [x, y] vectors to [rho, theta]
    """
    x = arr[0]
    y = arr[1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.array([rho, theta])


def pol2cart(arr):
    """
    convert array of [rho, theta] vectors to [x, y]
    """
    rho = arr[0]
    theta = arr[1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y])


def make_radial_matrix(r0, r1=None, norm=True, center=None, dtype=np.float, getxy=False):
    """
    Make a matrix of size **(r0, r1)** where the value of each element is the Euclidean
    distance to **center**. If **center**& is not given, it is assumed to be the middle of the matrix.
    If **norm** is True (default), the distance is normalized to half the radius, i.e. values
    will range from [-1, 1] for both axes. If only r0 is given, the matrix will be (r0, r0).
    If r1 is also given, the matrix will be (r0, r1).

    To make a circular binary mask of (r0, r0), use
        make_radial_matrix(r0) < 1

    Parameters
    ----------
    r0: float or int
        The width of the mask (and height if r1 == None).
    r1: float or int or None (default: None)
        The height of the mask, if provided.
    norm: bool (default: True)
        Normalize the distance such that 2/(r0, r1) equales a distance of 1.
    center: None or two-element list-like (default: None)
        Sets the origin of the matrix (uses the middle if this is set to None).
    dtype: np.dtype (default: np.float)
        Data type of the output matrix
    getxy: bool (default: False)
        If True, return x, y values instead of r

    Returns
    -------
    if getxy:
        tuple of (r0, 1) and (1, r1) ndarrays
    else:
        np.ndarray of shape (r0, r1)
    """
    if not r1:
        r1 = r0
    if r0 < 0 or r1 < 0:
        print("make_radial_matrix(): r0 < 0 or r1 < 0?")

    if center is not None and norm and sum(center)/len(center) > 1:
        raise ValueError("|center| should be < 1 if norm is set")

    if center is None:
        if norm:
            center = (0, 0)
        else:
            center = (r0 / 2.0, r1 / 2.0)

    if norm:
        r0v = np.linspace(-1-center[0], 1-center[0], r0).astype(dtype).reshape(-1, 1)
        r1v = np.linspace(-1-center[1], 1-center[1], r1).astype(dtype).reshape(1, -1)
    else:
        r0v = np.linspace(0-center[0], r0-center[0], r0).astype(dtype).reshape(-1, 1)
        r1v = np.linspace(0-center[1], r1-center[1], r1).astype(dtype).reshape(1, -1)

    if getxy:
        return r0v, r1v
    else:
        return np.sqrt(r0v**2 + r1v**2)


def radial_zernike(m, n, rho):
    """
    Make radial Zernike polynomial on coordinate grid **rho**.

    Parameters
    ----------
    m: int
        m-th radial Zernike index
    n: int
        n-th azimuthal Zernike index
    rho: 2D ndarray
        Radial coordinate grid

    Returns
    -------
    wf: 2D ndarray
        Radial polynomial with identical shape as **rho**

    Notes
    -----
    See https://en.wikipedia.org/wiki/Zernike_polynomials for details.
    """
    if np.mod(n-m, 2) == 1:
        return 0.0

    wf = 0.0
    for k in range(int((n - m)/2) + 1):
        wf += rho**(n - 2.0*k) * (-1.0)**k * fac(n-k) / (fac(k) * fac((n + m)/2.0 - k) * fac((n - m)/2.0 - k))

    return wf


def zernike(m, n, rho, phi, norm=False):
    """
    Calculate Zernike mode (m, n) on grid **rho** and **phi**.
    **rho** and **phi** must be radial and azimuthal coordinate grids of identical shape, respectively.

    Parameters
    ----------
    m: int
        m-th radial Zernike index
    n: int
        n-th azimuthal Zernike index
    rho: 2D ndarray
        Radial coordinate grid
    phi: 2D ndarray
        Azimuthal coordinate grid
    norm: bool (default: False)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    wf: 2D ndarray
        Wavefront described by Zernike mode (m, n). Same shape as rho and phi.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Zernike_polynomials for details.
    """
    nc = 1.0
    if norm:
        nc = np.sqrt(2 * (n + 1)/(1 + (m == 0)))

    if m > 0:
        return nc * radial_zernike(m, n, rho) * np.cos(m * phi)
    if m < 0:
        return nc * radial_zernike(-m, n, rho) * np.sin(-m * phi)

    wf = nc * radial_zernike(0, n, rho)
    return wf


def noll_to_zernike(j):
    """
    Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index, and m is the azimuthal Zernike index.

    Parameters
    ----------
    j: int
        j-th Zernike mode Noll index

    Returns
    -------
    (n, m): tuple
        Zernike azimuthal and radial indices

    Notes
    -----
    See <https://oeis.org/A176988>.
    """
    if j == 0:
        raise ValueError("Noll indices start at 1, 0 is invalid.")

    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return (n, m)


def zernike_noll(j, rho, phi, norm=False):
    """
    Calculate Noll Zernike mode **j** on grid **rho** and **phi**.
    **rho** and **phi** must be radial and azimuthal coordinate grids of identical shape, respectively.

    Parameters
    ----------
    j: int
        j-th Noll Zernike index
    rho: 2D ndarray
        Radial coordinate grid
    phi: 2D ndarray
        Azimuthal coordinate grid
    norm: bool (default: True)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    wf: 2D ndarray
        Wavefront described by Noll Zernike mode, j. Same shape as rho and phi.
    """
    n, m = noll_to_zernike(j)
    return zernike(m, n, rho, phi, norm)


def noll_normalization_vector(nmodes=30):
    """
    Calculate Noll normalization vector.
    This function calculates a **nmodes** element vector with Noll (i.e. unit variance)
    normalization constants for Zernike modes that have not already been normalized.

    Parameters
    ----------
    nmodes: int (default: 30)
        Size of normalization vector

    Returns
    -------
    1D ndarray of length nmodes
    """
    nolls = (noll_to_zernike(j+1) for j in range(nmodes))
    norms = [np.sqrt(2 * (n + 1)/(1 + (m == 0))) for n, m in nolls]
    return np.asanyarray(norms)


def noll_coefficient(l):
    """
    Calculate the Noll coefficent to normalize mode **l** to unit variance.

    Parameters
    ----------
    l: int
        Noll mode number

    Returns
    -------
    norm_coeff: float
        Noll normalization coefficient
    """
    if l < 0:
        raise ZernikeException("Noll modes start at l=1. l=%d is not valid." % l)

    n, m = noll_to_zernike(l)
    norm_coeff = np.sqrt(2 * (n + 1)/(1 + (m == 0)))
    return norm_coeff


def make_zernike_basis(nmodes, rad, modestart=2, calc_covmat=False):
    """
    Calculate a basis of **nmodes** Zernike modes with radius **rad**.
    This output of this function can be used as cache for other functions.

    Parameters
    ----------
    nmodes: int
        Number of modes to generate
    rad: float
        Radius of Zernike modes
    modestart: int (default: 2)
        Noll index of first mode to calculate. By default, start at 2 to ignore the first mode, piston.
    calc_covmat: bool (default: False)
        Calculate and return the covariance matrix for Zernike modes and its inverse.

    Returns
    -------
    Dict with the following entries:
        'modes' - list of Zernike modes
        'modesmat' - matrix of (nmodes, npixels)
        'covmat' - covariance matrix for the generated set of Zernike modes
        'covmat_in' - inverse of 'covmat'
        'mask' - binary mask to crop only the orthogonal portions of the calculated modes.
    """

    if nmodes <= 0:
        return {'modes': [], 'modesmat': [], 'covmat': 0, 'covmat_in': 0, 'mask': [[0]]}
    if rad <= 0:
        raise ValueError("radius should be > 0")
    if modestart <= 0:
        raise ValueError("**modestart** Noll index should be > 0")

    # Use vectors instead of a grid matrix
    rvec = (np.arange(2.0 * rad) - rad) / rad
    r0 = rvec.reshape(-1, 1)
    r1 = rvec.reshape(1, -1)
    grid_rad = make_radial_matrix(2*rad)
    grid_ang = np.arctan2(r0, r1)

    grid_mask = grid_rad <= 1

    # Build list of Zernike modes, these are *not* masked/cropped
    zern_modes = [zernike_noll(zmode, grid_rad, grid_ang) for zmode in range(modestart, nmodes+modestart)]

    # Convert modes to (nmodes, npixels) matrix
    zern_modes_mat = np.r_[zern_modes].reshape(nmodes, -1)

    covmat = covmat_in = None
    if calc_covmat:
        # Calculate covariance matrix
        covmat = np.array([[np.sum(zerni * zernj * grid_mask) for zerni in zern_modes] for zernj in zern_modes])
        # Invert covariance matrix using Moore-Penrose pseudo-inversion
        covmat_in = np.linalg.pinv(covmat)

    # Create and return dict
    return {'modes': zern_modes, 'modesmat': zern_modes_mat, 'covmat': covmat, 'covmat_in': covmat_in, 'mask': grid_mask}


def calc_slope(im, slopes_inv=None):
    """
    Calculate 2D slope of **im**, to be used to calculate unit Zernike
    influence on a Shack-Hartmann image. If **slopes_inv** is given, use that (2, N) matrix for
    fitting, otherwise generate and pseudo-invert slopes ourselves.

    An advantage of calculating image slopes in this way is that we do not need to know the derivatives
    of the basis set we are using. We instead calculate them as mapped into pixel space. This makes it easy
    to support different basis sets (e.g. mirror bending modes or annular zernikes) without also having to
    determine the analytic derivatives of the basis sets.  An additional advantage over the previous MMT SHWFS
    system is that this takes into account the integrated zernike slope across an aperture vs. the instantaneous
    slope at the aperture center.

    Parameters
    ----------
    im: 2D ndarray
        Image to fit slopes to
    slopes_inv: ndarray or None (default: None)
        Pre-computed pseudo-inverted slope matrix to fit with.  Set to None to auto-calculate.

    Returns
    -------
    tuple of (axis 0 slope, axis 1 slope)
    """

    if (slopes_inv is None):
        slopes = (np.indices(im.shape, dtype=float)/(np.r_[im.shape].reshape(-1, 1, 1))).reshape(2, -1)
        slopes = np.vstack([slopes, np.ones(slopes.shape[1])])
        slopes_inv = np.linalg.pinv(slopes)

    return np.dot(im.reshape(1, -1), slopes_inv).ravel()[:2]


def calc_influence_matrix(subaps, basis_func=make_zernike_basis, nbasis=20, cntr=None,
                          rad=-1.0, singval=0.5, subapsize=22.0, pixsize=0.1, modestart=2):
    """
    Given a sub-aperture array pattern, calculate a matrix that converts
    image shift vectors in pixels to basis polynomial amplitudes (default: Zernike)
    and also its inverse. The parameters **subapsize** and **pixsize** are used for
    absolute calibration.

    The data returned is a tuple of the following:
        1. Matrix to compute basis mode amplitudes from image shifts
        2. Matrix to compute image shifts from basis mode amplitudes
        3. The set of basis polynomials used
        4. The extent of the basis polynomials in units of **subaps**

    To calculate the above mentioned matrices, we measure the x, y-slope of all basis modes over
    all sub-apertures, giving a matrix `basisslopes_mat` that converts slopes for each basis polynomial matrix:
        subap_slopes_vec = basisslopes_mat . basis_amp_vec

    To obtain pixel shifts inside the sub-images we multiply these slopes in radians/subaperture by
        sfac = π * ap_width * pix_scale / 206265
    where ap_width is in pixels, pix_scale is arcsec/pixel, and 1/206265 converts arcsec to radians.
    The factor of π comes from the normalization of the integral of the basis modes over all radii and angles.

    We then have
        subap_shift_vec = sfac * basisslopes_mat . basis_amp_vec

    To get the inverse relation, we invert `basisslopes_mat`, giving:
        basis_amp_vec = (sfac * basisslopes_mat)^-1 . subap_shift_vec
        basis_amp_vec = basis_inv_mat . subap_shift_vec

    Parameters
    ----------
    subaps: list of 4-element list-likes
        List of subapertures formatted as (low0, high0, low1, high1)
    basis_func: function reference (default: make_zernike_basis)
        Function that generates an orthongal set of basis polynomials, e.g. Zernike polynomials.
    nbasis: int (default: 20)
        Number of basis modes to model
    cntr: 2-element list-like or None (default: None)
        Coordinate to center basis set around. If None, use calculated center of **subaps**.
    rad: float (default: -1.0)
        Radius of the aperture to use. If negative, used as fraction **-rad**, otherwise used as radius in pixels.
    singval: float (default: 1.0)
        Percentage of singular values to take into account when inverting the matrix
    subapsize: float (default: 22.0)
        Size of single Shack-Hartmann sub-aperture in detector pixels
    pixsize: float (default: 0.1)
        Detector pixel size in arcseconds

    Returns
    -------
    Tuple of (
        shift to basis polynomial matrix,
        basis polynomial to shift matrix,
        basis polynomials used,
        basis polynomial shape in units of **subaps**
    )
    """
    # we already know pixel size in arcsec so multiply by aperture width and convert to radians.
    sfac = np.pi * subapsize * pixsize / 206265.

    # Geometry: offset between subap pattern and Zernike modes
    sasize = np.median(subaps[:, 1::2] - subaps[:, ::2], axis=0).astype(int)

    if cntr is None:
        cntr = np.mean(subaps[:, ::2], axis=0).astype(int)

    # add 0.5 pixel to account for offset from where pixel is indexed and where its physical center is
    if rad < 0:
        pattrad = np.max(np.max(subaps[:, 1::2], 0) - np.min(subaps[:, ::2], 0)) / 2.0
        rad = int((pattrad * -rad) + 0.5)
    else:
        rad = int(rad + 0.5)
    saoffs = np.around(-cntr + np.r_[[rad, rad]]).astype(int)

    extent = cntr[1]-rad, cntr[1]+rad, cntr[0]-rad, cntr[0]+rad
    basis = basis_func(nbasis, rad, modestart=modestart)

    slopes = (np.indices(sasize, dtype=float)/(np.r_[sasize].reshape(-1, 1, 1))).reshape(2, -1)
    slopes = np.vstack([slopes, np.ones(slopes.shape[1])])
    slopes_inv = np.linalg.pinv(slopes)

    basisslopes = np.r_[
        [
            [
                calc_slope(
                    base[subap[0]+saoffs[0]:subap[1]+saoffs[0], subap[2]+saoffs[1]:subap[3]+saoffs[1]], slopes_inv=slopes_inv
                ) for subap in subaps
            ] for base in basis['modes']
        ]
    ].reshape(nbasis, -1)

    # np.linalg.pinv() takes the cutoff wrt the *maximum*, we want a cut-off
    # based on the cumulative sum, i.e. the total included power, which is
    # why we want to use svd() and not pinv().
    U, s, Vh = np.linalg.svd(basisslopes * sfac, full_matrices=False)
    cums = s.cumsum() / s.sum()
    nvec = np.argwhere(cums >= singval)[0][0]
    singval = cums[nvec]
    s[nvec+1:] = np.inf
    basis_inv_mat = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

    return basis_inv_mat, basisslopes*sfac, basis, extent


class ZernikeVector(MutableMapping):
    """
    Class to wrap and visualize a vector of Zernike polynomial coefficients. We build upon a MutableMapping
    class to provide a way to access/modify coefficients in a dict-like way.
    """
    __zernikelabels = {
        "Z01": "Piston (0, 0)",
        "Z02": "X Tilt (1, 1)",
        "Z03": "Y Tilt (1, -1)",
        "Z04": "Defocus (2, 0)",
        "Z05": "Primary Astig at 45˚ (2, -2)",
        "Z06": "Primary Astig at 0˚ (2, 2)",
        "Z07": "Primary Y Coma (3, -1)",
        "Z08": "Primary X Coma (3, 1)",
        "Z09": "Y Trefoil (3, -3)",
        "Z10": "X Trefoil (3, 3)",
        "Z11": "Primary Spherical (4, 0)",
        "Z12": "Secondary Astigmatism at 0˚ (4, 2)",
        "Z13": "Secondary Astigmatism at 45˚ (4, -2)",
        "Z14": "X Tetrafoil (4, 4)",
        "Z15": "Y Tetrafoil (4, -4)",
        "Z16": "Secondary X Coma (5, 1)",
        "Z17": "Secondary Y Coma (5, -1)",
        "Z18": "Secondary X Trefoil (5, 3)",
        "Z19": "Secondary Y Trefoil (5, -3)",
        "Z20": "X Pentafoil (5, 5)",
        "Z21": "Y Pentafoil (5, -5)",
        "Z22": "Secondary Spherical (6, 0)",
        "Z23": "Tertiary Astigmatism at 45˚ (6, -2)",
        "Z24": "Tertiary Astigmatism at 0˚ (6, 2)",
        "Z25": "Secondary X Trefoil (6, -4)",
        "Z26": "Secondary Y Trefoil (6, 4)",
        "Z27": "Hexafoil Y (6, -6)",
        "Z28": "Hexafoil X (6, 6)",
        "Z29": "Tertiary Y Coma (7, -1)",
        "Z30": "Tertiary X Coma (7, 1)",
        "Z31": "Tertiary Y Trefoil (7, -3)",
        "Z32": "Tertiary X Trefoil (7, 3)",
        "Z33": "Secondary Pentafoil Y (7, -5)",
        "Z34": "Secondary Pentafoil X (7, 5)",
        "Z35": "Heptafoil Y (7, -7)",
        "Z36": "Heptafoil X (7, 7)",
        "Z37": "Tertiary Spherical (8, 0)"
    }

    def __init__(self, coeffs=[], modestart=2, normalized=False, units=u.nm, **kwargs):
        """
        Parameters
        ----------
        coeffs: list-like (default: [])
            Vector of coefficients starting from **modestart**.
        modestart: int (default: 2)
            Noll mode number of the first included mode.
        normalized: bool (default: False)
            If True, coefficients are normalized to unit variance.  If False, coefficients
            reflect the phase amplitude of the mode.
        units: astropy.units.core.IrreducibleUnit or astropy.units.core.PrefixUnit (default: u.nm - nanometers)
            Units of the coefficients.
        **kwargs: kwargs
            Keyword arguments for setting terms individually, e.g. Z09=10.0.

        Attributes
        ----------
        modestart: int
            Noll mode number of the first included mode.
        normalized: bool
            If True, coefficients are normalized to unit variance.  If False, coefficients
            reflect the phase amplitude of the mode.
        coeffs: dict
            Contains the Zernike coefficients with keys of form "Z%02d"
        ignored: dict
            Used to store coefficients that are temporarily ignored. Managed via self.ignore()/self.restore().
        """
        self.modestart = modestart
        self.normalized = normalized
        self.coeffs = {}
        self.ignored = {}

        # now set the units
        self.units = units

        # first load from input array/list-like
        self.from_array(coeffs)

        # now load any keyword inputs
        input_dict = dict(**kwargs)
        for k in sorted(input_dict.keys()):
            self.__setitem__(k, input_dict[k])

    def __iter__(self):
        """
        If instance is accessed as an iterator, iterate over the dict of coefficients.
        """
        return iter(self.coeffs)

    def __contains__(self, val):
        """
        If testing for existence of a value, look in dict of coefficients for it.
        """
        return val in self.coeffs

    def __len__(self):
        """
        The length of the Zernike vector will be the mode number of the highest term minus modestart.
        The self.array property has the logic to work this out so use that length.
        """
        return len(self.array)

    def __getitem__(self, key):
        """
        Overload __getitem__ so that coefficients can be accessed in a dict-like manner. Add logic to validate
        keys and return amplitude of 0 if key is valid, but term not set in self.coeffs.
        """
        if key in self.coeffs:
            return self.coeffs[key]
        else:
            if self._valid_key(key):
                return 0.0 * self.units
            else:
                raise KeyError("Invalid Zernike term, %s" % key)

    def __setitem__(self, key, item):
        """
        Overload __setitem__ so that coefficients can be set in a dict-like manner.
        """
        if self._valid_key(key):
            # this is a hacky way to get, say, Z4 to become Z04 to maintain consistency
            l = self._key_to_l(key)
            key = self._l_to_key(l)
            self.coeffs[key] = u.Quantity(item, self.units)
        else:
            raise KeyError("Malformed Zernike mode key, %s" % key)

    def __delitem__(self, key):
        """
        Overload __delitem__ so that coefficients can be deleted in a dict-like manner.
        """
        if key in self.coeffs:
            del self.coeffs[key]

    def __repr__(self):
        """
        Overload __repr__ to print out coefficients in a nice way including units and descriptive labels.
        """
        s = ""
        if self.normalized:
            print("Normalized (Noll) Coefficients")
        else:
            print("Phase Amplitude Coefficients")
        for k in sorted(self.coeffs.keys()):
            if k in self.__zernikelabels:
                label = self.__zernikelabels[k]
                s += "%4s: %12s \t %s" % (k, "{0:0.03g}".format(self.coeffs[k]), label)
            else:
                s += "%4s: %12s" % (k, "{0:0.03g}".format(self.coeffs[k]))
            s += "\n"
        return s

    def __str__(self):
        """
        Do the same as __repr__
        """
        return self.__repr__()

    def __add__(self, zv):
        """
        Create + operator to add instances together or add a constant to get a new instance.
        Use set() to collect the unique set of keys and self.__getitem__() to use 0.0 as a default.
        """
        d = {}
        if isinstance(zv, ZernikeVector):
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = self.__getitem__(k) + zv[k]
        else:
            try:
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) + float(zv) * self.units
            except:
                raise ZernikeException("Invalid data-type, %s, for ZernikeVector + operation: zv = %s" % (type(zv), zv))
        return ZernikeVector(**d)

    def __radd__(self, zv):
        """
        Order doesn't matter for the + operator so __radd__ is same as __add__
        """
        return self.__add__(zv)

    def __sub__(self, zv):
        """
        Create - operator to substract an instance or a constant to get a new instant. Complement to __add__...
        """
        d = {}
        if isinstance(zv, ZernikeVector):
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = self.__getitem__(k) - zv[k]
        else:
            try:
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) - float(zv) * self.units
            except:
                raise ZernikeException("Invalid data-type, %s, for ZernikeVector - operation: zv = %s" % (type(zv), zv))
        return ZernikeVector(**d)

    def __rsub__(self, zv):
        """
        Complement to __sub__ so ZernikeVector can work on both sides of - operator.
        """
        d = {}
        if isinstance(zv, ZernikeVector):
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = zv[k] - self.__getitem__(k)
        else:
            try:
                for k in self.coeffs:
                    d[k] = float(zv) * self.units - self.__getitem__(k)
            except:
                raise ZernikeException("Invalid data-type, %s, for ZernikeVector - operation: zv = %s" % (type(zv), zv))
        return ZernikeVector(**d)

    def __mul__(self, zv):
        """
        Create * operator to scale ZernikeVector by a constant value or ZernikeVector.
        """
        d = {}
        if isinstance(zv, ZernikeVector):
            # keys that are in one, but not the other are valid and will result in 0's in the result.
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = self.__getitem__(k) * zv[k].to(self.units)
            outunits = self.units * self.units
        else:
            try:
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) * float(zv)
                outunits = self.units
            except:
                raise ZernikeException("Invalid data-type, %s, for ZernikeVector * operation: zv = %s" % (type(zv), zv))
        d['units'] = outunits
        return ZernikeVector(**d)

    def __rmul__(self, zv):
        """
        Multiplication works the same in any order so same as __mul__.
        """
        return self.__mul__(zv)

    def __div__(self, zv):
        """
        Still required for python 2.x...
        """
        return self.__truediv__(zv)

    def __truediv__(self, zv):
        """
        Create / operator to divide ZernikeVector by a constant value or other ZernikeVector. Only terms in both ZernikeVectors
        will be divided.
        """
        d = {}
        if isinstance(zv, ZernikeVector):
            # only meaningful to divide keys that exist in both cases. division by 0 otherwise ok and results in np.inf.
            keys = set(self.coeffs.keys()) & set(zv.coeffs.keys())
            for k in keys:
                d[k] = self.__getitem__(k) / zv[k].to(self.units)
            outunits = u.dimensionless_unscaled
        else:
            try:
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) / float(zv)
                outunits = self.units
            except:
                raise ZernikeException("Invalid data-type, %s, for ZernikeVector / operation: zv = %s" % (type(zv), zv))
        d['units'] = outunits
        return ZernikeVector(**d)

    def __rdiv__(self, zv):
        """
        Still required for python 2.x compatibility
        """
        return self.__rtruediv__(zv)

    def __rtruediv__(self, zv):
        """
        Implement __truediv__ for the right side of the operator as well.
        """
        d = {}
        if isinstance(zv, ZernikeVector):
            # only meaningful to divide keys that exist in both cases. division by 0 otherwise ok and results in np.inf.
            keys = set(self.coeffs.keys()) & set(zv.coeffs.keys())
            for k in keys:
                d[k] = zv[k].to(self.units) / self.__getitem__(k)
            outunits = u.dimensionless_unscaled
        else:
            try:
                for k in self.coeffs:
                    d[k] = float(zv) / self.__getitem__(k)
                outunits = 1.0 / self.units
            except:
                raise ZernikeException("Invalid data-type, %s, for ZernikeVector / operation: zv = %s" % (type(zv), zv))
        d['units'] = outunits
        return ZernikeVector(**d)

    def _valid_key(self, key):
        """
        Define valid format for coefficient keys.
        """
        if re.match(r"Z\d\d", key):
            return True
        else:
            return False

    def _key_to_l(self, key):
        """
        Parse key to get Noll mode number.
        """
        try:
            l = int(key.replace("Z", ""))
        except:
            raise ZernikeException("Malformed Zernike mode key, %s" % key)
        return l

    def _l_to_key(self, l):
        """
        Take Noll mode number and generate valid coefficient key.
        """
        key = "Z%02d" % l
        return key

    @property
    def units(self):
        """
        Return the coefficient units currently being used.
        """
        return self._units

    @units.setter
    def units(self, units):
        """
        When units are set, we need to go through each coefficient and perform the unit conversion to match.
        """
        self._units = units
        for k in self.coeffs:
            self.coeffs[k] = u.Quantity(self.coeffs[k], units)

    @property
    def array(self):
        """
        Return coefficients in the form of a 1D np.ndarray.
        """
        keys = sorted(self.coeffs.keys())
        last = self._key_to_l(keys[-1])
        arr = u.Quantity(np.zeros(last - self.modestart + 1), self.units)
        for k in keys:
            i = self._key_to_l(k) - self.modestart
            arr[i] = u.Quantity(self.coeffs[k], self.units)
        return arr

    @property
    def norm_array(self):
        """
        Return coefficients in the form of a 1D np.ndarray with each coefficient normalized to unit variance for its mode.
        """
        if self.normalized:
            return self.array
        else:
            # rather than repeat the logic used to normalize the coefficients, we use the existing methods
            # to do the work and grab the results in between.
            self.normalize()
            arr = self.array.copy()
            self.denormalize()
            return arr

    @property
    def peak2valley(self):
        """
        Return the peak-to-valley amplitude of the Zernike set.
        """
        x, y, r, p, ph = self.phase_map()
        return u.Quantity(ph.max() - ph.min(), self.units)

    @property
    def rms(self):
        """
        Return the RMS phase displacement of the Zernike set.
        """
        x, y, r, p, ph = self.phase_map()
        return u.Quantity(np.sqrt(np.mean(np.square(ph))), self.units)

    def normalize(self):
        """
        Normalize coefficients to unit variance for each mode.
        """
        if not self.normalized:
            self.normalized = True
            for k in self.coeffs:
                l = self._key_to_l(k)
                noll = noll_coefficient(l)
                self.coeffs[k] /= noll

    def denormalize(self):
        """
        Restore normalized coefficients to phase amplitude.
        """
        if self.normalized:
            self.normalized = False
            for k in self.coeffs:
                l = self._key_to_l(k)
                noll = noll_coefficient(l)
                self.coeffs[k] *= noll

    def from_array(self, coeffs, modestart=None):
        """
        Load coefficients from a provided list/array starting from modestart. Array is assumed to start
        from self.modestart if modestart is not provided.
        """
        if len(coeffs) > 0:
            if modestart is None:
                modestart = self.modestart
            for i, c in enumerate(coeffs):
                key = self._l_to_key(i + modestart)
                self.__setitem__(key, c)

    def ignore(self, key):
        """
        Set coefficient, key, aside for later recall if needed.
        """
        if self._valid_key(key) and key in self.coeffs:
            self.ignored[key] = self.coeffs[key]
            self.coeffs[key] = 0.0 * self.units

    def restore(self, key):
        """
        Restore coefficient, key, back into set of coefficients.
        """
        if self._valid_key(key) and key in self.ignored:
            self.coeffs[key] = self.ignored[key]
            del self.ignored[key]

    def total_phase(self, rho, phi):
        """
        Calculate total phase displacement at polar coordinates (rho, phi).
        """
        phase = 0.0
        for k, z in self.coeffs.items():
            l = self._key_to_l(k)
            if self.normalized:
                norm = noll_coefficient(l)
            else:
                norm = 1.0
            ph = z * norm * zernike_noll(l, rho, phi)
            phase += ph
        return phase

    def phase_map(self, n=400):
        """
        Calculate a 2D polar map of total phase displacements with sampling of n points along rho and phi vectors.
        """
        rho = np.linspace(0.0, 1.0, n)
        phi = np.linspace(0, 2*np.pi, n)
        [p, r] = np.meshgrid(phi, rho)
        x = r * np.cos(p)
        y = r * np.sin(p)
        ph = self.total_phase(r, p)
        return x, y, r, p, ph

    def plot_map(self):
        """
        Plot 2D map of total phase displacement.
        """
        x, y, r, p, ph = self.phase_map(n=400)
        fig = plt.pcolormesh(x, y, ph)
        fig.axes.set_axis_off()
        fig.axes.set_aspect(1.0)
        cbar = plt.colorbar()
        cbar.set_label(self.units.name, rotation=0)

    def plot_surface(self):
        """
        Plot total phase displacement as a 3D surface along with 2D contour map.
        """
        x, y, r, p, ph = self.phase_map(n=100)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, ph, rstride=1, cstride=1, linewidth=0, alpha=0.6, cmap='plasma')
        v = max(abs(ph.max().value), abs(ph.min().value))
        ax.set_zlim(-v*5, v*5)
        cset = ax.contourf(x, y, ph, zdir='z', offset=-v*5, cmap='plasma')
        ax.xaxis.set_ticks([-1, 0, 1])
        ax.yaxis.set_ticks([-1, 0, 1])
        cbar = fig.colorbar(cset, shrink=1, aspect=30)
        cbar.set_label(self.units.name, rotation=0)
