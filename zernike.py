# Licensed under GPL3 (see LICENSE)

"""
zernike.py -- A collection of functions and classes for performing wavefront analysis using Zernike polynomials.
Several of these routines were adapted from https://github.com/tvwerkhoven/libtim-py. They have been updated to make them
more applicable for MMTO usage and comments added to clarify what they do and how.
"""

import re

import numpy as np

from collections import MutableMapping
from scipy.misc import factorial as fac


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


def zernike_noll(j, rho, phi, norm=True):
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


def noll_normalization(nmodes=30):
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
                          rad=-1.0, singval=1.0, subapsize=22.0, pixsize=0.1):
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
    sasize = np.median(subaps[:, 1::2] - subaps[:, ::2], axis=0)
    if cntr is None:
        cntr = np.mean(subaps[:, ::2], axis=0).astype(int)

    # add 0.5 pixel to account for offset from where pixel is indexed and where its physical center is
    if rad < 0:
        pattrad = np.max(np.max(subaps[:, 1::2], 0) - np.min(subaps[:, ::2], 0)) / 2.0
        rad = int((pattrad * -rad) + 0.5)
    else:
        rad = int(rad + 0.5)
    saoffs = -cntr + np.r_[[rad, rad]]

    extent = cntr[1]-rad, cntr[1]+rad, cntr[0]-rad, cntr[0]+rad
    basis = basis_func(nbasis, rad, modestart=2)

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
    Class to wrap and visualize a vector of Zernike polynomial coefficients
    """
    __zernikelabels = {
        "Z1": "Piston (0, 0)",
        "Z2": "X Tilt (1, 1)",
        "Z3": "Y Tilt (1, -1)",
        "Z4": "Defocus (2, 0)",
        "Z5": "Primary Astig at 45˚ (2, -2)",
        "Z6": "Primary Astig at 0˚ (2, 2)",
        "Z7": "Primary Y Coma (3, -1)",
        "Z8": "Primary X Coma (3, 1)",
        "Z9": "Y Trefoil (3, -3)",
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

    def __init__(self, coeffs=[], modestart=2, **kwargs):
        self.modestart = modestart
        self.coeffs = {}
        self.ignored = {}

        # first load from input array/list-like
        if len(coeffs) > 0:
            for i, c in enumerate(coeffs):
                key = self._l_to_key(i + self.modestart)
                self.coeffs[key] = float(c)

        # now load any keyword inputs
        input_dict = dict(**kwargs)
        for k in sorted(input_dict.keys()):
            if self._valid_key(k):
                self.coeffs[k] = float(input_dict[k])

    def __iter__(self):
        return iter(self.coeffs)

    def __contains__(self, val):
        return value in self.coeffs

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        if key in self.coeffs:
            return self.coeffs[key]
        else:
            if self._valid_key(key):
                return 0.0
            else:
                raise KeyError("Invalid Zernike term, %s" % key)

    def __setitem__(self, key, item):
        if self._valid_key(key):
            self.coeffs[key] = float(item)
        else:
            raise KeyError("Malformed Zernike mode key, %s" % key)

    def __delitem__(self, key):
        if key in self.coeffs:
            del self.coeffs[key]

    def __repr__(self):
        s = ""
        for k in sorted(self.coeffs.keys()):
            if k in self.__zernikelabels:
                label = self.__zernikelabels[k]
                s += "%4s: %12.2f \t %s" % (k, self.coeffs[k], label)
            else:
                s += "%4s: %12.2f" % (k, self.coeffs[k])
            s += "\n"
        return s

    def _valid_key(self, key):
        if re.match('Z\d', key):
            return True
        else:
            return False

    def _key_to_l(self, key):
        try:
            l = int(key.replace("Z", ""))
        except:
            raise Exception("Malformed Zernike mode key, %s" % key)
        return l

    def _l_to_key(self, l):
        key = "Z%d" % l
        return l

    @property
    def array(self):
        keys = sorted(self.coeffs.keys())
        last = self._key_to_l(keys[-1])
        arr = np.zeros(last - self.modestart + 1)
        for k in keys:
            i = self._key_to_l(k) - self.modestart
            arr[i] = self.coeffs[k]
        return arr

    def ignore(self, key):
        if self._valid_key(key) and key in self.coeffs:
            self.ignore[key] = self.coeffs[key]
            self.coeffs[key] = 0.0

    def restore(self, key):
        if self._valid_key(key) and key in self.ignore:
            self.coeffs[key] = self.ignore[key]
            del self.ignore[key]
