# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
A collection of functions and classes for performing wavefront analysis using Zernike polynomials.
Several of these routines were adapted from https://github.com/tvwerkhoven/libtim-py. They have been updated to make them
more applicable for MMTO usage and comments added to clarify what they do and how.

Expressions for cartesian derivatives of the Zernike polynomials were adapted from:
        http://adsabs.harvard.edu/abs/2014OptLE..52....7N
"""

import re
import json
import copy

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D  # noqa

import lmfit
import numpy as np
import astropy.units as u

from collections.abc import MutableMapping
from math import factorial as fac

from mmtwfs.custom_exceptions import ZernikeException


__all__ = [
    "ZernikeVector",
    "cart2pol",
    "pol2cart",
    "R_mn",
    "dR_drho",
    "theta_m",
    "dtheta_dphi",
    "zernike",
    "dZ_dx",
    "dZ_dy",
    "noll_to_zernike",
    "zernike_noll",
    "zernike_slope_noll",
    "zernike_slopes",
    "noll_normalization_vector",
    "norm_coefficient",
    "noll_coefficient",
]


def cart2pol(arr):
    """
    Convert array of [x, y] vectors to [rho, theta]

    Parameters
    ----------
    arr : `~numpy.ndarray`
        2D array with ``x`` vector as 0th element and ``y`` vector as 1st element.

    Returns
    -------
    polarr : `~numpy.ndarray`
        2D array with ``rho`` as the 0th element and ``theta`` as the 1st element.
    """
    x = arr[0]
    y = arr[1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    polarr = np.array([rho, theta])
    return polarr


def pol2cart(polarr):
    """
    Convert array of [rho, theta] vectors to [x, y]

    Parameters
    ----------
    polarr : `~numpy.ndarray`
        2D array with ``rho`` as the 0th element and ``theta`` as the 1st element.

    Returns
    -------
    arr : `~numpy.ndarray`
        2D array with ``x`` vector as 0th element and ``y`` vector as 1st element.
    """
    rho = polarr[0]
    theta = polarr[1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    arr = np.array([x, y])
    return arr


def R_mn(m, n, rho, cache=None):
    """
    Make radial Zernike polynomial on coordinate grid **rho**.

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    n : int
        n-th radial Zernike index
    rho : 2D `~numpy.ndarray`
        Radial coordinate grid

    Returns
    -------
    wf : 2D `~numpy.ndarray`
        Radial polynomial with identical shape as **rho**

    Notes
    -----
    See https://en.wikipedia.org/wiki/Zernike_polynomials for details.
    """
    if cache is not None:
        if ("R_mn", n, m) in cache:
            return cache[("R_mn", n, m)]

    if np.mod(n - m, 2) == 1:
        return 0.0

    m = np.abs(m)
    wf = 0.0
    for k in range(int((n - m) / 2) + 1):
        wf += (
            rho ** (n - 2 * k)
            * (-1) ** k
            * fac(n - k)
            / (fac(k) * fac(int((n + m) / 2) - k) * fac(int((n - m) / 2) - k))
        )

    if cache is not None:
        cache[("R_mn", n, m)] = wf

    return wf


def dR_drho(m, n, rho, cache=None):
    """
    First derivative of Zernike radial polynomial, R(m, n, rho) calculated on coordinate grid **rho**.

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    n : int
        n-th radial Zernike index
    rho : 2D `~numpy.ndarray`
        Radial coordinate grid

    Returns
    -------
    dwf : 2D `~numpy.ndarray`
        Radial polynomial with identical shape as **rho**

    Notes
    -----
    See http://adsabs.harvard.edu/abs/2014OptLE..52....7N for details.
    """
    if cache is not None:
        if ("dR_drho", n, m) in cache:
            return cache[("dR_drho", n, m)]

    dR_mn = R_mn(m, n, rho, cache=cache) * (rho**2 * (n + 2.0) + m) / (
        rho * (1.0 - rho**2)
    ) - R_mn(m + 1, n + 1, rho, cache=cache) * (n + m + 2.0) / (1.0 - rho**2)

    if cache is not None:
        cache[("dR_drho", n, m)] = dR_mn

    return dR_mn


def theta_m(m, phi):
    """
    Calculate angular Zernike mode on coordinate grid **phi**

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    phi : 2D `~numpy.ndarray`
        Azimuthal coordinate grid

    Returns
    -------
    theta : 2D `~numpy.ndarray`
        Angular Zernike mode with identical shape as **phi**
    """
    am = np.abs(m)
    if m >= 0.0:
        theta = np.cos(am * phi)
    else:
        theta = np.sin(am * phi)
    return theta


def dtheta_dphi(m, phi):
    """
    Calculate the first derivative of the m-th Zernike angular mode on coordinate grid **phi**

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    phi : 2D `~numpy.ndarray`
        Azimuthal coordinate grid

    Returns
    -------
    dtheta : 2D `~numpy.ndarray`
        Angular slopes of mode, m, with identical shape as **phi**
    """
    am = np.abs(m)
    if m >= 0.0:
        dtheta = -am * np.sin(am * phi)
    else:
        dtheta = am * np.cos(am * phi)
    return dtheta


def zernike(m, n, rho, phi, norm=False, cache=None):
    """
    Calculate Zernike mode (m, n) on grid **rho** and **phi**.
    **rho** and **phi** must be radial and azimuthal coordinate grids of identical shape, respectively.

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    n : int
        n-th radial Zernike index
    rho : 2D `~numpy.ndarray`
         Radial coordinate grid
    phi : 2D `~numpy.ndarray`
        Azimuthal coordinate grid
    norm : bool (default: False)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    wf : 2D `~numpy.ndarray`
        Wavefront described by Zernike mode (m, n). Same shape as **rho** and **phi**.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Zernike_polynomials and http://adsabs.harvard.edu/abs/2014OptLE..52....7N for details.
    """
    nc = 1.0
    if norm:
        nc = norm_coefficient(m, n)

    wf = nc * R_mn(m, n, rho, cache=cache) * theta_m(m, phi)

    return wf


def dZ_dx(m, n, rho, phi, norm=False, cache=None):
    """
    Calculate the X slopes of Zernike mode (m, n) on grid **rho** and **phi**.

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    n : int
        n-th radial Zernike index
    rho : 2D `~numpy.ndarray`
        Radial coordinate grid
    phi : 2D `~numpy.ndarray`
        Azimuthal coordinate grid
    norm : bool (default: False)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    dwf : 2D `~numpy.ndarray`
        Wavefront slope in X described by Zernike mode (m, n). Same shape as **rho** and **phi**.

    Notes
    -----
    See http://adsabs.harvard.edu/abs/2014OptLE..52....7N for details.
    """
    if cache is not None:
        if ("dZ_dx", n, m) in cache:
            return cache[("dZ_dx", n, m)]

    nc = 1.0
    if norm:
        nc = norm_coefficient(m, n)

    dwf = (
        dR_drho(m, n, rho, cache=cache) * theta_m(m, phi) * np.cos(phi)
        - R_mn(m, n, rho, cache=cache) * dtheta_dphi(m, phi) * np.sin(phi) / rho
    )

    dwf *= nc

    if cache is not None:
        cache[("dZ_dx", n, m)] = dwf

    return dwf


def dZ_dy(m, n, rho, phi, norm=False, cache=None):
    """
    Calculate the Y slopes of Zernike mode (m, n) on grid **rho** and **phi**.

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    n : int
        n-th radial Zernike index
    rho : 2D `~numpy.ndarray`
        Radial coordinate grid
    phi : 2D `~numpy.ndarray`
        Azimuthal coordinate grid
    norm : bool (default: False)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    dwf : 2D `~numpy.ndarray`
        Wavefront slope in Y described by Zernike mode (m, n). Same shape as **rho** and **phi**.

    Notes
    -----
    See http://adsabs.harvard.edu/abs/2014OptLE..52....7N for details.
    """
    if cache is not None:
        if ("dZ_dy", n, m) in cache:
            return cache[("dZ_dy", n, m)]

    nc = 1.0
    if norm:
        nc = norm_coefficient(m, n)

    dwf = (
        dR_drho(m, n, rho, cache=cache) * theta_m(m, phi) * np.sin(phi)
        + R_mn(m, n, rho, cache=cache) * dtheta_dphi(m, phi) * np.cos(phi) / rho
    )

    dwf *= nc

    if cache is not None:
        cache[("dZ_dy", n, m)] = dwf

    return dwf


def noll_to_zernike(j):
    """
    Convert linear Noll index to tuple of Zernike indices.
    j is the linear Noll coordinate, n is the radial Zernike index, and m is the azimuthal Zernike index.

    Parameters
    ----------
    j : int
        j-th Zernike mode Noll index

    Returns
    -------
    (n, m) : tuple
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

    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return (n, m)


def zernike_noll(j, rho, phi, norm=False, cache=None):
    """
    Calculate Noll Zernike mode **j** on grid **rho** and **phi**.
    **rho** and **phi** must be radial and azimuthal coordinate grids of identical shape, respectively.

    Parameters
    ----------
    j : int
        j-th Noll Zernike index
    rho : 2D `~numpy.ndarray`
        Radial coordinate grid
    phi : 2D `~numpy.ndarray`
        Azimuthal coordinate grid
    norm : bool (default: True)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    wf : 2D `~numpy.ndarray`
        Wavefront described by Noll Zernike mode, j. Same shape as **rho** and **phi**.
    """
    n, m = noll_to_zernike(j)
    wf = zernike(m, n, rho, phi, norm, cache=cache)
    return wf


def zernike_slope_noll(j, rho, phi, norm=False, cache=None):
    """
    Calculate X/Y slopes for Noll Zernike mode **j** on grid **rho** and **phi**.
    **rho** and **phi** must be radial and azimuthal coordinate grids of identical shape, respectively.

    Parameters
    ----------
    j : int
        j-th Noll Zernike index
    rho : `~numpy.ndarray`
        Radial coordinate grid
    phi : `~numpy.ndarray`
        Azimuthal coordinate grid
    norm : bool (default: True)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    dwx, dwx : 2D `~numpy.ndarray`, 2D `~numpy.ndarray`
        X/Y wavefront slopes of Noll Zernike mode, j. Same shapes as **rho** and **phi**.
    """
    n, m = noll_to_zernike(j)
    dwx = dZ_dx(m, n, rho, phi, norm=norm, cache=cache)
    dwy = dZ_dy(m, n, rho, phi, norm=norm, cache=cache)
    return dwx, dwy


def zernike_slopes(zv, rho, phi, norm=False):
    """
    Calculate total slope of a set of Zernike modes on a polar coordinate grid, (rho, phi).

    Parameters
    ----------
    zv: dict-like or ZernikeVector
        ZernikeVector or dict-like with Zernikevector-compatible key naming scheme containing zernike polynomial coefficients.
    rho : `~numpy.ndarray`
        Radial coordinate grid
    phi : `~numpy.ndarray`
        Azimuthal coordinate grid
    norm : bool (default: True)
        Normalize modes to unit variance (i.e. Noll coefficients)

    Returns
    -------
    xslope, yslope: `~numpy.ndarray`, `~numpy.ndarray`
        Total X and Y slopes of zv at each (rho, phi) point. Same shapes as **rho** and **phi**.
    """
    xslope = 0.0
    yslope = 0.0
    cache = {}
    for k, v in zv.items():
        mode = int(k.replace("Z", ""))
        dwx, dwy = zernike_slope_noll(mode, rho, phi, norm=norm, cache=cache)
        xslope += v * dwx
        yslope += v * dwy
    return xslope, yslope


def noll_normalization_vector(nmodes=30):
    """
    Calculate Noll normalization vector.
    This function calculates a **nmodes** element vector with Noll (i.e. unit variance)
    normalization constants for Zernike modes that have not already been normalized.

    Parameters
    ----------
    nmodes : int (default: 30)
        Size of normalization vector

    Returns
    -------
    norms : 1D `~numpy.ndarray` of length nmodes
    """
    nolls = (noll_to_zernike(j + 1) for j in range(nmodes))
    norms = np.asanyarray([norm_coefficient(m, n) for n, m in nolls])
    return norms


def norm_coefficient(m, n):
    """
    Calculate the normalization coefficient for the (m, n) Zernike mode.

    Parameters
    ----------
    m : int
        m-th azimuthal Zernike index
    n : int
        n-th radial Zernike index

    Returns
    -------
    norm_coeff : float
        Noll normalization coefficient
    """
    norm_coeff = np.sqrt(2 * (n + 1) / (1 + (m == 0)))
    return norm_coeff


def noll_coefficient(ll):
    """
    Calculate the Noll coefficent to normalize mode **l** to unit variance.

    Parameters
    ----------
    ll : int
        Noll mode number

    Returns
    -------
    norm_coeff : float
        Noll normalization coefficient
    """
    if ll < 1:
        raise ZernikeException(f"Noll modes start at l=1. l={ll} is not valid.")

    n, m = noll_to_zernike(ll)
    norm_coeff = norm_coefficient(m, n)
    return norm_coeff


class ZernikeVector(MutableMapping):
    """
    Class to wrap and visualize a vector of Zernike polynomial coefficients. We build upon a
    `~collections.MutableMapping` class to provide a way to access/modify coefficients in a dict-like way.

    Attributes
    ----------
    modestart : int
        Noll mode number of the first included mode.
    normalized : bool
        If True, coefficients are normalized to unit variance.  If False, coefficients
        reflect the phase amplitude of the mode.
    coeffs : dict
        Contains the Zernike coefficients with keys of form "Z%02d"
    ignored : dict
        Used to store coefficients that are temporarily ignored. Managed via ``self.ignore()`` and ``self.restore()``.
    """

    __zernikelabels = {
        "Z01": "Piston (0, 0)",
        "Z02": "X Tilt (1, 1)",
        "Z03": "Y Tilt (1, -1)",
        "Z04": "Defocus (2, 0)",
        "Z05": "Primary Astig at 45° (2, -2)",
        "Z06": "Primary Astig at 0° (2, 2)",
        "Z07": "Primary Y Coma (3, -1)",
        "Z08": "Primary X Coma (3, 1)",
        "Z09": "Y Trefoil (3, -3)",
        "Z10": "X Trefoil (3, 3)",
        "Z11": "Primary Spherical (4, 0)",
        "Z12": "Secondary Astigmatism at 0° (4, 2)",
        "Z13": "Secondary Astigmatism at 45° (4, -2)",
        "Z14": "X Tetrafoil (4, 4)",
        "Z15": "Y Tetrafoil (4, -4)",
        "Z16": "Secondary X Coma (5, 1)",
        "Z17": "Secondary Y Coma (5, -1)",
        "Z18": "Secondary X Trefoil (5, 3)",
        "Z19": "Secondary Y Trefoil (5, -3)",
        "Z20": "X Pentafoil (5, 5)",
        "Z21": "Y Pentafoil (5, -5)",
        "Z22": "Secondary Spherical (6, 0)",
        "Z23": "Tertiary Astigmatism at 45° (6, -2)",
        "Z24": "Tertiary Astigmatism at 0° (6, 2)",
        "Z25": "Secondary X Trefoil (6, -4)",
        "Z26": "Secondary Y Trefoil (6, 4)",
        "Z27": "Y Hexafoil (6, -6)",
        "Z28": "X Hexafoil (6, 6)",
        "Z29": "Tertiary Y Coma (7, -1)",
        "Z30": "Tertiary X Coma (7, 1)",
        "Z31": "Tertiary Y Trefoil (7, -3)",
        "Z32": "Tertiary X Trefoil (7, 3)",
        "Z33": "Secondary Y Pentafoil (7, -5)",
        "Z34": "Secondary X Pentafoil (7, 5)",
        "Z35": "Y Heptafoil (7, -7)",
        "Z36": "X Heptafoil (7, 7)",
        "Z37": "Tertiary Spherical (8, 0)",
    }

    __shortlabels = {
        "Z01": "Piston",
        "Z02": "X Tilt",
        "Z03": "Y Tilt",
        "Z04": "Defocus",
        "Z05": "Astig 45°",
        "Z06": "Astig 0° ",
        "Z07": "Y Coma",
        "Z08": "X Coma",
        "Z09": "Y Tref",
        "Z10": "X Tref",
        "Z11": "Spher",
        "Z12": "Astig2 0°",
        "Z13": "Astig2 45°",
        "Z14": "X Tetra",
        "Z15": "Y Tetra",
        "Z16": "X Coma2",
        "Z17": "Y Coma2",
        "Z18": "X Tref2",
        "Z19": "Y Tref2",
        "Z20": "X Penta",
        "Z21": "Y Penta",
        "Z22": "Spher2",
        "Z23": "Astig3 45°",
        "Z24": "Astig3 0°",
        "Z25": "X Tref2",
        "Z26": "Y Tref2",
        "Z27": "Y Hexa",
        "Z28": "X Hexa",
        "Z29": "Y Coma3",
        "Z30": "X Coma3",
        "Z31": "Y Tref3",
        "Z32": "X Tref3",
        "Z33": "Y Penta2",
        "Z34": "X Penta2",
        "Z35": "Y Hept",
        "Z36": "X Hept",
        "Z37": "Spher3",
    }

    def __init__(
        self,
        coeffs=[],
        modestart=2,
        normalized=False,
        zmap=None,
        units=u.nm,
        errorbars=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        coeffs : list-like (default: [])
            Vector of coefficients starting from **modestart**.
        modestart : int (default: 2)
            Noll mode number of the first included mode.
        normalized : bool (default: False)
            If True, coefficients are normalized to unit variance (Noll coefficients).  If False, coefficients
            reflect the phase amplitude of the mode (fringe coefficients).
        zmap : dict
            When loading coefficients from an array this maps the coefficient keys to array indices.
        units : `~astropy.units.IrreducibleUnit` or `~astropy.units.PrefixUnit` (default: ``u.nm`` - nanometers)
            Units of the coefficients.
        errorbars = dict (default: None)
            Uncertainties for each coefficient.
        **kwargs : kwargs
            Keyword arguments for setting terms individually, e.g. Z09=10.0.
        """

        self.modestart = modestart
        self.normalized = normalized

        self.coeffs = dict()
        self.ignored = dict()

        # now set the units
        self.units = units

        if errorbars is None:
            self.errorbars = dict()
        elif isinstance(errorbars, dict):
            self.errorbars = errorbars
        else:
            raise ValueError(
                "Errorbars for ZernikeVectors must be in the form of a dict."
            )

        # coeffs can be either a list-like or a string which is a JSON filename
        if isinstance(coeffs, str):
            self.load(filename=coeffs)
        elif isinstance(coeffs, lmfit.minimizer.MinimizerResult):
            self.load_lmfit(coeffs)
        else:
            self.from_array(coeffs, zmap=zmap, errorbars=self.errorbars)

        # now load any keyword inputs
        input_dict = dict(**kwargs)
        for k in sorted(input_dict.keys()):
            self.__setitem__(k, input_dict[k])

        # make sure errorbar units are consistent
        if len(self.errorbars) > 0:
            for k, v in self.errorbars.items():
                self.errorbars[k] = u.Quantity(self.errorbars[k], self.units)

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
        if self._valid_key(key):
            return self.coeffs.get(key, 0.0 * self.units)
        else:
            raise KeyError(f"Invalid Zernike term, {key}")

    def __setitem__(self, key, item):
        """
        Overload __setitem__ so that coefficients can be set in a dict-like manner.
        """
        if self._valid_key(key):
            # this is a hacky way to get, say, Z4 to become Z04 to maintain consistency
            mode = self._key_to_l(key)
            key = self._l_to_key(mode)

            self.coeffs[key] = u.Quantity(item, self.units)
        else:
            raise KeyError(f"Malformed Zernike mode key, {key}")

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
        s = self.pretty_print(last=99)

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
        errorbars = {}
        if isinstance(zv, ZernikeVector):
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = self.__getitem__(k) + zv[k]
                if k in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.sqrt(
                        self.errorbars[k] ** 2 + zv.errorbars[k] ** 2
                    )
                elif k in self.errorbars and k not in zv.errorbars:
                    errorbars[k] = self.errorbars[k]
                elif k not in self.errorbars and k in zv.errorbars:
                    errorbars[k] = zv.errorbars[k]
        else:
            try:
                z = u.Quantity(zv, self.units)
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) + z
                    if k in self.errorbars:
                        errorbars[k] = self.errorbars[k]
            except Exception as e:
                raise ZernikeException(
                    f"Invalid data-type, {type(zv)}, for ZernikeVector + operation: zv = {zv} ({e})"
                )
        return ZernikeVector(units=self.units, errorbars=errorbars, **d)

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
        errorbars = {}
        if isinstance(zv, ZernikeVector):
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = self.__getitem__(k) - zv[k]
                if k in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.sqrt(
                        self.errorbars[k] ** 2 + zv.errorbars[k] ** 2
                    )
                elif k in self.errorbars and k not in zv.errorbars:
                    errorbars[k] = self.errorbars[k]
                elif k not in self.errorbars and k in zv.errorbars:
                    errorbars[k] = zv.errorbars[k]
        else:
            try:
                z = u.Quantity(zv, self.units)
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) - z
                    if k in self.errorbars:
                        errorbars[k] = self.errorbars[k]
            except Exception as e:
                raise ZernikeException(
                    f"Invalid data-type, {type(zv)}, for ZernikeVector - operation: zv = {zv} ({e})"
                )
        return ZernikeVector(units=self.units, errorbars=errorbars, **d)

    def __rsub__(self, zv):
        """
        Complement to __sub__ so ZernikeVector can work on both sides of - operator.
        """
        d = {}
        errorbars = {}
        if isinstance(zv, ZernikeVector):
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = zv[k] - self.__getitem__(k)
                if k in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.sqrt(
                        self.errorbars[k] ** 2 + zv.errorbars[k] ** 2
                    )
                elif k in self.errorbars and k not in zv.errorbars:
                    errorbars[k] = self.errorbars[k]
                elif k not in self.errorbars and k in zv.errorbars:
                    errorbars[k] = zv.errorbars[k]
        else:
            try:
                z = u.Quantity(zv, self.units)
                for k in self.coeffs:
                    d[k] = z - self.__getitem__(k)
                    if k in self.errorbars:
                        errorbars[k] = self.errorbars[k]
            except Exception as e:
                raise ZernikeException(
                    f"Invalid data-type, {type(zv)}, for ZernikeVector - operation: zv = {zv} ({e})"
                )
        return ZernikeVector(errorbars=errorbars, **d)

    def __mul__(self, zv):
        """
        Create * operator to scale ZernikeVector by a constant value or ZernikeVector.
        """
        d = {}
        errorbars = {}
        if isinstance(zv, ZernikeVector):
            # keys that are in one, but not the other are valid and will result in 0's in the result.
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            outunits = self.units * self.units
            for k in keys:
                d[k] = self.__getitem__(k) * zv[k].to(self.units)
                if k in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.sqrt(
                        (self.errorbars[k] / self.__getitem__(k)) ** 2
                        + (zv.errorbars[k] / zv[k]) ** 2
                    )
                elif k in self.errorbars and k not in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.abs(
                        self.errorbars[k] / self.__getitem__(k)
                    )
                elif k not in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.abs(zv.errorbars[k] / zv[k])
        else:
            try:
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) * zv
                    if k in self.errorbars:
                        errorbars[k] = (
                            np.abs(d[k] / self.__getitem__(k)) * self.errorbars[k]
                        )
                    outunits = d[k].unit
            except Exception as e:
                raise ZernikeException(
                    f"Invalid data-type, {type(zv)}, for ZernikeVector * operation: zv = {zv} ({e})"
                )
        return ZernikeVector(units=outunits, errorbars=errorbars, **d)

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
        errorbars = {}
        if isinstance(zv, ZernikeVector):
            # only meaningful to divide keys that exist in both cases. division by 0 otherwise ok and results in np.inf.
            keys = set(self.coeffs.keys()) & set(zv.coeffs.keys())
            outunits = u.dimensionless_unscaled
            for k in keys:
                d[k] = self.__getitem__(k) / zv[k].to(self.units)
                if k in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.sqrt(
                        (self.errorbars[k] / self.__getitem__(k)) ** 2
                        + (zv.errorbars[k] / zv[k]) ** 2
                    )
                elif k in self.errorbars and k not in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.abs(
                        self.errorbars[k] / self.__getitem__(k)
                    )
                elif k not in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.abs(zv.errorbars[k] / zv[k])
        else:
            try:
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) / float(zv)
                    if k in self.errorbars:
                        errorbars[k] = (
                            np.abs(d[k] / self.__getitem__(k)) * self.errorbars[k]
                        )
                outunits = self.units
            except Exception as e:
                raise ZernikeException(
                    f"Invalid data-type, {type(zv)}, for ZernikeVector / operation: zv = {zv} ({e})"
                )
        return ZernikeVector(units=outunits, errorbars=errorbars, **d)

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
        errorbars = {}
        if isinstance(zv, ZernikeVector):
            # only meaningful to divide keys that exist in both cases. division by 0 otherwise ok and results in np.inf.
            keys = set(self.coeffs.keys()) & set(zv.coeffs.keys())
            outunits = u.dimensionless_unscaled
            for k in keys:
                d[k] = zv[k].to(self.units) / self.__getitem__(k)
                if k in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.sqrt(
                        (self.errorbars[k] / self.__getitem__(k)) ** 2
                        + (zv.errorbars[k] / zv[k]) ** 2
                    )
                elif k in self.errorbars and k not in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.abs(
                        self.errorbars[k] / self.__getitem__(k)
                    )
                elif k not in self.errorbars and k in zv.errorbars:
                    errorbars[k] = np.abs(d[k]) * np.abs(zv.errorbars[k] / zv[k])
        else:
            try:
                for k in self.coeffs:
                    d[k] = float(zv) / self.__getitem__(k)
                    if k in self.errorbars:
                        errorbars[k] = (
                            np.abs(d[k] / self.__getitem__(k)) * self.errorbars[k]
                        )
                outunits = (1.0 / self.units).unit
            except Exception as e:
                raise ZernikeException(
                    f"Invalid data-type, {type(zv)}, for ZernikeVector / operation: zv = {zv} ({e})"
                )
        return ZernikeVector(units=outunits, errorbars=errorbars, **d)

    def __pow__(self, n):
        """
        Implement the pow() method and ** operator.
        """
        d = {}
        errorbars = {}
        try:
            outunits = (self.units) ** n
            for k in self.coeffs:
                d[k] = self.__getitem__(k) ** n
                if k in self.errorbars:
                    errorbars[k] = (
                        np.abs(d[k] * n / self.__getitem__(k)) * self.errorbars[k]
                    )
        except Exception as e:
            raise ZernikeException(
                f"Invalid data-type, {type(n)}, for ZernikeVector ** operation: n = {n} ({e})"
            )
        return ZernikeVector(units=outunits, errorbars=errorbars, **d)

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
            mode = int(key.replace("Z", ""))
        except Exception as e:
            raise ZernikeException(f"Malformed Zernike mode key, {key} ({e})")
        return mode

    def _l_to_key(self, mode):
        """
        Take Noll mode number and generate valid coefficient key.
        """
        key = "Z{0:02d}".format(mode)
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
        for k in self.coeffs:
            self.coeffs[k] = u.Quantity(self.coeffs[k], units)
            if k in self.errorbars:
                self.errorbars[k] = u.Quantity(self.errorbars[k], units)
        self._units = units

    @property
    def array(self):
        """
        Return coefficients in the form of a 1D np.ndarray.
        """
        keys = sorted(self.coeffs.keys())
        last = self._key_to_l(keys[-1])
        arrsize = max(0, last - self.modestart + 1)
        arr = u.Quantity(np.zeros(arrsize), self.units)
        for k in keys:
            i = self._key_to_l(k) - self.modestart
            if i >= 0:
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
        # ignore piston and tilts when calculating wavefront RMS
        orig_modestart = self.modestart
        if self.modestart < 4:
            self.modestart = 4
        norm_coeffs = self.norm_array
        # once coeffs are normalized, the RMS is simply the sqrt of the sum of the squares of the coefficients
        rms = np.sqrt(np.sum(norm_coeffs**2))
        self.modestart = orig_modestart
        return rms

    def pretty_print(self, last=22):
        """
        Overload __repr__ to print out coefficients in a nice way including units and descriptive labels.
        """
        s = ""
        if self.normalized:
            s += "Normalized (Noll) Coefficients\n"
        else:
            s += "Fringe Coefficients\n"

        keys = sorted(self.coeffs.keys())
        if len(keys) > 0:
            for k in keys:
                if self._key_to_l(k) <= last:
                    if k in self.__zernikelabels:
                        label = self.label(k)
                    else:
                        label = ""

                    if k in self.errorbars:
                        s += "{0:>4s}: {1:>24s} \t {2:s}".format(
                            k,
                            "{0:8.4g} ± {1:5.3g}".format(
                                self.coeffs[k].value, self.errorbars[k]
                            ),
                            label,
                        )
                    else:
                        s += "{0:>4s}: {1:>24s} \t {2:s}".format(
                            k, "{0:0.4g}".format(self.coeffs[k]), label
                        )

                    s += "\n"

            s += "\n"
            if self._key_to_l(keys[-1]) > last:
                hi_orders = ZernikeVector(
                    modestart=last + 1,
                    normalized=self.normalized,
                    units=self.units,
                    **self.coeffs,
                )
                s += "High Orders RMS: \t {0:0.3g}  {1:>3s} ➞ {2:>3s}\n".format(
                    hi_orders.rms, self._l_to_key(last + 1), keys[-1]
                )
            s += "Total RMS: \t {0:0.4g}\n".format(self.rms)

        return s

    def copy(self):
        """
        Make a new ZernikeVector with the same configuration and coefficients
        """
        new = ZernikeVector(
            modestart=self.modestart,
            normalized=self.normalized,
            errorbars=copy.deepcopy(self.errorbars),
            units=self.units,
            **self.coeffs,
        )
        return new

    def save(self, filename="zernike.json"):
        """
        Save Zernike vector to JSON format to retain units and normalization info.
        """
        outdict = {}
        outdict["units"] = self.units.to_string()
        outdict["normalized"] = self.normalized
        outdict["modestart"] = self.modestart
        outdict["errorbars"] = {}
        outdict["coeffs"] = {}
        for k, c in self.coeffs.items():
            outdict["coeffs"][k] = c.value
        for k, v in self.errorbars.items():
            outdict["errorbars"][k] = v.value

        with open(filename, "w") as f:
            json.dump(outdict, f, indent=4, separators=(",", ": "), sort_keys=True)

    def load(self, filename="zernike.json"):
        """
        Load ZernikeVector data from JSON format.
        """
        try:
            with open(filename, "r") as f:
                json_data = json.load(f)
        except IOError as e:
            msg = f"Missing JSON file, {filename}: {e}"
            raise ZernikeException(value=msg)

        if "units" in json_data:
            self.units = u.Unit(json_data["units"])

        if "normalized" in json_data:
            self.normalized = json_data["normalized"]

        if "modestart" in json_data:
            self.modestart = json_data["modestart"]

        self.coeffs = {}
        if "coeffs" in json_data:
            for k, v in json_data["coeffs"].items():
                self.__setitem__(k, v)

        self.errorbars = {}
        if "errorbars" in json_data:
            for k, v in json_data["coeffs"].items():
                self.errorbars[k] = u.Quantity(v, self.units)

    def load_lmfit(self, fit_report):
        """
        Load information from a lmfit.minimizer.MinimizerResult that is output from a wavefront fit.
        If there are reported errorbars, populate those, too.
        """
        has_errors = fit_report.errorbars
        for k, v in fit_report.params.items():
            self.__setitem__(k, v)
            if has_errors:
                self.errorbars[k] = u.Quantity(v.stderr, self.units)

    def label(self, key):
        """
        If defined, return the descriptive label for mode, 'key'
        """
        if key in self.__zernikelabels:
            return self.__zernikelabels[key]
        else:
            return key

    def shortlabel(self, key):
        """
        If defined, return the short label for mode, 'key'
        """
        if key in self.__shortlabels:
            return self.__shortlabels[key]
        else:
            return key

    def from_array(
        self, coeffs, zmap=None, modestart=None, errorbars=None, normalized=False
    ):
        """
        Load coefficients from a provided list/array starting from modestart. Array is assumed to start
        from self.modestart if modestart is not provided.
        """
        self.normalized = normalized
        if errorbars is None or not isinstance(errorbars, dict):
            self.errorbars = dict()
        else:
            self.errorbars = errorbars

        if len(coeffs) > 0:
            if modestart is None:
                modestart = self.modestart

            if zmap:
                for k in zmap:
                    mode = self._key_to_l(k)
                    if mode >= modestart:
                        self.__setitem__(k, coeffs[zmap[k]])
            else:
                for i, c in enumerate(coeffs):
                    key = self._l_to_key(i + modestart)
                    if c != 0.0:
                        self.__setitem__(key, c)

    def frac_error(self, key=None):
        """
        Calculate fractional size of the error bar for mode, key.
        """
        err = 0.0
        if key is not None and key in self.coeffs and self.coeffs[key].value != 0.0:
            err = np.abs(
                self.errorbars.get(key, 0.0 * self.units).value / self.coeffs[key].value
            )
        return err

    def normalize(self):
        """
        Normalize coefficients to unit variance for each mode.
        """
        if not self.normalized:
            self.normalized = True
            for k in self.coeffs:
                mode = self._key_to_l(k)
                noll = noll_coefficient(mode)
                self.coeffs[k] /= noll
                if k in self.errorbars:
                    self.errorbars[k] /= noll

    def denormalize(self):
        """
        Restore normalized coefficients to fringe coefficients.
        """
        if self.normalized:
            self.normalized = False
            for k in self.coeffs:
                mode = self._key_to_l(k)
                noll = noll_coefficient(mode)
                self.coeffs[k] *= noll
                if k in self.errorbars:
                    self.errorbars[k] *= noll

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

    def rotate(self, angle=0.0 * u.deg):
        """
        Rotate the ZernikeVector by an angle. Rotation matrix algorithm taken from https://arxiv.org/pdf/1302.7106.pdf.
        """
        # this is a hack to extend the vector to the largest supported term to make sure we include everything affected
        # by the rotation
        self.coeffs["Z99"] = 0.0
        a = self.array
        ang = u.Quantity(angle, u.rad)  # convert angle to radians

        # there must be a more concise way to do this, but this makes it clear what's going on
        rotm = np.zeros((len(a), len(a)))
        for r in np.arange(len(a)):
            n_i, m_i = noll_to_zernike(r + self.modestart)
            for c in np.arange(len(a)):
                n_j, m_j = noll_to_zernike(c + self.modestart)
                if n_i != n_j:
                    rotm[r, c] = 0.0
                elif m_i == m_j:
                    rotm[r, c] = np.cos(m_j * ang)
                elif m_i == -m_j and m_i != 0.0:
                    rotm[r, c] = np.sin(m_j * ang)
                elif np.abs(m_i) != np.abs(m_j):
                    rotm[r, c] = 0.0
                else:
                    rotm[r, c] = 1.0

        rot_arr = np.dot(rotm, a)
        # this will take back out the zero terms
        del self.coeffs["Z99"]

        # i think it's correct to just pass on the uncertainties unchanged since measurements are done in unrotated space.
        # might want to consider handling rotations in the pupil coordinates before doing a fit.
        self.from_array(rot_arr, errorbars=self.errorbars)

    def total_phase(self, rho, phi):
        """
        Calculate total phase displacement at polar coordinates (rho, phi).
        """
        phase = 0.0
        for k, z in self.coeffs.items():
            mode = self._key_to_l(k)
            if self.normalized:
                norm = noll_coefficient(mode)
            else:
                norm = 1.0
            ph = z * norm * zernike_noll(mode, rho, phi)
            phase += ph
        return phase

    def phase_map(self, n=400):
        """
        Calculate a 2D polar map of total phase displacements with sampling of n points along rho and phi vectors.
        """
        rho = np.linspace(0.0, 1.0, n)
        phi = np.linspace(0, 2 * np.pi, n)
        [p, r] = np.meshgrid(phi, rho)
        x = r * np.cos(p)
        y = r * np.sin(p)
        ph = self.total_phase(r, p)
        return x, y, r, p, ph

    def fringe_bar_chart(
        self, total=True, max_c=2000 * u.nm, title=None, last_mode=None
    ):
        """
        Plot a bar chart of the fringe amplitudes of the coefficients
        """
        # we want to plot bars for each of the modes we usually use and thus label.
        label_keys = sorted(self.__zernikelabels.keys())
        if last_mode is None:
            last_label = self._key_to_l(label_keys[-1])
        else:
            last_label = last_mode
        last_coeff = self._key_to_l(sorted(self.coeffs.keys())[-1])

        if last_coeff < 22:
            modes = label_keys[3:last_coeff]  # ignore piston and tilts in bar plot
        else:
            modes = label_keys[3:22]
        labels = [self.shortlabel(m) for m in modes]

        if self.normalized:
            coeffs = [
                self.__getitem__(m).value * noll_coefficient(self._key_to_l(m))
                for m in modes
            ]
            errorbars = [
                self.errorbars.get(m, 0.0 * self.units).value
                * noll_coefficient(self._key_to_l(m))
                for m in modes
            ]
        else:
            coeffs = [self.__getitem__(m).value for m in modes]
            errorbars = [self.errorbars.get(m, 0.0 * self.units).value for m in modes]

        # lump higher order terms into one RMS bin.
        if last_coeff > last_label:
            hi_orders = ZernikeVector(
                modestart=last_label + 1,
                normalized=self.normalized,
                units=self.units,
                **self.coeffs,
            )
            labels.append("Hi Ord RMS")
            coeffs.append(hi_orders.rms.value)
            errorbars.append(0.0)

        # add total RMS
        if total:
            labels.append("Total RMS")
            coeffs.append(self.rms.value)
            errorbars.append(0.0)

        max_c = u.Quantity(max_c, self.units).value
        cmap = cm.ScalarMappable(col.Normalize(-max_c, max_c), cm.coolwarm_r)
        cmap._A = []  # stupid matplotlib
        ind = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.set_label("Fringe Wavefront Amplitude per Zernike Mode")
        ax.bar(ind, coeffs, color=cmap.to_rgba(coeffs), yerr=errorbars)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(color="gray", linestyle="dotted")
        ax.xaxis.grid(color="gray", linestyle="dotted", lw=1)
        ax.set_axisbelow(True)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels, rotation=45, ha="right", size="x-small")
        ax.set_ylim(-max_c, max_c)
        ax.set_ylabel(f"Wavefront Amplitude ({self.units})")
        if title is not None:
            ax.set_title(title)
        cb = fig.colorbar(cmap, ax=ax)
        cb.set_label(f"{self.units}")
        return fig

    def bar_chart(
        self, residual=None, total=True, max_c=500 * u.nm, title=None, last_mode=None
    ):
        """
        Plot a bar chart of the coefficients and, optionally, a residual amount not included in the coefficients.
        """
        # we want to plot bars for each of the modes we usually use and thus label.
        label_keys = sorted(self.__zernikelabels.keys())
        if last_mode is None:
            last_label = self._key_to_l(label_keys[-1])
        else:
            last_label = last_mode

        last_coeff = self._key_to_l(sorted(self.coeffs.keys())[-1])
        if last_coeff < 22:
            modes = label_keys[3:last_coeff]  # ignore piston and tilts in bar plot
        else:
            modes = label_keys[3:22]
        labels = [self.shortlabel(m) for m in modes]

        if self.normalized:
            coeffs = [np.abs(self.__getitem__(m).value) for m in modes]
        else:
            coeffs = [
                np.abs(self.__getitem__(m).value) / noll_coefficient(self._key_to_l(m))
                for m in modes
            ]

        # lump higher order terms into one RMS bin.
        if last_coeff > last_label:
            hi_orders = ZernikeVector(
                modestart=last_label + 1,
                normalized=self.normalized,
                units=self.units,
                **self.coeffs,
            )
            labels.append("High Orders")
            coeffs.append(hi_orders.rms.value)

        # add total RMS
        if total:
            labels.append("Total")
            coeffs.append(self.rms.value)

        # add residual RMS of zernike fit
        if residual is not None:
            resid = u.Quantity(residual, self.units).value
            labels.append("Residual")
            coeffs.append(resid)

        max_c = u.Quantity(max_c, self.units).value
        cmap = cm.ScalarMappable(col.Normalize(0, max_c), cm.magma_r)
        cmap._A = []  # stupid matplotlib
        ind = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.set_label("RMS Wavefront Error per Zernike Mode")
        ax.bar(ind, coeffs, color=cmap.to_rgba(coeffs))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(color="gray", linestyle="dotted")
        ax.set_axisbelow(True)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels, rotation=45, ha="right", size="x-small")
        ax.set_ylim(0, max_c)
        ax.set_ylabel(f"RMS Wavefront Error ({self.units})")
        if title is not None:
            ax.set_title(title)
        cb = fig.colorbar(cmap, ax=ax)
        cb.set_label(f"{self.units}")
        return fig

    def plot_map(self):
        """
        Plot 2D map of total phase displacement.
        """
        x, y, r, p, ph = self.phase_map(n=400)
        fig, ax = plt.subplots()
        fig.set_label("Wavefront Map")
        vmin = u.Quantity(-1000, u.nm).to(self.units).value
        vmax = -vmin
        pmesh = ax.pcolormesh(x, y, ph, vmin=vmin, vmax=vmax, cmap=cm.RdBu)
        pmesh.axes.set_axis_off()
        pmesh.axes.set_aspect(1.0)
        cbar = fig.colorbar(pmesh)
        cbar.set_label(self.units.name, rotation=0)
        return fig

    def plot_surface(self):
        """
        Plot total phase displacement as a 3D surface along with 2D contour map.
        """
        x, y, r, p, ph = self.phase_map(n=100)
        fig = plt.figure(figsize=(8, 6))
        fig.set_label("3D Wavefront Map")
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            x, y, ph, rstride=1, cstride=1, linewidth=0, alpha=0.6, cmap="plasma"
        )
        v = max(abs(ph.max().value), abs(ph.min().value))
        ax.set_zlim(-v * 5, v * 5)
        # cset = ax.contourf(x, y, ph, zdir='z', offset=-v*5, cmap='plasma')
        ax.xaxis.set_ticks([-1, 0, 1])
        ax.yaxis.set_ticks([-1, 0, 1])
        cbar = fig.colorbar(surf, shrink=1, aspect=30, ax=ax)
        cbar.set_label(self.units.name, rotation=0)
        return fig
