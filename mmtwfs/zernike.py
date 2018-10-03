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
import numbers
from collections import MutableMapping

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import uncertainties

import astropy.units as u
from astropy.units import (Unit, dimensionless_unscaled, get_current_unit_registry, UnitBase, UnitsError, UnitTypeError)
from astropy.utils.misc import isiterable, InheritDocstrings

from scipy.special import factorial as fac

from .custom_exceptions import ZernikeException


__all__ = ['ZernikeVector', 'cart2pol', 'pol2cart', 'R_mn', 'dR_drho', 'theta_m', 'dtheta_dphi', 'zernike', 'dZ_dx', 'dZ_dy',
           'noll_to_zernike', 'zernike_noll', 'zernike_slope_noll', 'noll_normalization_vector', 'norm_coefficient',
           'noll_coefficient', 'zernike_influence_matrix', 'UQuantity']


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


def R_mn(m, n, rho):
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
    if np.mod(n-m, 2) == 1:
        return 0.0

    m = np.abs(m)
    wf = 0.0
    for k in range(int((n - m)/2) + 1):
        wf += rho**(n - 2.0*k) * (-1.0)**k * fac(n-k) / (fac(k) * fac((n + m)/2.0 - k) * fac((n - m)/2.0 - k))

    return wf


def dR_drho(m, n, rho):
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
    dR_mn = R_mn(m, n, rho) * (rho**2 * (n + 2.) + m) / (rho * (1. - rho**2)) - \
        R_mn(m+1, n+1, rho) * (n + m + 2.) / (1. - rho**2)

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


def zernike(m, n, rho, phi, norm=False):
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

    wf = nc * R_mn(m, n, rho) * theta_m(m, phi)

    return wf


def dZ_dx(m, n, rho, phi, norm=False):
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
    nc = 1.0
    if norm:
        nc = norm_coefficient(m, n)

    dwf = dR_drho(m, n, rho) * theta_m(m, phi) * np.cos(phi) - \
        R_mn(m, n, rho) * dtheta_dphi(m, phi) * np.sin(phi) / rho

    dwf *= nc

    return dwf


def dZ_dy(m, n, rho, phi, norm=False):
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
    nc = 1.0
    if norm:
        nc = norm_coefficient(m, n)

    dwf = dR_drho(m, n, rho) * theta_m(m, phi) * np.sin(phi) + \
        R_mn(m, n, rho) * dtheta_dphi(m, phi) * np.cos(phi) / rho

    dwf *= nc

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

    m = (-1)**j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return (n, m)


def zernike_noll(j, rho, phi, norm=False):
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
    wf = zernike(m, n, rho, phi, norm)
    return wf


def zernike_slope_noll(j, rho, phi, norm=False):
    """
    Calculate X/Y slopes for Noll Zernike mode **j** on grid **rho** and **phi**.
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
    dwx, dwx : 2D `~numpy.ndarray`, 2D `~numpy.ndarray`
        X/Y wavefront slopes of Noll Zernike mode, j. Same shapes as **rho** and **phi**.
    """
    n, m = noll_to_zernike(j)
    dwx = dZ_dx(m, n, rho, phi, norm=norm)
    dwy = dZ_dy(m, n, rho, phi, norm=norm)
    return dwx, dwy


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
    nolls = (noll_to_zernike(j+1) for j in range(nmodes))
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
    norm_coeff = np.sqrt(2 * (n + 1)/(1 + (m == 0)))
    return norm_coeff


def noll_coefficient(l):
    """
    Calculate the Noll coefficent to normalize mode **l** to unit variance.

    Parameters
    ----------
    l : int
        Noll mode number

    Returns
    -------
    norm_coeff : float
        Noll normalization coefficient
    """
    if l < 1:
        raise ZernikeException(f"Noll modes start at l=1. l={l} is not valid.")

    n, m = noll_to_zernike(l)
    norm_coeff = norm_coefficient(m, n)
    return norm_coeff


def zernike_influence_matrix(pup_coords, nmodes=20, modestart=2):
    """
    Calculate matrices to convert wavefront slopes to Zernike coefficients and to convert Zernike coefficients to
    wavefront slopes.  This method analytic derivatives to calculate the slopes for Zernike mode.  Adapting this methods
    for other basis sets would also require calculating the analytic derivatives for those sets.  This method also currently
    only calculates the slope at the aperture center.  It would be more correct to average over the aperture, but this isn't
    hugely important for the lower order modes we're most interested in.

    Parameters
    ----------
    pup_coords : 2-element tuple
        Pupil coordinates of the aperture centers.
    nmodes : int (default: 20)
        Number of Zernike modes to fit.
    modestart : int (default: 2)
        First mode to include in the set to fit.

    Returns
    -------
    matrices : tuple (2D `~numpy.ndarray`, 2D `~numpy.ndarray`)
        (slopes-to-zernike matrix, zernike-to-slope matrix)
    """
    x = pup_coords[0]
    y = pup_coords[1]
    rho, phi = cart2pol([x, y])
    zern_slopes = [zernike_slope_noll(zmode, rho, phi) for zmode in range(modestart, nmodes+modestart)]
    zern_slopes_mat = np.r_[zern_slopes].reshape(nmodes, -1)  # X slopes and then Y slopes for each mode

    # use SVD to set up optimized conversion matrices
    U, s, Vh = np.linalg.svd(zern_slopes_mat, full_matrices=False)

    # don't need to trim singular values for reasonable numbers of modes so fit all requested modes.
    zern_inv_mat = np.dot(Vh.T, np.dot(np.diag(1./s), U.T))

    matrices = (zern_inv_mat, zern_slopes_mat)
    return matrices


# subclass astropy.units.Quantity to add support for uncertainties.UFloat objects
class UQuantity(u.Quantity):

    # overload __new__ to fix the various data-type checks...
    def __new__(cls, value, unit=None, dtype=None, copy=True, order=None, subok=False, ndmin=0):

        if unit is not None:
            # convert unit first, to avoid multiple string->unit conversions
            unit = Unit(unit)
            # if we allow subclasses, allow a class from the unit.
            if subok:
                qcls = getattr(unit, '_quantity_class', cls)
                if issubclass(qcls, cls):
                    cls = qcls

        # optimize speed for Quantity with no dtype given, copy=False
        if isinstance(value, (u.Quantity, UQuantity)):
            if unit is not None and unit is not value.unit:
                value = value.to(unit)
                # the above already makes a copy (with float dtype)
                copy = False

            if type(value) is not cls and not (subok and
                                               isinstance(value, cls)):
                value = value.view(cls)

            if dtype is None:
                if not copy:
                    return value

                if not np.can_cast(np.float32, value.dtype):
                    dtype = float

            return np.array(value, dtype=dtype, copy=copy, order=order,
                            subok=True, ndmin=ndmin)

        # Maybe str, or list/tuple of Quantity? If so, this may set value_unit.
        # To ensure array remains fast, we short-circuit it.
        value_unit = None
        if not isinstance(value, np.ndarray):
            if isinstance(value, str):
                # The first part of the regex string matches any integer/float;
                # the second parts adds possible trailing .+-, which will break
                # the float function below and ensure things like 1.2.3deg
                # will not work.
                pattern = (r'\s*[+-]?'
                           r'((\d+\.?\d*)|(\.\d+)|([nN][aA][nN])|'
                           r'([iI][nN][fF]([iI][nN][iI][tT][yY]){0,1}))'
                           r'([eE][+-]?\d+)?'
                           r'[.+-]?')

                v = re.match(pattern, value)
                unit_string = None
                try:
                    value = float(v.group())

                except Exception:
                    raise TypeError('Cannot parse "{0}" as a {1}. It does not '
                                    'start with a number.'
                                    .format(value, cls.__name__))

                unit_string = v.string[v.end():].strip()
                if unit_string:
                    value_unit = Unit(unit_string)
                    if unit is None:
                        unit = value_unit  # signal no conversion needed below.

            elif (isiterable(value) and len(value) > 0 and
                  all(isinstance(v, (u.Quantity, UQuantity)) for v in value)):
                # Convert all quantities to the same unit.
                if unit is None:
                    unit = value[0].unit
                value = [q.to_value(unit) for q in value]
                value_unit = unit  # signal below that conversion has been done

        if value_unit is None:
            # If the value has a `unit` attribute and if not None
            # (for Columns with uninitialized unit), treat it like a quantity.
            value_unit = getattr(value, 'unit', None)
            if value_unit is None:
                # Default to dimensionless for no (initialized) unit attribute.
                if unit is None:
                    unit = cls._default_unit
                value_unit = unit  # signal below that no conversion is needed
            else:
                try:
                    value_unit = Unit(value_unit)
                except Exception as exc:
                    raise TypeError("The unit attribute {0!r} of the input could "
                                    "not be parsed as an astropy Unit, raising "
                                    "the following exception:\n{1}"
                                    .format(value.unit, exc))

                if unit is None:
                    unit = value_unit
                elif unit is not value_unit:
                    copy = False  # copy will be made in conversion at end

        value = np.array(value, dtype=dtype, copy=copy, order=order,
                         subok=False, ndmin=ndmin)

        # check that array contains numbers or long int objects
        if (value.dtype.kind in 'OSU' and
            not (value.dtype.kind == 'O' and
                 isinstance(value.item(() if value.ndim == 0 else 0),
                            (numbers.Number, uncertainties.UFloat)))):
            raise TypeError("The value must be a valid Python or "
                            "Numpy numeric type.")

        # by default, cast any integer, boolean, etc., to float
        if dtype is None and (not np.can_cast(np.float32, value.dtype)
                              or value.dtype.kind == 'O'):
            if not isinstance(value.item(() if value.ndim == 0 else 0), uncertainties.UFloat):
                value = value.astype(float)

        value = value.view(cls)
        value._set_unit(value_unit)
        if unit is value_unit:
            return value
        else:
            # here we had non-Quantity input that had a "unit" attribute
            # with a unit different from the desired one.  So, convert.
            return value.to(unit)

    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.

        Parameters
        ----------
        unit : UnitBase
            The unit for which the appropriate class should be returned

        Returns
        -------
        tuple :
            - `UQuantity` subclass
            - bool: True if subclasses of the given class are ok
        """
        return UQuantity, True

    def _to_own_unit(self, value, check_precision=True):
        try:
            _value = value.to_value(self.unit)
        except AttributeError:
            # We're not a Quantity, so let's try a more general conversion.
            # Plain arrays will be converted to dimensionless in the process,
            # but anything with a unit attribute will use that.
            try:
                _value = UQuantity(value).to_value(self.unit)
            except UnitsError as exc:
                # last chance: if this was not something with a unit
                # and is all 0, inf, or nan, we treat it as arbitrary unit.
                if (not hasattr(value, 'unit') and
                        can_have_arbitrary_unit(value)):
                    _value = value
                else:
                    raise exc

        if check_precision:
            value_dtype = getattr(value, 'dtype', None)
            if self.dtype != value_dtype and not value.dtype.kind == 'O':
                self_dtype_array = np.array(_value, self.dtype)
                value_dtype_array = np.array(_value, dtype=value_dtype,
                                             copy=False)
                if not np.all(np.logical_or(self_dtype_array ==
                                            value_dtype_array,
                                            np.isnan(value_dtype_array))):
                    raise TypeError("cannot convert value type to array type "
                                    "without precision loss")
        return _value


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
        "Z05": u"Primary Astig at 45° (2, -2)",
        "Z06": u"Primary Astig at 0° (2, 2)",
        "Z07": "Primary Y Coma (3, -1)",
        "Z08": "Primary X Coma (3, 1)",
        "Z09": "Y Trefoil (3, -3)",
        "Z10": "X Trefoil (3, 3)",
        "Z11": "Primary Spherical (4, 0)",
        "Z12": u"Secondary Astigmatism at 0° (4, 2)",
        "Z13": u"Secondary Astigmatism at 45° (4, -2)",
        "Z14": "X Tetrafoil (4, 4)",
        "Z15": "Y Tetrafoil (4, -4)",
        "Z16": "Secondary X Coma (5, 1)",
        "Z17": "Secondary Y Coma (5, -1)",
        "Z18": "Secondary X Trefoil (5, 3)",
        "Z19": "Secondary Y Trefoil (5, -3)",
        "Z20": "X Pentafoil (5, 5)",
        "Z21": "Y Pentafoil (5, -5)",
        "Z22": "Secondary Spherical (6, 0)",
        "Z23": u"Tertiary Astigmatism at 45° (6, -2)",
        "Z24": u"Tertiary Astigmatism at 0° (6, 2)",
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
        "Z37": "Tertiary Spherical (8, 0)"
    }

    __shortlabels = {
        "Z01": "Piston",
        "Z02": "X Tilt",
        "Z03": "Y Tilt",
        "Z04": "Defocus",
        "Z05": u"Astig 45°",
        "Z06": u"Astig 0° ",
        "Z07": "Y Coma",
        "Z08": "X Coma",
        "Z09": "Y Tref",
        "Z10": "X Tref",
        "Z11": "Spher",
        "Z12": u"Astig2 0°",
        "Z13": u"Astig2 45°",
        "Z14": "X Tetra",
        "Z15": "Y Tetra",
        "Z16": "X Coma2",
        "Z17": "Y Coma2",
        "Z18": "X Tref2",
        "Z19": "Y Tref2",
        "Z20": "X Penta",
        "Z21": "Y Penta",
        "Z22": "Spher2",
        "Z23": u"Astig3 45°",
        "Z24": u"Astig3 0°",
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
        "Z37": "Spher3"
    }

    def __init__(self, coeffs=[], modestart=2, normalized=False, zmap=None, units=u.nm, **kwargs):
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
        **kwargs : kwargs
            Keyword arguments for setting terms individually, e.g. Z09=10.0.
        """

        self.modestart = modestart
        self.normalized = normalized
        self.coeffs = {}
        self.ignored = {}

        # now set the units
        self.units = units

        # python 2.x compatibility hack
        try:
            basestring
        except NameError:
            basestring = str

        # coeffs can be either a list-like or a string which is a JSON filename
        if isinstance(coeffs, basestring):
            self.load(filename=coeffs)
        else:
            self.from_array(coeffs, zmap=zmap)

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
                raise KeyError(f"Invalid Zernike term, {key}")

    def __setitem__(self, key, item):
        """
        Overload __setitem__ so that coefficients can be set in a dict-like manner.
        """
        if self._valid_key(key):
            # this is a hacky way to get, say, Z4 to become Z04 to maintain consistency
            mode = self._key_to_l(key)
            key = self._l_to_key(mode)
            self.coeffs[key] = UQuantity(item, self.units)
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
        if isinstance(zv, ZernikeVector):
            keys = set(self.coeffs.keys()) | set(zv.coeffs.keys())
            for k in keys:
                d[k] = self.__getitem__(k) + zv[k]
        else:
            try:
                for k in self.coeffs:
                    d[k] = self.__getitem__(k) + float(zv) * self.units
            except Exception as e:
                raise ZernikeException(f"Invalid data-type, {type(zv)}, for ZernikeVector + operation: zv = {zv} ({e})")
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
            except Exception as e:
                raise ZernikeException(f"Invalid data-type, {type(zv)}, for ZernikeVector - operation: zv = {zv} ({e})")
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
            except Exception as e:
                raise ZernikeException(f"Invalid data-type, {type(zv)}, for ZernikeVector - operation: zv = {zv} ({e})")
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
            except Exception as e:
                raise ZernikeException(f"Invalid data-type, {type(zv)}, for ZernikeVector * operation: zv = {zv} ({e})")
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
            except Exception as e:
                raise ZernikeException(f"Invalid data-type, {type(zv)}, for ZernikeVector / operation: zv = {zv} ({e})")
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
            except Exception as e:
                raise ZernikeException(f"Invalid data-type, {type(zv)}, for ZernikeVector / operation: zv = {zv} ({e})")
        d['units'] = outunits
        return ZernikeVector(**d)

    def __pow__(self, n):
        """
        Implement the pow() method and ** operator.
        """
        d = {}
        try:
            for k in self.coeffs:
                d[k] = self.__getitem__(k).value ** float(n) * self.units
        except Exception as e:
            raise ZernikeException(f"Invalid data-type, {type(n)}, for ZernikeVector ** operation: n = {n} ({e})")
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
        self._units = units
        for k in self.coeffs:
            self.coeffs[k] = UQuantity(self.coeffs[k], units)

    @property
    def array(self):
        """
        Return coefficients in the form of a 1D np.ndarray.
        """
        keys = sorted(self.coeffs.keys())
        last = self._key_to_l(keys[-1])
        arrsize = max(0, last - self.modestart + 1)
        arr = UQuantity(np.zeros(arrsize), self.units)
        for k in keys:
            i = self._key_to_l(k) - self.modestart
            if i >= 0:
                arr[i] = UQuantity(self.coeffs[k], self.units)
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
        return UQuantity(ph.max() - ph.min(), self.units)

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
                        s += "{0:>4s}: {1:>12s} \t {2:s}".format(k, "{0:0.03g}".format(self.coeffs[k]), self.label(k))
                    else:
                        s += "{0:>4s}: {1:>12s}".format(k, "{0:0.03g}".format(self.coeffs[k]))
                    s += "\n"

            s += "\n"
            if self._key_to_l(keys[-1]) > last:
                hi_orders = ZernikeVector(modestart=last+1, normalized=self.normalized, units=self.units, **self.coeffs)
                s += "High Orders RMS: \t {0:0.03g}  {1:>3s} ➞ {2:>3s}\n".format(hi_orders.rms, self._l_to_key(last+1), keys[-1])
            s += "Total RMS: \t \t {0:0.03g}\n".format(self.rms)

        return s

    def copy(self):
        """
        Make a new ZernikeVector with the same configuration and coefficients
        """
        new = ZernikeVector(modestart=self.modestart, normalized=self.normalized, units=self.units, **self.coeffs)
        return new

    def save(self, filename="zernike.json"):
        """
        Save Zernike vector to JSON format to retain units and normalization info.
        """
        outdict = {}
        outdict['units'] = self.units.to_string()
        outdict['normalized'] = self.normalized
        outdict['modestart'] = self.modestart
        outdict['coeffs'] = {}
        for k, c in self.coeffs.items():
            outdict['coeffs'][k] = c.value
        with open(filename, 'w') as f:
            json.dump(outdict, f, indent=4, separators=(',', ': '), sort_keys=True)

    def load(self, filename="zernike.json"):
        """
        Load ZernikeVector data from JSON format.
        """
        try:
            with open(filename, 'r') as f:
                json_data = json.load(f)
        except IOError as e:
            msg = "Missing JSON file: %s" % filename
            raise ZernikeException(value=msg)

        if 'units' in json_data:
            self.units = u.Unit(json_data['units'])

        if 'normalized' in json_data:
            self.normalized = json_data['normalized']

        if 'modestart' in json_data:
            self.modestart = json_data['modestart']

        self.coeffs = {}
        if 'coeffs' in json_data:
            for k, v in json_data['coeffs'].items():
                self.__setitem__(k, v)

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

    def from_array(self, coeffs, zmap=None, modestart=None, normalized=False):
        """
        Load coefficients from a provided list/array starting from modestart. Array is assumed to start
        from self.modestart if modestart is not provided.
        """
        self.normalized = normalized
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
        self.coeffs['Z99'] = 0.0
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
        del self.coeffs['Z99']
        self.from_array(rot_arr)

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
        phi = np.linspace(0, 2*np.pi, n)
        [p, r] = np.meshgrid(phi, rho)
        x = r * np.cos(p)
        y = r * np.sin(p)
        ph = self.total_phase(r, p)
        return x, y, r, p, ph

    def fringe_bar_chart(self, total=True, max_c=2000*u.nm, title=None, last_mode=None):
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

        if last_coeff < 21:
            modes = label_keys[3:last_coeff]  # ignore piston and tilts in bar plot
        else:
            modes = label_keys[3:22]
        labels = [self.shortlabel(m) for m in modes]

        if self.normalized:
            coeffs = [self.__getitem__(m).value * noll_coefficient(self._key_to_l(m)) for m in modes]
        else:
            coeffs = [self.__getitem__(m).value for m in modes]

        # lump higher order terms into one RMS bin.
        if last_coeff > last_label:
            hi_orders = ZernikeVector(modestart=last_label+1, normalized=self.normalized, units=self.units, **self.coeffs)
            labels.append("Hi Ord RMS")
            coeffs.append(hi_orders.rms.value)

        # add total RMS
        if total:
            labels.append("Total RMS")
            coeffs.append(self.rms.value)

        max_c = u.Quantity(max_c, self.units).value
        cmap = cm.ScalarMappable(col.Normalize(-max_c, max_c), cm.coolwarm_r)
        cmap._A = []  # stupid matplotlib
        ind = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.set_label("Fringe Wavefront Amplitude per Zernike Mode")
        ax.bar(ind, coeffs, color=cmap.to_rgba(coeffs))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(color='gray', linestyle='dotted')
        ax.xaxis.grid(color='gray', linestyle='dotted', lw=1)
        ax.set_axisbelow(True)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels, rotation=45, ha='right', size='x-small')
        ax.set_ylim(-max_c, max_c)
        ax.set_ylabel(f"Wavefront Amplitude ({self.units})")
        if title is not None:
            ax.set_title(title)
        cb = fig.colorbar(cmap)
        cb.set_label("%s" % self.units)
        return fig

    def bar_chart(self, residual=None, total=True, max_c=500*u.nm, title=None, last_mode=None):
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
        if last_coeff < 21:
            modes = label_keys[3:last_coeff]  # ignore piston and tilts in bar plot
        else:
            modes = label_keys[3:22]
        labels = [self.shortlabel(m) for m in modes]

        if self.normalized:
            coeffs = [np.abs(self.__getitem__(m).value) for m in modes]
        else:
            coeffs = [np.abs(self.__getitem__(m).value) / noll_coefficient(self._key_to_l(m)) for m in modes]

        # lump higher order terms into one RMS bin.
        if last_coeff > last_label:
            hi_orders = ZernikeVector(modestart=last_label+1, normalized=self.normalized, units=self.units, **self.coeffs)
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(color='gray', linestyle='dotted')
        ax.set_axisbelow(True)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels, rotation=45, ha='right', size='x-small')
        ax.set_ylim(0, max_c)
        ax.set_ylabel(f"RMS Wavefront Error ({self.units})")
        if title is not None:
            ax.set_title(title)
        cb = fig.colorbar(cmap)
        cb.set_label("%s" % self.units)
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
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, ph, rstride=1, cstride=1, linewidth=0, alpha=0.6, cmap='plasma')
        v = max(abs(ph.max().value), abs(ph.min().value))
        ax.set_zlim(-v*5, v*5)
        cset = ax.contourf(x, y, ph, zdir='z', offset=-v*5, cmap='plasma')
        ax.xaxis.set_ticks([-1, 0, 1])
        ax.yaxis.set_ticks([-1, 0, 1])
        cbar = fig.colorbar(cset, shrink=1, aspect=30)
        cbar.set_label(self.units.name, rotation=0)
        return fig
