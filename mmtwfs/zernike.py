# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
A collection of functions and classes for performing wavefront analysis using Zernike polynomials.
Several of these routines were adapted from https://github.com/tvwerkhoven/libtim-py.
They have been updated to make them more applicable for MMTO usage and comments added to
clarify what they do and how.

Expressions for cartesian derivatives of the Zernike polynomials were adapted from:
        http://adsabs.harvard.edu/abs/2014OptLE..52....7N
"""

from typing import Any
from collections.abc import MutableMapping

import numpy as np

from math import factorial as fac

from mmtwfs.custom_exceptions import ZernikeException


__all__ = [
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


def cart2pol(arr: np.ndarray) -> np.ndarray:
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


def pol2cart(polarr: np.ndarray) -> np.ndarray:
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


def R_mn(
        m: int,
        n: int,
        rho: np.ndarray,
        cache: dict[Any, Any] | None = None,
    ) -> float | np.ndarray:
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

    m = abs(m)
    wf = np.zeros_like(rho, dtype=np.float64)
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


def dR_drho(
        m: int,
        n: int,
        rho: np.ndarray,
        cache: dict[Any, Any] | None = None
    ) -> np.ndarray:
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


def theta_m(
        m: int,
        phi: np.ndarray
    ) -> np.ndarray:
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


def dtheta_dphi(
        m: int,
        phi: np.ndarray
    ) -> np.ndarray:
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


def zernike(
        m: int,
        n: int,
        rho: np.ndarray,
        phi: np.ndarray,
        norm: bool = False,
        cache: dict[Any, Any] | None = None
    ) -> np.ndarray:
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


def dZ_dx(
        m: int,
        n: int,
        rho: np.ndarray,
        phi: np.ndarray,
        norm: bool = False,
        cache: dict[Any, Any] | None = None
    ) -> np.ndarray:
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


def dZ_dy(
        m: int,
        n: int,
        rho: np.ndarray,
        phi: np.ndarray,
        norm: bool = False,
        cache: dict[Any, Any] | None = None
    ) -> np.ndarray:
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


def noll_to_zernike(j: int) -> tuple[int, int]:
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


def zernike_noll(
        j: int,
        rho: np.ndarray,
        phi: np.ndarray,
        norm: bool = False,
        cache: dict[Any, Any] | None = None
    ) -> np.ndarray:
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


def zernike_slope_noll(
        j: int,
        rho: np.ndarray,
        phi: np.ndarray,
        norm: bool = False,
        cache: dict[Any, Any] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
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


def zernike_slopes(
        zv: MutableMapping,
        rho: np.ndarray,
        phi: np.ndarray,
        norm: bool = False,
        use_cache: bool = True,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
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
    cache: dict[Any, Any] | None = None
    xslope = np.zeros_like(rho, dtype=np.float64)
    yslope = np.zeros_like(rho, dtype=np.float64)
    if use_cache:
        cache = {}
    for k, v in zv.items():
        mode = int(k.replace("Z", ""))
        dwx, dwy = zernike_slope_noll(mode, rho, phi, norm=norm, cache=cache)
        xslope += v * dwx
        yslope += v * dwy
    return (xslope, yslope)


def noll_normalization_vector(nmodes: int = 30) -> np.ndarray:
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
    nolls = [noll_to_zernike(j + 1) for j in range(nmodes)]
    norms = np.asanyarray([norm_coefficient(m, n) for n, m in nolls])
    return norms


def norm_coefficient(m: int , n: int) -> float:
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


def noll_coefficient(ll: int) -> float:
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
