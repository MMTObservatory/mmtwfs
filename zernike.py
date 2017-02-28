import numpy as np

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
    See <http://www.opt.indiana.edu/vsg/library/vsia/vsia-2000_taskforce/tops4_2.html>.
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


def calc_zern_infmat(subaps, nzern=20, zerncntr=None, zernrad=-1.0, singval=1.0, subapsize=22.0, pixsize=0.1):
    """
    Given a sub-aperture array pattern, calculate a matrix that converts
    image shift vectors in pixels to Zernike amplitudes and also its inverse.
    The parameters **focus**, **wavelen**, **subapsize** and **pixsize** are
    used for absolute calibration. If these are provided, the shifts in
    pixel are translated to Zernike amplitudes where amplitude has unit
    variance, i.e. the normalization used by Noll (1976).

    The data returned is a tuple of the following:
        1. Matrix to compute Zernike modes from image shifts
        2. Matrix to image shifts from Zernike amplitudes
        3. The set of Zernike polynomials used
        4. The extent of the Zernike basis in units of **subaps**

    To calculate the above mentioned matrices, we measure the x, y-slope of all Zernike modes over
    all sub-apertures, giving a matrix `zernslopes_mat` that converts slopes for each Zernike matrix:
        subap_slopes_vec = zernslopes_mat . zern_amp_vec

    To obtain pixel shifts inside the sub images we multiply these slopes in radians/subaperture by
        sfac = π * ap_width * pix_scale / 206265
    where ap_width is in pixels, pix_scale is arcsec/pixel, and 1/206265 converts arcsec to radians.
    The factor of π comes from the normalization of the integral of the Zernike modes over all radii and angles.

    We then have
        subap_shift_vec = sfac * zernslopes_mat . zern_amp_vec

    To get the inverse relation, we invert `zernslopes_mat`, giving:
        zern_amp_vec = (sfac * zernslopes_mat)^-1 . subap_shift_vec
        zern_amp_vec = zern_inv_mat . subap_shift_vec

    Parameters
    ----------
    subaps: list of 4-element list-likes
        List of subapertures formatted as (low0, high0, low1, high1)
    nzern: int (default: 20)
        Number of Zernike modes to model
    zerncenter: 2-element list-like or None (default: None)
        Coordinate to center Zernike set around. If None, use calculated center of **subaps**.
    zernrad: float (default: -1.0)
        Radius of the aperture to use. If negative, used as fraction **-zernrad**, otherwise used as radius in pixels.
    singval: float (default: 1.0)
        Percentage of singular values to take into account when inverting the matrix
    subapsize: float (default: 22.0)
        Size of single Shack-Hartmann sub-aperture in detector pixels
    pixsize: float (default: 0.1)
        Detector pixel size in arcseconds

    Returns
    -------
    Tuple of (
        shift to Zernike matrix,
        Zernike to shift matrix,
        Zernike polynomials used,
        Zernike base shape in units of **subaps**
    )
    """
    # we already know pixel size in arcsec so multiply by aperture width and convert to radians.
    sfac = np.pi * subapsize * pixsize / 206265.

    # Geometry: offset between subap pattern and Zernike modes
    sasize = np.median(subaps[:, 1::2] - subaps[:, ::2], axis=0)
    if zerncntr is None:
        zerncntr = np.mean(subaps[:, ::2], axis=0).astype(int)

    if zernrad < 0:
        pattrad = np.max(np.max(subaps[:, 1::2], 0) - np.min(subaps[:, ::2], 0))/2.0
        rad = int((pattrad*-zernrad)+0.5)
    else:
        rad = int(zernrad+0.5)
    saoffs = -zerncntr + np.r_[[rad, rad]]

    extent = zerncntr[1]-rad, zerncntr[1]+rad, zerncntr[0]-rad, zerncntr[0]+rad
    zbasis = make_zernike_basis(nzern, rad, modestart=2)

    slopes = (np.indices(sasize, dtype=float)/(np.r_[sasize].reshape(-1, 1, 1))).reshape(2, -1)
    slopes = np.vstack([slopes, np.ones(slopes.shape[1])])
    slopes_inv = np.linalg.pinv(slopes)

    zernslopes = np.r_[
        [
            [
                calc_slope(
                    zbase[subap[0]+saoffs[0]:subap[1]+saoffs[0], subap[2]+saoffs[1]:subap[3]+saoffs[1]], slopes_inv=slopes_inv
                ) for subap in subaps
            ] for zbase in zbasis['modes']
        ]
    ].reshape(nzern, -1)

    # np.linalg.pinv() takes the cutoff wrt the *maximum*, we want a cut-off
    # based on the cumulative sum, i.e. the total included power, which is
    # why we use svd() and not pinv().
    U, s, Vh = np.linalg.svd(zernslopes * sfac, full_matrices=False)
    cums = s.cumsum() / s.sum()
    nvec = np.argwhere(cums >= singval)[0][0]
    singval = cums[nvec]
    s[nvec+1:] = np.inf
    zern_inv_mat = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

    return zern_inv_mat, zernslopes*sfac, zbasis, extent
