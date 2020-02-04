# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import astropy.units as u

from matplotlib.testing.decorators import cleanup

from ..zernike import ZernikeVector, noll_normalization_vector, noll_coefficient, R_mn, norm_coefficient, zernike, \
    dZ_dx, dZ_dy, noll_to_zernike
from ..custom_exceptions import ZernikeException


def test_R_mn():
    r = R_mn(2, 1, np.ones(25).reshape((5, 5)))
    assert(r == 0.0)


def test_norm():
    rho = 0.5
    phi = 0.5
    m, n = 0, 2
    nc = norm_coefficient(m, n)
    for f in [zernike, dZ_dx, dZ_dy]:
        z1 = f(m, n, rho, phi, norm=False)
        z2 = f(m, n, rho, phi, norm=True)
        assert(np.isclose(z2/z1, nc))


def test_bogus_noll_to_zernike():
    try:
        noll_to_zernike(0)
    except ValueError:
        assert True
    else:
        assert False


def test_bogus_coefficient():
    try:
        noll_coefficient(0)
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_setkeys():
    zv = ZernikeVector()
    for k in ["Z2", "bazz"]:
        try:
            zv[k] = 100.0
        except KeyError:
            assert True
        except Exception as e:
            assert(e is not None)
            assert False
        else:
            assert False


def test_bogus_getkeys():
    zv = ZernikeVector()
    for k in ["Z2", "bazz"]:
        try:
            zv[k]
        except KeyError:
            assert True
        except Exception as e:
            assert(e is not None)
            assert False
        else:
            assert False


def test_bogus_add():
    z1 = ZernikeVector(Z04=1000)
    try:
        z1 + "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_radd():
    z1 = ZernikeVector(Z04=1000)
    try:
        "bazz" + z1
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_sub():
    z1 = ZernikeVector(Z04=1000)
    try:
        z1 - "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_rsub():
    z1 = ZernikeVector(Z04=1000)
    try:
        "bazz" - z1
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_div():
    z1 = ZernikeVector(Z04=1000)
    try:
        z1 / "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_rdiv():
    z1 = ZernikeVector(Z04=1000)
    try:
        "bazz" / z1
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_mul():
    z1 = ZernikeVector(Z04=1000)
    try:
        z1 * "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_pow():
    z1 = ZernikeVector(Z04=1000)
    try:
        z1 ** "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_bogus_key():
    zv = ZernikeVector()
    try:
        zv._key_to_l("bazz")
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_repr():
    zv = ZernikeVector(Z04=1000, Z05=500, modestart=2)
    assert(len(repr(zv)) > 0)
    zv.normalize()
    assert(len(repr(zv)) > 0)
    zv['Z99'] = 1.0
    assert(len(repr(zv)) > 0)


def test_str():
    zv = ZernikeVector(Z04=1000, Z05=500, modestart=2)
    assert(len(str(zv)) > 0)


def test_pprint():
    zv = ZernikeVector(Z04=1000, Z05=500, Z80=100, modestart=2)
    s = zv.pretty_print()
    assert(len(s) > 0)


def test_zernike_del():
    zv = ZernikeVector(Z04=1000, Z05=500, modestart=2)
    assert(len(zv) == 4)
    del zv['Z05']
    assert(len(zv) == 3)


def test_zernike_add():
    z1 = ZernikeVector(Z04=1000, errorbars={'Z04': 10})
    z2 = ZernikeVector(Z06=500, errorbars={'Z06': 10})
    a1 = z1 + z2
    a2 = z2 + z1
    assert(a1 == a2)


def test_zernike_add_scalar():
    z1 = ZernikeVector(Z04=1000)
    z2 = 100.
    a1 = z1 + z2
    a2 = z2 + z1
    assert(a1 == a2)


def test_zernike_mult_scalar():
    z1 = ZernikeVector(Z04=1000)
    z2 = 3.
    a1 = z1 * z2
    a2 = z2 * z1
    assert(a1 == a2)


def test_zernike_mult():
    z1 = ZernikeVector(Z04=1000, errorbars={'Z04': 10})
    z2 = ZernikeVector(Z06=100, errorbars={'Z06': 10})
    a1 = z1 * z2
    a2 = z2 * z1
    assert(a1 == a2)


def test_zernike_sub():
    z1 = ZernikeVector(Z04=1000, errorbars={'Z04': 10})
    z2 = ZernikeVector(Z06=500, errorbars={'Z06': 10})
    a1 = z1 - z2
    a2 = z2 - z1
    assert(a1 == -1*a2)
    a2 = z1.__rsub__(z2)
    assert(a1 == -1*a2)


def test_zernike_sub_scalar():
    z1 = ZernikeVector(Z04=1000)
    z2 = 500.
    a1 = z1 - z2
    a2 = z2 - z1
    assert(a1 == -1*a2)


def test_zernike_div_scalar():
    z1 = ZernikeVector(Z04=1000)
    z2 = 500.
    a1 = z1 / z2
    a2 = z2 / z1
    assert(a1 == 1. / a2)
    # test python2.x methods
    a1 = z1.__div__(z2)
    a2 = z1.__rdiv__(z2)
    assert(a1 == 1. / a2)


def test_zernike_div():
    z1 = ZernikeVector(Z04=1000, errorbars={'Z04': 10})
    z2 = ZernikeVector(Z04=100, errorbars={'Z04': 10})
    z3 = ZernikeVector(Z04=100, errorbars={'Z06': 10})
    a1 = z1 / z2
    a2 = z2 / z1
    a3 = z1 / z3
    a4 = z3 / z1
    assert(a1 == 1. / a2)
    assert(a3 == 1. / a4)
    a2 = z1.__rtruediv__(z2)
    a3 = z1.__rtruediv__(z3)
    a4 = z3.__rtruediv__(z1)
    assert(a1 == 1. / a2)
    assert(a1['Z04'] == 1. / a3['Z04'])
    assert(a4['Z04'] == 1. / a3['Z04'])


def test_zernike_div_nan():
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z06=100)
    a1 = z1 / z2
    a2 = z2 / z1
    assert(a1 == 1. / a2)


def test_zernike_pow():
    amp = 1000
    z1 = ZernikeVector(Z04=amp, errorbars={'Z04': 10})
    z2 = z1 ** 2
    assert(amp**2 == z2['Z04'].value)


def test_p2v():
    zv = ZernikeVector(Z04=1000)
    p2v = zv.peak2valley
    assert(p2v > 0.0)


def test_rms():
    zv = ZernikeVector(Z04=1000)
    rms = zv.rms
    assert(rms > 0.0)


def test_from_array():
    zv = ZernikeVector(coeffs=[0, 0, 1000], modestart=2)
    assert(len(zv) == 3)
    assert(zv['Z04'].value == 1000.0)
    zv2 = ZernikeVector(coeffs=[500., 500.], zmap={'Z05': 0, 'Z06': 1}, modestart=5)
    assert(len(zv2) == 2)
    assert(zv2['Z06'].value == 500.0)


def test_noll_vector():
    arr = noll_normalization_vector(nmodes=20)
    assert(len(arr) == 20)


def test_ignore():
    zv = ZernikeVector(Z04=1000, Z05=500, modestart=2)
    assert(len(zv) == 4)
    zv.ignore('Z05')
    assert(zv['Z05'].value == 0.0)
    zv.restore('Z05')
    assert(len(zv) == 4)
    assert(zv['Z05'].value == 500.0)


def test_rotate():
    zv = ZernikeVector(Z05=500)
    a = zv['Z05'].value
    zv.rotate(angle=180*u.deg)
    b = zv['Z05'].value
    assert(a == b)
    zv.rotate(angle=90*u.deg)
    c = zv['Z05'].value
    assert(a == -c)


def test_loadsave():
    zv = ZernikeVector(Z04=100, Z05=200, Z06=-500)
    zv.save(filename="test.json")
    zv2 = ZernikeVector(coeffs="test.json")
    for c in zv2:
        assert(zv2[c] == zv[c])
    try:
        ZernikeVector(coeffs="bogus.json")
    except ZernikeException:
        assert True
    except Exception as e:
        assert(e is not None)
        assert False
    else:
        assert False


def test_labels():
    zv = ZernikeVector(Z04=100, Z05=200, Z06=-500)
    long = zv.label('Z04')
    ls = zv.shortlabel('Z04')
    assert(len(long) > len(ls))
    ll = zv.label('Z80')
    lls = zv.shortlabel('Z80')
    assert(len(ll) == len(lls))


@cleanup
def test_plots():
    zv = ZernikeVector(Z04=100, Z05=200, Z06=-500)
    f1 = zv.bar_chart(title="bar chart", residual=100.)
    assert(f1 is not None)
    f2 = zv.plot_map()
    assert(f2 is not None)
    f3 = zv.plot_surface()
    assert(f3 is not None)
    f4 = zv.fringe_bar_chart(title="fringe bar chart")
    assert(f4 is not None)
    zv.normalize()
    zv['Z99'] = 100.0 * u.nm
    f1 = zv.bar_chart()
    assert(f1 is not None)
    f2 = zv.plot_map()
    assert(f2 is not None)
    f3 = zv.plot_surface()
    assert(f3 is not None)
    f4 = zv.fringe_bar_chart()
    assert(f4 is not None)
