# Licensed under GPL3 (see LICENSE)
# coding=utf-8

import astropy.units as u

from matplotlib.testing.decorators import cleanup

from ..zernike import ZernikeVector
from ..custom_exceptions import ZernikeException


def test_bogus_setkeys():
    zv = ZernikeVector()
    for k in ["Z2", "bazz"]:
        try:
            zv[k] = 100.0
        except KeyError:
            assert True
        except Exception as e:
            assert False
        else:
            assert False

def test_bogus_getkeys():
    zv = ZernikeVector()
    for k in ["Z2", "bazz"]:
        try:
            a = zv[k]
        except KeyError:
            assert True
        except Exception as e:
            assert False
        else:
            assert False

def test_bogus_add():
    z1 = ZernikeVector(Z04=1000)
    try:
        z2 = z1 + "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_bogus_radd():
    z1 = ZernikeVector(Z04=1000)
    try:
        z2 = "bazz" + z1
    except ZernikeException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_bogus_sub():
    z1 = ZernikeVector(Z04=1000)
    try:
        z2 = z1 - "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_bogus_rsub():
    z1 = ZernikeVector(Z04=1000)
    try:
        z2 = "bazz" - z1
    except ZernikeException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_bogus_div():
    z1 = ZernikeVector(Z04=1000)
    try:
        z2 = z1 / "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_bogus_rdiv():
    z1 = ZernikeVector(Z04=1000)
    try:
        z2 = "bazz" / z1
    except ZernikeException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_bogus_mul():
    z1 = ZernikeVector(Z04=1000)
    try:
        z2 = z1 * "bazz"
    except ZernikeException:
        assert True
    except Exception as e:
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
        assert False
    else:
        assert False

def test_repr():
    zv = ZernikeVector(Z04=1000, Z05=500, modestart=2)
    assert(len(repr(zv)) > 0)

def test_zernike_del():
    zv = ZernikeVector(Z04=1000, Z05=500, modestart=2)
    assert(len(zv) == 4)
    del zv['Z05']
    assert(len(zv) == 3)

def test_zernike_add():
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z06=500)
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
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z06=100)
    a1 = z1 * z2
    a2 = z2 * z1
    assert(a1 == a2)

def test_zernike_sub():
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z06=500)
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
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z04=100)
    a1 = z1 / z2
    a2 = z2 / z1
    assert(a1 == 1. / a2)
    a2 = z1.__rtruediv__(z2)
    assert(a1 == 1. / a2)

def test_zernike_div_nan():
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z06=100)
    a1 = z1 / z2
    a2 = z2 / z1
    assert(a1 == 1. / a2)

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

def test_labels():
    zv = ZernikeVector(Z04=100, Z05=200, Z06=-500)
    l = zv.label('Z04')
    ls = zv.shortlabel('Z04')
    assert(len(l) > len(ls))

@cleanup
def test_plots():
    zv = ZernikeVector(Z04=100, Z05=200, Z06=-500)
    f1 = zv.bar_chart()
    assert(f1 is not None)
    f2 = zv.plot_map()
    assert(f2 is not None)
    f3 = zv.plot_surface()
    assert(f3 is not None)
