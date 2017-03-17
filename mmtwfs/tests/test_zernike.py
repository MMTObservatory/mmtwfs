# Licensed under GPL3 (see LICENSE)
# coding=utf-8

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