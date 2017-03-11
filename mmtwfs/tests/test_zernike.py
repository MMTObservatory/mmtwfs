# Licensed under GPL3 (see LICENSE)
# coding=utf-8

from ..zernike import ZernikeVector
from ..custom_exceptions import WFSConfigException


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
    z2 = ZernikeVector(Z04=100)
    a1 = z1 * z2
    a2 = z2 * z1
    assert(a1 == a2)

def test_zernike_sub():
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z06=500)
    a1 = z1 - z2
    a2 = z2 - z1
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

def test_zernike_div():
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z04=100)
    a1 = z1 / z2
    a2 = z2 / z1
    assert(a1 == 1. / a2)

def test_zernike_div_nan():
    z1 = ZernikeVector(Z04=1000)
    z2 = ZernikeVector(Z06=100)
    a1 = z1 / z2
    a2 = z2 / z1
    assert(a1 == 1. / a2)
