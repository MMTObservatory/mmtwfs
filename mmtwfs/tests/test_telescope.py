# Licensed under GPL3 (see LICENSE)
# coding=utf-8

import os
import pkg_resources
import filecmp

import astropy.units as u

from ..config import mmt_config
from ..telescope import MMT
from ..zernike import ZernikeVector
from ..custom_exceptions import WFSConfigException

def test_telescope():
    for s in mmt_config['secondary']:
        t = MMT(secondary=s)
        assert(t.secondary.diameter == mmt_config['secondary'][s]['diameter'])

def test_pupil_mask():
    for s in mmt_config['secondary']:
        t = MMT(secondary=s)
        mask = t.pupil_mask(size=400)
        assert(mask.shape == (400, 400))
        assert(mask.max() == 1.0)
        assert(mask.min() == 0.0)

def test_bogus_pupil_mask():
    for s in mmt_config['secondary']:
        t = MMT(secondary=s)
        try:
            mask = t.pupil_mask(size=600)
        except WFSConfigException:
            assert True
        except Exception as e:
            assert False
        else:
            assert False

def test_psf():
    for s in mmt_config['secondary']:
        t = MMT(secondary=s)
        zv = ZernikeVector(Z04=500*u.nm)
        p = t.psf(zv=zv)
        p_im = p[0].data
        assert(p_im.max() < 1.0)

def test_force_file():
    t = MMT()
    # define a zernike vector with AST45 of -1000 nm and check if the correction equals the forces required to bend
    # +1000 nm of AST45 into the mirror.
    zv = ZernikeVector(Z05=-1000)
    f_table = t.bending_forces(zv=zv, gain=1.0)
    t.to_rcell(f_table, filename="forcefile")
    test_file = pkg_resources.resource_filename("mmtwfs", os.path.join("data", "test_data", "AST45_p1000.frc"))
    assert(filecmp.cmp("forcefile", test_file))
