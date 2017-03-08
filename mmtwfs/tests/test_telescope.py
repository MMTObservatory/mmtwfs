# Licensed under GPL3 (see LICENSE)
# coding=utf-8

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
