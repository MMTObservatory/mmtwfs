# Licensed under GPL3 (see LICENSE)
# coding=utf-8

from ..config import mmt_config
from ..telescope import MMT


def test_telescope():
    for s in mmt_config['secondary']:
        t = MMT(secondary=s)
        assert(t.secondary.diameter == mmt_config['secondary'][s]['diameter'])
