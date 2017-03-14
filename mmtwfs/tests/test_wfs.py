# Licensed under GPL3 (see LICENSE)
# coding=utf-8

from ..config import mmt_config
from ..wfs import WFSFactory
from ..custom_exceptions import WFSConfigException


def test_wfses():
    for s in mmt_config['wfs']:
        wfs = WFSFactory(wfs=s, test="foo")
        assert(wfs.test == "foo")

def test_bogus_wfs():
    try:
        wfs = WFSFactory(wfs="bazz")
    except WFSConfigException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False
