# Licensed under GPL3 (see LICENSE)
# coding=utf-8

from ..config import mmt_config
from ..secondary import SecondaryFactory
from ..custom_exceptions import WFSConfigException


def test_secondaries():
    for s in mmt_config['secondary']:
        sec = SecondaryFactory(secondary=s, test="foo")
        assert(sec.test == "foo")

def test_bogus_secondary():
    try:
        sec = SecondaryFactory(secondary="bazz")
    except WFSConfigException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False
