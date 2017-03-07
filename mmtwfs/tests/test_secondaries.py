# Licensed under GPL3 (see LICENSE)
# coding=utf-8

from ..config import mmt_config
from ..secondary import SecondaryFactory


def test_secondaries():
    for s in mmt_config['secondary']:
        sec = SecondaryFactory(secondary=s, test="foo")
        assert(sec.test == "foo")
