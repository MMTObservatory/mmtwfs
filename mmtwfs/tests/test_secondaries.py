# Licensed under GPL3 (see LICENSE)
# coding=utf-8

from ..config import mmt_config
from ..secondary import SecondaryFactory
from ..custom_exceptions import WFSConfigException, WFSCommandException


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

def test_connect():
    s = SecondaryFactory(secondary='f5')
    try:
        s.connect()
    except:
        assert(not s.connected)
    finally:
        s.disconnect()
    assert(not s.connected)

def test_focus():
    s = SecondaryFactory(secondary='f5')
    cmd = s.focus(200.3)
    assert("200.3" in cmd)

def test_m1spherical():
    s = SecondaryFactory(secondary='f5')
    cmd = s.m1spherical(200.3)
    assert("200.3" in cmd)

def test_cc():
    s = SecondaryFactory(secondary='f5')
    cmd = s.cc('x', 200.3)
    assert("200.3" in cmd)
    cmd = s.cc('y', 200.3)
    assert("200.3" in cmd)
    try:
        cmd = s.cc('z', 200.3)
    except WFSCommandException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_zc():
    s = SecondaryFactory(secondary='f5')
    cmd = s.zc('x', 200.3)
    assert("200.3" in cmd)
    cmd = s.zc('y', 200.3)
    assert("200.3" in cmd)
    try:
        cmd = s.zc('z', 200.3)
    except WFSCommandException:
        assert True
    except Exception as e:
        assert False
    else:
        assert False

def test_clear():
    s = SecondaryFactory(secondary='f5')
    cmd = s.clear_m1spherical()
    assert("0.0" in cmd)
    cmds = s.clear_wfs()
    for c in cmds:
        assert("0.0" in c)
