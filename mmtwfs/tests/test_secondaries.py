# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

from unittest.mock import patch, MagicMock

from mmtwfs.config import mmtwfs_config
from mmtwfs.secondary import SecondaryFactory
from mmtwfs.custom_exceptions import WFSConfigException, WFSCommandException


def test_secondaries():
    for s in mmtwfs_config["secondary"]:
        sec = SecondaryFactory(secondary=s, test="foo")
        assert sec.test == "foo"


def test_bogus_secondary():
    try:
        sec = SecondaryFactory(secondary="bazz")
        assert sec is not None
    except WFSConfigException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False


def test_focus():
    s = SecondaryFactory(secondary="f5")
    cmd = s.focus(200.3)
    assert "200.3" in cmd


def test_m1spherical():
    s = SecondaryFactory(secondary="f5")
    cmd = s.m1spherical(200.3)
    assert "200.3" in cmd


def test_cc():
    s = SecondaryFactory(secondary="f5")
    cmd = s.cc("x", 200.3)
    assert "200.3" in cmd
    cmd = s.cc("y", 200.3)
    assert "200.3" in cmd
    try:
        cmd = s.cc("z", 200.3)
    except WFSCommandException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False


def test_zc():
    s = SecondaryFactory(secondary="f5")
    cmd = s.zc("x", 200.3)
    assert "200.3" in cmd
    cmd = s.zc("y", 200.3)
    assert "200.3" in cmd
    try:
        cmd = s.zc("z", 200.3)
    except WFSCommandException:
        assert True
    except Exception as e:
        assert e is not None
        assert False
    else:
        assert False


def test_clear():
    s = SecondaryFactory(secondary="f5")
    cmd = s.clear_m1spherical()
    assert "0.0" in cmd
    cmds = s.clear_wfs()
    for c in cmds:
        assert "0.0" in c


def test_disconnect():
    """Test disconnect method"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    s.disconnect()
    assert s.connected is False


def test_hex_sock_success():
    """Test hex_sock with successful connection"""
    s = SecondaryFactory(secondary="f5")
    s.host = "localhost"
    s.port = 5000
    with patch("socket.socket") as mock_socket:
        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        result = s.hex_sock()
        assert result == mock_sock_instance
        mock_sock_instance.connect.assert_called_once_with(("localhost", 5000))


def test_hex_sock_failure():
    """Test hex_sock with connection failure"""
    s = SecondaryFactory(secondary="f5")
    s.host = "localhost"
    s.port = 5000
    with patch("socket.socket") as mock_socket:
        mock_sock_instance = MagicMock()
        mock_sock_instance.connect.side_effect = Exception("Connection refused")
        mock_socket.return_value = mock_sock_instance
        result = s.hex_sock()
        assert result is None


def test_connect_success():
    """Test connect with successful connection"""
    s = SecondaryFactory(secondary="f5")
    with patch("mmtwfs.secondary.srvlookup") as mock_srvlookup:
        mock_srvlookup.return_value = ("hexapod.mmto.arizona.edu", 5000)
        with patch.object(s, "hex_sock") as mock_hex_sock:
            mock_sock = MagicMock()
            mock_hex_sock.return_value = mock_sock
            s.connect()
            assert s.connected is True
            mock_sock.shutdown.assert_called_once()
            mock_sock.close.assert_called_once()


def test_connect_failure():
    """Test connect with failed connection"""
    s = SecondaryFactory(secondary="f5")
    with patch("mmtwfs.secondary.srvlookup") as mock_srvlookup:
        mock_srvlookup.return_value = ("hexapod.mmto.arizona.edu", 5000)
        with patch.object(s, "hex_sock") as mock_hex_sock:
            mock_hex_sock.return_value = None
            s.connect()
            assert s.connected is False


def test_inc_offset_connected():
    """Test inc_offset when connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    with patch.object(s, "hex_sock") as mock_hex_sock:
        mock_sock = MagicMock()
        mock_hex_sock.return_value = mock_sock
        cmd = s.inc_offset("wfs", "z", 100.0)
        assert "offset_inc wfs z 100.0" in cmd
        mock_sock.sendall.assert_any_call(b"offset_inc wfs z 100.0\n")
        mock_sock.sendall.assert_any_call(b"apply_offsets\n")
        mock_sock.recv.assert_called_once()
        mock_sock.shutdown.assert_called_once()
        mock_sock.close.assert_called_once()


def test_cc_connected():
    """Test cc when connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    with patch.object(s, "hex_sock") as mock_hex_sock:
        mock_sock = MagicMock()
        mock_hex_sock.return_value = mock_sock
        cmd = s.cc("x", 10.0)
        assert "offset_cc wfs tx 10.0" in cmd
        mock_sock.sendall.assert_any_call(b"offset_cc wfs tx 10.0\n")
        mock_sock.sendall.assert_any_call(b"apply_offsets\n")


def test_zc_connected():
    """Test zc when connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    with patch.object(s, "hex_sock") as mock_hex_sock:
        mock_sock = MagicMock()
        mock_hex_sock.return_value = mock_sock
        cmd = s.zc("y", 5.0)
        assert "offset_zc wfs ty 5.0" in cmd
        mock_sock.sendall.assert_any_call(b"offset_zc wfs ty 5.0\n")
        mock_sock.sendall.assert_any_call(b"apply_offsets\n")


def test_correct_coma_connected():
    """Test correct_coma when connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    with patch.object(s, "cc") as mock_cc:
        mock_cc.return_value = "cmd"
        result = s.correct_coma(10.0, 20.0)
        assert result == (10.0, 20.0)
        mock_cc.assert_any_call("x", 10.0)
        mock_cc.assert_any_call("y", 20.0)


def test_correct_coma_not_connected():
    """Test correct_coma when not connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = False
    result = s.correct_coma(10.0, 20.0)
    assert result == (10.0, 20.0)


def test_recenter_connected():
    """Test recenter when connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    with patch.object(s, "zc") as mock_zc:
        mock_zc.return_value = "cmd"
        result = s.recenter(10.0, 20.0)
        assert result == (10.0, 20.0)
        mock_zc.assert_any_call("x", 20.0)
        mock_zc.assert_any_call("y", 10.0)


def test_recenter_not_connected():
    """Test recenter when not connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = False
    result = s.recenter(10.0, 20.0)
    assert result == (10.0, 20.0)


def test_clear_m1spherical_connected():
    """Test clear_m1spherical when connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    with patch.object(s, "hex_sock") as mock_hex_sock:
        mock_sock = MagicMock()
        mock_hex_sock.return_value = mock_sock
        cmd = s.clear_m1spherical()
        assert "offset m1spherical z 0.0" in cmd
        mock_sock.sendall.assert_any_call(b"offset m1spherical z 0.0\n")
        mock_sock.sendall.assert_any_call(b"apply_offsets\n")


def test_clear_wfs_connected():
    """Test clear_wfs when connected"""
    s = SecondaryFactory(secondary="f5")
    s.connected = True
    with patch.object(s, "hex_sock") as mock_hex_sock:
        mock_sock = MagicMock()
        mock_hex_sock.return_value = mock_sock
        cmds = s.clear_wfs()
        assert len(cmds) == 5  # tx, ty, x, y, z
        for c in cmds:
            assert "0.0" in c
