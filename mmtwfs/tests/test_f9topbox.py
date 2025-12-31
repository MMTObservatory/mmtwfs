# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

from unittest.mock import patch, MagicMock

from mmtwfs.f9topbox import CompMirror


class TestCompMirror:
    """Test suite for CompMirror class"""

    def test_init(self):
        """Test CompMirror initialization"""
        cm = CompMirror()
        assert cm.connected is False
        assert cm.host is None
        assert cm.port is None

        cm2 = CompMirror(host="localhost", port=5000)
        assert cm2.host == "localhost"
        assert cm2.port == 5000

    def test_disconnect(self):
        """Test disconnect method"""
        cm = CompMirror()
        cm.connected = True
        cm.disconnect()
        assert cm.connected is False

    def test_netsock_success(self):
        """Test netsock with successful connection"""
        cm = CompMirror(host="localhost", port=5000)
        with patch("socket.socket") as mock_socket:
            mock_sock_instance = MagicMock()
            mock_socket.return_value = mock_sock_instance
            result = cm.netsock()
            assert result == mock_sock_instance
            mock_sock_instance.connect.assert_called_once_with(("localhost", 5000))

    def test_netsock_failure(self):
        """Test netsock with connection failure"""
        cm = CompMirror(host="localhost", port=5000)
        with patch("socket.socket") as mock_socket:
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect.side_effect = Exception("Connection refused")
            mock_socket.return_value = mock_sock_instance
            result = cm.netsock()
            assert result is None

    def test_connect_success(self):
        """Test connect with successful connection"""
        cm = CompMirror(host="localhost", port=5000)
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_netsock.return_value = mock_sock
            cm.connect()
            assert cm.connected is True
            mock_sock.shutdown.assert_called_once()
            mock_sock.close.assert_called_once()

    def test_connect_failure(self):
        """Test connect with failed connection"""
        cm = CompMirror(host="localhost", port=5000)
        with patch.object(cm, "netsock") as mock_netsock:
            mock_netsock.return_value = None
            cm.connect()
            assert cm.connected is False

    def test_connect_with_srvlookup(self):
        """Test connect using srvlookup when host/port not specified"""
        cm = CompMirror()
        with patch("mmtwfs.f9topbox.srvlookup") as mock_srvlookup:
            mock_srvlookup.return_value = ("topbox.mmto.arizona.edu", 5000)
            with patch.object(cm, "netsock") as mock_netsock:
                mock_sock = MagicMock()
                mock_netsock.return_value = mock_sock
                cm.connect()
                mock_srvlookup.assert_called_once_with("_lampbox._tcp.mmto.arizona.edu")
                assert cm.host == "topbox.mmto.arizona.edu"
                assert cm.port == 5000

    def test_get_mirror_not_connected(self):
        """Test get_mirror when not connected"""
        cm = CompMirror()
        state = cm.get_mirror()
        assert state == "N/A"

    def test_get_mirror_out(self):
        """Test get_mirror returns out state"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"OUT"
            mock_netsock.return_value = mock_sock
            state = cm.get_mirror()
            assert state == "out"
            mock_sock.sendall.assert_called_once_with(b"get_mirror\n")

    def test_get_mirror_in(self):
        """Test get_mirror returns in state"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"IN"
            mock_netsock.return_value = mock_sock
            state = cm.get_mirror()
            assert state == "in"

    def test_get_mirror_busy(self):
        """Test get_mirror returns busy state"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"BUSY"
            mock_netsock.return_value = mock_sock
            state = cm.get_mirror()
            assert state == "busy"

    def test_get_mirror_error(self):
        """Test get_mirror handles error response"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"X"
            mock_netsock.return_value = mock_sock
            state = cm.get_mirror()
            # X is error but doesn't change state from initial "N/A"
            assert state == "N/A"

    def test_move_mirror_not_connected(self):
        """Test _move_mirror when not connected"""
        cm = CompMirror()
        state = cm._move_mirror("in")
        assert state == "N/A"

    def test_move_mirror_invalid_command(self):
        """Test _move_mirror with invalid command"""
        cm = CompMirror()
        # Don't set connected=True, invalid command returns N/A without socket ops
        state = cm._move_mirror("invalid")
        assert state == "N/A"

    def test_move_mirror_success(self):
        """Test _move_mirror with successful move"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"0"
            mock_netsock.return_value = mock_sock
            state = cm._move_mirror("in")
            assert state == "in"
            mock_sock.sendall.assert_any_call(b"set_mirror_exclusive in\n")

    def test_move_mirror_timeout(self):
        """Test _move_mirror with timeout response"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"1"
            mock_netsock.return_value = mock_sock
            state = cm._move_mirror("out")
            assert state == "N/A"

    def test_move_mirror_error(self):
        """Test _move_mirror with error response"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"X"
            mock_netsock.return_value = mock_sock
            state = cm._move_mirror("in")
            assert state == "N/A"

    def test_mirror_in(self):
        """Test mirror_in method"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"0"
            mock_netsock.return_value = mock_sock
            state = cm.mirror_in()
            assert state == "in"

    def test_mirror_out(self):
        """Test mirror_out method"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "netsock") as mock_netsock:
            mock_sock = MagicMock()
            mock_sock.recv.return_value = b"0"
            mock_netsock.return_value = mock_sock
            state = cm.mirror_out()
            assert state == "out"

    def test_toggle_mirror_from_in(self):
        """Test toggle_mirror when mirror is in"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "get_mirror") as mock_get:
            mock_get.return_value = "in"
            with patch.object(cm, "mirror_out") as mock_out:
                mock_out.return_value = "out"
                state = cm.toggle_mirror()
                assert state == "out"
                mock_out.assert_called_once()

    def test_toggle_mirror_from_out(self):
        """Test toggle_mirror when mirror is out"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "get_mirror") as mock_get:
            mock_get.return_value = "out"
            with patch.object(cm, "mirror_in") as mock_in:
                mock_in.return_value = "in"
                state = cm.toggle_mirror()
                assert state == "in"
                mock_in.assert_called_once()

    def test_toggle_mirror_unknown_state(self):
        """Test toggle_mirror with unknown state"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        with patch.object(cm, "get_mirror") as mock_get:
            mock_get.return_value = "busy"
            state = cm.toggle_mirror()
            assert state == "busy"

    def test_move_mirror_invalid_connected(self):
        """Test _move_mirror with invalid command when connected (logs error)"""
        cm = CompMirror(host="localhost", port=5000)
        cm.connected = True
        # Invalid command doesn't match "in" or "out", so logs error and returns N/A
        state = cm._move_mirror("bogus")
        assert state == "N/A"
