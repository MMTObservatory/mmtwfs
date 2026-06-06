# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

from unittest.mock import patch, MagicMock

from mmtwfs.utils import srvlookup


class TestSrvlookup:
    """Test suite for srvlookup function"""

    def test_srvlookup_success(self):
        """Test successful SRV lookup"""
        with patch("mmtwfs.utils.resolver.resolve") as mock_resolve:
            mock_response = MagicMock()
            mock_record = MagicMock()
            mock_record.target.to_text.return_value = "server.example.com"
            mock_record.port = 5000
            mock_response.__getitem__ = MagicMock(return_value=mock_record)
            mock_resolve.return_value = mock_response

            host, port = srvlookup("_test._tcp.example.com")
            assert host == "server.example.com"
            assert port == 5000
            mock_resolve.assert_called_once_with("_test._tcp.example.com", "SRV")

    def test_srvlookup_failure(self):
        """Test SRV lookup failure returns None"""
        with patch("mmtwfs.utils.resolver.resolve") as mock_resolve:
            mock_resolve.side_effect = Exception("DNS lookup failed")

            host, port = srvlookup("_test._tcp.example.com")
            assert host is None
            assert port is None
