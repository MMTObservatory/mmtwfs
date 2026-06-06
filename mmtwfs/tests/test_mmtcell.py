# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile

import pytest

from mmtwfs.mmtcell import Cell


class TestCell:
    """Test suite for Cell class"""

    def test_init(self):
        """Test Cell initialization"""
        cell = Cell()
        assert cell.writer is None
        assert cell.reader is None
        assert cell.timeout == 3
        assert cell.read_width == 30000

    def test_is_connected_false(self):
        """Test is_connected property when not connected"""
        cell = Cell()
        assert cell.is_connected is False

    def test_is_connected_true(self):
        """Test is_connected property when connected"""
        cell = Cell()
        cell.reader = MagicMock()
        cell.writer = MagicMock()
        assert cell.is_connected is True

    def test_reset(self):
        """Test reset method"""
        cell = Cell()
        cell.reader = MagicMock()
        cell.writer = MagicMock()
        result = cell.reset()
        assert cell.reader is None
        assert cell.writer is None
        assert result is None

    def test_connect_timeout(self):
        """Test connect with timeout"""
        cell = Cell()

        async def run_test():
            with patch("asyncio.open_connection", new_callable=AsyncMock) as mock_open:
                mock_open.side_effect = asyncio.TimeoutError()
                with pytest.raises(asyncio.TimeoutError):
                    await cell.connect()

        asyncio.run(run_test())

    def test_connect_refused(self):
        """Test connect with connection refused"""
        cell = Cell()

        async def run_test():
            with patch("asyncio.open_connection", new_callable=AsyncMock) as mock_open:
                mock_open.side_effect = ConnectionRefusedError()
                with pytest.raises(ConnectionRefusedError):
                    await cell.connect()

        asyncio.run(run_test())

    def test_disconnect_when_not_connected(self):
        """Test disconnect when not connected"""
        cell = Cell()

        async def run_test():
            result = await cell.disconnect()
            assert result is None

        asyncio.run(run_test())

    def test_send_when_not_connected(self):
        """Test send when not connected"""
        cell = Cell()

        async def run_test():
            result = await cell.send("test message")
            assert result is None

        asyncio.run(run_test())

    def test_recv_when_not_connected(self):
        """Test recv when not connected"""
        cell = Cell()

        async def run_test():
            result = await cell.recv()
            assert result is None

        asyncio.run(run_test())

    def test_ident_when_not_connected(self):
        """Test ident when not connected"""
        cell = Cell()

        async def run_test():
            result = await cell.ident()
            assert result is None

        asyncio.run(run_test())

    def test_send_force_file_not_connected(self):
        """Test send_force_file when not connected"""
        cell = Cell()

        async def run_test():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".frc", delete=False) as f:
                f.write("1\t100.0\n")
                forcefile = Path(f.name)

            try:
                result = await cell.send_force_file(forcefile)
                assert result is None
            finally:
                forcefile.unlink()

        asyncio.run(run_test())

    def test_send_force_file_not_exists(self):
        """Test send_force_file with non-existent file"""
        cell = Cell()
        cell.reader = MagicMock()
        cell.writer = MagicMock()

        async def run_test():
            result = await cell.send_force_file(Path("/nonexistent/file.frc"))
            assert result is None

        asyncio.run(run_test())

    def test_disconnect_when_connected(self):
        """Test disconnect when connected"""
        cell = Cell()

        async def run_test():
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            cell.reader = MagicMock()
            cell.writer = mock_writer

            result = await cell.disconnect()
            mock_writer.close.assert_called_once()
            assert result is None

        asyncio.run(run_test())

    def test_send_when_connected(self):
        """Test send when connected"""
        cell = Cell()

        async def run_test():
            mock_writer = MagicMock()
            mock_writer.write = MagicMock()
            mock_writer.drain = AsyncMock()
            cell.reader = MagicMock()
            cell.writer = mock_writer

            result = await cell.send("test message")
            mock_writer.write.assert_called_once_with(b"test message")
            assert result is None

        asyncio.run(run_test())

    def test_recv_at_eof(self):
        """Test recv when at end of file"""
        cell = Cell()

        async def run_test():
            mock_reader = MagicMock()
            mock_reader.at_eof = MagicMock(return_value=True)
            cell.reader = mock_reader
            cell.writer = MagicMock()

            with pytest.raises(Exception) as exc_info:
                await cell.recv()
            assert "No data available" in str(exc_info.value)

        asyncio.run(run_test())

    def test_recv_when_connected(self):
        """Test recv when connected and data available"""
        cell = Cell()

        async def run_test():
            mock_reader = MagicMock()
            mock_reader.at_eof = MagicMock(return_value=False)
            mock_reader.read = AsyncMock(return_value=b"response data")
            cell.reader = mock_reader
            cell.writer = MagicMock()

            result = await cell.recv()
            assert result == "response data"

        asyncio.run(run_test())
