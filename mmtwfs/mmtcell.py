# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

import asyncio
import logging

log = logging.getLogger("Cell")
log.setLevel(logging.DEBUG)


class Cell:
    """
    Async class to manage communications with the MMT primary mirror cell's control computer

    Attributes
    ----------
    writer : asyncio connection stream handler
        Handles writing and draining the stream
    reader : asyncio connection stream handler
        Handles reading the stream
    timeout : int
        Timeout for connecting to indiserver using future wait for
    read_width : int
        Amount to read from the stream per read
    """
    def __init__(self):
        self.writer = None
        self.reader = None
        self.timeout = 3
        self.read_width = 30000

    async def connect(self):
        """
        Connect to the cell
        """
        log.debug('Connecting to the cell')
        future = asyncio.open_connection("mmtcell", 5810)
        # Handle timeouts or cell not available
        try:
            self.reader, self.writer = await asyncio.wait_for(
                future,
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            log.debug("Timed out trying to connect to the cell.")
            raise
        except ConnectionRefusedError:
            log.debug("Connection to cell refused.")
            raise

        return None

    async def disconnect(self):
        """
        Disconnect from the cell

        If connected, this will close the writer and wait for it to close.
        Then it will reset the read/write stream handlers.
        """
        if self.is_connected:
            log.debug("Disconnecting from the cell")
            self.writer.close()
            await self.writer.wait_closed()

            # reset stream handlers
            self.reset()

        return None

    def reset(self):
        """
        Reset the stream handlers
        """
        log.debug("Resetting stream handlers")
        self.reader = self.writer = None
        return None

    @property
    def is_connected(self):
        """
        Returns True is writer and reader stream handlers have been initialized
        """
        return self.writer is not None and self.reader is not None

    async def send(self, msg):
        """
        Send message to cell

        Parameters
        ----------
        msg : str
            Message to send to shell

        Returns
        -------
        None
        """
        log.debug(f"Sending message: {msg}")
        # the cell code predates UTF-8 by many years so encode to ascii to be safe
        msg = msg.encode('ascii', 'ignore')
        self.writer.write(msg)
        await self.writer.drain()

        return None

    async def recv(self):
        """
        Receive message from the cell
        """
        log.debug("Receiving message from cell")
        if self.reader.at_eof():
            raise Exception("No data available from cell connection")

        response = await self.reader.read(self.read_width)
        response = response.decode('ascii')
        log.debug(f"Received from cell: {response}")

        return response

