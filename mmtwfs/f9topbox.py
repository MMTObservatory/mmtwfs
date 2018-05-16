# Licensed under GPL3
# coding=utf-8

"""
Classes and utilities for controlling components of the MMTO's F/9 topbox
"""
import socket

import logging
import logging.handlers

from .utils import srvlookup

log = logging.getLogger("F/9 TopBox")
log.setLevel(logging.INFO)


__all__ = ['CompMirror']


class CompMirror(object):
    """
    Defines how to query and command the comparison mirror within the F/9 topbox
    """
    def __init__(self, host=None, port=None):
        # get host/port for topbox communication. if not specified, use srvlookup to get from MMTO DNS.
        if host is None and port is None:
            self.host, self.port = srvlookup("_lampbox._tcp.mmto.arizona.edu")
        else:
            self.host = host
            self.port = port

        # use this boolean to determine if commands are actually to be sent
        self.connected = False

    def connect(self):
        """
        Set state to connected so that commands will be sent
        """
        if self.host is not None and not self.connected:
            sock = self.netsock()
            if sock is None:
                self.connected = False
            else:
                log.info("Successfully connected to F/9 topbox.")
                self.connected = True
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()

    def disconnect(self):
        """
        Set state to disconnected
        """
        self.connected = False

    def netsock(self):
        """
        Set up socket for communicating with the topbox
        """
        try:
            topbox_server = (self.host, self.port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(topbox_server)
        except Exception as e:
            log.error(f"Error connecting to topbox server. Remaining disconnected...: {e}")
            return None
        return sock

    def get_mirror(self):
        """
        Query current status of the comparison mirror
        """
        state = "N/A"
        if self.connected:
            sock = self.netsock()
            sock.sendall(b"get_mirror\n")
            result = sock.recv(4096).decode('utf8')
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            if "OUT" in result:
                state = "out"
                log.debug("Comparison mirror is OUT.")
            if "IN" in result:
                state = "in"
                log.debug("Comparison mirror is IN.")
            if "BUSY" in result:
                state = "busy"
                log.debug("Comparison mirror is BUSY.")
            if "X" in result:
                log.error("Error querying comparison mirror status.")
        else:
            log.warning("Topbox not connected. Can't get comparison mirror status.")
        return state

    def _move_mirror(self, cmd):
        """
        Send network command to topbox to move the comparison mirror in or out
        """
        state = "N/A"
        if "in" in cmd or "out" in cmd:
            if self.connected:
                sock = self.netsock()
                netcmd = f"set_mirror_exclusive {cmd}\n"
                sock.sendall(netcmd.encode("utf8"))
                result = sock.recv(4096).decode('utf8')
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
                if "X" in result:
                    log.error(f"Error sending comparison mirror command: {cmd}.")
                if "1" in result:
                    log.error(f"Comparison mirror command, {cmd}, timed out.")
                if "0" in result:
                    log.info(f"Comparison mirror successfully moved {cmd}.")
                    state = cmd
            else:
                log.warning("Topbox not connected. Can't send motion command.")
        else:
            log.error(f"Invalid comparison mirror command, {cmd}, send to topbox. Must be 'in' or 'out'")
        return state

    def mirror_in(self):
        """
        Sends command to move comparison mirror in
        """
        state = self._move_mirror("in")
        return state

    def mirror_out(self):
        """
        Sends command to move comparison mirror out
        """
        state = self._move_mirror("out")
        return state

    def toggle_mirror(self):
        """
        Checks comparison mirror state and sends appropriate command to toggle its state
        """
        status = self.get_mirror()
        if status == "in":
            status = self.mirror_out()
        elif status == "out":
            status = self.mirror_in()
        else:
            log.warning(f"Cannot toggle comparison mirror status, {status}.")
        return status
