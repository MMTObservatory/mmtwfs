# Licensed under GPL3
# coding=utf-8

"""
Classes and utilities for optical modeling and controlling the position of the secondary mirrors of the MMTO.
"""
import socket
import sys

import logging
import logging.handlers
log = logging.getLogger("")
log.setLevel(logging.INFO)

import astropy.units as u

from .utils import srvlookup
from .config import recursive_subclasses, merge_config, mmt_config
from .custom_exceptions import WFSConfigException, WFSCommandException


def SecondaryFactory(secondary="f5", config={}, **kwargs):
    """
    Build and return proper Secondary sub-class instance based on the value of 'secondary'.
    """
    config = merge_config(config, dict(**kwargs))
    secondary = secondary.lower()

    types = recursive_subclasses(Secondary)
    secondaries = [t.__name__.lower() for t in types]
    sec_map = dict(list(zip(secondaries, types)))

    if secondary not in secondaries:
        raise WFSConfigException(value="Specified secondary, %s, not valid or not implemented." % secondary)

    sec_cls = sec_map[secondary](config=config)
    return sec_cls


class Secondary(object):
    """
    Defines configuration pattern and methods common to all secondary mirror systems
    """
    def __init__(self, config={}):
        key = self.__class__.__name__.lower()
        self.__dict__.update(merge_config(mmt_config['secondary'][key], config))

        # get host/port to use for hexapod communication
        self.host, self.port = srvlookup(self.hexserv)

        # use this boolean to determine if corrections are actually to be sent
        self.connected = False

    def inc_offset(self, offset, axis, value):
        """
        Apply an incremental 'offset' of 'value' to 'axis'.
        """
        cmd = "offset_inc %s %s %f\n" % (offset, axis, value)
        if self.connected:
            sock = self.hex_sock()
            sock.sendall(cmd.encode("utf8"))
            sock.sendall(b"apply_offsets\n")
            result = sock.recv(4096)
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        return cmd

    def connect(self):
        """
        Set state to connected so that calculated corrections will be sent to the relevant systems
        """
        if self.host is not None:
            sock = self.hex_sock()
            if sock is None:
                self.connected = False
            else:
                self.connected = True
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()

    def hex_sock(self):
        """
        Set up socket for communicating with the hexapod
        """
        try:
            hex_server = (self.host, self.port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(hex_server)
        except Exception as e:
            log.error("Error connecting to hexapod server. Remaining disconnected...: %s" % e)
            return None
        return sock

    def disconnect(self):
        """
        Set state to disconnected so that corrections will be calculated, but not sent
        """
        self.connected = False

    def focus(self, foc):
        """
        Move hexapod by 'foc' microns in Z to correct focus
        """
        foc_um = u.Quantity(foc, u.um).value  # focus command must be in microsn
        log.info("Moving %s hexapod %s in Z..." % (self.__class__.__name__.lower(), foc))
        cmd = self.inc_offset("wfs", "z", foc_um)
        return cmd

    def m1spherical(self, foc):
        """
        Move hexapod by 'foc' microns in Z to correct focus. This is a special case to differentiate focus
        commands that help correct spherical aberration from normal focus commands.
        """
        foc_um = u.Quantity(foc, u.um).value  # focus command must be in microsn
        log.info("Moving %s hexapod %s in Z to correct spherical aberration..." % (self.__class__.__name__.lower(), foc))
        cmd = self.inc_offset("m1spherical", "z", foc_um)
        return cmd

    def cc(self, axis, tilt):
        """
        Move hexapod by 'tilt' arcsec about its center of curvature along 'axis'.
        This corrects coma with no image movement.
        """
        tilt = u.Quantity(tilt, u.arcsec).value
        axis = axis.lower()
        if axis not in ['x', 'y']:
            msg = "Invalid axis %s send to hexapod. Only 'x' and 'y' are valid for center-of-curvature offsets." % axis
            raise WFSCommandException(value=msg)

        log.info("Moving %s hexapod %.3f arcsec about the center of curvature along the %s axis..." % (
            self.__class__.__name__.lower(),
            tilt,
            axis)
        )
        cmd = "offset_cc wfs t%s %f\n" % (axis, tilt)
        if self.connected:
            sock = self.hex_sock()
            sock.sendall(cmd.encode("utf8"))
            sock.sendall(b"apply_offsets\n")
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        return cmd

    def zc(self, axis, tilt):
        """
        Move hexapod by 'tilt' arcsec about its center of curvature along axis.
        This corrects coma with no image movement.
        """
        tilt = u.Quantity(tilt, u.arcsec).value
        axis = axis.lower()
        if axis not in ['x', 'y']:
            msg = "Invalid axis %s send to hexapod. Only 'x' and 'y' are valid for zero-coma offsets." % axis
            raise WFSCommandException(value=msg)

        log.info("Moving %s hexapod %.3f arcsec about the zero-coma point along the %s axis..." % (
            self.__class__.__name__.lower(),
            tilt,
            axis)
        )
        cmd = "offset_zc wfs t%s %f\n" % (axis, tilt)
        if self.connected:
            sock = self.hex_sock()
            sock.sendall(cmd.encode("utf8"))
            sock.sendall(b"apply_offsets\n")
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        return cmd

    def correct_coma(self, cc_x_corr, cc_y_corr):
        """
        Apply calculated tilts to correct coma
        """
        if self.connected:
            self.cc('x', cc_x_corr)
            self.cc('y', cc_y_corr)
        return cc_x_corr, cc_y_corr

    def recenter(self, az, el):
        """
        Apply calculated az/el offsets using ZC tilts
        """
        if self.connected:
            self.zc('x', el)
            self.zc('y', az)
        return az, el

    def clear_m1spherical(self):
        """
        When clearing forces from the primary mirror, also need to clear any focus offsets applied to secondary to help
        correct spherical aberration.
        """
        log.info("Resetting hexapod's spherical aberration offset to 0...")
        cmd = "offset m1spherical z 0.0\n"
        if self.connected:
            sock = self.hex_sock()
            sock.sendall(cmd.encode("utf8"))
            sock.sendall(b"apply_offsets\n")
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        return cmd

    def clear_wfs(self):
        """
        Clear the 'wfs' offsets that get populated by WFS corrections.
        """
        log.info("Resetting hexapod's WFS offsets to 0...")
        axes = ['tx', 'ty', 'x', 'y', 'z']
        cmds = []
        if self.connected:
            sock = self.hex_sock()
            for ax in axes:
                cmd = "offset wfs %s 0.0\n" % ax
                cmds.append(cmd)
                sock.sendall(cmd.encode("utf8"))
            sock.sendall(b"apply_offsets\n")
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        return cmds

class F5(Secondary):
    """
    Defines configuration and methods specific to the F/5 secondary system
    """
    pass


class F9(Secondary):
    """
    Defines configuration and methods specific to the F/9 secondary system
    """
    pass
