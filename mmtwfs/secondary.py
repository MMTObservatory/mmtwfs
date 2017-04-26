# Licensed under GPL3
# coding=utf-8

"""
Classes and utilities for optical modeling and controlling the position of the secondary mirrors of the MMTO.
"""
import socket
import sys

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

        # use this boolean to determine if corrections are actually to be sent
        self.connected = False
        self.sock = None

    def inc_offset(self, offset, axis, value):
        """
        Apply an incremental 'offset' of 'value' to 'axis'.
        """
        cmd = "offset_inc %s %s %f" % (offset, axis, value)
        if self.connected:
            self.sock.sendall(cmd)
            self.sock.sendall("apply_offsets")
        return cmd

    def connect(self):
        """
        Set state to connected so that calculated corrections will be sent to the relevant systems
        """
        self.connected = True
        try:
            hex_server = (self.host, self.port)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(hex_server)
        except Exception as e:
            print("Error connecting to hexapod server. Remaining disconnected...: %s" % e)
            self.connected = False

    def disconnect(self):
        """
        Set state to disconnected so that corrections will be calculated, but not sent
        """
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
                self.sock = None
            except Exception as e:
                print("Error closing connection to hexapod server: %s" % e)

    def focus(self, foc):
        """
        Move hexapod by 'foc' microns in Z to correct focus
        """
        print("Moving %s hexapod %.3f microns in Z..." % (self.__class__.__name__.lower(), foc))
        cmd = self.inc_offset("wfs", "z", foc)
        return cmd

    def m1spherical(self, foc):
        """
        Move hexapod by 'foc' microns in Z to correct focus. This is a special case to differentiate focus
        commands that help correct spherical aberration from normal focus commands.
        """
        print("Moving %s hexapod %.3f microns in Z to correct spherical aberration..." % (self.__class__.__name__.lower(), foc))
        cmd = self.inc_offset("m1spherical", "z", foc)
        return cmd

    def clear_m1spherical(self):
        """
        When clearing forces from the primary mirror, also need to clear any focus offsets applied to secondary to help
        correct spherical aberration.
        """
        print("Moving %s hexapod %.3f microns in Z to correct spherical aberration..." % (self.__class__.__name__.lower(), foc))
        cmd = "offset m1spherical z 0.0"
        if self.connected:
            self.sock.sendall(cmd)
            self.sock.sendall("apply_offsets")
        return cmd

    def cc(self, axis, tilt):
        """
        Move hexapod by 'tilt' arcsec about its center of curvature along 'axis'.
        This corrects coma with no image movement.
        """
        tilt = tilt.lower()
        if tilt not in ['x', 'y']:
            msg = "Invalid axis %s send to hexapod. Only 'x' and 'y' are valid for center-of-curvature offsets." % axis
            raise WFSCommandException(value=msg)

        print("Moving %s hexapod %.3f arcsec about the center of curvature along the %s..." % (
            self.__class__.__name__.lower(),
            tilt,
            axis)
        )
        cmd = "offset_cc wfs tilt%s %f" % (axis, tilt)
        if self.connected:
            self.sock.sendall(cmd)
            self.sock.sendall("apply_offsets")
        return cmd

    def zc(self, axis, tilt):
        """
        Move hexapod by 'tilt' arcsec about its center of curvature along axis.
        This corrects coma with no image movement.
        """
        tilt = tilt.lower()
        if tilt not in ['x', 'y']:
            msg = "Invalid axis %s send to hexapod. Only 'x' and 'y' are valid for zero-coma offsets." % axis
            raise WFSCommandException(value=msg)

        print("Moving %s hexapod %.3f arcsec about the zero-coma point along the %s..." % (
            self.__class__.__name__.lower(),
            tilt,
            axis)
        )
        cmd = "offset_zc wfs tilt%s %f" % (axis, tilt)
        if self.connected:
            self.sock.sendall(cmd)
            self.sock.sendall("apply_offsets")
        return cmd


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
