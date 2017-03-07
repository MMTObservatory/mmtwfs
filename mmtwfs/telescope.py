# Licensed under GPL3
# coding=utf-8

import numpy as np

import astropy.units as u
from astropy.io import ascii

from .config import merge_config, mmt_config
from .custom_exceptions import WFSConfigException
from .secondary import SecondaryFactory


class MMT(object):
    """
    Defines configuration and methods that pertain to the MMT's telescope and primary mirror systems
    """
    def __init__(self, secondary="f5", config={}, **kwargs):
        config = merge_config(config, dict(**kwargs))
        self.__dict__.update(merge_config(mmt_config['telescope'], config))

        self.secondary = SecondaryFactory(secondary=secondary)

        self.radius = self.diameter / 2.
        self.nmperrad = self.radius.to(u.nm).value
        self.nmperasec = self.nmperrad / 206265.

        # ratio of the size of the central obstruction of the secondary to the size of the primary
        self.obscuration = self.secondary.diameter / self.diameter

        # load table of finite element coordinates
        self.nodecoor = self.load_bcv_coordinates()
        self.n_node = len(self.nodecoor)

        # load table of actuator coordinates
        self.actcoor = self.load_actuator_coordinates()
        self.n_act = len(self.actcoor)

        # load actuator influence matrix that provides the surface displacement caused by 1 lb of force by
        # each actuator at each of self.node finite element node positions.
        self.surf2act = self.load_influence_matrix()

    def load_influence_matrix(self):
        """
        The influence of each actuator on the mirror surface has been modeled via finite element analysis.
        This method loads the influence matrix that resulted from this analysis and maps for each actuator
        the influence of 1 lb of force on the mirror surface at each finite element node.  This matrix is
        stored in a binary file for compactness and speed of loading.
        """
        surf2act = np.fromfile(self.surf2act_file, dtype=np.float32).reshape(self.n_node, self.n_act)
        return surf2act

    def load_actuator_coordinates(self):
        """
        The actuator IDs and X/Y positions in mm are stored in a simple ASCII table.  Load it using
        astropy.io.ascii, convert to units of mirror radius, and add polar coordinates.
        """
        coord = ascii.read(self.actuator_file, names=["act_id", "act_x", "act_y", "act_type"])
        for ax in ["act_x", "act_y"]:
            coord[ax] /= self.radius.to(u.mm).value
        coord['act_rho'] = np.sqrt(coord['act_x']**2 + coord['act_y']**2)
        coord['act_phi'] = np.arctan2(coord['act_y'], coord['act_x'])
        coord['act_phi'].unit = u.radian

        return coord

    def load_bcv_coordinates(self):
        """
        The BCV finite element nodes IDs and X/Y/Z positions in mm are stored in a simple ASCII table.  Load it
        using astropy.io.ascii, convert to units of mirror radius, and add polar coordinates.
        """
        coord = ascii.read(self.nodecoor_file, names=["bcv_id", "bcv_x", "bcv_y", "bcv_z"])
        for ax in ["bcv_x", "bcv_y"]:
            coord[ax] /= self.radius.to(u.mm).value
        coord['bcv_rho'] = np.sqrt(coord['bcv_x']**2 + coord['bcv_y']**2)
        coord['bcv_phi'] = np.arctan2(coord['bcv_y'], coord['bcv_x'])
        coord['bcv_phi'].unit = u.radian

        return coord
