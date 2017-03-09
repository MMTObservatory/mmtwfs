# Licensed under GPL3
# coding=utf-8

import warnings
import numpy as np

import astropy.units as u
from astropy.io import ascii

from .config import merge_config, mmt_config
from .custom_exceptions import WFSConfigException
from .secondary import SecondaryFactory
from .zernike import ZernikeVector

# we need to wrap the poppy import in a context manager to trap its whinging about missing pysynphot stuff that we don't use.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import poppy


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

        # create model of MMTO pupil including secondary and secondary support obstructions
        self.pupil = self._pupil_model()

    def _pupil_model(self):
        """
        Use poppy to create a model of the pupil given the configured primary and secondary mirrors.
        """
        primary = poppy.CircularAperture(radius=self.radius.to(u.m).value)
        secondary = poppy.SecondaryObscuration(
            secondary_radius=self.secondary.diameter.to(u.m).value / 2,
            n_supports=self.n_supports,
            support_width=self.support_width.to(u.m).value,
            support_angle_offset=self.support_offset.to(u.deg).value
        )
        pup_model = poppy.CompoundAnalyticOptic(opticslist=[primary, secondary], name="MMTO")
        return pup_model

    def pupil_mask(self, size=400):
        """
        Use the pupil model to make a pupil mask that can be used as a kernel for finding pupil-like things in images
        """
        if size >= 500:
            msg = "WFS pupil sizes are currently restricted to 500 pixels in diameter or less."
            raise WFSConfigException(value=msg)

        # not sure how to get the image data out directly, but the to_fits() method gives me a path...
        pup_im = self.pupil.to_fits(npix=size)[0].data
        return pup_im

    def psf(self, zv=ZernikeVector(), wavelength=550.*u.nm, pixscale=0.01, fov=1.0):
        """
        Take a ZernikeVector and calculate resulting MMTO PSF at given wavelength.
        """
        # poppy wants the wavelength in meters
        try:
            w = wavelength.to(u.m)
        except AttributeError:
            w = wavelength  # if no unit provided, assumed meters

        # poppy wants the piston term so whack it in there if modestart isn't already 1
        if zv.modestart != 1:
            zv.modestart = 1
            zv['Z01'] = 0.0

        # poppy wants coeffs in meters
        zv.units = u.m

        # poppy wants Noll normalized coefficients
        coeffs = zv.norm_array

        osys = poppy.OpticalSystem()
        osys.add_pupil(self.pupil)
        wfe = poppy.ZernikeWFE(radius=self.radius.to(u.m).value, coefficients=coeffs)
        osys.add_pupil(wfe)
        osys.add_detector(pixelscale=pixscale, fov_arcsec=fov)
        psf = osys.calc_psf(w)
        return psf

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
