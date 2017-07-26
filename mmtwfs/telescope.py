# Licensed under GPL3
# coding=utf-8

import os
import warnings

import logging
import logging.handlers
log = logging.getLogger("")
log.setLevel(logging.INFO)

import numpy as np
from scipy.misc import imrotate

import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy import visualization

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col

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

        # load actuator influence matrix that provides the surface displacement caused by 1 N of force by
        # each actuator at each of self.node finite element node positions.
        self.surf2act = self.load_influence_matrix()

        # create model of MMTO pupil including secondary and secondary support obstructions
        self.pupil = self._pupil_model()

        # use this boolean to determine if corrections are actually to be sent
        self.connected = False

        # keep track of last and total forces. a blank ZernikeVector will generate the appropriate format
        # table with all forces set to 0.
        self.last_forces = self.bending_forces(zv=ZernikeVector())
        self.total_forces = self.bending_forces(zv=ZernikeVector())
        self.last_m1focus = 0.0 * u.um
        self.total_m1focus = 0.0 * u.um

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

    def connect(self):
        """
        Set state to connected so that calculated corrections will be sent to the relevant systems
        """
        self.connected = True

    def disconnect(self):
        """
        Set state to disconnected so that corrections will be calculated, but not sent
        """
        self.connected = False

    def pupil_mask(self, rotation=0.0, size=400):
        """
        Use the pupil model to make a pupil mask that can be used as a kernel for finding pupil-like things in images
        """
        if size >= 500:
            msg = "WFS pupil sizes are currently restricted to 500 pixels in diameter or less."
            raise WFSConfigException(value=msg)

        rotation = u.Quantity(rotation, u.deg)

        # not sure how to get the image data out directly, but the to_fits() method gives me a path...
        pup_im = imrotate(self.pupil.to_fits(npix=size)[0].data.astype(float), rotation.value)
        pup_im = pup_im / pup_im.max()
        return pup_im

    def psf(self, zv=ZernikeVector(), wavelength=550.*u.nm, pixscale=0.02, fov=1.0, plot=True):
        """
        Take a ZernikeVector and calculate resulting MMTO PSF at given wavelength.
        """
        # poppy wants the wavelength in meters
        try:
            w = wavelength.to(u.m).value
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
        psf_fig = None
        if plot:
            im = psf[0].data
            psf_fig, ax = plt.subplots()
            psf_fig.set_label("PSF at {0:0.0f}".format(wavelength))
            norm = visualization.mpl_normalize.ImageNormalize(stretch=visualization.LinearStretch())
            ims = ax.imshow(psf[0].data, extent=[-fov/2, fov/2, -fov/2, fov/2], cmap=cm.magma, norm=norm)
            ax.set_xlabel("arcsec")
            ax.set_ylabel("arcsec")
            cb = psf_fig.colorbar(ims)
            cb.set_label("Fraction of Total Flux")
        return psf, psf_fig

    def bending_forces(self, zv=ZernikeVector(), gain=0.5):
        """
        Given a ZernikeVector (or similar object describing a 2D polynomial surface), calculate the actuator forces required
        to correct for the surface displacement it describes.
        """
        # we don't want to bend any tilts...
        if 'Z02' in zv:
            zv['Z02'] = 0.0
        if 'Z03' in zv:
            zv['Z03'] = 0.0

        # convert to nm...
        zv.units = u.nm

        # make sure we're not Noll normalized...
        zv.denormalize()

        # need to rotate the wavefront -90 degrees to match the BCV angle convention of +Y being 0 deg.
        zv.rotate(-90*u.deg)

        # get surface displacements at the BCV node positions. multiply the wavefront amplitude by 0.5 to account for reflection
        # off the surface.
        surf_corr = -0.5 * gain * zv.total_phase(self.nodecoor['bcv_rho'], self.nodecoor['bcv_phi'])
        if isinstance(surf_corr, float):  # means we got 0.0 from zv.total_phase()
            force_vec = np.zeros(self.n_act)
        else:
            force_vec = np.dot(surf_corr, self.surf2act).value  # remove the units that got passed through

        # return an astropy.table.Table so we can easily package actuator ID along with the force. its write() method
        # also provides a lot of flexibility in providing outputs that match the old system.
        t = Table([self.actcoor['act_id'], force_vec], names=['actuator', 'force'])
        return t

    def to_rcell(self, t, filename="zfile"):
        """
        Take table generated by bending_forces() and write it to a file of a format that matches the old SHWFS system
        """
        t.write(filename, format="ascii.no_header", delimiter="\t", formats={'force': ".1f"})

    def calculate_primary_corrections(self, zv, mask=[], gain=0.5):
        """
        Take ZernikeVector as input and determine corrections to apply to primary/secondary
        """
        # leave out tilts, focus, and coma from force calcs to start with
        def_mask = ['Z02', 'Z03', 'Z04', 'Z07', 'Z08']
        def_mask.extend(mask)
        mask = list(set(def_mask))
        zv_masked = gain * zv.copy()
        for k in mask:
            zv_masked.ignore(k)

        # to reduce the amount of force required to remove spherical aberration, we offset the r**2 part of that term by
        # bending focus into the primary and then offsetting that by adjusting the secondary.  this has the effect of
        # reducing by ~1/4 to 1/3 the total force required to correct a given amount of spherical aberration.
        #
        # this same scheme can also be extended to the higher order spherical terms as well, Z22 and Z37.
        #
        # for reference:
        #   Z04 ~ 2r**2 - 1
        #   Z11 ~ 6r**4 - 6r**2 + 1
        #   Z22 ~ 20r**6 - 30r**4 + 12r**2 - 1
        #   Z37 ~ 70r**8 - 140r**6 + 90r**4 - 20r**2 + 1
        #
        zv_masked['Z04'] = 0.5*(6.0 * zv_masked['Z11'] - 12.0 * zv_masked['Z22'] + 20.0 * zv_masked['Z37'])

        m1focus_corr = -zv_masked['Z04'] / self.secondary.focus_trans

        t = self.bending_forces(zv=zv_masked, gain=gain)

        return t, m1focus_corr

    def correct_primary(self, t, m1focus_corr, filename="zfile"):
        """
        Take force table and focus offset calculated by self.calculate_primary_corrections() and apply them, if connected.
        """
        if self.connected:
            self.secondary.m1spherical(m1focus_corr)
            self.to_rcell(t, filename=filename)
            log.info("Sending forces from %s..." % filename)
            os.system("/mmt/scripts/cell_send_forces %s" % filename)

        self.last_forces = t.copy(copy_data=True)
        self.last_m1focus = m1focus_corr.copy()
        self.total_forces['force'] += t['force']
        self.total_m1focus += m1focus_corr
        return t, m1focus_corr

    def undo_last(self, zfilename="zfile_undo"):
        """
        Undo the last set of corrections.
        """
        log.info("Undoing last set of primary mirror corrections...")
        self.last_forces['force'] *= -1
        self.last_m1focus *= -1
        if self.connected:
            self.secondary.m1spherical(self.last_m1focus)
            self.to_rcell(self.last_forces, filename=zfilename)
            os.system("/mmt/scripts/cell_send_forces %s" % zfilename)

        self.total_m1focus += self.last_m1focus
        self.total_forces['force'] += self.last_forces['force']
        return self.last_forces.copy(), self.last_m1focus.copy()

    def clear_forces(self):
        """
        Clear applied forces from primary mirror and clear any m1spherical offsets from secondary hexapod
        """
        log.info("Clearing forces and spherical aberration focus offsets...")
        if self.connected:
            self.secondary.clear_m1spherical()
            os.system("/mmt/scripts/cell_clear_forces")

        # the 'last' corrections are negations of the current total. reset the totals to 0.
        self.last_forces = self.total_forces.copy(copy_data=True)
        self.last_forces['force'] *= -1
        self.last_m1focus = -self.total_m1focus
        self.total_forces = self.bending_forces(zv=ZernikeVector())
        self.total_m1focus = 0.0

        return self.last_forces.copy(), self.last_m1focus.copy()

    def load_influence_matrix(self):
        """
        The influence of each actuator on the mirror surface has been modeled via finite element analysis.
        This method loads the influence matrix that resulted from this analysis and maps for each actuator
        the influence of 1 lb of force on the mirror surface at each finite element node.  This matrix is
        stored in a binary file for compactness and speed of loading.
        """
        surf2act = np.fromfile(self.surf2act_file, dtype=np.float32).reshape(self.n_act, self.n_node).transpose()
        return surf2act

    def load_actuator_coordinates(self):
        """
        The actuator IDs and X/Y positions in mm are stored in a simple ASCII table.  Load it using
        astropy.io.ascii, convert to units of mirror radius, and add polar coordinates.
        """
        coord = ascii.read(self.actuator_file, names=["act_id", "act_x", "act_y", "act_type"])
        for ax in ["act_x", "act_y"]:
            coord[ax] /= self.bcv_radius.to(u.mm).value
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
            coord[ax] /= self.bcv_radius.to(u.mm).value
        coord['bcv_rho'] = np.sqrt(coord['bcv_x']**2 + coord['bcv_y']**2)
        coord['bcv_phi'] = np.arctan2(coord['bcv_y'], coord['bcv_x'])
        coord['bcv_phi'].unit = u.radian

        return coord

    def plot_forces(self, t, m1focus=None, limit=100.):
        """
        Plot actuator forces given force table as output from self.bending_forces()
        """
        coords = self.actcoor
        r_fac = 0.5 * self.diameter / self.bcv_radius  # adjust for slight difference
        cmap = cm.ScalarMappable(col.Normalize(-1*limit, limit), cm.bwr)
        cmap._A = []  # grr stupid matplotlib
        fig, ax = plt.subplots()
        fig.set_label("M1 Actuator Forces")
        xcor, ycor = coords['act_x']/r_fac, coords['act_y']/r_fac
        ax.scatter(xcor, ycor, color=cmap.to_rgba(t['force']))
        for i, (x, y) in enumerate(zip(xcor, ycor)):
            ax.text(x, y+0.02, t['actuator'][i],  horizontalalignment='center', verticalalignment='bottom', size='xx-small')

        ax.set_aspect(1.0)
        circle1 = plt.Circle((0, 0), 1.0, fill=False, color='black', alpha=0.2)
        circle2 = plt.Circle((0, 0), 0.9/6.5, fill=False, color='black', alpha=0.2)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        if m1focus is not None:
            ax.set_title("M1 Focus Offset: {0:0.1f}".format(m1focus))
        ax.set_axis_off()
        cb = fig.colorbar(cmap)
        cb.set_label("Actuator Force (N)")
        return fig
