# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
config.py - Configuration data and utility functions
"""

import os
import pkg_resources

import astropy.units as u


__all__ = ['recursive_subclasses', 'merge_config', 'mmtwfs_config']


def recursive_subclasses(cls):
    """
    The __subclasses__() method only goes one level deep, but various classes can be separated by multiple
    inheritance layers. This function recursively walks through the inheritance tree and returns a list of all
    subclasses at all levels that inherit from the given class.

    Parameters
    ----------
    cls: any python Class
        Python class to get list of subclasses for

    Returns
    -------
    all_subclasses: list
        List of all subclasses that ultimately inherit from cls
    """
    all_subclasses = []

    top_subclasses = cls.__subclasses__()
    all_subclasses.extend(top_subclasses)

    for s in top_subclasses:
        all_subclasses.extend(recursive_subclasses(s))

    return all_subclasses


def merge_config(*dicts):
    """
    This takes a list of python dicts and merges them into a single dict.  It is set up to
    assure that later arguments will take precedence over earlier ones by default.

    Parameters
    ----------
    dicts: list
        List of dicts to merge into a single dict

    Returns
    -------
    updated: dict
        Arguments combined into a single dict.  Later arguments take precedence over earlier arguments
        and dicts take precedence over non-dict values.
    """
    updated = {}

    # grab all of the keys
    keys = set()
    for o in dicts:
        keys = keys.union(set(o))

    for key in keys:
        values = [o[key] for o in dicts if key in o]
        # find values that are dicts so we can recurse through them
        maps = [value for value in values if isinstance(value, dict)]
        if maps:
            updated[key] = merge_config(*maps)
        else:
            # if not a dict, then return the last value we have since later arguments
            # take precendence
            updated[key] = values[-1]
    return updated


"""
MMTO optics numbers are taken from http://www.mmto.org/sites/default/files/mmt_conv7_2.pdf.
FLWO optics numbers are taken from Deb Woods' memo. Unsure if the description is published online somewhere...
"""
mmtwfs_config = {
    "telescope": {
        "mmt": {
            "diameter": 6502.4 * u.mm,  # primary diameter
            "bcv_radius": 3228.5 * u.mm,  # radius to use when normalizing BCV finite element coordinates
            "n_supports": 4,  # number of secondary support struts
            "support_width": 0.12 * u.m,  # width of support struts in meters
            "support_offset": 45. * u.deg,  # offset of support struts in degrees
            "psf_pixel_scale": 0.02,  # arcsec/pixel
            "psf_fov": 1.0,  # arcsec
            # influence matrix to map actuator forces to surface displacement
            "surf2act_file": pkg_resources.resource_filename(__name__, os.path.join("data", "Surf2ActTEL_32.bin")),
            # coordinates of finite element nodes used in surf2act
            "nodecoor_file": pkg_resources.resource_filename(__name__, os.path.join("data", "bcv_node_coordinates.dat")),
            # coordinates of the force actuators
            "actuator_file": pkg_resources.resource_filename(__name__, os.path.join("data", "actuator_coordinates.dat")),
            "zern_map": {  # map the old zernike mode indexing scheme to the Noll scheme used in ZernikeVector
                "Z02": 0,
                "Z03": 1,
                "Z04": 2,
                "Z05": 3,
                "Z06": 4,
                "Z07": 5,
                "Z08": 6,
                "Z09": 8,
                "Z10": 9,
                "Z11": 7,
                "Z12": 11,
                "Z13": 10,
                "Z14": 13,
                "Z15": 12,
                "Z16": 16,
                "Z17": 17,
                "Z18": 14,
                "Z19": 15,
                "Z22": 18
            }
        },
        "flwo12": {
            "diameter": 1219.225 * u.mm,  # primary diameter
            "n_supports": 4,  # number of secondary support struts
            "support_width": 0.03 * u.m,  # width of support struts in meters
            "support_offset": 0. * u.deg,  # offset of support struts in degrees
            "psf_pixel_scale": 0.02,  # arcsec/pixel
            "psf_fov": 1.0  # arcsec
        }
    },
    "secondary": {
        "f5": {
            "telescope": "mmt",
            "hexserv": "_hexapod._tcp.mmto.arizona.edu",
            "diameter": 1688.0 * u.mm,  # clear aperture of secondary
            "plate_scale": 0.167 * u.mm / u.arcsec,  # plate scale of the focal plane (this is for spectroscopic mode)
            "theta_cc": 79.0 * u.nm / u.arcsec,  # nm of coma per arcsec of center-of-curvature tilt.
            "cc_trans": 24.97 * u.um / u.arcsec,  # um of hexapod translation per arcsec of center-of-curvature tilt.
            "zc_trans": 9.453 * u.um / u.arcsec,  # um of hexapod translation per arcsec of zero-coma tilt.
            "focus_trans": 40.8 * u.nm / u.um  # nm of defocus per um of hexapod Z (focus) translation.
        },
        "f9": {
            "telescope": "mmt",
            "hexserv": "_hexapod._tcp.mmto.arizona.edu",
            "diameter": 1006.7 * u.mm,
            "plate_scale": 0.284 * u.mm / u.arcsec,
            "theta_cc": 44.4 * u.nm / u.arcsec,
            "cc_trans": 13.6 * u.um / u.arcsec,
            "zc_trans": 5.86 * u.um / u.arcsec,
            "focus_trans": 34.7 * u.nm / u.um
        },
        "flwo12": {
            "telescope": "flwo12",
            "diameter": 310.295 * u.mm,  # clear aperture of secondary
            "plate_scale": 0.0455 * u.mm / u.arcsec  # plate scale of the focal plane
        }
    },
    "wfs": {
        "f5": {
            "name": "Hecto WFS",
            "telescope": "mmt",  # telescope used with WFS system
            "secondary": "f5",  # secondary used with WFS system
            "default_mode": "hecto",
            "eff_wave": 600 * u.nm,  # effective wavelength of the thruput response of the system
            "cor_coords": [251.0, 267.0],  # image coordinates of the center of rotation
            "find_fwhm": 9.0,  # FWHM for DAOfind kernel
            "find_thresh": 5.0,  # threshold for DAOfind
            "cen_thresh": 0.7,  # threshold for finding peaks in correlation image used for pupil registration
            "cen_sigma": 5.0,  # sigma of smoothing kernel used on data before pupil registration
            "cen_tol": 50.,  # distance from cor_coords allowed for a wavefront analysis to be considered potentially valid
            "rotation": 234.0 * u.deg,  # rotation of aperture locations w.r.t. the primary mirror
            "lenslet_pitch": 600 * u.um,  # width of each lenslet
            "lenslet_fl": 40 * u.mm,  # focal length of each lenslet_fl
            "pix_um": 20 * u.um,  # pixel size in micrometers
            "pix_size": 0.135 * u.arcsec,  # arcsec per WFS detector pixel
            "pup_size": 450,  # pixels
            "pup_inner": 45,  # inner obscuration radius in pixels
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 21,  # number of zernike modes to fit
            "az_parity": -1,  # E/W flip in image motion
            "el_parity": -1,  # N/S flip in image motion
            "wfs_mask": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "f5_mask.fits")),
            "reference_file": pkg_resources.resource_filename(
                __name__,
                os.path.join("data", "ref_images", "f5_hecto_ref.fits")
            ),
            "aberr_table_file": pkg_resources.resource_filename(
                __name__,
                os.path.join("data", "f5zernfield_std_curvedsurface.TXT")
            ),
            "modes": {
                "megacam": {
                    "label": "Megacam",
                    "ref_zern": {
                        "Z04": -468. * u.nm,  # defocus
                        "Z11": -80. * u.nm  # primary spherical
                    },
                },
                "hecto": {
                    "label": "Hecto",
                    "ref_zern": {
                        "Z04": -2810. * u.nm,
                        "Z11": -150. * u.nm
                    },
                },
                "mmtcam": {
                    "label": "MMTCam",
                    "ref_zern": {
                        "Z04": -500. * u.nm,
                        "Z11": -150. * u.nm
                    },
                },
                "maestro": {
                    "label": "Maestro",
                    "ref_zern": {
                        "Z04": -2820. * u.nm,
                        "Z11": -150. * u.nm
                    },
                },
                "swirc": {
                    "label": "SWIRC",
                    "ref_zern": {
                        "Z04": -2017. * u.nm,
                        "Z11": -1079. * u.nm
                    }
                }
            }
        },
        "f9": {
            "name": "F/9 WFS with Apogee Camera",
            "telescope": "mmt",  # telescope used with WFS system
            "secondary": "f9",
            "default_mode": "blue",
            "eff_wave": 780 * u.nm,  # effective wavelength of the thruput response of the system
            "lampsrv": "_lampbox._tcp.mmto.arizona.edu",
            "cor_coords": [255.0, 255.0],
            "find_fwhm": 7.0,
            "find_thresh": 5.0,
            "cen_thresh": 0.7,
            "cen_sigma": 5.0,
            "cen_tol": 50.,
            "rotation": -225. * u.deg,
            "lenslet_pitch": 625 * u.um,  # width of each lenslet
            "lenslet_fl": 45 * u.mm,  # focal length of each lenslet_fl
            "pix_um": 20 * u.um,  # pixel size in micrometers
            "pix_size": 0.119 * u.arcsec,  # old KX260e detector with 20 um pixels
            "pup_size": 420,  # pupil outer diameter in pixels
            "pup_inner": 25,  # inner obscuration radius in pixels
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 21,  # number of zernike modes to fit
            "az_parity": -1,  # E/W flip in image motion
            "el_parity": 1,  # N/S flip in image motion
            "wfs_mask": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "oldf9_mask.fits")),
            "reference_file": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "f9_ref.fits")),
            "modes": {
                "blue": {
                    "label": "Blue Channel",
                    "ref_zern": {
                        "Z04": 7982. * u.nm
                    },
                },
                "red": {
                    "label": "Red Channel",
                    "ref_zern": {
                        "Z04": 7982. * u.nm
                    },
                },
                "spol": {
                    "label": "SPOL",
                    "ref_zern": {
                        "Z04": -308. * u.nm
                    },
                }
            }
        },
        "newf9": {
            "name": "F/9 WFS with SBIG Camera",
            "telescope": "mmt",  # telescope used with WFS system
            "secondary": "f9",
            "default_mode": "blue",
            "eff_wave": 600 * u.nm,  # effective wavelength of the thruput response of the system
            "lampsrv": "_lampbox._tcp.mmto.arizona.edu",
            "cor_coords": [376.0, 434.0],
            "find_fwhm": 12.0,
            "find_thresh": 5.0,
            "cen_thresh": 0.7,
            "cen_sigma": 6.0,
            "cen_tol": 150.,
            "rotation": -225. * u.deg,
            "lenslet_pitch": 625 * u.um,  # width of each lenslet
            "lenslet_fl": 45 * u.mm,  # focal length of each lenslet_fl
            "pix_um": 5.4 * u.um * 3,  # pixel size in micrometers
            "pix_size": 0.09639 * u.arcsec,  # SBIG STT-8300 with 5.4 um pixels binned 3x3
            "pup_size": 570,  # pupil outer diameter in pixels
            "pup_inner": 25,  # inner obscuration radius in pixels
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 21,  # number of zernike modes to fit
            "az_parity": 1,  # E/W flip in image motion
            "el_parity": -1,  # N/S flip in image motion
            "wfs_mask": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "newf9_mask.fits")),
            "reference_file": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "f9_new_ref.fits")),
            "modes": {
                "blue": {
                    "label": "Blue Channel",
                    "ref_zern": {
                        "Z04": 8282. * u.nm
                    },
                },
                "red": {
                    "label": "Red Channel",
                    "ref_zern": {
                        "Z04": 8282. * u.nm
                    },
                },
                "spol": {
                    "label": "SPOL",
                    "ref_zern": {
                        "Z04": -308. * u.nm
                    },
                }
            }
        },
        "mmirs": {
            "name": "MMIRS WFS",
            "telescope": "mmt",  # telescope used with WFS system
            "secondary": "f5",
            "default_mode": None,
            "eff_wave": 750 * u.nm,  # effective wavelength of the thruput response of the system
            "cor_coords": [245.0, 253.0],
            "find_fwhm": 6.0,
            "find_thresh": 4.0,
            "cen_thresh": 0.7,
            "cen_sigma": 6.0,
            "cen_tol": 50.,
            "rotation": 180. * u.deg,  # this is referenced to camera2. camera1 is camera2+180, but is flipped by image acq
            "lenslet_pitch": 600 * u.um,  # width of each lenslet
            "lenslet_fl": 40 * u.mm,  # focal length of each lenslet_fl
            "pix_um": 13 * u.um * 2,  # pixel size in micrometers, always binned 2x2
            "pix_size": 0.156 * u.arcsec,
            "pup_size": 345,  # pixels
            "pup_inner": 40,
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 21,  # number of zernike modes to fit
            "az_parity": 1,  # E/W flip in image motion
            "el_parity": 1,  # N/S flip in image motion
            "wfs_mask": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "mmirs_mask.fits")),
            "aberr_table_file": pkg_resources.resource_filename(__name__, os.path.join("data", "mmirszernfield.tab")),
            "modes": {
                "mmirs1": {
                    "label": "Camera 1",
                    "ref_zern": {
                        "Z04": -3176. * u.nm
                    },
                    "reference_file": pkg_resources.resource_filename(
                        __name__,
                        os.path.join("data", "ref_images", "mmirs_camera1_ref.fits")
                    ),
                },
                "mmirs2": {
                    "label": "Camera 2",
                    "ref_zern": {
                        "Z04": 1059. * u.nm
                    },
                    "reference_file": pkg_resources.resource_filename(
                        __name__,
                        os.path.join("data", "ref_images", "mmirs_camera2_ref.fits")
                    ),
                }
            }
        },
        "binospec": {
            "name": "Binospec WFS",
            "telescope": "mmt",  # telescope used with WFS system
            "secondary": "f5",
            "default_mode": "binospec",
            "eff_wave": 600 * u.nm,  # effective wavelength of the thruput response of the system
            "cor_coords": [269.0, 252.0],
            "find_fwhm": 7.0,
            "find_thresh": 5.0,
            "cen_thresh": 0.7,
            "cen_sigma": 6.0,
            "cen_tol": 100.,
            "rotation": 180. * u.deg,  # per j. kansky 9/26/2017
            "lenslet_pitch": 600 * u.um,  # width of each lenslet
            "lenslet_fl": 40 * u.mm,  # focal length of each lenslet_fl
            "pix_um": 13 * u.um * 2,  # pixel size in micrometers, always binned 2x2
            "pix_size": 0.156 * u.arcsec,
            "pup_size": 300,  # pixels
            "pup_inner": 45,
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 21,  # number of zernike modes to fit
            "az_parity": 1,  # E/W flip in image motion
            "el_parity": 1,  # N/S flip in image motion
            "wfs_mask": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "bino_mask.fits")),
            "aberr_table_file": pkg_resources.resource_filename(__name__, os.path.join("data", "f5zernfield_flatsurface.tab")),
            "modes": {
                "binospec": {
                    "label": "Binospec",
                    "ref_zern": {
                        "Z04": 0.0 * u.nm
                    },
                    "reference_file": pkg_resources.resource_filename(
                        __name__,
                        os.path.join("data", "ref_images", "binospec_ref.fits")
                    )
                }
            }
        },
        "flwo12": {
            "name": "FLWO 1.2m WFS",
            "telescope": "flwo12",  # telescope used with WFS system
            "secondary": "flwo12",
            "default_mode": "default",
            "eff_wave": 600 * u.nm,  # effective wavelength of the thruput response of the system
            "cor_coords": [255.0, 255.0],
            "find_fwhm": 7.0,
            "find_thresh": 5.0,
            "cen_thresh": 0.7,
            "cen_sigma": 6.0,
            "cen_tol": 75.,
            "rotation": 0. * u.deg,
            "lenslet_pitch": 400. * u.um,  # width of each lenslet
            "lenslet_fl": 53 * u.mm,  # focal length of each lenslet_fl
            "pix_um": 20 * u.um,  # pixel size in micrometers
            "pix_size": 0.24 * u.arcsec,
            "pup_size": 420,  # pupil outer diameter in pixels
            "pup_inner": 40,  # inner obscuration radius in pixels
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 21,  # number of zernike modes to fit
            "az_parity": -1,  # E/W flip in image motion
            "el_parity": 1,  # N/S flip in image motion
            "wfs_mask": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "flwo_mask.fits")),
            "reference_file": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "LED2sec_22.fits")),
            "modes": {
                "default": {
                    "label": "FLWO 1.2m WFS",
                    "ref_zern": {
                        "Z04": 0. * u.nm
                    },
                }
            }
        }
    }
}
