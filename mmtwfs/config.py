# Licensed under GPL3 (see LICENSE)
# coding=utf-8

"""
config.py - Configuration data and utility functions
"""

import os
import pkg_resources

import astropy.units as u


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
Optics numbers are taken from http://www.mmto.org/sites/default/files/mmt_conv7_2.pdf
"""
mmt_config = {
    "telescope": {
        "diameter": 6502.4 * u.mm,  # primary diameter
        "bcv_radius": 3228.5 * u.mm,  # radius to use when normalizing BCV finite element coordinates
        "n_supports": 4,  # number of secondary support struts
        "support_width": 0.12 * u.m,  # width of support struts in meters
        "support_offset": 45. * u.deg,  # offset of support struts in degrees
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
    "secondary": {
        "f5": {
            "hexserv": "_hexapod._tcp.mmto.arizona.edu",
            "diameter": 1688.0 * u.mm,  # clear aperture of secondary
            "plate_scale": 0.167 * u.mm / u.arcsec, # plate scale of the focal plane (this is for spectroscopic mode)
            "theta_cc": 79.0 * u.nm / u.arcsec,  # nm of coma per arcsec of center-of-curvature tilt.
            "cc_trans": 24.97 * u.um / u.arcsec,  # um of hexapod translation per arcsec of center-of-curvature tilt.
            "zc_trans": 9.453 * u.um / u.arcsec,  # um of hexapod translation per arcsec of zero-coma tilt.
            "focus_trans": 40.8 * u.nm / u.um  # nm of defocus per um of hexapod Z (focus) translation.
        },
        "f9": {
            "hexserv": "_hexapod._tcp.mmto.arizona.edu",
            "diameter": 1006.7 * u.mm,
            "plate_scale": 0.284 * u.mm / u.arcsec,
            "theta_cc": 44.4 * u.nm / u.arcsec,
            "cc_trans": 13.6 * u.um / u.arcsec,
            "zc_trans": 5.86 * u.um / u.arcsec,
            "focus_trans": 34.7 * u.nm / u.um
        }
    },
    "wfs": {
        "f5": {
            "name": "F/5 WFS",
            "secondary": "f5",  # secondary used with WFS system
            "default_mode": "hecto",
            "cor_coords": [255.0, 255.0],  # image coordinates of the center of rotation
            "find_fwhm": 7.0,  # FWHM for DAOfind kernel
            "find_thresh": 5.0,  # threshold for DAOfind
            "rotation": 234.0 * u.deg,  # rotation of aperture locations w.r.t. the primary mirror
            "pix_size": 0.135 * u.arcsec,  # arcsec per WFS detector pixel
            "pup_size": 460,  # pixels
            "pup_inner": 65,  # inner obscuration radius in pixels
            "pup_offset": [-0.42, 0.47],  # [x, y] pupil offset from center of reference aperture pattern
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 36,  # number of zernike modes to fit
            "init_scale": 1.0,  # initial guess for scale of aperture grid w.r.t. reference
            "reference_file": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "f5_hecto_ref.fits")),
            "aberr_table_file": pkg_resources.resource_filename(__name__, os.path.join("data", "f5zernfield_std_curvedsurface.TXT")),
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
            "secondary": "f9",
            "default_mode": "blue",
            "lampsrv": "_lampbox._tcp.mmto.arizona.edu",
            "cor_coords": [255.0, 255.0],
            "find_fwhm": 7.0,
            "find_thresh": 5.0,
            "rotation": -225. * u.deg,
            "pix_size": 0.119 * u.arcsec,  # old KX260e detector with 20 um pixels
            "pup_size": 440,  # pupil outer diameter in pixels
            "pup_inner": 55,  # inner obscuration radius in pixels
            "pup_offset": [0.4, 0.75],  # [x, y] pupil offset from center of reference aperture pattern
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 36,  # number of zernike modes to fit
            "init_scale": 0.95,
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
            "secondary": "f9",
            "default_mode": "blue",
            "lampsrv": "_lampbox._tcp.mmto.arizona.edu",
            "cor_coords": [415.0, 425.0],
            "find_fwhm": 12.0,
            "find_thresh": 7.0,
            "rotation": -225. * u.deg,
            "pix_size": 0.09639 * u.arcsec,  # SBIG STT-8300 with 5.4 um pixels binned 3x3
            "pup_size": 530,  # pupil outer diameter in pixels
            "pup_inner": 65,  # inner obscuration radius in pixels
            "pup_offset": [0.0, 0.5],  # [x, y] pupil offset from center of reference aperture pattern
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 36,  # number of zernike modes to fit
            "init_scale": 0.95,
            "reference_file": pkg_resources.resource_filename(__name__, os.path.join("data", "ref_images", "f9_new_ref.fits")),
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
        "mmirs": {
            "name": "MMIRS WFS",
            "secondary": "f5",
            "default_mode": None,
            "cor_coords": [255.0, 255.0],
            "find_fwhm": 7.0,
            "find_thresh": 5.0,
            "rotation": 180. * u.deg,  # this is referenced to camera2. camera1 is camera2+180, but is flipped by image acq
            "pix_size": 0.156 * u.arcsec,
            "pup_size": 310,  # pixels
            "pup_inner": 50,
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 36,  # number of zernike modes to fit
            "init_scale": 1.0,
            "aberr_table_file": pkg_resources.resource_filename(__name__, os.path.join("data", "mmirszernfield.tab")),
            "modes": {
                "mmirs1": {
                    "label": "Camera 1",
                    "pup_offset": [0.75, 0.5],  # [x, y] pupil offset from center of reference aperture pattern
                    "ref_zern": {
                        "Z04": -1325. * u.nm
                    },
                    "reference_file": pkg_resources.resource_filename(
                        __name__,
                        os.path.join("data", "ref_images", "mmirs_camera1_ref.fits")
                    ),
                },
                "mmirs2": {
                    "label": "Camera 2",
                    "pup_offset": [0.75, 0.0],  # [x, y] pupil offset from center of reference aperture pattern
                    "ref_zern": {
                        "Z04": 1912. * u.nm
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
            "secondary": "f5",
            "default_mode": None,
            "cor_coords": [255.0, 255.0],
            "find_fwhm": 7.0,
            "find_thresh": 5.0,
            "rotation": 180. * u.deg,  # per j. kansky 9/26/2017
            "pix_size": 0.156 * u.arcsec,
            "pup_size": 300,  # pixels
            "pup_inner": 50,
            "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
            "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
            "nzern": 36,  # number of zernike modes to fit
            "init_scale": 1.0,
            "aberr_table_file": pkg_resources.resource_filename(__name__, os.path.join("data", "f5zernfield_flatsurface.tab")),
            "modes": {
                "binospec": {
                    "label": "Binospec",
                    "pup_offset": [0.5, 0.0],  # [x, y] pupil offset from center of reference aperture pattern in subap units
                    "ref_zern": {
                        "Z04": 0.0 * u.nm
                    },
                    "reference_file": pkg_resources.resource_filename(
                        __name__,
                        os.path.join("data", "ref_images", "binospec_ref.fits")
                    )
                }
            }
        }
    }
}
