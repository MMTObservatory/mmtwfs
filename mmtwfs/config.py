# Licensed under GPL3 (see LICENSE)

"""
config.py - Configuration data and utility functions
"""

import os

import astropy.units as u

wfsroot = os.environ.get("WFSROOT")


def default_wfsroot(directory=None):
    """
    The MMT WFS code, configuration, and data files are all configured to live in a directory
    defined by the environment variable $WFSROOT.  Map that to "wfsroot" internally and use this
    function to set it explicitly.
    """
    global wfsroot
    if directory is not None:
        wfsroot = directory
    return wfsroot


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
            updated[key] = merge_data(*maps)
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
        "n_act": 104,  # number of actuator nodes
        "n_node": 3222,  # number of finite element nodes that sample mirror surface
        "surf2act": os.path.join(wfsroot, "Surf2ActTEL_32.bin"),  # influence matrix to map actuator forces to surface displacement
        "nodecoor": os.path.join(wfsroot, "bcv_node_coordinates.dat")  # coordinates of finite element nodes used in surf2act
    },
    "secondary": {
        "f5": {
            "diameter": 1688.0 * u.mm,  # clear aperture of secondary
            "theta_cc": 79.0 * u.nm / u.arcsec,  # nm of coma per arcsec of center-of-curvature tilt.
            "cc_trans": 25.0 * u.um / u.arcsec,  # um of hexapod translation per arcsec of center-of-curvature tilt.
            "zc_trans": 9.45 * u.um / u.arcsec,  # um of hexapod translation per arcsec of zero-coma tilt.
            "focus_trans": 40.8 * u.nm / u.um  # nm of defocus per um of hexapod Z (focus) translation.
        },
        "f9": {
            "diameter": 1006.7 * u.mm,
            "theta_cc": 44.4 * u.nm / u.arcsec,
            "cc_trans": 13.6 * u.um / u.arcsec,
            "zc_trans": 5.86 * u.um / u.arcsec,
            "focus_trans": 34.7 * u.nm / u.um
        }
    }
    "wfs": {
        "m1_gain": 0.5,  # default gain to apply to primary mirror corrections
        "m2_gain": 1.0,  # default gain to apply to secondary mirror corrections
        "f5": {
            "secondary": "f5",  # secondary used with WFS system
            "rotation": 234.0 * u.deg,  # rotation of aperture locations w.r.t. the primary mirror
            "pix_size": 0.135 * u.arcsec,  # arcsec per WFS detector pixel
            "pup_size": 450,  # pixels
            "reference_zern": {
                "megacam": {
                    "Z04": -468. * u.nm,  # defocus
                    "Z11": -80. * u.nm  # primary spherical
                },
                "hecto": {
                    "Z04": -2810. * u.nm,
                    "Z11": -150. * u.nm
                },
                "maestro": {
                    "Z04": -2820. * u.nm,
                    "Z11": -150. * u.nm
                },
                "swirc": {
                    "Z04": -2017. * u.nm,
                    "Z11": -1079. * u.nm
                }
            }
        },
        "f9": {
            "secondary": "f9",
            "rotation": -225. * u.deg,
            "pix_size": 0.119 * u.arcsec,  # old KX260e detector with 20 um pixels
            "pup_size": 420,  # pixels
            "reference_zern": {
                "blue": {
                    "Z04": 7982. * u.nm
                },
                "red": {
                    "Z04": 7982. * u.nm
                },
                "spol": {
                    "Z04": -308. * u.nm
                }
            }
        },
        "mmirs": {
            "secondary": "f5",
            "rotation": 180. * u.deg,  # this is referenced to camera2. camera1 is camera2+180, but is flipped by image acq
            "pix_size": 0.156 * u.arcsec,
            "pup_size": 360,  # pixels
            "reference_zern": {
                "mmirs1": {
                    "Z04": -2918. * u.nm
                },
                "mmirs2": {
                    "Z04": 1912. * u.nm
                }
            }
        }
    }
}
