#!/usr/bin/env python

from datetime import datetime
import traceback
import functools
import multiprocessing
from multiprocessing import Pool

import pytz
import time
import os
import sys
from pathlib import Path

import argparse

import numpy as np
import scipy
import pandas as pd

import matplotlib
from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt

from astropy import stats
from astropy.io import fits, ascii
from astropy.table import Column
from astropy.time import Time

import astropy.units as u
from astropy.io import fits
from mmtwfs.wfs import WFSFactory

import logging


log = logging.getLogger('WFS Reanalyze')
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

tz = pytz.timezone("America/Phoenix")

# instantiate all of the WFS systems...
wfs_keys = ['f9', 'newf9', 'f5', 'mmirs', 'binospec']
wfs_systems = {}
wfs_names = {}
for w in wfs_keys:
    wfs_systems[w] = WFSFactory(wfs=w)
    wfs_names[w] = wfs_systems[w].name

# give mmirs a default
wfs_systems['mmirs'].default_mode = 'mmirs1'

# map f9 to oldf9
wfs_systems['oldf9'] = wfs_systems['f9']

# loosen mmirs centering tolerance to deal with past camera misalignments
wfs_systems['mmirs'].cen_tol = 120.
wfs_systems['binospec'].cen_tol = 120.


def check_image(f, wfskey=None):
    hdr = {}
    with fits.open(f, output_verify="ignore") as hdulist:
        for h in hdulist:
            hdr.update(h.header)
        data = hdulist[-1].data

    if 'CURR_TEMP' in hdr:
        hdr['OSSTEMP'] = hdr['CURR_TEMP']

    # if wfskey is None, figure out which WFS from the header info...
    if wfskey is None:
        # check for MMIRS
        if 'WFSNAME' in hdr:
            if 'mmirs' in hdr['WFSNAME']:
                wfskey = 'mmirs'
        if 'mmirs' in f.name:
            wfskey = 'mmirs'

        # check for binospec
        if 'bino' in f.name or 'wfs_ff_cal_img' in f.name:
            wfskey = 'binospec'
        if 'ORIGIN' in hdr:
            if 'Binospec' in hdr['ORIGIN']:
                wfskey = 'binospec'

        # check for new F/9
        if 'f9wfs' in f.name:
            wfskey = 'newf9'
        if 'OBSERVER' in hdr:
            if 'F/9 WFS' in hdr['OBSERVER']:
                wfskey = 'newf9'
        if wfskey is None and 'CAMERA' in hdr:
            if 'F/9 WFS' in hdr['CAMERA']:
                wfskey = 'newf9'

        # check for old F/9
        if 'INSTRUME' in hdr:
            if 'Apogee' in hdr['INSTRUME']:
                wfskey = 'oldf9'
        if 'DETECTOR' in hdr:
            if 'Apogee' in hdr['DETECTOR']:
                wfskey = 'oldf9'

        # check for F/5 (hecto)
        if wfskey is None and 'SEC' in hdr:  # mmirs has SEC in header as well and is caught above
            if 'F5' in hdr['SEC']:
                wfskey = 'f5'

        # some early F/5 data had no real id in their headers...
        if wfskey is None and Path(f.parent / "F5").exists():
            wfskey = 'f5'

    if wfskey is None:
        # if wfskey is still None at this point, whinge.
        log.error(f"Can't determine WFS for {f.name}...")

    if 'AIRMASS' not in hdr:
        if 'SECZ' in hdr:
            hdr['AIRMASS'] = hdr['SECZ']
        else:
            hdr['AIRMASS'] = np.nan

    if 'EXPTIME' not in hdr:
        hdr['EXPTIME'] = np.nan

    # we need to fix the headers in all cases to have a proper DATE-OBS entry with
    # properly formatted FITS timestamp.  in the meantime, this hack gets us what we need
    # for analysis in pandas.
    dtime = None
    if 'DATEOBS' in hdr:
        dateobs = hdr['DATEOBS']
        if 'UT' in hdr:
            ut = hdr['UT'].strip()
        elif 'TIME-OBS' in hdr:
            ut = hdr['TIME-OBS']
        else:
            ut = "07:00:00"  # midnight
        timestring = dateobs + " " + ut + " UTC"
        if '-' in timestring:
            dtime = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S %Z")
        else:
            dtime = datetime.strptime(timestring, "%a %b %d %Y %H:%M:%S %Z")

    else:
        if wfskey == "oldf9":
            d = hdr['DATE-OBS']
            if '/' in d:
                day, month, year = d.split('/')
                year = str(int(year) + 1900)
                timestring = year + "-" + month + "-" + day + " " + hdr['TIME-OBS'] + " UTC"
            else:
                timestring = d + " " + hdr['TIME-OBS'] + " UTC"
            dtime = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S %Z")
        else:
            if 'DATE-OBS' in hdr:
                timestring = hdr['DATE-OBS'] + " UTC"
                try:
                    dtime = datetime.strptime(timestring, "%Y-%m-%dT%H:%M:%S.%f %Z")
                except:
                    dtime = datetime.strptime(timestring, "%Y-%m-%dT%H:%M:%S %Z")
                # mmirs uses local time in this header pre-2019
                if wfskey == 'mmirs' and dtime < datetime.fromisoformat("2019-01-01T12:00:00"):
                    local_dt = tz.localize(dtime)
                    dtime = local_dt.astimezone(pytz.utc)
            else:
                if 'MJD' in hdr:
                    dtime = Time(hdr['MJD'], format='mjd').to_datetime()
                else:
                    dt = datetime.fromtimestamp(f.stat().st_ctime)
                    local_dt = tz.localize(dt)
                    dtime = local_dt.astimezone(pytz.utc)

    if dtime is None:
        log.error(f"No valid timestamp in header for {f.name}...")
        obstime = None
    else:
        obstime = dtime.isoformat().replace('+00:00', '')

    hdr['WFSKEY'] = wfskey
    hdr['OBS-TIME'] = obstime
    return data, hdr


def process_image(f, force=False):
    """
    Process FITS file, f, to get info we want from the header and then analyse it with the
    appropriate WFS instance. Return results in a comma-separated line that will be collected
    and saved in a CSV file.
    """
    if "Ref" in str(f) or "sog" in str(f):
        return None

    outfile = f.parent / (f.stem + ".output")
    if not force and Path.exists(outfile):
        log.info(f"Already processed {f.name}, loading previous data...")
        with open(outfile, 'r') as fp:
            line = fp.readlines()[0]
            return line

    try:
        data, hdr = check_image(f)
    except Exception as e:
        log.error(f"Problem checking {f}: {e}")
        return None

    wfskey = hdr['WFSKEY']
    obstime = hdr['OBS-TIME']
    airmass = hdr['AIRMASS']
    exptime = hdr['EXPTIME']
    az = hdr.get('AZ', np.nan)
    el = hdr.get('EL', np.nan)
    tiltx = hdr.get('TILTX', np.nan)
    tilty = hdr.get('TILTY', np.nan)
    transx = hdr.get('TRANSX', np.nan)
    transy = hdr.get('TRANSY', np.nan)
    focus = hdr.get('FOCUS', np.nan)
    if np.isnan(focus) and 'TRANSZ' in hdr:
        focus = hdr.get('TRANSZ', np.nan)
    osst = hdr.get('OSSTEMP', np.nan)
    if 'OUT_T' in hdr:
        outt = hdr.get('OUT_T', np.nan)
    else:
        outt = hdr.get('T_OUT', np.nan)
    if 'CHAM_T' in hdr:
        chamt = hdr.get('CHAM_T', np.nan)
    else:
        chamt = hdr.get('T_CHAM', np.nan)

    # being conservative here and only using data that has proper slope determination
    # and wavefront solution. also want to get statistics on the quality of the wavefront fits.
    try:
        results = wfs_systems[wfskey].measure_slopes(str(f), plot=False)
    except:
        log.error(f"Problem analyzing {f.name}...")
        results = {}
        results['slopes'] = None

    if results['slopes'] is not None:
        try:
            zresults = wfs_systems[wfskey].fit_wavefront(results, plot=False)
            zv = zresults['zernike']
            focerr = wfs_systems[wfskey].calculate_focus(zv)
            cc_x_err, cc_y_err = wfs_systems[wfskey].calculate_cc(zv)
            line = f"{obstime},{wfskey},{f.name},{exptime},{airmass},{az},{el},{osst},{outt},{chamt},{tiltx},{tilty},{transx},{transy},{focus},{focerr.value},{cc_x_err.value},{cc_y_err.value},{results['xcen']},{results['ycen']},{results['seeing'].value},{results['raw_seeing'].value},{results['fwhm']},{zresults['zernike_rms'].value},{zresults['residual_rms'].value}\n"
            zfile = f.parent / (f.stem + ".reanalyze.zernike")
            zresults['zernike'].save(filename=zfile)
            spotfile = f.parent / (f.stem + ".spots.csv")
            results['spots'].write(spotfile, overwrite=True)
            with open(outfile, 'w') as fp:
                fp.write(line)
            return line
        except Exception as e:
            log.error(f"Problem fitting wavefront for {f.name}: {e}")
            return None
    else:
        return None


def main():
    """
    Take directories as argument and go through each one to process files in parallel using Pool.map()
    """
    parser = argparse.ArgumentParser(description='Utility for parallelized batch reprocessing of WFS data')

    parser.add_argument(
        '-r', '--rootdir',
        metavar="<root dir>",
        help="Directory containing WFS data. Defaults to current working directory.",
        default="."
    )

    parser.add_argument(
        '-d', '--dirs',
        metavar="<glob>",
        help="Glob of directories to process. Defaults to *.",
        default="*"
    )

    parser.add_argument(
        '--forcedir',
        help="Force rebuild of CSV for a directory",
        action='store_true'
    )

    parser.add_argument(
        '--force',
        help="Force reanalysis of individual data files",
        action="store_true"
    )

    parser.add_argument(
        '-n', '--nproc',
        metavar="<# processes>",
        help="Number of parallel processes. Defaults to half number of available cores.",
        default=int(multiprocessing.cpu_count()/2)  # MKL uses a lot of threads so best to limit Pool to half available cores
    )

    args = parser.parse_args()

    rootdir = Path(args.rootdir)

    log.info(f"Using {args.nproc} cores...")

    dirs = sorted(list(rootdir.glob(args.dirs)))  # pathlib, where have you been all my life!
    csv_header = "time,wfs,file,exptime,airmass,az,el,osst,outt,chamt,tiltx,tilty,transx,transy,focus,focerr,cc_x_err,cc_y_err,xcen,ycen,seeing,raw_seeing,fwhm,wavefront_rms,residual_rms\n"
    slow = False
    for d in dirs:
        if d.is_dir():
            if not args.forcedir and Path.exists(d / "reanalyze_results.csv"):
                log.info(f"Already processed {d.name}...")
            else:
                try:
                    lines = []
                    lines.append(csv_header)
                    night = int(d.name)  # valid WFS directories are ints of the form YYYYMMDD. if not this form, int barfs
                    fitsfiles = d.glob("*.fits")
                    log.info(f"Processing {d}...")
                    if slow:
                        plines = []
                        for f in fitsfiles:
                            log.debug(f"Processing {f}...")
                            l = process_image(f, force=args.force)
                            plines.append(l)
                    else:
                        with Pool(processes=args.nproc) as pool:
                            plines = pool.map(process_image, fitsfiles)  # plines comes out in same order as fitslines!

                    plines = list(filter(None.__ne__, plines))  # trim out any None entries
                    if len(plines) > 0:
                        lines.extend(plines)
                        with open(d / "reanalyze_results.csv", "w") as f:
                            f.writelines(lines)

                except ValueError as e:  # this means running int(d.name) failed so it's not a valid directory...
                    log.warn(f"Skipping %s... ({e})" % d.name)
