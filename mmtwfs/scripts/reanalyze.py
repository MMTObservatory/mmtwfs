#!/usr/bin/env python

from datetime import datetime
import concurrent.futures
import multiprocessing
from functools import partial
import pytz
from pathlib import Path
import argparse
import warnings
import logging

import coloredlogs

import matplotlib

import numpy as np

import astropy.units as u
from astropy.time import Time

from astropy.io import fits
from mmtwfs.wfs import WFSFactory
from mmtwfs.zernike import ZernikeVector

from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

# Force non-interactive backend
matplotlib.use('Agg')

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

log = logging.getLogger('WFS Reanalyze')
coloredlogs.install(level='INFO', logger=log)

tz = pytz.timezone("America/Phoenix")


# instantiate all of the WFS systems...
CSV_HEADER = "time,wfs,file,exptime,airmass,az,el,osst,outt,chamt,tiltx,tilty,"\
    "transx,transy,focus,focerr,cc_x_err,cc_y_err,xcen,ycen,seeing,raw_seeing,"\
    "vlt_seeing,raw_vlt_seeing,ellipticity,fwhm,wavefront_rms,residual_rms,method,delivered_iq\n"
CSV_COLUMNS = CSV_HEADER.strip().split(",")


def delivered_iq(wfskey, zv, seeing):
    """
    FWHM in arcsec of the delivered PSF at 500 nm: the optics PSF from the measured wavefront convolved with the
    seeing as observed. NaN when there's no usable seeing or the PSF can't be calculated.
    """
    if not np.isfinite(seeing) or seeing <= 0:
        return np.nan
    try:
        psf, _ = wfs_systems[wfskey].telescope.psf(zv, band="500nm", seeing=seeing * u.arcsec, plot=False)
    except Exception as e:
        log.error(f"Problem calculating delivered IQ: {e}")
        return np.nan
    return psf.delivered_fwhm.to_value(u.arcsec)


def upgrade_cached_line(line, zfile=None):
    """
    Bring a line from an .output file written by an older version up to the current CSV_HEADER. Lines from before
    the method column were all full analyses. Lines from before the delivered_iq column get it from the wavefront
    saved in zfile.
    """
    fields = line.strip().split(",")
    if len(fields) == len(CSV_COLUMNS) - 2:
        fields.append("full")
    if len(fields) == len(CSV_COLUMNS) - 1:
        iq = np.nan
        if fields[CSV_COLUMNS.index("method")] == "full" and zfile is not None and Path(zfile).exists():
            try:
                zv = ZernikeVector()
                zv.load(filename=zfile)
                seeing = float(fields[CSV_COLUMNS.index("raw_vlt_seeing")])
                iq = delivered_iq(fields[CSV_COLUMNS.index("wfs")], zv, seeing)
            except Exception as e:
                log.error(f"Problem calculating delivered IQ from {zfile}: {e}")
        fields.append(str(iq))
        return ",".join(fields) + "\n"
    return line


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
wfs_systems['oldf9'].cen_tol = 75.
wfs_systems['f5'].cen_tol = 75.


def check_image(f, wfskey=None):
    hdr = {}
    with fits.open(f, ignore_missing_simple=True) as hdulist:
        hdulist.verify('silentfix')
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

        if wfskey == 'mmirs':
            if 'CAMERA' not in hdr:
                if hdr['WFSNAME'] == 'mmirs1':
                    hdr['CAMERA'] = 1
                else:
                    hdr['CAMERA'] = 2

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

        # the new pyindi interface specifies F/5 WFS in the header
        if 'INSTRUME' in hdr:
            if 'F/5 WFS' in hdr['INSTRUME']:
                wfskey = 'f5'

        # also try for early old F/9 data...
        if wfskey is None and Path(f.parent / "F9").exists():
            wfskey = 'oldf9'

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
        elif wfskey == "newf9":
            par_dir = f.parent.name
            if "20" in par_dir:
                timestring = f"{par_dir} {hdr['UT']} UTC"
                dtime = datetime.strptime(timestring, "%Y%m%d %H:%M:%S %Z")
            elif 'MJD' in hdr:
                dtime = Time(hdr['MJD'], format='mjd').to_datetime()
                log.info(f"newf9 got {dtime} from {hdr['MJD']}...")
            else:
                dt = datetime.fromtimestamp(f.stat().st_ctime)
                local_dt = tz.localize(dt)
                dtime = local_dt.astimezone(pytz.utc)
        else:
            if 'DATE-OBS' in hdr:
                timestring = hdr['DATE-OBS'] + " UTC"
                try:
                    dtime = datetime.strptime(timestring, "%Y-%m-%dT%H:%M:%S.%f %Z")
                except Exception:
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


def process_image(f, force=False, retry_failed=False):
    """
    Process FITS file, f, to get info we want from the header and then analyse it with the
    appropriate WFS instance. Return results in a comma-separated line that will be collected
    and saved in a CSV file.

    With retry_failed, files marked as failed by an earlier run are analyzed again while cached
    results for good files are reused. The failed marker is removed first and is recreated if
    the file fails again.
    """
    if "Ref" in str(f) or "sog" in str(f) or "coadded" in str(f):
        return None

    outfile = f.parent / (f.stem + ".output")
    failed = f.parent / (f.stem + ".failed")
    if not force and Path.exists(outfile):
        log.info(f"Already processed {f.name}, loading previous data...")
        with open(outfile, 'r') as fp:
            lines = fp.readlines()

        if len(lines) > 0:
            line = upgrade_cached_line(lines[0], zfile=f.parent / (f.stem + ".reanalyze.zernike"))
            if line != lines[0]:
                with open(outfile, 'w') as fp:
                    fp.write(line)
            return line

    if not force and Path.exists(failed):
        if not retry_failed:
            log.info(f"Already failed processing {f.name}, skipping...")
            return None
        log.info(f"Retrying {f.name}, which failed previously...")
        failed.unlink()

    try:
        data, hdr = check_image(f)
    except Exception as e:
        log.error(f"Problem checking {f}: {e}")
        failed.touch()  # mark this file as failed
        return None

    log.info(f"Processing {f.name}...")
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
    except Exception as e:
        log.error(f"Problem analyzing {f.name}: {e}")
        failed.touch()
        results = {}
        results['slopes'] = None

    if results['slopes'] is not None:
        try:
            zresults = wfs_systems[wfskey].fit_wavefront(results, plot=False)
            zv = zresults['zernike']
            focerr = wfs_systems[wfskey].calculate_focus(zv)
            cc_x_err, cc_y_err = wfs_systems[wfskey].calculate_cc(zv)
            iq = delivered_iq(wfskey, zresults['zernike'], results['raw_vlt_seeing'].to_value(u.arcsec))
            line = f"{obstime},{wfskey},{f.name},{exptime},{airmass},{az},{el},{osst},{outt}," \
                f"{chamt},{tiltx},{tilty},{transx},{transy},{focus},{focerr.value},{cc_x_err.value}," \
                f"{cc_y_err.value},{results['xcen']},{results['ycen']},{results['seeing'].value}," \
                f"{results['raw_seeing'].value},{results["vlt_seeing"].value},{results["raw_vlt_seeing"].value},"\
                f"{results['ellipticity']},{results['fwhm']},{zresults['zernike_rms'].value}," \
                f"{zresults['residual_rms'].value},full,{iq}\n"
            zfile = f.parent / (f.stem + ".reanalyze.zernike")
            zresults['zernike'].save(filename=zfile)
            spotfile = f.parent / (f.stem + ".spots.csv")
            results['spots'].write(spotfile, overwrite=True)
            fits.writeto(f.parent / (f.stem + ".coadded_spot.fits"), results["coadded_spot"], overwrite=True)
            with open(outfile, 'w') as fp:
                fp.write(line)
            return line
        except Exception as e:
            log.error(f"Problem fitting wavefront for {f.name}: {e}")
            failed.touch()
            return None
    elif results.get('focus_only', False):
        try:
            nan = np.nan
            # the M2 focus error before the fallback's extra gain and clipping, comparable to full-analysis rows
            focerr = wfs_systems[wfskey].calculate_focus(results['zernike'].copy())
            # only record the pupil center if it was measured rather than the nominal fallback
            if results['grid'].get('center_measured', True):
                xcen, ycen = results['grid']['center']
            else:
                xcen, ycen = nan, nan
            line = f"{obstime},{wfskey},{f.name},{exptime},{airmass},{az},{el},{osst},{outt}," \
                f"{chamt},{tiltx},{tilty},{transx},{transy},{focus},{focerr.value},{nan},{nan}," \
                f"{xcen},{ycen},{nan},{nan},{nan},{nan},{nan},{nan},{nan},{nan},periodicity,{nan}\n"
            results['zernike'].save(filename=f.parent / (f.stem + ".periodicity.zernike"))
            with open(outfile, 'w') as fp:
                fp.write(line)
            return line
        except Exception as e:
            log.error(f"Problem saving focus-only results for {f.name}: {e}")
            failed.touch()
            return None
    else:
        failed.touch()  # mark this file as failed
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
        nargs='+',
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
        '--retry-failed',
        help="Reanalyze files that failed in earlier runs, reusing cached results for the rest. "
             "Rebuilds the CSV for each directory.",
        action="store_true"
    )

    parser.add_argument(
        '-n', '--nproc',
        metavar="<# processes>",
        help="Number of parallel processes. Defaults to half number of available cores.",
        type=int,
        default=int(multiprocessing.cpu_count()/2)  # MKL uses a lot of threads so best to limit Pool to half available cores
    )

    args = parser.parse_args()

    rootdir = Path(args.rootdir)

    log.info(f"Using {args.nproc} cores...")

    dirs = sorted(list(args.dirs))  # pathlib, where have you been all my life!
    csv_header = CSV_HEADER

    log.info(f"Found {len(dirs)} directories to process...")

    for d in dirs:
        d = rootdir / d
        if d.is_dir():
            if not (args.forcedir or args.retry_failed) and Path.exists(d / "reanalyze_results.csv"):
                log.info(f"Already processed {d.name}...")
            else:
                try:
                    lines = []
                    lines.append(csv_header)
                    _ = int(d.name)  # valid WFS directories are ints of the form YYYYMMDD. if not this form, int barfs
                    fitsfiles = sorted(list(d.glob("*.fits")))
                    log.info(f"Processing {len(fitsfiles)} images in {d}...")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nproc) as pool:
                        process = partial(process_image, force=args.force, retry_failed=args.retry_failed)
                        plines = pool.map(process, fitsfiles, timeout=300)  # plines comes out in same order as fitslines!

                    plines = [line for line in plines if line is not None]  # trim out any None entries

                    if len(plines) > 0:
                        lines.extend(plines)
                        with open(d / "reanalyze_results.csv", "w") as f:
                            f.writelines(lines)

                except ValueError as e:  # this means running int(d.name) failed so it's not a valid directory...
                    log.warn(f"Skipping %s... ({e})" % d.name)
