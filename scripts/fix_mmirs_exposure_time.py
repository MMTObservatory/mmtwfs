#!/usr/bin/env python

import sys
import math
import argparse

from datetime import datetime
from pathlib import Path

from astropy.io import fits

import logging


log = logging.getLogger('Fix MMIRS')
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

def main():
    parser = argparse.ArgumentParser(description='Utility for fixing missing exposure times in MMIRS WFS images.')

    parser.add_argument(
        'rootdir',
        metavar="<WFS data directory>",
        help="Directory containing MMIRS WFS data to fix.",
        default="."
    )
    parser.add_argument(
        '--dryrun',
        help="Calculate new exposure times, but don't rewrite files.",
        action="store_true"
    )

    args = parser.parse_args()

    rootdir = Path(args.rootdir)
    files = sorted(list(rootdir.glob("mmirs*.fits")))

    if len(files) < 1:
        log.error(f"No MMIRS WFS data found in {str(rootdir)}")
        return

    timedict = {}
    for f in files:
        with fits.open(f) as hdulist:
            hdr = hdulist[-1].header
            data = hdulist[-1].data
        timedict[str(f)] = hdr['DATE-OBS']
    log.debug(timedict)

    sec = 0.
    for i in range(0, len(files)):
        if i < len(files)-1:
            t1 = datetime.strptime(timedict[str(files[i])], "%Y-%m-%dT%H:%M:%S")
            t2 = datetime.strptime(timedict[str(files[i+1])], "%Y-%m-%dT%H:%M:%S")
        else:  # handle last file
            t1 = datetime.strptime(timedict[str(files[i-1])], "%Y-%m-%dT%H:%M:%S")
            t2 = datetime.strptime(timedict[str(files[i])], "%Y-%m-%dT%H:%M:%S")
        diff = t2-t1

        # exposure times are almost always in multiples of 5 sec unless the exposures are very short
        diff_sec = 5 * math.floor(diff.seconds/5)

        # mmirs wfs exposures should almost never be more than 3 min during normal operations.
        # large gaps are assumed to be the end of a track so 200 seems a good cutoff to reject
        # those and use the previous time diff instead.
        if diff_sec < 200:
            sec = diff_sec

        f = files[i]
        with fits.open(f) as hdulist:
            changed = False
            for h in hdulist:
                if 'EXPTIME' in h.header:
                    if h.header['EXPTIME'] == 0.0:
                        if args.dryrun:
                            log.info(f"DRYRUN -- Setting EXPTIME to {sec} in {str(f)}..")
                        else:
                            log.info(f"Setting EXPTIME to {sec} in {str(f)}..")
                        h.header['EXPTIME'] = sec
                        changed = True
                    else:
                        log.info(f"EXPTIME already set to {h.header['EXPTIME']} for {str(f)}")
            if changed and not args.dryrun:
                hdulist.writeto(f, overwrite=True)


if __name__ == "__main__":
    main()
