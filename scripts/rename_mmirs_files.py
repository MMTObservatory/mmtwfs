#!/usr/bin/env python

import argparse

from pathlib import Path
from parse import parse
from astropy.io import fits

import logging


log = logging.getLogger('Rename MMIRS')
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

def main():
    parser = argparse.ArgumentParser(description='Utility for renaming MMIRS images based on the original numbering.')

    parser.add_argument(
        '-r', '--rootdir',
        metavar="<WFS data directory>",
        help="Directory containing MMIRS WFS data to fix.",
        default="."
    )
    parser.add_argument(
        '-s', '--savedir',
        help="Directory for saving renamed files."
        default="."
    )
    parser.add_argument(
        '-d', '--dryrun',
        help="Determine new filenames, but don't rewrite files.",
        action="store_true"
    )

    args = parser.parse_args()

    rootdir = Path(args.rootdir)
    files = sorted(list(rootdir.glob("mmirs*.fits")))

    if len(files) < 1:
        log.error(f"No MMIRS WFS data found in {str(rootdir)}")
        return

    fmt = "wfs{:d}.fits"
    for f in files:
        with fits.open(f) as hdulist:
            h = hdulist[0].header
        if 'FITSNAME' in h:
            orig_name = Path(h.header['FITSNAME']).name
            log.info(f"{f.name} was originally {orig_name}.")
            orig_num = parse(gmt, new_name)
            new_name = "mmirs_wfs_{:04d}.fits".format(orig_num)
            new_path = f.parent / new_name
        if not args.dryrun:
            try:
                hdulist.writeto(new_path, overwrite=False)
            except:
                log.error(f"{new_path.name} already exists!")


if __name__ == "__main__":
    main()
