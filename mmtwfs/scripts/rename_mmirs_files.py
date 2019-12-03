#!/usr/bin/env python

import sys
import argparse

from pathlib import Path
from parse import parse
from astropy.io import fits

import logging


log = logging.getLogger('Rename MMIRS')
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
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
        help="Directory for saving renamed files.",
        default="."
    )
    parser.add_argument(
        '-d', '--dryrun',
        help="Determine new filenames, but don't rewrite files.",
        action="store_true"
    )

    args = parser.parse_args()

    rootdir = Path(args.rootdir)
    files = sorted(list(rootdir.glob("mmirs_wfs_0*.fits")))

    if len(files) < 1:
        log.error(f"No MMIRS WFS data found in {str(rootdir)}")
        return

    fmt = "wfs{:d}.fits"
    for f in files:
        with fits.open(f) as hdulist:
            h = hdulist[0].header
            if 'FITSNAME' in h:
                orig_name = Path(h['FITSNAME']).name
                log.info(f"{f.name} was originally {orig_name}.")
                orig_num = parse(fmt, orig_name)[0]
                new_name = "mmirs_wfs_rename_{:04d}.fits".format(orig_num)
                new_path = Path(args.savedir) / new_name
            if not args.dryrun:
                try:
                    hdulist.writeto(new_path, overwrite=False, output_verify="silentfix")
                    log.info(f"Writing {f.name} to {str(new_path)}...")
                except Exception as e:
                    log.error(f"Error writing {new_path.name}: {e}")
            else:
                log.info(f"Would rename {f.name} to {str(new_path)}...")

if __name__ == "__main__":
    main()
