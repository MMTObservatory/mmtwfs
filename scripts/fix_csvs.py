#!/usr/bin/env python

import sys
from pathlib import Path
from astropy.io import ascii, fits
from astropy.table import Column

csv_list = Path(sys.argv[1])

with open(csv_list) as f:
    csvs = f.readlines()

for csv in csvs:
    csv_path = csv_list.parent / csv.rstrip()
    t = ascii.read(str(csv_path))

    if not 'exptime' in t.columns:
        exptimes = []
        for r in t:
            with fits.open(csv_path.parent / r['file']) as hdulist:
                exptime = hdulist[-1].header['EXPTIME']
                exptimes.append(exptime)
        c = Column(name='exptime', data=exptimes)
        t.add_column(c)
        t.write(csv_path, format="ascii.csv", overwrite=True)
        print(f"writing {str(csv_path)}")

