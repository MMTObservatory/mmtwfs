# mmtwfs/data/test_data/nightreport/make_fixture.py
"""
Rebuild the nightreport test fixture from the archived 20261003 MMIRS night: header-only copies of frames 0318-0340
(end of HIP102074, the 06:42 UTC M1 clear, start of HIP109911 with focus/coma/M1 sends and gain changes), their
reanalyze and online outputs, and the wfs.log lines between them with polling noise removed.

Usage: python make_fixture.py /Volumes/Samsung_4TB/wfsdat/20261003
"""
import shutil
import sys
import warnings
from pathlib import Path

from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

SUFFIXES = [".output", ".failed", ".reanalyze.zernike", ".fits.forces", ".fits.masked.zernike", ".fits.zernike"]


def main(src):
    src = Path(src)
    dst = Path(__file__).parent / src.name
    dst.mkdir(exist_ok=True)
    for i in range(318, 341):
        stem = f"mmirs_wfs_{i:04d}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            with fits.open(src / f"{stem}.fits", ignore_missing_simple=True) as hl:
                hdr = hl[0].header.copy()
        for key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "BSCALE", "BZERO", "EXTEND"):
            hdr.remove(key, ignore_missing=True, remove_all=True)
        fits.PrimaryHDU(header=hdr).writeto(dst / f"{stem}.fits", overwrite=True)
        for suffix in SUFFIXES:
            if (src / f"{stem}{suffix}").exists():
                shutil.copy(src / f"{stem}{suffix}", dst / f"{stem}{suffix}")
    keep = []
    with open(src / "wfs.log", errors="replace") as fp:
        for line in fp:
            stamp = line[:19]
            if "2026-10-02 23:40:00" <= stamp < "2026-10-03 01:58:00" and " - tornado." not in line:
                keep.append(line)
    (dst / "wfs.log").write_text("".join(keep))


if __name__ == "__main__":
    main(sys.argv[1])
