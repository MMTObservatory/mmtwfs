# mmtwfs/outputs.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Column layout of the per-frame reanalyze outputs: the one-line ``<stem>.output`` files and the ``reanalyze_results.csv``
that collects them. Kept apart from mmtwfs.scripts.reanalyze because importing that script builds every WFS.
"""

CSV_HEADER = "time,wfs,file,exptime,airmass,az,el,osst,outt,chamt,tiltx,tilty,"\
    "transx,transy,focus,focerr,cc_x_err,cc_y_err,xcen,ycen,seeing,raw_seeing,"\
    "vlt_seeing,raw_vlt_seeing,ellipticity,fwhm,wavefront_rms,residual_rms,method,delivered_iq\n"
CSV_COLUMNS = CSV_HEADER.strip().split(",")
STRING_COLUMNS = ("time", "wfs", "file", "method")


def read_output(path):
    """
    Parse a one-line ``.output`` file into a dict keyed by CSV_COLUMNS. Lines written before the ``method`` and
    ``delivered_iq`` columns existed are padded with "full" and NaN. A "None", written for a header value the frame
    lacks, reads as NaN. Returns None for an empty file.
    """
    text = path.read_text().strip()
    if not text:
        return None
    fields = text.split(",")
    ncols = len(CSV_COLUMNS)
    if len(fields) == ncols - 2:
        fields.append("full")
    if len(fields) == ncols - 1:
        fields.append("nan")
    if len(fields) != ncols:
        raise ValueError(f"{path}: expected {ncols} fields, got {len(fields)}")
    row = dict(zip(CSV_COLUMNS, fields))
    for col in CSV_COLUMNS:
        if col not in STRING_COLUMNS:
            row[col] = float("nan" if row[col] == "None" else row[col])
    return row
