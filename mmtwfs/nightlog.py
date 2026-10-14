# mmtwfs/nightlog.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Build a structured record of one night of WFS operations from the wfssrv log (wfs.log), the archived WFS frames, and
the reanalyze outputs. This module only builds data; mmtwfs.scripts.nightreport renders it.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from astropy.table import Table

log = logging.getLogger("NightLog")

# wfs.log timestamps are MST. Arizona doesn't observe DST.
LOCAL_UTC_OFFSET = timedelta(hours=7)

SETTLE_DEFAULTS = {
    "focus_thresh": 10.0,  # um
    "coma_thresh": 2.0,  # arcsec
    "consecutive": 2,  # frames
    "iq_abs": 0.05,  # arcsec
    "iq_frac": 0.05,
    "slew_sep": 60.0,  # arcsec
}

TIME = "datetime64[ms]"
SEND_COLUMNS = [
    ("time", TIME), ("kind", str), ("file", str), ("value", float), ("value2", float), ("forces_file", str),
    ("fmax", float), ("fmin", float), ("applied", str), ("frac", float),
]
EVENT_COLUMNS = [("time", TIME), ("kind", str), ("detail", str), ("file", str)]
ANALYSIS_COLUMNS = [("time", TIME), ("file", str), ("spher", bool)]

NUM = r"-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"
LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) - (.+?) - ([A-Z]+) - (.*)$")
PATTERNS = [
    ("analyze", re.compile(r"^Analyzing (?P<path>\S+?)\.\.\.$")),
    ("spher_all", re.compile(r"^Ignoring all spherical terms")),
    ("focus", re.compile(rf"^Sending (?P<v>{NUM}) um focus to secondary")),
    ("coma", re.compile(rf"^Sending (?P<v>{NUM}) arcsec/(?P<v2>{NUM}) arcsec CC_X/CC_Y to secondary")),
    ("m1", re.compile(r"^Sending forces from (?P<path>\S+?)\.\.\.$")),
    ("m1_full", re.compile(r"^\.\.\.forces successfully applied in full")),
    ("m1_partial", re.compile(rf"^\.\.\.applied (?P<v>{NUM})% of requested forces")),
    ("m1_rejected", re.compile(r"^\.\.\.(?:forces rejected|unable to apply forces)")),
    ("m1_unexpected", re.compile(r"^\.\.\.got unexpected reply from cell command")),
    ("m1focus", re.compile(rf"^Moving \w+ hexapod (?P<v>{NUM}) um in Z to correct spherical aberration")),
    ("fmax", re.compile(rf"^Maximum force (?P<v>{NUM}) N")),
    ("fmin", re.compile(rf"^Minimum force (?P<v>{NUM}) N")),
    ("clear_m1", re.compile(r"^Clearing forces and spherical aberration focus offsets")),
    ("clear_m2", re.compile(r"^Resetting hexapod's WFS offsets to 0")),
    ("bad_gain", re.compile(r"^Invalid M[12] gain")),
    ("m1_gain", re.compile(rf"^Setting M1 gain to (?P<v>{NUM})")),
    ("m2_gain", re.compile(rf"^Setting M2 gain to (?P<v>{NUM})")),
    ("wfs_select", re.compile(r"^Setting (?P<wfs>\w+)$")),
    ("connect", re.compile(r" (?:is|already) (?P<state>connected|disconnected)\.?$")),
    ("validity", re.compile(r"(?P<msg>only focus corrections valid|wavefront fit too poor)")),
]
CELL_REPLIES = {"m1_full": ("full", 1.0), "m1_rejected": ("rejected", 0.0), "m1_unexpected": ("unexpected", 1.0)}
ERROR_LOGGERS = ("Telescope", "Secondary")


def _default(dtype):
    if dtype is str:
        return ""
    kind = np.dtype(dtype).kind
    if kind == "b":
        return False
    if kind == "M":
        return np.datetime64("NaT", "ms")
    if kind in "iu":
        return -1
    return np.nan


def _table(rows, columns):
    """
    Build a Table with a fixed schema from a list of dicts. Missing keys get a type-appropriate null, and an empty
    list still yields every column with the right dtype.
    """
    return Table({
        name: np.array([r.get(name, _default(dtype)) for r in rows], dtype=dtype)
        for name, dtype in columns
    })


def night_window(night):
    """
    UTC (start, end) of the night archived in directory YYYYMMDD: local noon the day before to local noon on the
    directory date.
    """
    end = datetime.strptime(night, "%Y%m%d") + timedelta(hours=12) + LOCAL_UTC_OFFSET
    return end - timedelta(days=1), end


def _match(msg):
    for kind, pattern in PATTERNS:
        m = pattern.search(msg)
        if m:
            return kind, m
    return None, None


@dataclass
class LogData:
    sends: Table
    events: Table
    analyses: Table
    n_lines: int = 0
    n_outside: int = 0
    n_unrecognized: int = 0


def empty_logdata():
    return LogData(_table([], SEND_COLUMNS), _table([], EVENT_COLUMNS), _table([], ANALYSIS_COLUMNS))


def parse_log(logfile, night):
    """
    Parse a wfssrv log into sends, events, and online analyses, keeping only lines inside the night window.
    """
    start, end = night_window(night)
    sends, events, analyses = [], [], []
    n_lines = n_outside = n_unrecognized = 0
    connected = None
    last_m1 = None
    prev_kind = None

    def current_file():
        return analyses[-1]["file"] if analyses else ""

    with open(logfile, errors="replace") as fp:
        for line in fp:
            if " - tornado." in line or " - poppy - " in line:
                continue
            m = LINE_RE.match(line.rstrip("\n"))
            if m is None:
                continue  # continuation line of a multi-line message
            n_lines += 1
            stamp, ms, logger, level, msg = m.groups()
            t = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") + timedelta(milliseconds=int(ms)) + LOCAL_UTC_OFFSET
            if not start <= t < end:
                n_outside += 1
                continue
            kind, mm = _match(msg)

            if kind == "analyze":
                analyses.append(dict(time=t, file=Path(mm["path"]).name, spher=False))
            elif kind == "spher_all":
                if analyses:
                    analyses[-1]["spher"] = True
            elif kind == "focus":
                sends.append(dict(time=t, kind="focus", file=current_file(), value=float(mm["v"])))
            elif kind == "coma":
                sends.append(dict(time=t, kind="coma", file=current_file(), value=float(mm["v"]),
                                  value2=float(mm["v2"])))
            elif kind == "m1":
                forces_file = Path(mm["path"]).name
                last_m1 = dict(time=t, kind="m1", file=forces_file.removesuffix(".forces"), forces_file=forces_file,
                               applied="unknown", frac=1.0)
                sends.append(last_m1)
            elif kind in CELL_REPLIES or kind == "m1_partial":
                if last_m1 is not None and last_m1["applied"] == "unknown":
                    if kind == "m1_partial":
                        last_m1["applied"], last_m1["frac"] = "partial", float(mm["v"]) / 100.0
                    else:
                        last_m1["applied"], last_m1["frac"] = CELL_REPLIES[kind]
            elif kind == "m1focus":
                src = last_m1["file"] if last_m1 is not None else current_file()
                sends.append(dict(time=t, kind="m1focus", file=src, value=float(mm["v"])))
            elif kind in ("fmax", "fmin"):
                if last_m1 is not None:
                    last_m1[kind] = float(mm["v"])
            elif kind == "clear_m1":
                events.append(dict(time=t, kind="clear_m1"))
            elif kind == "clear_m2":
                if connected is not False:
                    events.append(dict(time=t, kind="clear_m2"))
            elif kind in ("m1_gain", "m2_gain"):
                if prev_kind != "bad_gain":
                    events.append(dict(time=t, kind=kind, detail=mm["v"]))
            elif kind == "wfs_select":
                events.append(dict(time=t, kind="wfs_select", detail=mm["wfs"]))
            elif kind == "connect":
                connected = mm["state"] == "connected"
                events.append(dict(time=t, kind="connect", detail=mm["state"]))
            elif kind == "validity":
                events.append(dict(time=t, kind="validity", detail=mm["msg"], file=current_file()))
            elif kind is None and level == "ERROR" and logger in ERROR_LOGGERS and not msg.startswith("\t"):
                events.append(dict(time=t, kind="error", detail=msg.strip()))
            elif kind is None:
                n_unrecognized += 1
            prev_kind = kind

    return LogData(
        sends=_table(sends, SEND_COLUMNS),
        events=_table(events, EVENT_COLUMNS),
        analyses=_table(analyses, ANALYSIS_COLUMNS),
        n_lines=n_lines,
        n_outside=n_outside,
        n_unrecognized=n_unrecognized,
    )
