# mmtwfs/nightlog.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Build a structured record of one night of WFS operations from the wfssrv log (wfs.log), the archived WFS frames, and
the reanalyze outputs. This module only builds data; mmtwfs.scripts.nightreport renders it.
"""

import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

from mmtwfs.outputs import read_output
from mmtwfs.wfs import WFSFactory
from mmtwfs.zernike import ZernikeVector

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


# frame column <- .output column. The .output "focus" column is the hexapod position (TRANSZ); "focerr" is the
# measured focus correction (um, at M2 gain 1), which is what settling is judged on.
FROM_OUTPUT = {"focus": "focerr", "cc_x_err": "cc_x_err", "cc_y_err": "cc_y_err", "wavefront_rms": "wavefront_rms",
               "residual_rms": "residual_rms", "raw_vlt_seeing": "raw_vlt_seeing", "delivered_iq": "delivered_iq"}
HEADER_FLOATS = {"exptime": "EXPTIME", "raoff": "RAOFF", "decoff": "DECOFF", "az": "AZ", "el": "EL",
                 "transx": "TRANSX", "transy": "TRANSY", "transz": "TRANSZ", "tiltx": "TILTX", "tilty": "TILTY"}
FRAME_COLUMNS = (
    [("time", TIME), ("file", str), ("wfs", str), ("catid", str), ("catra", float), ("catdec", float),
     ("ra", float), ("dec", float)]
    + [(k, float) for k in HEADER_FLOATS]
    + [("status", str)]
    + [(k, float) for k in FROM_OUTPUT]
    + [("method", str), ("spher", np.int8), ("visit", int), ("focus_unc", float), ("m1_resid_rms", float),
       ("m1_force_rms", float)]
)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _ra_deg(value):
    """RA from a header: float hours or a sexagesimal string, returned in degrees."""
    if isinstance(value, str):
        return Angle(value, unit=u.hourangle).degree
    return _float(value) * 15.0


def _dec_deg(value):
    if isinstance(value, str):
        return Angle(value, unit=u.deg).degree
    return _float(value)


def frame_wfs(name, hdr):
    """
    Which WFS took a frame, using the same clues as mmtwfs.scripts.reanalyze.check_image. The old Apogee F/9 maps to
    the "f9" WFS key.
    """
    if "mmirs" in name or "mmirs" in str(hdr.get("WFSNAME", "")):
        return "mmirs"
    if "wfs_ff_cal_img" in name or "bino" in name:
        return "binospec"
    if "f9wfs" in name or "F/9 WFS" in str(hdr.get("OBSERVER", "")) or "F/9 WFS" in str(hdr.get("CAMERA", "")):
        return "newf9"
    if "Apogee" in str(hdr.get("INSTRUME", "")) or "Apogee" in str(hdr.get("DETECTOR", "")):
        return "f9"
    return "unknown"


def _read_header(path):
    """Merge the headers of every HDU, as reanalyze does. Binospec keeps DATE-OBS in HDU 1."""
    hdr = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        with fits.open(path, ignore_missing_simple=True) as hdulist:
            for h in hdulist:
                hdr.update(h.header)
    return hdr


def _frame_time(hdr):
    dateobs = str(hdr.get("DATE-OBS", ""))
    if "T" in dateobs:
        return datetime.fromisoformat(dateobs.rstrip("Z"))
    if "MJD" in hdr:
        return Time(float(hdr["MJD"]), format="mjd", scale="utc").datetime
    return None


# reanalyze skips these: reference images, the SOG camera, and derived products such as *.coadded_spot.fits
DERIVED_MARKERS = ("Ref", "sog", "coadded")


def read_frames(nightdir):
    """
    One row per WFS frame in a night directory, failed frames included, joined with its reanalyze output.
    """
    rows = []
    counts = {"n_unreadable": 0, "n_untimed": 0}
    for f in sorted(nightdir.glob("*.fits")):
        if any(s in f.name for s in DERIVED_MARKERS):
            continue  # same skip rule as reanalyze; Binospec names have dots in their timestamps
        try:
            hdr = _read_header(f)
        except Exception as e:  # astropy raises several types for corrupt files
            log.warning(f"Can't read {f}: {e}")
            counts["n_unreadable"] += 1
            continue
        t = _frame_time(hdr)
        if t is None:
            counts["n_untimed"] += 1
            continue
        stem = f.name.removesuffix(".fits")
        row = dict(
            time=t, file=f.name, wfs=frame_wfs(f.name, hdr), catid=str(hdr.get("CATID", "") or "").strip(),
            catra=_ra_deg(hdr.get("CATRA")), catdec=_dec_deg(hdr.get("CATDEC")),
            ra=_ra_deg(hdr.get("RA")), dec=_dec_deg(hdr.get("DEC")),
            spher=-1, visit=-1,
        )
        row.update({k: _float(hdr.get(key)) for k, key in HEADER_FLOATS.items()})
        output = None
        outfile = nightdir / f"{stem}.output"
        if outfile.exists():
            try:
                output = read_output(outfile)
            except ValueError as e:
                log.warning(str(e))
        if output is not None:
            row["status"] = "ok"
            row.update({k: output[col] for k, col in FROM_OUTPUT.items()})
            row["method"] = output["method"]
        elif (nightdir / f"{stem}.failed").exists():
            row["status"] = "failed"
        else:
            row["status"] = "unanalyzed"
        rows.append(row)
    rows.sort(key=lambda r: r["time"])
    return _table(rows, FRAME_COLUMNS), counts


def _seconds(dt):
    return float(dt / np.timedelta64(1, "s"))


def _sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation in arcsec between two positions given in degrees."""
    r1, d1, r2, d2 = np.radians([ra1, dec1, ra2, dec2])
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) * 3600.0)


def _target(row):
    """Target position in degrees: the catalog position, or the pointing with the offsets removed."""
    if np.isfinite(row["catra"]) and np.isfinite(row["catdec"]):
        return row["catra"], row["catdec"]
    raoff = row["raoff"] if np.isfinite(row["raoff"]) else 0.0
    decoff = row["decoff"] if np.isfinite(row["decoff"]) else 0.0
    dec = row["dec"] - decoff / 3600.0
    return row["ra"] - raoff / 3600.0 / np.cos(np.radians(dec)), dec


def _same_target(a, b, params):
    if np.isfinite(a["catra"]) and np.isfinite(b["catra"]):
        return (round(a["catra"], 4), round(a["catdec"], 4), a["catid"]) == \
               (round(b["catra"], 4), round(b["catdec"], 4), b["catid"])
    return _sep_arcsec(*_target(a), *_target(b)) <= params["slew_sep"]


def build_visits(frames, params):
    """
    Split time-ordered frames into visits: runs of consecutive frames on the same target. Sets frames["visit"] and
    returns (visit rows, offset events).
    """
    visits, offsets = [], []
    idx = -1
    for i in range(len(frames)):
        row = frames[i]
        if i == 0 or not _same_target(frames[i - 1], row, params):
            idx += 1
            visits.append(dict(visit=idx, wfs=row["wfs"], catid=row["catid"], first=i))
        elif (np.isfinite(row["raoff"]) and np.isfinite(frames[i - 1]["raoff"])
              and (abs(row["raoff"] - frames[i - 1]["raoff"]) > 0.01
                   or abs(row["decoff"] - frames[i - 1]["decoff"]) > 0.01)):
            offsets.append(dict(time=row["time"], kind="offset", detail=f"{row['raoff']:.2f},{row['decoff']:.2f}",
                                file=row["file"]))
        frames["visit"][i] = idx

    for n, v in enumerate(visits):
        vf = frames[frames["visit"] == v["visit"]]
        first = vf[0]
        v.pop("first")
        v.update(start=first["time"], end=vf["time"][-1], n_frames=len(vf),
                 n_ok=int(np.sum(vf["status"] == "ok")), n_failed=int(np.sum(vf["status"] == "failed")),
                 el=first["el"])
        ok_times = vf["time"][vf["status"] == "ok"]
        v["t_first_ok"] = _seconds(ok_times[0] - first["time"]) if len(ok_times) else np.nan
        if n == 0:
            v.update(gap=np.nan, d_az=np.nan, d_el=np.nan, slew_sep=np.nan)
        else:
            prev = frames[frames["visit"] == v["visit"] - 1][-1]
            v.update(gap=_seconds(first["time"] - prev["time"]),
                     d_az=float((first["az"] - prev["az"] + 180.0) % 360.0 - 180.0),
                     d_el=float(first["el"] - prev["el"]),
                     slew_sep=_sep_arcsec(*_target(prev), *_target(first)))
    return visits, offsets


_WFS_CACHE = {}


def get_wfs(key):
    """A WFS instance per key, built once. Only used offline; never connected."""
    if key not in _WFS_CACHE:
        _WFS_CACHE[key] = WFSFactory(wfs=key)
    return _WFS_CACHE[key]


def gain_at(events, kind, time, default):
    """The most recent gain of this kind logged at or before time, or the WFS default."""
    sel = events[(events["kind"] == kind) & (events["time"] <= time)]
    return float(sel["detail"][-1]) if len(sel) else default


def m1_resid_rms(zv):
    """RMS (nm) of the M1-actionable terms. Z04 holds the spherical-focus offset that M2 takes, so it's excluded."""
    z = zv.copy()
    z.ignore("Z04")
    return float(z.rms.value)


def _rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan


def calculate_primary_at(wfs, zv, mask, gain):
    """wfs.calculate_primary at a specific M1 gain, leaving the WFS's own gain untouched."""
    orig = wfs.m1_gain
    wfs.m1_gain = gain
    try:
        return wfs.calculate_primary(zv.copy(), mask=mask)
    finally:
        wfs.m1_gain = orig


def _load_zv(path):
    zv = ZernikeVector()
    zv.load(filename=path)
    return zv


def focus_uncertainty(zv, wfs):
    """Focus uncertainty (um) implied by the Z04 errorbar, on the same scale as calculate_focus at M2 gain 1."""
    z = zv.copy()
    z.denormalize()
    err = z.errorbars.get("Z04")
    if err is None:
        return np.nan
    return abs(float(u.Quantity(err, z.units).value / wfs.secondary.focus_trans.value))


def m1_frame_metrics(frames, nightdir, events):
    """
    Fill frames["focus_unc"], frames["m1_resid_rms"] and frames["m1_force_rms"] for ok frames. For M1, use the
    online .masked.zernike, which is exactly what fed the proposed forces. Otherwise recompute from
    .reanalyze.zernike with the frame's spherical mask and the M1 gain in effect.
    """
    for i in range(len(frames)):
        row = frames[i]
        if row["status"] != "ok" or row["wfs"] == "unknown":
            continue
        try:
            wfs = get_wfs(row["wfs"])
            gain = gain_at(events, "m1_gain", row["time"], wfs.m1_gain)
            zfile = nightdir / f"{row['file'].removesuffix('.fits')}.reanalyze.zernike"
            zv_full = _load_zv(zfile) if zfile.exists() else None
            if zv_full is not None:
                frames["focus_unc"][i] = focus_uncertainty(zv_full, wfs)
            masked_file = nightdir / f"{row['file']}.masked.zernike"
            if masked_file.exists():
                zv_masked = _load_zv(masked_file)
                forces = wfs.telescope.bending_forces(zv=zv_masked.copy(), gain=gain)
            elif zv_full is not None:
                mask = ["Z11", "Z22"] if row["spher"] == 1 else ["Z22"]
                forces, _, zv_masked = calculate_primary_at(wfs, zv_full, mask, gain)
            else:
                continue
            frames["m1_resid_rms"][i] = m1_resid_rms(zv_masked)
            frames["m1_force_rms"][i] = _rms(forces["force"])
        except Exception as e:
            log.warning(f"M1 metrics failed for {row['file']}: {e}")
