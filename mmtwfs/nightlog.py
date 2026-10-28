# mmtwfs/nightlog.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Build a structured record of one night of WFS operations from the wfssrv log (wfs.log), the archived WFS frames, and
the reanalyze outputs. This module only builds data; mmtwfs.scripts.nightreport renders it.
"""

import functools
import logging
import re
import warnings
from dataclasses import dataclass, field
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
from mmtwfs.telescope import MMT
from mmtwfs.config import mmtwfs_config
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
EVENT_COLUMNS = [("time", TIME), ("kind", str), ("detail", str), ("file", str), ("wfs", str)]
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
    ("wfs_restart", re.compile(r"^restarting (?P<wfs>\w+)$")),
    ("connect", re.compile(r"^(?P<name>.+?) (?:is|already) (?P<state>connected|disconnected)\.?$")),
    ("validity", re.compile(r"(?P<msg>only focus corrections valid|wavefront fit too poor)")),
]
CELL_REPLIES = {"m1_full": ("full", 1.0), "m1_rejected": ("rejected", 0.0), "m1_unexpected": ("unexpected", 0.0)}
ERROR_LOGGERS = ("Telescope", "Secondary")
WFS_NAMES = {key: cfg["name"] for key, cfg in mmtwfs_config["wfs"].items()}  # "Setting <key>" -> "<name> is connected."


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
    connected = {}  # wfssrv keeps a connected flag per WFS object, keyed here by WFS name
    last_connected = None  # most recent connect state of any WFS, for resets before any WFS is selected
    selected = selected_key = None  # the selected WFS's name and its config key
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
            kind, mm = _match(msg)
            if kind == "connect":
                last_connected = connected[mm["name"]] = mm["state"] == "connected"
            elif kind == "wfs_select":
                selected_key = mm["wfs"]
                selected = WFS_NAMES.get(selected_key, selected_key)
            if not start <= t < end:
                n_outside += 1  # connection state and WFS selection still carry into the window
                continue

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
                state = connected.get(selected) if selected is not None else last_connected
                if state is not False:
                    events.append(dict(time=t, kind="clear_m2"))
            elif kind in ("m1_gain", "m2_gain"):
                if prev_kind != "bad_gain":  # wfssrv sets the gain on the selected WFS only
                    events.append(dict(time=t, kind=kind, detail=mm["v"], wfs=selected_key or ""))
            elif kind == "wfs_select":
                events.append(dict(time=t, kind="wfs_select", detail=mm["wfs"]))
            elif kind == "wfs_restart":  # a fresh WFS object, back at its default gains
                events.append(dict(time=t, kind="wfs_restart", detail=mm["wfs"], wfs=mm["wfs"]))
            elif kind == "connect":
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
               "residual_rms": "residual_rms", "raw_vlt_seeing": "raw_vlt_seeing", "vlt_seeing": "vlt_seeing",
               "delivered_iq": "delivered_iq"}
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
    the "f9" WFS key. Only needed for frames without a reanalyze output, which records the WFS itself.
    """
    if "mmirs" in name or "mmirs" in str(hdr.get("WFSNAME", "")):
        return "mmirs"
    if "wfs_ff_cal_img" in name or "bino" in name or "Binospec" in str(hdr.get("ORIGIN", "")):
        return "binospec"
    if "f9wfs" in name or "F/9 WFS" in str(hdr.get("OBSERVER", "")) or "F/9 WFS" in str(hdr.get("CAMERA", "")):
        return "newf9"
    if "Apogee" in str(hdr.get("INSTRUME", "")) or "Apogee" in str(hdr.get("DETECTOR", "")):
        return "f9"
    if "f5wfs" in name or "F/5 WFS" in str(hdr.get("INSTRUME", "")) or "F5" in str(hdr.get("SEC", "")):
        return "f5"
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


OUTPUT_WFS = {"oldf9": "f9"}  # reanalyze's names that differ from the WFS config keys

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
            t = _frame_time(hdr)
            if t is not None:
                coords = dict(catra=_ra_deg(hdr.get("CATRA")), catdec=_dec_deg(hdr.get("CATDEC")),
                              ra=_ra_deg(hdr.get("RA")), dec=_dec_deg(hdr.get("DEC")))
        except Exception as e:  # corrupt files and malformed header values raise several types
            log.warning(f"Can't read {f}: {e}")
            counts["n_unreadable"] += 1
            continue
        if t is None:
            counts["n_untimed"] += 1
            continue
        stem = f.name.removesuffix(".fits")
        row = dict(
            time=t, file=f.name, wfs=frame_wfs(f.name, hdr), catid=str(hdr.get("CATID", "") or "").strip(),
            spher=-1, visit=-1, **coords,
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
            row["wfs"] = OUTPUT_WFS.get(output["wfs"], output["wfs"])
        elif (nightdir / f"{stem}.failed").exists():
            row["status"] = "failed"
        else:
            row["status"] = "unanalyzed"
        rows.append(row)
    rows.sort(key=lambda r: r["time"])
    return _table(rows, FRAME_COLUMNS), counts


FOCUS_ONLY_DROPPED = ("cc_x_err", "cc_y_err", "delivered_iq")
ALL_VALID_RESID, FOCUS_VALID_RESID = 4000.0, 7000.0  # nm; wfssrv's residual RMS limits for its corrections


def apply_validity(frames, events, analyses):
    """
    Drop the measurements wfssrv itself rejected. A wavefront residual above 7000 nm logs "wavefront fit too poor"
    and no correction is valid, so the frame's status becomes "invalid". Between 4000 and 7000 nm only focus is
    valid, so coma and delivered IQ are set to NaN. For frames wfssrv analyzed, the verdicts come from the wfs.log
    validity events. Frames it never analyzed (no log, or wfssrv down) get them from the reanalyzed residual RMS
    with wfssrv's limits.
    """
    validity = events[events["kind"] == "validity"]
    logged = dict(zip(validity["file"], validity["detail"]))
    online = set(analyses["file"])
    verdict = {}
    for f, resid in zip(frames["file"], frames["residual_rms"]):
        if f in online:
            if f in logged:
                verdict[f] = logged[f]
        elif resid > FOCUS_VALID_RESID:
            verdict[f] = "wavefront fit too poor"
        elif resid >= ALL_VALID_RESID:
            verdict[f] = "only focus corrections valid"
    frames.replace_column("status", frames["status"].astype("U10"))  # sized to the longest status present, maybe "ok"
    for i, f in enumerate(frames["file"]):
        if frames["status"][i] != "ok" or f not in verdict:
            continue
        if verdict[f] == "wavefront fit too poor":
            frames["status"][i] = "invalid"
        else:
            for col in FOCUS_ONLY_DROPPED:
                frames[col][i] = np.nan


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
    if a["catid"] and b["catid"]:  # the catalog ID stays fixed for the whole of a visit
        return a["catid"] == b["catid"]
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
            # the new RAOFF/DECOFF header values: the pointing moved within the visit
            offsets.append(dict(time=row["time"], kind="offset", file=row["file"],
                                detail=f"RA {row['raoff']:+.2f} Dec {row['decoff']:+.2f} arcsec"))
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


def gain_at(events, kind, time, default, wfs=None):
    """
    The most recent gain of this kind logged at or before time, or the WFS default. wfssrv keeps a gain per WFS, so
    with wfs given only that WFS's gains count (or ones logged before any WFS was selected), and a restart of that WFS
    returns it to the default.
    """
    sel = (events["kind"] == kind) & (events["time"] <= time)
    if wfs is not None:
        sel &= (events["wfs"] == wfs) | (events["wfs"] == "")
        restarts = events["time"][(events["kind"] == "wfs_restart") & (events["wfs"] == wfs) & (events["time"] <= time)]
        if len(restarts):
            sel &= events["time"] > restarts[-1]
    gains = events["detail"][sel]
    return float(gains[-1]) if len(gains) else default


@functools.lru_cache
def default_gain(key, kind, fallback):
    """A WFS's configured gain, or fallback when the WFS is unknown or can't be built."""
    if key == "unknown":
        return fallback
    try:
        return getattr(get_wfs(key), kind)
    except Exception as e:
        log.warning(f"Can't build WFS {key}: {e}")
        return fallback


def selected_wfs(events, time):
    """The WFS wfssrv had selected at time, or None."""
    sel = events["detail"][(events["kind"] == "wfs_select") & (events["time"] <= time)]
    return str(sel[-1]) if len(sel) else None


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


def load_masked_zv(path):
    """
    An online .masked.zernike in the WFS frame. wfssrv saves the zv_masked that calculate_primary returns, and
    bending_forces has already rotated that vector -90 deg in place into the BCV frame, so undo the rotation.
    """
    zv = _load_zv(path)
    zv.rotate(90 * u.deg)
    return zv


def focus_uncertainty(zv, wfs):
    """Focus uncertainty (um) implied by the Z04 errorbar, on the same scale as calculate_focus at M2 gain 1."""
    z = zv.copy()
    z.denormalize()
    err = z.errorbars.get("Z04")
    if err is None:
        return np.nan
    return abs(float(u.Quantity(err, z.units).value / wfs.secondary.focus_trans.value))


def m1_frame_metrics(frames, nightdir, events, analyses=None):
    """
    Fill frames["focus_unc"], frames["m1_resid_rms"] and frames["m1_force_rms"] for ok frames. For M1, use the
    online .masked.zernike, which is exactly what fed the proposed forces. Otherwise recompute from
    .reanalyze.zernike with the frame's spherical mask and the M1 gain in effect when wfssrv analyzed it.
    """
    analysed_at = {} if analyses is None else dict(zip(analyses["file"], analyses["time"]))
    for i in range(len(frames)):
        row = frames[i]
        if row["status"] != "ok" or row["wfs"] == "unknown":
            continue
        try:
            wfs = get_wfs(row["wfs"])
            gain = gain_at(events, "m1_gain", analysed_at.get(row["file"], row["time"]), wfs.m1_gain, row["wfs"])
            zfile = nightdir / f"{row['file'].removesuffix('.fits')}.reanalyze.zernike"
            zv_full = _load_zv(zfile) if zfile.exists() else None
            if zv_full is not None:
                frames["focus_unc"][i] = focus_uncertainty(zv_full, wfs)
            masked_file = nightdir / f"{row['file']}.masked.zernike"
            if masked_file.exists():
                zv_masked = load_masked_zv(masked_file)
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


F9_KEYS = ("f9", "newf9")


def first_run(mask, k):
    """Index of the element that starts the first run of k consecutive True values, or None."""
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run == k:
            return i - k + 1
    return None


def _settle_time(times, mask, k, t0):
    i = first_run(mask, k)
    return (np.nan, None) if i is None else (_seconds(times[i] - t0), i)


def _send_totals(sends, forces, telescope):
    vec = None
    for s in sends[sends["kind"] == "m1"]:
        f = forces.get(s["forces_file"])
        if f is not None:
            vec = s["frac"] * f if vec is None else vec + s["frac"] * f
    totals = dict(
        n_sends=len(sends),
        net_focus=float(np.sum(sends["value"][sends["kind"] == "focus"])),
        net_m1focus=float(np.sum(sends["value"][sends["kind"] == "m1focus"])),
        net_cc_x=float(np.sum(sends["value"][sends["kind"] == "coma"])),
        net_cc_y=float(np.sum(sends["value2"][sends["kind"] == "coma"])),
        m1_sent_rms=np.nan, m1_sent_surf_rms=np.nan,
    )
    if vec is not None:
        totals["m1_sent_rms"] = _rms(vec)
        totals["m1_sent_surf_rms"] = float(np.std(vec @ telescope.load_act2surf()))
    return totals


def visit_settle(vframes, vsends, params, forces, telescope):
    """
    Settle times, the errors the slew left behind, and the corrections sent for one visit. Times are seconds from
    the visit's first frame.
    """
    k, F, C = params["consecutive"], params["focus_thresh"], params["coma_thresh"]
    t0 = vframes["time"][0]
    ok = vframes[vframes["status"] == "ok"]
    out = dict(settle_def="last_send" if vframes["wfs"][0] in F9_KEYS else "thresh/conv")

    afocus = np.abs(np.asarray(ok["focus"], dtype=float))
    out["focus_settle_thresh"], i_settle = _settle_time(ok["time"], afocus < F, k, t0)
    out["focus_unc_at_settle"] = float(ok["focus_unc"][i_settle]) if i_settle is not None else np.nan
    n = len(ok)
    if n >= k:
        plateau = np.nanmedian(afocus[n // 2:])
        out["focus_settle_conv"], _ = _settle_time(ok["time"], afocus <= plateau + F, k, t0)
        iq = np.asarray(ok["delivered_iq"], dtype=float)
        iq_plateau = np.nanmedian(iq[n // 2:])
        tol = max(params["iq_abs"], params["iq_frac"] * iq_plateau)
        out["iq_settle_conv"], _ = _settle_time(ok["time"], iq <= iq_plateau + tol, k, t0)
    else:
        out["focus_settle_conv"] = out["iq_settle_conv"] = np.nan
    coma_ok = (np.abs(ok["cc_x_err"]) < C) & (np.abs(ok["cc_y_err"]) < C)
    out["coma_settle_thresh"], _ = _settle_time(ok["time"], coma_ok, k, t0)

    first = ok[0] if n else None
    for key, col in (("first_focus", "focus"), ("first_focus_unc", "focus_unc"), ("first_cc_x", "cc_x_err"),
                     ("first_cc_y", "cc_y_err"), ("first_wavefront_rms", "wavefront_rms"),
                     ("first_iq", "delivered_iq"), ("m1_resid_first", "m1_resid_rms"),
                     ("m1_force_first", "m1_force_rms")):
        out[key] = float(first[col]) if first is not None else np.nan
    out["m1_resid_last"] = float(ok["m1_resid_rms"][-1]) if n else np.nan

    if len(vsends):
        out["settle_last_send"] = _seconds(np.max(vsends["time"]) - t0)
    else:
        out["settle_last_send"] = _seconds(ok["time"][0] - t0) if n else np.nan

    frame_time = dict(zip(vframes["file"], vframes["time"]))
    if i_settle is not None:
        cutoff = ok["time"][i_settle]
        src = np.array([frame_time.get(f, np.datetime64("NaT", "ms")) for f in vsends["file"]], dtype=TIME)
        in_settle = vsends[src <= cutoff]
    else:
        in_settle = vsends
    for window, s in (("settle", in_settle), ("visit", vsends)):
        for key, value in _send_totals(s, forces, telescope).items():
            out[f"{key}_{window}"] = value
    return out


FIT_TERMS = ["Z04", "Z05", "Z06"] + [f"Z{j:02d}" for j in range(9, 23)]
M1_COLUMNS = [("time", TIME), ("fmax", float), ("fmin", float), ("frms", float), ("surf_rms", float)] + \
    [(t, float) for t in FIT_TERMS]
M2_CHANNELS = ["wfs_z", "wfs_cc_x", "wfs_cc_y", "m1spherical_z"]
M2_COLUMNS = [("time", TIME)] + [(c, float) for c in M2_CHANNELS]


@functools.cache
def get_mmt():
    return MMT()


def read_forces(path, act_ids):
    """A .forces file (act_id<TAB>force, as written by MMT.to_rcell) as a vector aligned to act_ids."""
    data = np.loadtxt(path, ndmin=2)
    lookup = dict(zip(data[:, 0].astype(int), data[:, 1]))
    return np.array([lookup.get(int(a), 0.0) for a in act_ids])


@functools.cache
def _surface_basis(telescope):
    rho, phi = telescope.nodecoor["bcv_rho"], telescope.nodecoor["bcv_phi"]
    cols = []
    for term in FIT_TERMS:
        zv = ZernikeVector(**{term: 1.0})
        zv.rotate(-90 * u.deg)  # same rotation bending_forces applies, so fitted terms are in the WFS frame
        cols.append(np.asarray(zv.total_phase(rho, phi), dtype=float))
    return np.array(cols).T


def fit_surface_zernikes(surface, telescope):
    """Least-squares Zernike fit to an M1 surface (nm at the BCV nodes), returned as wavefront nm (2 x surface)."""
    coeffs, *_ = np.linalg.lstsq(_surface_basis(telescope), surface, rcond=None)
    return dict(zip(FIT_TERMS, 2.0 * coeffs))


def _m1_snapshot(time, state, telescope):
    surface = state @ telescope.load_act2surf()
    row = dict(time=time, fmax=float(state.max()), fmin=float(state.min()), frms=_rms(state),
               surf_rms=float(np.std(surface)))
    row.update(fit_surface_zernikes(surface, telescope))
    return row


def m1_accounting(sends, events, forces, telescope):
    """
    Running total of the forces applied to M1, starting from zero (M1 is raised in a cleared state every night).
    Each m1 send adds frac x its .forces vector, as MMT.correct_primary does. clear_m1 resets the total.
    """
    state = np.zeros(telescope.n_act)
    rows, maps = [], []
    steps = [(s["time"], "send", s) for s in sends[sends["kind"] == "m1"]]
    steps += [(e["time"], "clear", e) for e in events[events["kind"] == "clear_m1"]]
    steps.sort(key=lambda s: (s[0], s[1] == "send"))
    for time, what, item in steps:
        if what == "clear":
            if np.any(state):
                maps.append((time, "before clear", state.copy()))
            state = np.zeros(telescope.n_act)
        else:
            vec = forces.get(item["forces_file"])
            if vec is None:
                continue  # reported as a missing_forces flag
            state = state + item["frac"] * vec
        rows.append(_m1_snapshot(time, state, telescope))
    if np.any(state):
        maps.append((steps[-1][0], "end of night", state.copy()))
    return _table(rows, M1_COLUMNS), maps


def m2_accounting(sends, events):
    """Running sums of the WFS offsets sent to the hexapod, per channel, reset by the matching clear."""
    state = dict.fromkeys(M2_CHANNELS, 0.0)
    steps = [(s["time"], s["kind"], s) for s in sends if s["kind"] in ("focus", "coma", "m1focus")]
    steps += [(e["time"], e["kind"], e) for e in events if e["kind"] in ("clear_m1", "clear_m2")]
    steps.sort(key=lambda s: s[0])
    rows = []
    for time, kind, item in steps:
        if kind == "focus":
            state["wfs_z"] += item["value"]
        elif kind == "coma":
            state["wfs_cc_x"] += item["value"]
            state["wfs_cc_y"] += item["value2"]
        elif kind == "m1focus":
            state["m1spherical_z"] += item["value"]
        elif kind == "clear_m1":
            state["m1spherical_z"] = 0.0
        elif kind == "clear_m2":
            state.update(wfs_z=0.0, wfs_cc_x=0.0, wfs_cc_y=0.0)
        rows.append(dict(time=time, **state))
    return _table(rows, M2_COLUMNS)


def m1_consistency(sends, events, analyses, nightdir, m1):
    """
    Compare the fitted cumulative M1 terms with what the sends should have produced, -sum(frac x gain x masked
    Zernike term), over the last span between clears that had M1 sends. Nights normally end with a clear, so the end
    of the night itself would only compare an empty mirror. Returns {term: (fitted, expected)}.
    """
    m1_sends = sends[sends["kind"] == "m1"]
    if len(m1) == 0 or len(m1_sends) == 0:
        return {}
    clears = np.sort(events["time"][events["kind"] == "clear_m1"])
    last_send = np.max(m1_sends["time"])
    before, after = clears[clears <= last_send], clears[clears > last_send]
    since = before[-1] if len(before) else None
    until = after[0] if len(after) else None
    in_span = np.ones(len(m1), dtype=bool) if until is None else m1["time"] < until
    if not np.any(in_span):
        return {}
    expected = dict.fromkeys(FIT_TERMS, 0.0)
    analysed_at = dict(zip(analyses["file"], analyses["time"]))
    for s in m1_sends:
        if since is not None and s["time"] < since:
            continue
        masked = nightdir / f"{s['file']}.masked.zernike"
        if not masked.exists():
            continue
        zv = load_masked_zv(masked)
        t = analysed_at.get(s["file"], s["time"])
        key = selected_wfs(events, t)
        gain = gain_at(events, "m1_gain", t, default_gain(key, "m1_gain", 0.5) if key else 0.5, key)
        for term in FIT_TERMS:
            if term in zv.coeffs:
                expected[term] -= s["frac"] * gain * zv[term].value
    last = m1[in_span][-1]
    return {term: (float(last[term]), expected[term]) for term in FIT_TERMS}


FLAG_COLUMNS = [("time", TIME), ("kind", str), ("file", str), ("detail", str), ("visit", int)]
M1_FOCUS_GATE = 150.0  # um; wfssrv blocks M1 corrections when pending focus exceeds this (signed comparison)


def _spher_detail(files, frames, events, analysed_at, nightdir):
    """How much spherical was being ignored, and the forces a full correction would have needed."""
    z11 = fmax = frms = 0.0
    wfs_of = dict(zip(frames["file"], frames["wfs"]))
    time_of = {**dict(zip(frames["file"], frames["time"])), **analysed_at}
    for f in files:
        stem = f.removesuffix(".fits")
        zfile = nightdir / f"{stem}.reanalyze.zernike"
        if not zfile.exists():
            zfile = nightdir / f"{f}.zernike"
        if not zfile.exists() or wfs_of.get(f, "unknown") == "unknown":
            continue
        try:
            zv = _load_zv(zfile)
            wfs = get_wfs(wfs_of[f])
            gain = gain_at(events, "m1_gain", time_of[f], wfs.m1_gain, wfs_of[f])
            t, _, _ = calculate_primary_at(wfs, zv, ["Z22"], gain)
        except Exception as e:
            log.warning(f"Couldn't size ignored spherical for {f}: {e}")
            continue
        z11 = max(z11, abs(float(zv["Z11"].value)) if "Z11" in zv.coeffs else 0.0)
        fmax = max(fmax, float(np.max(np.abs(t["force"]))))
        frms = max(frms, _rms(t["force"]))
    return z11, fmax, frms


def find_flags(frames, sends, events, analyses, forces, nightdir):
    """Events worth a reviewer's attention, each tied to the visit it happened in."""
    visit_of = dict(zip(frames["file"], frames["visit"]))
    analysed_at = dict(zip(analyses["file"], analyses["time"]))  # wfssrv used the gains in effect at analysis
    flags = []

    def add(time, kind, file, detail):
        if file in visit_of:
            visit = int(visit_of[file])
        else:  # no frame named: use the visit of the latest frame at or before the event
            idx = int(np.searchsorted(frames["time"], time, side="right")) - 1
            visit = int(frames["visit"][idx]) if idx >= 0 else -1
        flags.append(dict(time=time, kind=kind, file=file, detail=detail, visit=visit))

    spher = np.asarray(analyses["spher"], dtype=bool)
    i = 0
    while i < len(analyses):
        if not spher[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(analyses) and spher[j + 1]:
            j += 1
        files = list(analyses["file"][i:j + 1])
        z11, fmax, frms = _spher_detail(files, frames, events, analysed_at, nightdir)
        add(analyses["time"][i], "spher_ignored", files[0],
            f"{analyses['time'][i]} to {analyses['time'][j]} UTC, {len(files)} frames, Z11 up to {z11:.0f} nm; "
            f"full correction would need max |F| {fmax:.0f} N, RMS {frms:.0f} N")
        i = j + 1

    for row in frames[frames["status"] == "ok"]:
        # mirror wfssrv: pending focus (m2_gain x measured correction) above +150 um blocks M1. The gate is signed.
        default = default_gain(row["wfs"], "m2_gain", 1.0)
        pending = gain_at(events, "m2_gain", analysed_at.get(row["file"], row["time"]), default, row["wfs"]) * row["focus"]
        if pending > M1_FOCUS_GATE:
            add(row["time"], "m1_blocked_focus", row["file"], f"pending focus {pending:.0f} um")
    for e in events[events["kind"] == "validity"]:
        add(e["time"], "validity", e["file"], e["detail"])
    for e in events[events["kind"] == "error"]:
        add(e["time"], "error", e["file"], e["detail"])
    for s in sends[sends["kind"] == "m1"]:
        if s["applied"] != "full":
            add(s["time"], "m1_partial", s["file"], f"{s['applied']} ({s['frac']:.0%} assumed applied)")
        if s["forces_file"] not in forces:
            add(s["time"], "missing_forces", s["file"], f"{s['forces_file']} not in the archive")
    flags.sort(key=lambda f: f["time"])
    return _table(flags, FLAG_COLUMNS)


VISIT_COLUMNS = [
    ("visit", int), ("wfs", str), ("catid", str), ("start", TIME), ("end", TIME), ("n_frames", int), ("n_ok", int),
    ("n_failed", int), ("el", float), ("d_az", float), ("d_el", float), ("slew_sep", float), ("gap", float),
    ("t_first_ok", float), ("settle_def", str), ("focus_settle_thresh", float), ("focus_unc_at_settle", float),
    ("focus_settle_conv", float), ("iq_settle_conv", float), ("coma_settle_thresh", float),
    ("settle_last_send", float), ("first_focus", float), ("first_focus_unc", float), ("first_cc_x", float),
    ("first_cc_y", float), ("first_wavefront_rms", float), ("first_iq", float), ("m1_resid_first", float),
    ("m1_resid_last", float), ("m1_force_first", float),
] + [(f"{k}_{w}", int if k == "n_sends" else float)
     for w in ("settle", "visit")
     for k in ("n_sends", "net_focus", "net_m1focus", "net_cc_x", "net_cc_y", "m1_sent_rms", "m1_sent_surf_rms")] \
  + [("flags", str)]


@dataclass
class Night:
    path: Path
    params: dict
    frames: Table
    visits: Table
    sends: Table
    events: Table
    analyses: Table
    flags: Table
    m1: Table
    m2: Table
    m1_maps: list = field(default_factory=list)
    m1_check: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def load_night(path, params=None):
    """
    Everything the nightly report needs for the night archived in directory `path` (named YYYYMMDD).
    """
    path = Path(path)
    params = {**SETTLE_DEFAULTS, **(params or {})}
    notes = []

    frames, counts = read_frames(path)
    # a stale copy of another night can be archived with this one; keep the frames to the window the log uses
    start, end = (np.datetime64(t, "ms") for t in night_window(path.name))
    inside = (frames["time"] >= start) & (frames["time"] < end)
    counts["n_frames_outside"] = int(np.sum(~inside))
    frames = frames[inside]
    if counts["n_frames_outside"]:
        notes.append(f"{counts['n_frames_outside']} frames outside the night window were ignored")
    logfile = path / "wfs.log"
    if logfile.exists():
        logdata = parse_log(logfile, path.name)
    else:
        logdata = empty_logdata()
        notes.append("wfs.log missing: no sends, clears, or cumulative accounting")
    counts.update(n_log_lines=logdata.n_lines, n_outside=logdata.n_outside, n_unrecognized=logdata.n_unrecognized)
    if logdata.n_outside:
        notes.append(f"{logdata.n_outside} log lines outside the night window were ignored")
    if counts["n_unreadable"] or counts["n_untimed"]:
        notes.append(f"skipped {counts['n_unreadable']} unreadable and {counts['n_untimed']} untimed frames")
    if len(frames) and not np.any(frames["status"] == "ok"):
        notes.append("no reanalyze outputs: frame measurements and settle times are missing")

    sends, analyses = logdata.sends, logdata.analyses
    spher_of = dict(zip(analyses["file"], analyses["spher"]))
    for i, f in enumerate(frames["file"]):
        if f in spher_of:
            frames["spher"][i] = int(bool(spher_of[f]))
    apply_validity(frames, logdata.events, analyses)

    visit_rows, offsets = build_visits(frames, params)
    events = logdata.events
    if offsets:
        events = _table(sorted([dict(zip(events.colnames, r)) for r in events] + offsets, key=lambda e: e["time"]),
                        EVENT_COLUMNS)
    m1_frame_metrics(frames, path, events, analyses)

    mmt = get_mmt()
    forces = {}
    for ff in set(sends["forces_file"][sends["kind"] == "m1"]):
        if (path / ff).exists():
            forces[ff] = read_forces(path / ff, mmt.actcoor["act_id"])

    visit_of = dict(zip(frames["file"], frames["visit"]))
    send_visit = np.array([visit_of.get(f, -1) for f in sends["file"]], dtype=int)
    flags = find_flags(frames, sends, events, analyses, forces, path)
    for v in visit_rows:
        vframes = frames[frames["visit"] == v["visit"]]
        v.update(visit_settle(vframes, sends[send_visit == v["visit"]], params, forces, mmt))
        v["flags"] = ",".join(sorted(set(flags["kind"][flags["visit"] == v["visit"]])))

    m1, maps = m1_accounting(sends, events, forces, mmt)
    return Night(
        path=path, params=params, frames=frames, visits=_table(visit_rows, VISIT_COLUMNS), sends=sends,
        events=events, analyses=analyses, flags=flags, m1=m1, m2=m2_accounting(sends, events), m1_maps=maps,
        m1_check=m1_consistency(sends, events, analyses, path, m1), counts=counts, warnings=notes,
    )
