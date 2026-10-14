# Nightly WFS Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `nightreport` console script to mmtwfs. For one archived night it builds a self-contained HTML report and
data files covering slews, settle times, corrections sent, clears, and cumulative M1/M2 state.

**Architecture:**
- `mmtwfs/nightlog.py` turns a night directory into tables and returns them in a `Night` dataclass from `load_night()`.
  Its inputs are `wfs.log`, the FITS headers, the reanalyze `.output` files, and the online `.forces` and
  `.masked.zernike` files. wfssrv will reuse this in Phase 2.
- `mmtwfs/scripts/nightreport.py` renders a `Night` to HTML, CSV, and JSON, and holds the CLI.
- `mmtwfs/outputs.py` takes over reanalyze's CSV column constants, so importing them doesn't build five WFS objects.

**Tech Stack:** Python ≥3.13, numpy, astropy (Table, fits, coordinates), matplotlib (Agg), pytest. No new
dependencies.

**Spec:** `docs/superpowers/specs/2026-10-14-nightly-wfs-report-design.md`

## Global Constraints

- Python ≥3.13. flake8 clean at `--max-line-length=127`.
- No new runtime dependencies.
- No network, Redis, or hardware access anywhere in `nightlog.py` or `nightreport.py`.
- Never edit the legacy `f9` block in `mmtwfs/config.py`. This code only reads WFS configs.
- Log timestamps are MST, a fixed UTC−7 with no DST.
- The night window for directory `YYYYMMDD` is local noon on YYYYMMDD−1 to local noon on YYYYMMDD, which is 19:00 UTC
  on YYYYMMDD−1 to 19:00 UTC on YYYYMMDD.
- M1 and M2 cumulative state starts at zero for every night directory.
- Times in all tables are UTC `numpy.datetime64[ms]`. Durations are float seconds.
- All output files are written to a temp name and then `os.replace`d. `nightreport_summary.json` is written last,
  because its presence means "done".
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Pushing happens only when the user asks.
- Run tests with the mmtwfs conda env's python, e.g. `python -m pytest`. Base tox is broken.

## Review Focus

1. **A night with frames but no `wfs.log`, or with a log but no reanalyze outputs.** A report must still be written,
   with a warning, and not a traceback.
2. **A log that runs past the night**, like 20261005, which continues to 2026-10-07. Lines outside the window must not
   create sends or clears. They must not move the M1 state.
3. **The cell rejects or partially applies forces** (`Bad force data`, `applied N%`). M1 accounting must follow
   `MMT.correct_primary` exactly: `frac` × forces, with `frac` = 1 for an unexpected reply.
4. **A clear logged while wfssrv is disconnected.** The `Resetting hexapod's WFS offsets` line still appears, but it
   must not reset the M2 sums.
5. **Binospec headers**: `DATE-OBS` only in HDU 1, and sexagesimal `RA`/`DEC` strings. These must still produce
   correctly timed frames and visits.

Every item has a test in the task that owns that code. The tests are in Tasks 2, 3, 7, and 9.

---

### Task 1: Move the reanalyze CSV columns into `mmtwfs/outputs.py` and add `read_output`

**Files:**
- Create: `mmtwfs/outputs.py`
- Modify: `mmtwfs/scripts/reanalyze.py:41-44` (remove the constant definitions and import them instead)
- Test: `mmtwfs/tests/test_outputs.py`

**Interfaces:**
- Produces:
  - `CSV_HEADER: str`
  - `CSV_COLUMNS: list[str]`
  - `read_output(path: Path) -> dict | None`. It returns a dict keyed by `CSV_COLUMNS`. Numeric columns are floats
    and `time`/`wfs`/`file`/`method` are strings. An empty file returns `None`, and a wrong field count raises
    `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# mmtwfs/tests/test_outputs.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math

import pytest

from mmtwfs.outputs import CSV_COLUMNS, CSV_HEADER, read_output

LINE = ("2026-10-05T01:54:18,mmirs,mmirs_wfs_0001.fits,12.0,1.1959,188.66,56.74,12,11.53,12.3,244.43,167.91,"
        "-559.21,2188.5,9052.31,-0.0,5.511,0.0,256.99,232.50,4.886,5.439,2.442,2.719,0.083,16.34,1387.4,2220.4,"
        "full,2.805")


def test_columns_match_header():
    assert CSV_COLUMNS == CSV_HEADER.strip().split(",")
    assert CSV_COLUMNS[-2:] == ["method", "delivered_iq"]


def test_read_output(tmp_path):
    p = tmp_path / "a.output"
    p.write_text(LINE + "\n")
    row = read_output(p)
    assert row["file"] == "mmirs_wfs_0001.fits"
    assert row["method"] == "full"
    # "focus" is the hexapod focus position (TRANSZ); "focerr" is the measured focus correction
    assert row["focus"] == pytest.approx(9052.31)
    assert row["focerr"] == pytest.approx(-0.0)
    assert row["delivered_iq"] == pytest.approx(2.805)
    assert isinstance(row["raw_vlt_seeing"], float)


def test_read_output_pads_old_lines(tmp_path):
    p = tmp_path / "old.output"
    p.write_text(",".join(LINE.split(",")[:-2]) + "\n")
    row = read_output(p)
    assert row["method"] == "full"
    assert math.isnan(row["delivered_iq"])


def test_read_output_empty_and_bad(tmp_path):
    empty = tmp_path / "e.output"
    empty.write_text("")
    assert read_output(empty) is None
    bad = tmp_path / "b.output"
    bad.write_text("1,2,3\n")
    with pytest.raises(ValueError):
        read_output(bad)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest mmtwfs/tests/test_outputs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmtwfs.outputs'`

- [ ] **Step 3: Implement**

```python
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
    ``delivered_iq`` columns existed are padded with "full" and NaN. Returns None for an empty file.
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
            row[col] = float(row[col])
    return row
```

In `mmtwfs/scripts/reanalyze.py`, delete the four lines that define `CSV_HEADER` and `CSV_COLUMNS` (lines 41–44), and
add `from mmtwfs.outputs import CSV_HEADER, CSV_COLUMNS` next to the other `mmtwfs` imports. The names stay importable
from `mmtwfs.scripts.reanalyze`, and `test_reanalyze.py` relies on that.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_outputs.py mmtwfs/tests/test_reanalyze.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/outputs.py mmtwfs/tests/test_outputs.py mmtwfs/scripts/reanalyze.py
git commit -m "move reanalyze output columns into mmtwfs.outputs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Log parser

**Files:**
- Create: `mmtwfs/nightlog.py`
- Test: `mmtwfs/tests/test_nightlog.py`

**Interfaces:**
- Produces:
  - `LOCAL_UTC_OFFSET: timedelta`
  - `SETTLE_DEFAULTS: dict`
  - `night_window(night: str) -> (datetime, datetime)`, giving naive UTC datetimes.
  - `parse_log(logfile: Path, night: str) -> LogData`.
  - `LogData` dataclass:
    - `sends: Table`, with columns `SEND_COLUMNS`: `time, kind, file, value, value2, forces_file, fmax, fmin, applied,
      frac`.
    - `events: Table`, with columns `EVENT_COLUMNS`: `time, kind, detail, file`.
    - `analyses: Table`, with columns `ANALYSIS_COLUMNS`: `time, file, spher`.
    - `n_lines: int`, `n_outside: int`, `n_unrecognized: int`.
  - `_table(rows: list[dict], columns: list[tuple[str, dtype]]) -> Table`. It keeps the schema even when `rows` is
    empty.
- Send kinds are `m1`, `m1focus`, `focus`, and `coma`. `file` is a frame basename such as `mmirs_wfs_0325.fits`.
- Event kinds are `clear_m1`, `clear_m2`, `m1_gain`, `m2_gain`, `wfs_select`, `connect`, `validity`, and `error`. Task 4
  adds `offset`. A gain event's `detail` is the gain as a string.

- [ ] **Step 1: Write the failing tests**

```python
# mmtwfs/tests/test_nightlog.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from datetime import datetime

import numpy as np
import pytest

from mmtwfs.nightlog import night_window, parse_log

# real lines from /Volumes/Samsung_4TB/wfsdat/20261003/wfs.log (2026-10-03 01:44 MST = 08:44 UTC)
SEND_BLOCK = """\
2026-10-03 01:44:42,100 - WFSsrv - INFO - Analyzing /home/mmtop/wfsdat/mmirs_wfs_0325.fits...
2026-10-03 01:44:42,101 - WFSsrv - INFO - Only ignoring the high-order spherical terms ['Z22']...
2026-10-03 01:44:45,000 - tornado.access - INFO - 200 GET /files?format=json (128.196.100.10) 3.60ms
2026-10-03 01:44:47,935 - WFSsrv - INFO - Sending 35.7 um focus to secondary...
2026-10-03 01:44:49,777 - Secondary - INFO - Moving f5 hexapod -1.688 arcsec about the center of curvature along the x axis...
2026-10-03 01:44:49,777 - WFSsrv - INFO - Sending -1.7 arcsec/-9.7 arcsec CC_X/CC_Y to secondary...
2026-10-03 01:44:51,077 - WFSsrv - INFO - M1 corrections...
2026-10-03 01:44:51,078 - Telescope - INFO - Sending forces from /home/mmtop/wfsdat/mmirs_wfs_0325.fits.forces...
2026-10-03 01:44:51,078 - Telescope - INFO - Using command, /mmt/scripts/cell_send_forces /home/mmtop/wfsdat/mmirs_wfs_0325.fits.forces, to apply forces...
2026-10-03 01:44:51,343 - Telescope - INFO - ...forces successfully applied in full.
2026-10-03 01:44:51,343 - Secondary - INFO - Moving f5 hexapod 37.35271826100699 um in Z to correct spherical aberration...
2026-10-03 01:44:51,344 - WFSsrv - INFO - Maximum force 62.3 N
2026-10-03 01:44:51,344 - WFSsrv - INFO - Minimum force -80.59 N
2026-10-03 01:44:51,344 - WFSsrv - INFO - Adjusting M1 focus by 37.4 um
"""

BAD_FORCE_BLOCK = """\
2026-10-03 04:00:00,000 - WFSsrv - INFO - Analyzing /home/mmtop/wfsdat/mmirs_wfs_0475.fits...
2026-10-03 04:00:03,185 - Telescope - INFO - Sending forces from /home/mmtop/wfsdat/mmirs_wfs_0475.fits.forces...
2026-10-03 04:00:03,337 - Telescope - ERROR - ...got unexpected reply from cell command: Bad force data

2026-10-03 04:00:03,337 - Telescope - ERROR - \t stderr: 
2026-10-03 04:00:03,337 - Secondary - INFO - Moving f5 hexapod -0.1904877834978918 um in Z to correct spherical aberration...
2026-10-03 04:00:03,338 - WFSsrv - INFO - Maximum force 1.98 N
"""


def _write(tmp_path, text):
    p = tmp_path / "wfs.log"
    p.write_text(text)
    return p


def test_night_window():
    start, end = night_window("20261003")
    assert start == datetime(2026, 10, 2, 19, 0)
    assert end == datetime(2026, 10, 3, 19, 0)


def test_parse_send_block(tmp_path):
    d = parse_log(_write(tmp_path, SEND_BLOCK), "20261003")
    assert list(d.analyses["file"]) == ["mmirs_wfs_0325.fits"]
    assert d.analyses["time"][0] == np.datetime64("2026-10-03T08:44:42.100")
    assert not d.analyses["spher"][0]
    kinds = list(d.sends["kind"])
    assert kinds == ["focus", "coma", "m1", "m1focus"]
    focus, coma, m1, m1focus = d.sends
    assert focus["value"] == pytest.approx(35.7)
    assert focus["file"] == "mmirs_wfs_0325.fits"
    assert (coma["value"], coma["value2"]) == pytest.approx((-1.7, -9.7))
    assert m1["forces_file"] == "mmirs_wfs_0325.fits.forces"
    assert m1["file"] == "mmirs_wfs_0325.fits"
    assert m1["applied"] == "full" and m1["frac"] == 1.0
    assert (m1["fmax"], m1["fmin"]) == pytest.approx((62.3, -80.59))
    assert m1focus["value"] == pytest.approx(37.3527, rel=1e-4)
    assert m1focus["file"] == "mmirs_wfs_0325.fits"
    assert len(d.events) == 0


def test_unexpected_cell_reply_is_frac_one_and_not_an_error_event(tmp_path):
    d = parse_log(_write(tmp_path, BAD_FORCE_BLOCK), "20261003")
    m1 = d.sends[d.sends["kind"] == "m1"][0]
    assert m1["applied"] == "unexpected"
    assert m1["frac"] == 1.0
    # the reply and the "\t stderr:" line are part of the send, not separate error events
    assert len(d.events[d.events["kind"] == "error"]) == 0


@pytest.mark.parametrize("reply, applied, frac", [
    ("...applied 42.0% of requested forces.", "partial", 0.42),
    ("...forces rejected!", "rejected", 0.0),
    ("...unable to apply forces!", "rejected", 0.0),
])
def test_cell_replies(tmp_path, reply, applied, frac):
    text = ("2026-10-03 01:00:00,000 - Telescope - INFO - Sending forces from /x/mmirs_wfs_0001.fits.forces...\n"
            f"2026-10-03 01:00:00,100 - Telescope - ERROR - {reply}\n")
    m1 = parse_log(_write(tmp_path, text), "20261003").sends[0]
    assert (m1["applied"], m1["frac"]) == (applied, pytest.approx(frac))


def test_unknown_reply(tmp_path):
    text = "2026-10-03 01:00:00,000 - Telescope - INFO - Sending forces from /x/mmirs_wfs_0001.fits.forces...\n"
    m1 = parse_log(_write(tmp_path, text), "20261003").sends[0]
    assert (m1["applied"], m1["frac"]) == ("unknown", 1.0)


def test_clears_gains_and_connection(tmp_path):
    text = """\
2026-10-02 23:42:01,546 - WFSsrv - INFO - Setting M1 gain to 0.5
2026-10-02 23:42:01,547 - WFSsrv - INFO - Setting M2 gain to 1.0
2026-10-02 23:42:05,000 - WFSsrv - WARNING - Invalid M1 gain: 1.5
2026-10-02 23:42:05,001 - WFSsrv - INFO - Setting M1 gain to 1.5
2026-10-02 23:42:10,710 - WFS - INFO - Clearing WFS corrections from M1 and m1spherical offsets from M2.
2026-10-02 23:42:10,717 - Telescope - INFO - Clearing forces and spherical aberration focus offsets...
2026-10-02 23:42:10,718 - Secondary - INFO - Resetting hexapod's spherical aberration offset to 0...
2026-10-02 23:42:11,000 - WFS - INFO - Clearing WFS offsets from M2's hexapod.
2026-10-02 23:42:11,001 - Secondary - INFO - Resetting hexapod's WFS offsets to 0...
2026-10-02 23:50:00,000 - WFSsrv - INFO - MMIRS WFS is disconnected.
2026-10-02 23:51:00,000 - Secondary - INFO - Resetting hexapod's WFS offsets to 0...
2026-10-02 23:52:00,000 - WFSsrv - INFO - MMIRS WFS is connected.
2026-10-02 23:53:00,000 - WFSsrv - INFO - Setting mmirs
"""
    d = parse_log(_write(tmp_path, text), "20261003")
    ev = d.events
    assert list(ev["kind"]) == ["m1_gain", "m2_gain", "clear_m1", "clear_m2", "connect", "connect", "wfs_select"]
    assert list(ev["detail"][:2]) == ["0.5", "1.0"]
    assert list(ev["detail"][4:]) == ["disconnected", "connected", "mmirs"]


def test_validity_spher_and_errors(tmp_path):
    text = """\
2026-10-02 19:14:40,000 - WFSsrv - INFO - Analyzing /home/mmtop/wfsdat/mmirs_wfs_0014.fits...
2026-10-02 19:14:40,001 - WFSsrv - INFO - Ignoring all spherical terms ['Z11', 'Z22']...
2026-10-02 19:14:49,775 - WFSsrv - WARNING - /home/mmtop/wfsdat/mmirs_wfs_0014.fits: only focus corrections valid.
2026-10-02 19:15:00,000 - WFSsrv - ERROR - /home/mmtop/wfsdat/mmirs_wfs_0014.fits: wavefront fit too poor; no valid corrections
2026-10-02 19:16:00,000 - WFSsrv - ERROR - Wavefront measurement failed: /home/mmtop/wfsdat/mmirs_wfs_0015.fits
2026-10-02 19:17:00,000 - Secondary - ERROR - hexapod socket timed out
"""
    d = parse_log(_write(tmp_path, text), "20261003")
    assert bool(d.analyses["spher"][0])
    ev = d.events
    assert list(ev["kind"]) == ["validity", "validity", "error"]
    assert ev["file"][0] == "mmirs_wfs_0014.fits"
    assert ev["detail"][2] == "hexapod socket timed out"


def test_night_window_drops_outside_lines(tmp_path):
    text = """\
2026-10-02 11:59:00,000 - Telescope - INFO - Clearing forces and spherical aberration focus offsets...
2026-10-02 12:00:00,000 - Telescope - INFO - Clearing forces and spherical aberration focus offsets...
2026-10-07 21:13:07,013 - WFSsrv - INFO - Sending 10.0 um focus to secondary...
"""
    d = parse_log(_write(tmp_path, text), "20261003")
    assert len(d.events) == 1
    assert d.events["time"][0] == np.datetime64("2026-10-02T19:00:00.000")
    assert len(d.sends) == 0
    assert d.n_outside == 2


def test_empty_log_keeps_schema(tmp_path):
    d = parse_log(_write(tmp_path, ""), "20261003")
    assert len(d.sends) == 0 and "frac" in d.sends.colnames
    assert len(d.events) == 0 and "detail" in d.events.colnames
    assert d.sends["time"].dtype == np.dtype("datetime64[ms]")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmtwfs.nightlog'`

- [ ] **Step 3: Implement**

```python
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
        return np.datetime64("NaT")
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py
git commit -m "parse wfssrv logs into sends, events, and analyses

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Frame reader

**Files:**
- Modify: `mmtwfs/nightlog.py` (append)
- Test: `mmtwfs/tests/test_nightlog.py` (append)

**Interfaces:**
- Consumes: `read_output` (Task 1), `_table` (Task 2).
- Produces:
  - `read_frames(nightdir: Path) -> (Table, dict)`. The table has `FRAME_COLUMNS`:
    - from the header: `time, file, wfs, exptime, catid, catra, catdec, ra, dec, raoff, decoff, az, el, transx,
      transy, transz, tiltx, tilty`
    - `status`
    - from the output: `focus`, which is the `.output` **`focerr`** column, the measured M2 focus correction in µm at
      gain 1 after errorbar shrinkage. The `.output` `focus` column is the hexapod position (`TRANSZ`) and isn't used.
      Also `cc_x_err, cc_y_err` (measured coma corrections, ″), `wavefront_rms, residual_rms, raw_vlt_seeing,
      delivered_iq, method`.
    - `focus_unc`: the focus uncertainty in µm, from the Z04 errorbar. NaN until Task 5.
    - `spher`, an int8: 1, 0, or −1 for "not analyzed online"
    - `visit`, an int, −1 until Task 4
    - `m1_resid_rms`, `m1_force_rms`, NaN until Task 5

    Rows are sorted by `time`. `catra`, `catdec`, `ra`, and `dec` are in **degrees**. The dict is
    `{"n_unreadable": int, "n_untimed": int}`.
  - `frame_wfs(name: str, hdr: dict) -> str`, returning one of `mmirs`, `binospec`, `newf9`, `f9`, or `unknown`.

- [ ] **Step 1: Write the failing tests**

```python
# append to mmtwfs/tests/test_nightlog.py
from astropy.io import fits

from mmtwfs.nightlog import frame_wfs, read_frames

OUTPUT_LINE = ("2026-10-03T08:44:38,mmirs,{name},12.0,1.2,188.6,56.7,12,11.5,12.3,244.4,167.9,-559.2,2188.5,"
               "9052.3,{focus},5.5,0.4,-0.3,256.9,232.5,4.9,5.4,2.4,0.9,0.08,16.3,1387.4,2220.4,full,1.05")


def _frame(d, name, hdr0, hdr1=None, data=True):
    h0 = fits.Header(list(hdr0.items()))
    hdus = [fits.PrimaryHDU(np.zeros((2, 2), dtype=np.int16) if data else None, header=h0)]
    if hdr1 is not None:
        hdus.append(fits.ImageHDU(np.zeros((2, 2), dtype=np.int16), header=fits.Header(list(hdr1.items()))))
    fits.HDUList(hdus).writeto(d / name)


MMIRS_HDR = {"DATE-OBS": "2026-10-03T08:44:38", "WFSNAME": "mmirs2", "EXPTIME": 12.0, "CATID": "HIP109911",
             "CATRA": 22.26, "CATDEC": 30.1, "RA": 22.26, "DEC": 30.09, "RAOFF": 1.9, "DECOFF": -34.6,
             "AZ": 80.0, "EL": 60.0, "TRANSX": -600.0, "TRANSY": 2400.0, "TRANSZ": 9000.0, "TILTX": 260.0,
             "TILTY": 160.0}


def test_frame_wfs():
    assert frame_wfs("mmirs_wfs_0001.fits", {}) == "mmirs"
    assert frame_wfs("wfs_ff_cal_img_2025.02.03T034813.592.fits", {}) == "binospec"
    assert frame_wfs("f9wfs_20240426-212258.fits", {}) == "newf9"
    assert frame_wfs("x.fits", {"INSTRUME": "Apogee"}) == "f9"
    assert frame_wfs("x.fits", {}) == "unknown"


def test_read_frames_status_and_columns(tmp_path):
    _frame(tmp_path, "mmirs_wfs_0002.fits", MMIRS_HDR)
    _frame(tmp_path, "mmirs_wfs_0001.fits", {**MMIRS_HDR, "DATE-OBS": "2026-10-03T08:44:00"})
    _frame(tmp_path, "mmirs_wfs_0003.fits", {**MMIRS_HDR, "DATE-OBS": "2026-10-03T08:45:00"})
    _frame(tmp_path, "mmirs_wfs_0001.coadded_spot.fits", {})
    (tmp_path / "mmirs_wfs_0002.output").write_text(OUTPUT_LINE.format(name="mmirs_wfs_0002.fits", focus=12.5))
    (tmp_path / "mmirs_wfs_0001.failed").write_text("")
    frames, counts = read_frames(tmp_path)
    assert list(frames["file"]) == ["mmirs_wfs_0001.fits", "mmirs_wfs_0002.fits", "mmirs_wfs_0003.fits"]
    assert list(frames["status"]) == ["failed", "ok", "unanalyzed"]
    assert frames["focus"][1] == pytest.approx(12.5)  # from the .output focerr column, not the hexapod position
    assert frames["cc_x_err"][1] == pytest.approx(5.5)
    assert np.isnan(frames["focus"][0])
    assert frames["catra"][1] == pytest.approx(22.26 * 15)
    assert frames["catid"][1] == "HIP109911"
    assert frames["wfs"][1] == "mmirs"
    assert list(frames["spher"]) == [-1, -1, -1]
    assert counts == {"n_unreadable": 0, "n_untimed": 0}


def test_read_frames_binospec_headers(tmp_path):
    # Binospec keeps DATE-OBS/EXPTIME in HDU 1 and writes sexagesimal RA/DEC
    hdr0 = {"MJD": 60709.15877, "CATID": "SN2025kg", "CATRA": 3.70788333, "CATDEC": -22.50586944,
            "RA": "03:42:30.083", "DEC": "-22:29:59.850", "RAOFF": 331.3, "DECOFF": -63.6, "AZ": 206.3, "EL": 31.3}
    _frame(tmp_path, "wfs_ff_cal_img_2025.02.03T034813.592.fits", hdr0,
           hdr1={"DATE-OBS": "2025-02-03T03:48:13.592", "EXPTIME": 20.0})
    frames, _ = read_frames(tmp_path)
    assert frames["time"][0] == np.datetime64("2025-02-03T03:48:13.592")
    assert frames["exptime"][0] == 20.0
    assert frames["wfs"][0] == "binospec"
    assert frames["ra"][0] == pytest.approx((3 + 42 / 60 + 30.083 / 3600) * 15)
    assert frames["dec"][0] == pytest.approx(-(22 + 29 / 60 + 59.85 / 3600))


def test_read_frames_mjd_fallback_and_untimed(tmp_path):
    _frame(tmp_path, "mmirs_wfs_0001.fits", {"MJD": 61316.26998, "CATID": "X"})
    _frame(tmp_path, "mmirs_wfs_0002.fits", {"CATID": "X"})
    (tmp_path / "mmirs_wfs_0003.fits").write_text("not a fits file")
    frames, counts = read_frames(tmp_path)
    assert len(frames) == 1
    assert counts == {"n_unreadable": 1, "n_untimed": 1}


def test_read_frames_empty_dir(tmp_path):
    frames, _ = read_frames(tmp_path)
    assert len(frames) == 0 and "delivered_iq" in frames.colnames
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v -k frame`
Expected: FAIL with `ImportError: cannot import name 'frame_wfs'`

- [ ] **Step 3: Implement**

Add these imports at the top of `mmtwfs/nightlog.py`:

```python
import warnings

import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

from mmtwfs.outputs import read_output
```

Then append:

```python
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


def read_frames(nightdir):
    """
    One row per WFS frame in a night directory, failed frames included, joined with its reanalyze output.
    """
    rows = []
    counts = {"n_unreadable": 0, "n_untimed": 0}
    for f in sorted(nightdir.glob("*.fits")):
        if f.name.count(".") > 1:
            continue  # derived products such as *.coadded_spot.fits
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py
git commit -m "read WFS frame headers and reanalyze outputs for a night

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Visits and offsets

**Files:**
- Modify: `mmtwfs/nightlog.py` (append)
- Test: `mmtwfs/tests/test_nightlog.py` (append)

**Interfaces:**
- Consumes: the `frames` Table from Task 3.
- Produces: `build_visits(frames: Table, params: dict) -> (list[dict], list[dict])`.
  - It sets `frames["visit"]` in place.
  - It returns `(visit_rows, offset_events)`. Each visit row has `visit, wfs, catid, start, end, n_frames, n_ok,
    n_failed, el, d_az, d_el, slew_sep, gap, t_first_ok`. `start` and `end` are datetime64. `gap` and `t_first_ok` are
    seconds, or NaN. `slew_sep` is in arcsec.
  - Each offset event is `dict(time, kind="offset", detail="<raoff>,<decoff>", file)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to mmtwfs/tests/test_nightlog.py
from mmtwfs.nightlog import FRAME_COLUMNS, SETTLE_DEFAULTS, _table, build_visits


def _frames(specs):
    """specs: list of (iso time, catid, catra_deg, catdec_deg, raoff, decoff, status, el)"""
    rows = [dict(time=np.datetime64(t), file=f"f{i:03d}.fits", wfs="mmirs", catid=c, catra=ra, catdec=dec,
                 ra=ra, dec=dec, raoff=ro, decoff=do, status=s, el=el, az=100.0)
            for i, (t, c, ra, dec, ro, do, s, el) in enumerate(specs)]
    return _table(rows, FRAME_COLUMNS)


def test_build_visits_splits_on_target_and_lists_offsets():
    frames = _frames([
        ("2026-10-03T06:40:00", "A", 300.0, 40.0, 0.0, 0.0, "ok", 50.0),
        ("2026-10-03T06:41:00", "A", 300.0, 40.0, 0.0, 0.0, "ok", 50.5),
        ("2026-10-03T06:42:00", "A", 300.0, 40.0, 2.0, -30.0, "ok", 51.0),
        ("2026-10-03T08:44:00", "B", 333.0, 30.0, 0.0, 0.0, "failed", 60.0),
        ("2026-10-03T08:44:30", "B", 333.0, 30.0, 0.0, 0.0, "ok", 60.1),
    ])
    visits, offsets = build_visits(frames, SETTLE_DEFAULTS)
    assert list(frames["visit"]) == [0, 0, 0, 1, 1]
    assert [v["catid"] for v in visits] == ["A", "B"]
    a, b = visits
    assert np.isnan(a["gap"]) and np.isnan(a["slew_sep"])
    assert b["gap"] == pytest.approx(2 * 3600 + 2 * 60)
    assert b["t_first_ok"] == pytest.approx(30.0)
    assert b["d_el"] == pytest.approx(60.0 - 51.0)
    assert b["n_frames"] == 2 and b["n_ok"] == 1 and b["n_failed"] == 1
    assert b["slew_sep"] > 20 * 3600
    assert [(o["file"], o["detail"]) for o in offsets] == [("f002.fits", "2.00,-30.00")]


def test_build_visits_fallback_without_catalog():
    nan = np.nan
    frames = _frames([
        ("2026-10-03T06:40:00", "", nan, nan, 0.0, 0.0, "ok", 50.0),
        ("2026-10-03T06:41:00", "", nan, nan, 0.0, 0.0, "ok", 50.0),
        ("2026-10-03T06:42:00", "", nan, nan, 0.0, 0.0, "ok", 50.0),
    ])
    frames["ra"] = [300.0, 300.0 + 10 / 3600, 301.0]
    frames["dec"] = [40.0, 40.0, 40.0]
    visits, _ = build_visits(frames, SETTLE_DEFAULTS)
    assert list(frames["visit"]) == [0, 0, 1]


def test_build_visits_empty():
    frames = _table([], FRAME_COLUMNS)
    visits, offsets = build_visits(frames, SETTLE_DEFAULTS)
    assert visits == [] and offsets == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v -k visits`
Expected: FAIL with `ImportError: cannot import name 'build_visits'`

- [ ] **Step 3: Implement**

```python
# append to mmtwfs/nightlog.py

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py
git commit -m "split a night's frames into target visits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Per-frame M1 metrics and the gain timeline

**Files:**
- Modify: `mmtwfs/nightlog.py` (append)
- Test: `mmtwfs/tests/test_nightlog.py` (append)

**Interfaces:**
- Consumes: `frames` (Task 3), `events` (Task 2).
- Produces:
  - `get_wfs(key: str) -> WFS`, a cached `WFSFactory(wfs=key)`.
  - `gain_at(events: Table, kind: str, time: datetime64, default: float) -> float`.
  - `m1_resid_rms(zv: ZernikeVector) -> float`, giving nm and excluding Z04.
  - `calculate_primary_at(wfs, zv, mask, gain) -> (forces Table, m1focus, zv_masked)`. This is
    `wfs.calculate_primary` run with `wfs.m1_gain` temporarily set to `gain`.
  - `focus_uncertainty(zv: ZernikeVector, wfs) -> float`: |Z04 errorbar| / `wfs.secondary.focus_trans`, in µm, or
    NaN if there's no Z04 errorbar.
  - `m1_frame_metrics(frames: Table, nightdir: Path, events: Table) -> None`. It fills `frames["m1_resid_rms"]`,
    `frames["m1_force_rms"]`, and `frames["focus_unc"]` in place for `ok` frames.

- [ ] **Step 1: Write the failing tests**

```python
# append to mmtwfs/tests/test_nightlog.py
from mmtwfs.nightlog import EVENT_COLUMNS, gain_at, get_wfs, m1_frame_metrics, m1_resid_rms
from mmtwfs.zernike import ZernikeVector


def test_gain_at():
    ev = _table([dict(time=np.datetime64("2026-10-03T08:00:00"), kind="m1_gain", detail="0.2"),
                 dict(time=np.datetime64("2026-10-03T09:00:00"), kind="m1_gain", detail="0.5"),
                 dict(time=np.datetime64("2026-10-03T08:30:00"), kind="m2_gain", detail="1.0")], EVENT_COLUMNS)
    assert gain_at(ev, "m1_gain", np.datetime64("2026-10-03T07:00:00"), 0.5) == 0.5
    assert gain_at(ev, "m1_gain", np.datetime64("2026-10-03T08:10:00"), 0.5) == 0.2
    assert gain_at(ev, "m1_gain", np.datetime64("2026-10-03T09:00:00"), 0.5) == 0.5
    assert gain_at(ev, "m2_gain", np.datetime64("2026-10-03T08:40:00"), 0.7) == 1.0


def test_m1_resid_rms_ignores_focus():
    assert m1_resid_rms(ZernikeVector(Z04=1000.0, Z05=100.0)) == pytest.approx(
        ZernikeVector(Z05=100.0).rms.value)


def test_m1_frame_metrics_prefers_masked_file(tmp_path):
    frames = _frames([("2026-10-03T08:44:38", "B", 333.0, 30.0, 0.0, 0.0, "ok", 60.0)])
    masked = ZernikeVector(Z05=200.0, Z11=50.0)
    masked.save(filename=tmp_path / "f000.fits.masked.zernike")
    ev = _table([dict(time=np.datetime64("2026-10-03T08:00:00"), kind="m1_gain", detail="0.2")], EVENT_COLUMNS)
    m1_frame_metrics(frames, tmp_path, ev)
    expected = get_wfs("mmirs").telescope.bending_forces(zv=masked.copy(), gain=0.2)["force"]
    assert frames["m1_force_rms"][0] == pytest.approx(np.sqrt(np.mean(np.asarray(expected) ** 2)))
    assert frames["m1_resid_rms"][0] == pytest.approx(m1_resid_rms(masked))


def test_m1_frame_metrics_recomputes_from_reanalyze(tmp_path):
    frames = _frames([("2026-10-03T08:44:38", "B", 333.0, 30.0, 0.0, 0.0, "ok", 60.0)])
    frames["spher"][0] = 1
    ZernikeVector(Z05=200.0, Z11=300.0).save(filename=tmp_path / "f000.reanalyze.zernike")
    m1_frame_metrics(frames, tmp_path, _table([], EVENT_COLUMNS))
    assert frames["m1_force_rms"][0] > 0
    # spher=1 masks Z11, so only the Z05 term survives
    assert frames["m1_resid_rms"][0] == pytest.approx(ZernikeVector(Z05=200.0).rms.value, rel=0.01)
    assert get_wfs("mmirs").m1_gain == 0.5  # temporary gain restored


def test_focus_uncertainty_from_reanalyze_errorbars(tmp_path):
    frames = _frames([("2026-10-03T08:44:38", "B", 333.0, 30.0, 0.0, 0.0, "ok", 60.0)])
    ZernikeVector(Z04=-1452.0, Z05=100.0, errorbars={"Z04": 114.4}).save(
        filename=tmp_path / "f000.reanalyze.zernike")
    m1_frame_metrics(frames, tmp_path, _table([], EVENT_COLUMNS))
    focus_trans = get_wfs("mmirs").secondary.focus_trans.value  # 40.8 nm/um
    assert frames["focus_unc"][0] == pytest.approx(114.4 / focus_trans, rel=1e-3)


def test_m1_frame_metrics_missing_files_leave_nan(tmp_path):
    frames = _frames([("2026-10-03T08:44:38", "B", 333.0, 30.0, 0.0, 0.0, "ok", 60.0)])
    m1_frame_metrics(frames, tmp_path, _table([], EVENT_COLUMNS))
    assert np.isnan(frames["m1_force_rms"][0])
```

`ZernikeVector.save` writes errorbars and modestart, and `load` restores them. If the errorbar round trip in
`test_focus_uncertainty_from_reanalyze_errorbars` doesn't work this way, read `zernike.py` `save`/`load` and adjust the
test's construction, not `focus_uncertainty`. `calculate_primary` uses `frac_error`, and that's 0 when there
are no errorbars, so no terms shrink. The recompute test's `rel=0.01` covers that.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v -k "gain_at or m1_ or focus_uncertainty"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add the imports `from mmtwfs.wfs import WFSFactory` and `from mmtwfs.zernike import ZernikeVector` at the top, then
append:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass. If `test_m1_resid_rms_ignores_focus` fails because `ZernikeVector.ignore` doesn't exist for an
absent key, check `zernike.py:1316`. It's guarded by `key in self.coeffs`, so it shouldn't.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py
git commit -m "compute per-frame M1 residual and proposed force RMS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Settle times and per-visit corrections

**Files:**
- Modify: `mmtwfs/nightlog.py` (append)
- Test: `mmtwfs/tests/test_nightlog.py` (append)

**Interfaces:**
- Consumes: visit rows (Task 4), frames with `visit` and the M1 columns (Tasks 4 and 5), and `sends` (Task 2).
- Produces:
  - `first_run(mask, k) -> int | None`.
  - `visit_settle(vframes: Table, vsends: Table, params: dict, forces: dict, telescope) -> dict`. `forces` maps
    `forces_file` to a force vector aligned to `telescope.actcoor["act_id"]`. The returned keys are:
    - `focus_settle_thresh, focus_unc_at_settle, focus_settle_conv, iq_settle_conv, coma_settle_thresh,
      settle_last_send, settle_def`
    - `first_focus, first_focus_unc, first_cc_x, first_cc_y, first_wavefront_rms, first_iq`
    - `m1_resid_first, m1_resid_last, m1_force_first`
    - for each window `w` in (`settle`, `visit`): `n_sends_<w>`, `net_focus_<w>`, `net_m1focus_<w>`, `net_cc_x_<w>`,
      `net_cc_y_<w>`, `m1_sent_rms_<w>`, `m1_sent_surf_rms_<w>`

    Durations are seconds from the visit's first frame, or NaN.

- [ ] **Step 1: Write the failing tests**

```python
# append to mmtwfs/tests/test_nightlog.py
from mmtwfs.nightlog import SEND_COLUMNS, first_run, visit_settle


class _FakeTelescope:
    def __init__(self, n_act=3, n_node=4):
        self._a2s = np.ones((n_act, n_node))

    def load_act2surf(self):
        return self._a2s


def _visit_frames(focus, iq=None, status=None, cc=None):
    t0 = np.datetime64("2026-10-03T08:44:00")
    n = len(focus)
    rows = []
    for i in range(n):
        rows.append(dict(time=t0 + np.timedelta64(30 * i, "s"), file=f"f{i:03d}.fits", wfs="mmirs",
                         status=(status or ["ok"] * n)[i], focus=focus[i], focus_unc=1.0,
                         cc_x_err=(cc or [0.0] * n)[i], cc_y_err=0.0, wavefront_rms=500.0,
                         delivered_iq=(iq or [1.0] * n)[i], m1_resid_rms=100.0 - i, m1_force_rms=5.0, visit=0))
    return _table(rows, FRAME_COLUMNS)


def test_first_run():
    assert first_run(np.array([False, True, False, True, True]), 2) == 3
    assert first_run(np.array([True, True]), 2) == 0
    assert first_run(np.array([True, False, True]), 2) is None


def test_focus_settle_threshold_needs_k_consecutive():
    vf = _visit_frames([80.0, 5.0, 40.0, 8.0, 6.0, 4.0])
    s = visit_settle(vf, _table([], SEND_COLUMNS), SETTLE_DEFAULTS, {}, _FakeTelescope())
    assert s["focus_settle_thresh"] == pytest.approx(90.0)  # frame 3 starts the first run of two
    assert s["first_focus"] == 80.0
    assert s["settle_def"] == "thresh/conv"


def test_focus_never_settles():
    vf = _visit_frames([80.0, 60.0, 50.0])
    s = visit_settle(vf, _table([], SEND_COLUMNS), SETTLE_DEFAULTS, {}, _FakeTelescope())
    assert np.isnan(s["focus_settle_thresh"])


def test_convergence_definitions():
    # plateau of |focus| in the last half is 30, so <= 40 counts as converged even though > F
    vf = _visit_frames([100.0, 90.0, 35.0, 32.0, 30.0, 29.0], iq=[2.0, 1.8, 1.2, 1.0, 1.0, 1.01])
    s = visit_settle(vf, _table([], SEND_COLUMNS), SETTLE_DEFAULTS, {}, _FakeTelescope())
    assert s["focus_settle_conv"] == pytest.approx(60.0)
    assert s["iq_settle_conv"] == pytest.approx(90.0)


def test_no_ok_frames():
    vf = _visit_frames([np.nan, np.nan], status=["failed", "failed"])
    s = visit_settle(vf, _table([], SEND_COLUMNS), SETTLE_DEFAULTS, {}, _FakeTelescope())
    assert np.isnan(s["focus_settle_thresh"]) and np.isnan(s["first_focus"])


def test_send_windows_and_f9_last_send():
    vf = _visit_frames([80.0, 40.0, 5.0, 4.0])
    t0 = vf["time"][0]
    sends = _table([
        dict(time=t0 + np.timedelta64(5, "s"), kind="focus", file="f000.fits", value=40.0),
        dict(time=t0 + np.timedelta64(6, "s"), kind="m1", file="f000.fits", forces_file="f000.fits.forces", frac=0.5),
        dict(time=t0 + np.timedelta64(35, "s"), kind="coma", file="f001.fits", value=1.0, value2=-2.0),
        dict(time=t0 + np.timedelta64(95, "s"), kind="focus", file="f003.fits", value=2.0),
    ], SEND_COLUMNS)
    forces = {"f000.fits.forces": np.array([2.0, 2.0, 2.0])}
    s = visit_settle(vf, sends, SETTLE_DEFAULTS, forces, _FakeTelescope())
    assert s["focus_settle_thresh"] == pytest.approx(60.0)
    assert s["n_sends_settle"] == 3 and s["n_sends_visit"] == 4
    assert s["net_focus_settle"] == 40.0 and s["net_focus_visit"] == 42.0
    assert (s["net_cc_x_visit"], s["net_cc_y_visit"]) == (1.0, -2.0)
    assert s["m1_sent_rms_visit"] == pytest.approx(1.0)  # 0.5 * 2 N on every actuator
    assert s["settle_last_send"] == pytest.approx(95.0)

    vf["wfs"] = "newf9"
    assert visit_settle(vf, sends, SETTLE_DEFAULTS, forces, _FakeTelescope())["settle_def"] == "last_send"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v -k "settle or first_run or send_windows or no_ok"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# append to mmtwfs/nightlog.py

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
        src = np.array([frame_time.get(f, np.datetime64("NaT")) for f in vsends["file"]], dtype=TIME)
        in_settle = vsends[src <= cutoff]
    else:
        in_settle = vsends
    for window, s in (("settle", in_settle), ("visit", vsends)):
        for key, value in _send_totals(s, forces, telescope).items():
            out[f"{key}_{window}"] = value
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py
git commit -m "measure settle times and corrections sent per visit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Cumulative M1/M2 accounting and the surface Zernike fit

**Files:**
- Modify: `mmtwfs/nightlog.py` (append)
- Test: `mmtwfs/tests/test_nightlog.py` (append)

**Interfaces:**
- Consumes: `sends`, `events` (Task 2); `gain_at`, `_load_zv`, `get_wfs` (Task 5).
- Produces:
  - `FIT_TERMS: list[str]`, which is Z04, Z05, Z06, and Z09–Z22.
  - `get_mmt() -> MMT`, cached.
  - `read_forces(path: Path, act_ids) -> np.ndarray`.
  - `fit_surface_zernikes(surface: np.ndarray, telescope) -> dict[str, float]`, giving wavefront nm.
  - `m1_accounting(sends, events, forces: dict, telescope) -> (Table, list[tuple])`. The table has
    `M1_COLUMNS = time, fmax, fmin, frms, surf_rms, *FIT_TERMS`. The list holds `(time, label, force vector)` maps,
    taken before each clear and at the end of the night.
  - `m2_accounting(sends, events) -> Table`, with `M2_COLUMNS = time, wfs_z, wfs_cc_x, wfs_cc_y, m1spherical_z`.
  - `m1_consistency(sends, events, analyses, nightdir, m1: Table) -> dict[str, tuple[float, float]]`. It maps each
    term to (fitted, expected) at the end of the night.

- [ ] **Step 1: Write the failing tests**

```python
# append to mmtwfs/tests/test_nightlog.py
from mmtwfs.nightlog import (FIT_TERMS, fit_surface_zernikes, get_mmt, m1_accounting, m2_accounting,
                             read_forces)


def test_read_forces_aligns_by_actuator_id(tmp_path):
    p = tmp_path / "x.forces"
    p.write_text("101\t5.0\n1\t-2.5\n")
    vec = read_forces(p, [1, 2, 101])
    assert list(vec) == [-2.5, 0.0, 5.0]


@pytest.mark.parametrize("term, amp, rel", [("Z05", 200.0, 0.01), ("Z09", -100.0, 0.01), ("Z11", 150.0, 0.05)])
def test_surface_fit_recovers_bent_term(term, amp, rel):
    mmt = get_mmt()
    f = mmt.bending_forces(zv=ZernikeVector(**{term: amp}), gain=1.0)
    fit = fit_surface_zernikes(np.asarray(f["force"]) @ mmt.load_act2surf(), mmt)
    assert set(fit) == set(FIT_TERMS)
    assert fit[term] == pytest.approx(-amp, rel=rel)


def _t(minutes):
    return np.datetime64("2026-10-03T08:00:00") + np.timedelta64(minutes, "m")


def test_m1_accounting_sends_clears_and_missing():
    mmt = get_mmt()
    n = mmt.n_act
    sends = _table([
        dict(time=_t(1), kind="m1", forces_file="a.forces", frac=1.0),
        dict(time=_t(2), kind="m1", forces_file="b.forces", frac=0.5),
        dict(time=_t(3), kind="m1", forces_file="missing.forces", frac=1.0),
        dict(time=_t(5), kind="m1", forces_file="a.forces", frac=1.0),
    ], SEND_COLUMNS)
    events = _table([dict(time=_t(4), kind="clear_m1"), dict(time=_t(4), kind="clear_m2")], EVENT_COLUMNS)
    forces = {"a.forces": np.full(n, 2.0), "b.forces": np.full(n, 4.0)}
    m1, maps = m1_accounting(sends, events, forces, mmt)
    assert len(m1) == 4  # three applied sends and one clear; the missing file adds no snapshot
    assert m1["frms"][1] == pytest.approx(4.0)  # 2 + 0.5 * 4
    assert m1["frms"][2] == 0.0
    assert m1["frms"][3] == pytest.approx(2.0)
    assert [label for _, label, _ in maps] == ["before clear", "end of night"]
    assert maps[0][2] == pytest.approx(np.full(n, 4.0))


def test_m2_accounting_resets_per_channel():
    sends = _table([
        dict(time=_t(1), kind="focus", value=10.0),
        dict(time=_t(2), kind="coma", value=1.0, value2=-1.0),
        dict(time=_t(3), kind="m1focus", value=5.0),
        dict(time=_t(5), kind="focus", value=3.0),
        dict(time=_t(7), kind="m1focus", value=2.0),
    ], SEND_COLUMNS)
    events = _table([dict(time=_t(4), kind="clear_m1"), dict(time=_t(6), kind="clear_m2")], EVENT_COLUMNS)
    m2 = m2_accounting(sends, events)
    last = m2[-1]
    assert (last["wfs_z"], last["wfs_cc_x"], last["m1spherical_z"]) == (0.0, 0.0, 2.0)
    after_m1_clear = m2[m2["time"] == _t(4)][0]
    assert (after_m1_clear["wfs_z"], after_m1_clear["m1spherical_z"]) == (10.0, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v -k "forces or surface or accounting"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add the imports `import functools` and `from mmtwfs.telescope import MMT`, then append:

```python
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
    Compare the fitted cumulative M1 terms at the end of the night with what the sends should have produced since
    the last clear: -sum(frac x gain x masked Zernike term). Returns {term: (fitted, expected)}.
    """
    if len(m1) == 0:
        return {}
    clears = events["time"][events["kind"] == "clear_m1"]
    since = np.max(clears) if len(clears) else np.datetime64("NaT")
    expected = dict.fromkeys(FIT_TERMS, 0.0)
    analysed_at = dict(zip(analyses["file"], analyses["time"]))
    for s in sends[sends["kind"] == "m1"]:
        if not np.isnat(since) and s["time"] < since:
            continue
        masked = nightdir / f"{s['file']}.masked.zernike"
        if not masked.exists():
            continue
        zv = _load_zv(masked)
        gain = gain_at(events, "m1_gain", analysed_at.get(s["file"], s["time"]), 0.5)
        for term in FIT_TERMS:
            if term in zv.coeffs:
                expected[term] -= s["frac"] * gain * zv[term].value
    last = m1[-1]
    return {term: (float(last[term]), expected[term]) for term in FIT_TERMS}
```

`m1_consistency` falls back to a gain of 0.5 only when the log has no gain line before that send. 0.5 is the
`m1_gain` default in every MMT WFS config.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass. `m1_consistency` is exercised by Task 9's integration test.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py
git commit -m "track cumulative M1 forces and M2 offsets through a night

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Flags

**Files:**
- Modify: `mmtwfs/nightlog.py` (append)
- Test: `mmtwfs/tests/test_nightlog.py` (append)

**Interfaces:**
- Consumes: `frames`, `sends`, `events`, `analyses`, `forces`; `get_wfs`, `gain_at`, `calculate_primary_at`,
  `_load_zv`.
- Produces:
  - `FLAG_COLUMNS = time, kind, file, detail, visit`.
  - `find_flags(frames, sends, events, analyses, forces, nightdir) -> Table`. The flag kinds are `spher_ignored`,
    `m1_blocked_focus`, `validity`, `m1_partial`, `missing_forces`, and `error`. `visit` comes from the flag's file,
    or else from the latest frame at or before the flag's time, or −1.

- [ ] **Step 1: Write the failing test**

```python
# append to mmtwfs/tests/test_nightlog.py
from mmtwfs.nightlog import ANALYSIS_COLUMNS, find_flags


def test_find_flags(tmp_path):
    frames = _visit_frames([200.0, 5.0, 4.0])
    analyses = _table([
        dict(time=frames["time"][0], file="f000.fits", spher=True),
        dict(time=frames["time"][1], file="f001.fits", spher=True),
        dict(time=frames["time"][2], file="f002.fits", spher=False),
    ], ANALYSIS_COLUMNS)
    ZernikeVector(Z05=100.0, Z11=800.0).save(filename=tmp_path / "f000.reanalyze.zernike")
    sends = _table([
        dict(time=frames["time"][1], kind="m1", file="f001.fits", forces_file="f001.fits.forces",
             applied="partial", frac=0.4),
        dict(time=frames["time"][2], kind="m1", file="f002.fits", forces_file="f002.fits.forces",
             applied="full", frac=1.0),
    ], SEND_COLUMNS)
    events = _table([dict(time=frames["time"][2], kind="error", detail="hexapod socket timed out"),
                     dict(time=frames["time"][0], kind="validity", detail="only focus corrections valid",
                          file="f000.fits")], EVENT_COLUMNS)
    forces = {"f001.fits.forces": np.zeros(104)}
    flags = find_flags(frames, sends, events, analyses, forces, tmp_path)
    kinds = sorted(flags["kind"])
    assert kinds == sorted(["spher_ignored", "m1_blocked_focus", "validity", "m1_partial", "missing_forces",
                            "error"])
    spher = flags[flags["kind"] == "spher_ignored"][0]
    assert "2 frames" in spher["detail"] and "Z11 up to 800 nm" in spher["detail"]
    assert flags[flags["kind"] == "missing_forces"][0]["file"] == "f002.fits"
    assert set(flags["visit"]) == {0}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v -k find_flags`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# append to mmtwfs/nightlog.py

FLAG_COLUMNS = [("time", TIME), ("kind", str), ("file", str), ("detail", str), ("visit", int)]
M1_FOCUS_GATE = 150.0  # um; wfssrv blocks M1 corrections when pending focus exceeds this (signed comparison)


def _spher_detail(files, frames, events, nightdir):
    """How much spherical was being ignored, and the forces a full correction would have needed."""
    z11 = fmax = frms = 0.0
    wfs_of = dict(zip(frames["file"], frames["wfs"]))
    time_of = dict(zip(frames["file"], frames["time"]))
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
            gain = gain_at(events, "m1_gain", time_of[f], wfs.m1_gain)
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
        z11, fmax, frms = _spher_detail(files, frames, events, nightdir)
        add(analyses["time"][i], "spher_ignored", files[0],
            f"{analyses['time'][i]} to {analyses['time'][j]} UTC, {len(files)} frames, Z11 up to {z11:.0f} nm; "
            f"full correction would need max |F| {fmax:.0f} N, RMS {frms:.0f} N")
        i = j + 1

    for row in frames[frames["status"] == "ok"]:
        # mirror wfssrv: pending focus (m2_gain x measured correction) above +150 um blocks M1. The gate is signed.
        default = get_wfs(row["wfs"]).m2_gain if row["wfs"] != "unknown" else 1.0
        pending = gain_at(events, "m2_gain", row["time"], default) * row["focus"]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py
git commit -m "flag ignored spherical, blocked M1, partial sends, and errors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `load_night`, the real-data fixture, and the integration test

**Files:**
- Modify: `mmtwfs/nightlog.py` (append)
- Create: `mmtwfs/data/test_data/nightreport/make_fixture.py`
- Create: `mmtwfs/data/test_data/nightreport/20261003/` (generated)
- Test: `mmtwfs/tests/test_nightlog.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces:
  - `Night` dataclass with fields `path, params, frames, visits, sends, events, analyses, flags, m1, m2, m1_maps,
    m1_check, counts, warnings`.
  - `load_night(path, params=None) -> Night`.
  - `VISIT_COLUMNS`.

- [ ] **Step 1: Build the fixture from the archive**

```python
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
```

Run: `python mmtwfs/data/test_data/nightreport/make_fixture.py /Volumes/Samsung_4TB/wfsdat/20261003 && du -sh mmtwfs/data/test_data/nightreport && ls mmtwfs/data/test_data/nightreport/20261003 | wc -l`

Expected: under 400 KB, and about 23 `.fits` files plus their outputs and `wfs.log`. If it's over 1 MB, stop and
report it. The fixture has to stay small.

Check the counts that Step 2's test relies on:
`grep -c 'Sending forces' mmtwfs/data/test_data/nightreport/20261003/wfs.log` should be 6 (frames 0325 and 0330–0334),
and `grep -c 'Clearing forces and' mmtwfs/data/test_data/nightreport/20261003/wfs.log` should be 1. If either differs,
use the real count in the test and note it in the ledger.

- [ ] **Step 2: Write the failing integration test**

```python
# append to mmtwfs/tests/test_nightlog.py
import importlib

from mmtwfs.nightlog import load_night

FIXTURE = importlib.resources.files("mmtwfs") / "data" / "test_data" / "nightreport" / "20261003"


@pytest.fixture(scope="module")
def night():
    return load_night(FIXTURE)


def test_fixture_visits(night):
    assert list(night.visits["catid"]) == ["HIP102074", "HIP109911"]
    second = night.visits[1]
    assert second["gap"] > 3600  # 06:42 -> 08:44 UTC
    assert second["n_sends_visit"] > 0


def test_fixture_sends_and_clear(night):
    assert int(np.sum(night.sends["kind"] == "m1")) == 6
    assert int(np.sum(night.events["kind"] == "clear_m1")) == 1
    clear_time = night.events["time"][night.events["kind"] == "clear_m1"][0]
    assert clear_time == np.datetime64("2026-10-03T06:42:10.717")
    # the M1 state starts at zero and the clear precedes every send in the fixture
    assert night.m1["frms"][0] == 0.0
    assert night.m1["frms"][-1] > 0.0


def test_fixture_gain_changes_applied(night):
    gains = night.events[night.events["kind"] == "m1_gain"]
    assert list(gains["detail"]) == ["0.5", "0.2", "0.5", "0.2"]


def test_fixture_m1_consistency(night):
    checked = 0
    for term, (fitted, expected) in night.m1_check.items():
        if abs(expected) > 50:
            assert fitted == pytest.approx(expected, rel=0.25), term
            checked += 1
    assert checked > 0


def test_fixture_m2_sums_match_sends(night):
    focus = night.sends["value"][night.sends["kind"] == "focus"]
    assert night.m2["wfs_z"][-1] == pytest.approx(np.sum(focus))


def test_fixture_frames_have_m1_metrics(night):
    full = night.frames[(night.frames["status"] == "ok") & (night.frames["method"] == "full")]
    assert len(full) > 0
    assert np.all(np.isfinite(full["m1_force_rms"]))
    assert np.all(np.isfinite(full["focus_unc"]))


def test_load_night_without_log_or_outputs(tmp_path):
    nd = tmp_path / "20261003"
    nd.mkdir()
    for f in FIXTURE.iterdir():
        if f.name.endswith(".fits"):
            (nd / f.name).write_bytes(f.read_bytes())
    night = load_night(nd)
    assert len(night.visits) == 2
    assert any("wfs.log missing" in w for w in night.warnings)
    assert any("no reanalyze outputs" in w for w in night.warnings)
    assert len(night.m1) == 0


def test_load_night_empty(tmp_path):
    nd = tmp_path / "20261003"
    nd.mkdir()
    night = load_night(nd)
    assert len(night.frames) == 0 and len(night.visits) == 0
```

`test_fixture_gain_changes_applied` lists the M1 gain lines in the fixture window. From the archive, those are 0.5 at
23:42:01, 0.2 at 01:47:04, 0.5 at 01:48:26, and 0.2 at 01:56:49 local time. Check them against
`grep 'M1 gain' .../20261003/wfs.log` after building the fixture, and correct the list if it differs.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v -k "fixture or load_night"`
Expected: FAIL with `ImportError: cannot import name 'load_night'`.

- [ ] **Step 4: Implement**

Add `from dataclasses import field` to the dataclass import, then append:

```python
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

    visit_rows, offsets = build_visits(frames, params)
    events = logdata.events
    if offsets:
        events = _table(sorted([dict(zip(events.colnames, r)) for r in events] + offsets, key=lambda e: e["time"]),
                        EVENT_COLUMNS)
    m1_frame_metrics(frames, path, events)

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightlog.py -v`
Expected: all pass. If `test_fixture_m1_consistency` fails, don't loosen 25% silently. Print `night.m1_check`, and stop
and report the per-term numbers. The spec makes changing this tolerance a decision that gets recorded.

- [ ] **Step 6: Commit**

```bash
git add mmtwfs/nightlog.py mmtwfs/tests/test_nightlog.py mmtwfs/data/test_data/nightreport
git commit -m "assemble a night with load_night; add a trimmed real-night fixture

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Report rendering (HTML, CSV, JSON)

**Files:**
- Create: `mmtwfs/scripts/nightreport.py`
- Test: `mmtwfs/tests/test_nightreport.py`

**Interfaces:**
- Consumes: `Night`, `load_night`, `get_mmt`, `FIT_TERMS` (Task 9).
- Produces:
  - `summarize(night) -> dict`, which is JSON-safe: NaN becomes None.
  - `write_report(night) -> None`. It writes `nightreport.html` and `nightreport_{frames,visits,events,m1,m2}.csv`,
    then `nightreport_summary.json` last, all atomically into `night.path`.

- [ ] **Step 1: Write the failing tests**

```python
# mmtwfs/tests/test_nightreport.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib
import json
import shutil

import pytest

from mmtwfs.nightlog import load_night
from mmtwfs.scripts.nightreport import summarize, write_report

FIXTURE = importlib.resources.files("mmtwfs") / "data" / "test_data" / "nightreport" / "20261003"
OUTPUTS = ["nightreport.html", "nightreport_frames.csv", "nightreport_visits.csv", "nightreport_events.csv",
           "nightreport_m1.csv", "nightreport_m2.csv", "nightreport_summary.json"]


@pytest.fixture
def nightdir(tmp_path):
    d = tmp_path / "20261003"
    shutil.copytree(FIXTURE, d)
    return d


def test_write_report_files_and_contents(nightdir):
    night = load_night(nightdir)
    write_report(night)
    for name in OUTPUTS:
        assert (nightdir / name).exists(), name
    html = (nightdir / "nightreport.html").read_text()
    for heading in ("Summary", "Night timeline", "M1 surface", "Slew response", "Visits", "Events"):
        assert heading in html
    assert html.count("<img ") >= 3
    assert 'src="http' not in html and "<script" not in html  # self-contained
    summary = json.loads((nightdir / "nightreport_summary.json").read_text())
    assert summary["night"] == "20261003"
    assert summary["n_visits"] == 2
    assert summary["n_sends"]["m1"] == 6
    assert summary["params"]["focus_thresh"] == 10.0


def test_rerun_is_identical(nightdir):
    write_report(load_night(nightdir))
    first = {n: (nightdir / n).read_text() for n in OUTPUTS if n.endswith(".csv")}
    write_report(load_night(nightdir))
    assert first == {n: (nightdir / n).read_text() for n in first}


def test_summary_is_json_safe_for_empty_night(tmp_path):
    d = tmp_path / "20261003"
    d.mkdir()
    night = load_night(d)
    summary = summarize(night)
    json.dumps(summary, allow_nan=False)
    write_report(night)
    assert "No WFS frames" in (d / "nightreport.html").read_text()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightreport.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmtwfs.scripts.nightreport'`

- [ ] **Step 3: Implement**

```python
# mmtwfs/scripts/nightreport.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Nightly WFS report: render a night built by mmtwfs.nightlog.load_night into a self-contained HTML page plus CSV and
JSON data files in the night's directory.
"""

import base64
import html
import io
import json
import logging
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402

from mmtwfs.nightlog import FIT_TERMS, get_mmt  # noqa: E402
from mmtwfs.version import version  # noqa: E402

log = logging.getLogger("NightReport")

VISIT_TABLE_COLUMNS = [
    ("catid", "{}"), ("start", "{}"), ("gap", "{:.0f}"), ("t_first_ok", "{:.0f}"), ("focus_settle_thresh", "{:.0f}"),
    ("focus_settle_conv", "{:.0f}"), ("iq_settle_conv", "{:.0f}"), ("coma_settle_thresh", "{:.0f}"),
    ("settle_last_send", "{:.0f}"), ("first_focus", "{:.1f}"), ("first_cc_x", "{:.2f}"), ("first_cc_y", "{:.2f}"),
    ("first_iq", "{:.2f}"), ("n_sends_settle", "{}"), ("net_focus_settle", "{:.1f}"), ("n_sends_visit", "{}"),
    ("net_focus_visit", "{:.1f}"), ("m1_sent_rms_visit", "{:.1f}"), ("flags", "{}"),
]
EVENT_TABLE_COLUMNS = [("time", "{}"), ("kind", "{}"), ("file", "{}"), ("detail", "{}")]


def _atomic_write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_table(table, path):
    tmp = path.with_name(path.name + ".tmp")
    table.write(tmp, format="ascii.csv", overwrite=True)
    os.replace(tmp, path)


def _clean(value):
    """Make a value JSON-safe: numpy scalars to Python, NaN to None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _median(column):
    values = np.asarray(column, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def summarize(night):
    f, v = night.frames, night.visits
    ok = f[f["status"] == "ok"]
    check = [abs(fit - exp) for fit, exp in night.m1_check.values()]
    return _clean({
        "night": night.path.name,
        "mmtwfs_version": version,
        "params": night.params,
        "wfs": sorted(set(f["wfs"])),
        "start": str(f["time"][0]) if len(f) else None,
        "end": str(f["time"][-1]) if len(f) else None,
        "n_frames": len(f),
        "n_ok": int(np.sum(f["status"] == "ok")),
        "n_failed": int(np.sum(f["status"] == "failed")),
        "n_unanalyzed": int(np.sum(f["status"] == "unanalyzed")),
        "n_visits": len(v),
        "n_clears": dict(Counter(k for k in night.events["kind"] if k.startswith("clear"))),
        "n_sends": dict(Counter(night.sends["kind"])),
        "n_flags": dict(Counter(night.flags["kind"])),
        "median_raw_vlt_seeing": _median(ok["raw_vlt_seeing"]),
        "median_delivered_iq": _median(ok["delivered_iq"]),
        "median_focus_settle_thresh": _median(v["focus_settle_thresh"]),
        "median_focus_settle_conv": _median(v["focus_settle_conv"]),
        "median_gap": _median(v["gap"]),
        "median_t_first_ok": _median(v["t_first_ok"]),
        "m1_check_max_diff": max(check) if check else None,
        "warnings": night.warnings,
        "counts": night.counts,
    })


def _img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _mark_clears(ax, events):
    for t in events["time"][events["kind"] == "clear_m1"]:
        ax.axvline(t, color="C3", ls="--", lw=1)
    for t in events["time"][events["kind"] == "clear_m2"]:
        ax.axvline(t, color="C4", ls=":", lw=1)


def timeline_figure(night):
    f, ok = night.frames, night.frames[night.frames["status"] == "ok"]
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(12, 18))
    ax = axes[0]
    ax.plot(f["time"], f["el"], ".", ms=3, color="0.4")
    for v in night.visits:
        ax.axvspan(v["start"], v["end"], color=f"C{v['visit'] % 2}", alpha=0.15)
        ax.text(v["start"], 1.02, v["catid"], transform=ax.get_xaxis_transform(), fontsize=7, rotation=30)
    if len(night.flags):
        ax.plot(night.flags["time"], np.full(len(night.flags), np.nanmax(f["el"])), "v", color="C3", label="flag")
    ax.set_ylabel("EL (deg)")
    ax = axes[1]
    ax.errorbar(ok["time"], ok["focus"], yerr=ok["focus_unc"], fmt=".", ms=3, lw=0.5)
    focus_sends = night.sends["time"][night.sends["kind"] == "focus"]
    ax.plot(focus_sends, np.zeros(len(focus_sends)), "|", ms=10, color="C2", label="focus sent")
    ax.set_ylabel("focus error (um)")
    ax = axes[2]
    ax.plot(ok["time"], ok["cc_x_err"], ".", ms=3, label="CC x")
    ax.plot(ok["time"], ok["cc_y_err"], ".", ms=3, label="CC y")
    ax.set_ylabel("coma (arcsec)")
    ax = axes[3]
    ax.plot(ok["time"], ok["raw_vlt_seeing"], ".", ms=3, label="raw seeing")
    ax.plot(ok["time"], ok["delivered_iq"], ".", ms=3, label="delivered IQ")
    ax.set_ylabel("arcsec")
    ax = axes[4]
    for ch in ("wfs_z", "m1spherical_z", "wfs_cc_x", "wfs_cc_y"):
        ax.step(night.m2["time"], night.m2[ch], where="post", label=ch)
    ax.set_ylabel("cumulative M2\n(um / arcsec)")
    ax = axes[5]
    ax.step(night.m1["time"], night.m1["frms"], where="post", label="force RMS (N)")
    ax.step(night.m1["time"], night.m1["surf_rms"], where="post", label="surface RMS (nm)")
    ax.set_ylabel("cumulative M1")
    ax = axes[6]
    for term in FIT_TERMS:
        if len(night.m1) and np.max(np.abs(night.m1[term])) > 10:
            ax.step(night.m1["time"], night.m1[term], where="post", label=term)
    ax.set_ylabel("cumulative M1\nwavefront (nm)")
    ax.set_xlabel("UTC")
    for ax in axes:
        _mark_clears(ax, night.events)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper right")
    return fig


def surface_figures(night):
    figs = []
    mmt = get_mmt()
    for time, label, vec in night.m1_maps:
        fig = mmt.plot_force_influence(Table([mmt.actcoor["act_id"], vec], names=["actuator", "force"]))
        fig.axes[0].set_title(f"{label}, {time} UTC", fontsize=9)
        figs.append(fig)
    return figs


def slew_figure(night):
    v = night.visits
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(v["d_el"], v["first_focus"], "o")
    axes[0, 0].set(xlabel="ΔEL (deg)", ylabel="first-frame focus (um)")
    axes[0, 1].plot(v["el"], v["first_focus"], "o")
    axes[0, 1].set(xlabel="EL (deg)", ylabel="first-frame focus (um)")
    axes[1, 0].plot(v["d_el"], v["focus_settle_thresh"], "o")
    axes[1, 0].set(xlabel="ΔEL (deg)", ylabel="focus settle (s)")
    x = np.arange(len(v))
    axes[1, 1].bar(x - 0.2, v["gap"], 0.4, label="gap")
    axes[1, 1].bar(x + 0.2, v["t_first_ok"], 0.4, label="time to first good frame")
    axes[1, 1].set(xlabel="visit", ylabel="s")
    axes[1, 1].legend(fontsize=7)
    return fig


def _cell(value, fmt):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return html.escape(fmt.format(value))


def _html_table(table, columns):
    head = "".join(f"<th>{name}</th>" for name, _ in columns)
    rows = "".join("<tr>" + "".join(f"<td>{_cell(r[name], fmt)}</td>" for name, fmt in columns) + "</tr>"
                   for r in table)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


CSS = """body{font-family:system-ui,sans-serif;margin:16px;color:#222;background:#fff}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}
th{background:#f0f0f0}.warn{color:#a40}img{max-width:100%}"""


def render_html(night, summary):
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>WFS night {summary['night']}</title>"
             f"<style>{CSS}</style></head><body><h1>WFS night {summary['night']}</h1>"]
    parts.append("<h2>Summary</h2><table>" + "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(val))}</td></tr>"
        for k, val in summary.items() if k not in ("warnings", "params", "counts")) + "</table>")
    parts += [f"<p class='warn'>{html.escape(w)}</p>" for w in night.warnings]
    if len(night.frames) == 0:
        parts.append("<p>No WFS frames in this night directory.</p></body></html>")
        return "".join(parts)
    parts.append("<h2>Night timeline</h2>" + _img(timeline_figure(night)))
    parts.append("<h2>M1 surface</h2>")
    maps = surface_figures(night)
    parts.append("".join(_img(fig) for fig in maps) if maps else "<p>No M1 forces applied.</p>")
    parts.append("<h2>Slew response</h2>" + _img(slew_figure(night)))
    parts.append("<p>Gaps are as seen by the WFS: slew, acquisition, and WFS restart together.</p>")
    parts.append("<h2>Visits</h2><p>Times in seconds from each visit's first frame.</p>"
                 + _html_table(night.visits, VISIT_TABLE_COLUMNS))
    events = Table(night.events, copy=True)
    flags = Table(night.flags, copy=True)
    flags.remove_column("visit")
    combined = vstack([events, flags]) if len(flags) else events
    combined.sort("time")
    parts.append("<h2>Events</h2>" + _html_table(combined, EVENT_TABLE_COLUMNS))
    parts.append("</body></html>")
    return "".join(parts)


def write_report(night):
    out = night.path
    _write_table(night.frames, out / "nightreport_frames.csv")
    _write_table(night.visits, out / "nightreport_visits.csv")
    events = vstack([night.sends, night.events], metadata_conflicts="silent") if len(night.sends) else night.events
    _write_table(events, out / "nightreport_events.csv")
    _write_table(night.m1, out / "nightreport_m1.csv")
    _write_table(night.m2, out / "nightreport_m2.csv")
    summary = summarize(night)
    _atomic_write(out / "nightreport.html", render_html(night, summary))
    _atomic_write(out / "nightreport_summary.json", json.dumps(summary, indent=2, allow_nan=False))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightreport.py -v`
Expected: all pass. Then open the generated HTML once to look at it:
`python -c "from mmtwfs.nightlog import load_night; from mmtwfs.scripts.nightreport import write_report; write_report(load_night('<copy of fixture>'))"`
and then `open <copy>/nightreport.html`.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/scripts/nightreport.py mmtwfs/tests/test_nightreport.py
git commit -m "render nightly WFS reports to HTML, CSV, and JSON

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: CLI and entry point

**Files:**
- Modify: `mmtwfs/scripts/nightreport.py` (append `night_dirs`, `run_night`, `main`)
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `mmtwfs/tests/test_nightreport.py` (append)

**Interfaces:**
- Produces:
  - `night_dirs(paths: list[str]) -> list[Path]`. A path whose name is 8 digits is a night. Any other directory is a
    parent, and its 8-digit subdirectories are the nights. The result is sorted.
  - `main(argv=None) -> int`, returning 0 if every night succeeded and 1 otherwise.

- [ ] **Step 1: Write the failing tests**

```python
# append to mmtwfs/tests/test_nightreport.py
from unittest.mock import patch

from mmtwfs.scripts.nightreport import main, night_dirs


def test_night_dirs(tmp_path):
    for name in ("20261003", "20261004", "notes"):
        (tmp_path / name).mkdir()
    assert [p.name for p in night_dirs([str(tmp_path)])] == ["20261003", "20261004"]
    assert [p.name for p in night_dirs([str(tmp_path / "20261004")])] == ["20261004"]


def test_main_skips_done_nights_unless_forced(nightdir):
    assert main([str(nightdir)]) == 0
    stamp = (nightdir / "nightreport_summary.json").stat().st_mtime_ns
    assert main([str(nightdir)]) == 0
    assert (nightdir / "nightreport_summary.json").stat().st_mtime_ns == stamp
    assert main(["--forcedir", "--focus-thresh", "20", str(nightdir)]) == 0
    assert json.loads((nightdir / "nightreport_summary.json").read_text())["params"]["focus_thresh"] == 20.0


def test_main_error_on_one_night_returns_one(tmp_path, nightdir):
    bad = tmp_path / "20261004"
    bad.mkdir()
    # nights run in sorted order: 20261003 (good) then 20261004 (fails)
    with patch("mmtwfs.scripts.nightreport.load_night", side_effect=[load_night(nightdir), RuntimeError("boom")]):
        assert main([str(bad), str(nightdir)]) == 1
    assert (nightdir / "nightreport_summary.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest mmtwfs/tests/test_nightreport.py -v -k "night_dirs or main"`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement**

Add `import argparse`, `import concurrent.futures`, `import re`, `from pathlib import Path`, and
`from mmtwfs.nightlog import SETTLE_DEFAULTS, load_night` (extend the existing import) at the top, then append:

```python
NIGHT_RE = re.compile(r"^\d{8}$")


def night_dirs(paths):
    nights = []
    for p in map(Path, paths):
        if NIGHT_RE.match(p.name) and p.is_dir():
            nights.append(p)
        elif p.is_dir():
            nights.extend(d for d in p.iterdir() if d.is_dir() and NIGHT_RE.match(d.name))
    return sorted(nights)


def run_night(path, params):
    """Build and write one night's report. Returns True on success."""
    try:
        write_report(load_night(path, params))
        log.info(f"Wrote report for {path.name}")
        return True
    except Exception:
        log.exception(f"Report failed for {path}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Nightly WFS report from wfs.log, frames, and reanalyze outputs")
    parser.add_argument("dirs", nargs="+", metavar="DIR",
                        help="night directory (YYYYMMDD) or a parent containing night directories")
    parser.add_argument("--forcedir", action="store_true", help="rebuild reports that already exist")
    parser.add_argument("--focus-thresh", type=float, default=SETTLE_DEFAULTS["focus_thresh"],
                        help="focus settle threshold (um)")
    parser.add_argument("--coma-thresh", type=float, default=SETTLE_DEFAULTS["coma_thresh"],
                        help="coma settle threshold (arcsec)")
    parser.add_argument("--consecutive", type=int, default=SETTLE_DEFAULTS["consecutive"],
                        help="consecutive frames that must meet a threshold")
    parser.add_argument("--iq-abs", type=float, default=SETTLE_DEFAULTS["iq_abs"],
                        help="absolute IQ plateau tolerance (arcsec)")
    parser.add_argument("--iq-frac", type=float, default=SETTLE_DEFAULTS["iq_frac"],
                        help="fractional IQ plateau tolerance")
    parser.add_argument("-n", "--nproc", type=int, default=1, help="nights to process in parallel")
    args = parser.parse_args(argv)

    params = {**SETTLE_DEFAULTS, "focus_thresh": args.focus_thresh, "coma_thresh": args.coma_thresh,
              "consecutive": args.consecutive, "iq_abs": args.iq_abs, "iq_frac": args.iq_frac}
    todo = []
    for night in night_dirs(args.dirs):
        if not args.forcedir and (night / "nightreport_summary.json").exists():
            log.info(f"Already reported {night.name}")
        else:
            todo.append(night)

    if args.nproc > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.nproc) as pool:
            results = list(pool.map(run_night, todo, [params] * len(todo)))
    else:
        results = [run_night(night, params) for night in todo]
    return 0 if all(results) else 1
```

In `pyproject.toml`, under `[project.scripts]`, add `nightreport = "mmtwfs.scripts.nightreport:main"` after the
`reanalyze` line. Then reinstall the entry point with `pip install -e . --no-deps`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest mmtwfs/tests/test_nightreport.py -v && nightreport --help`
Expected: tests pass and the help text prints.

- [ ] **Step 5: Run the full suite and flake8**

Run: `python -m pytest mmtwfs -q 2>&1 | tail -3 && flake8 mmtwfs --count --max-line-length=127`
Expected: everything passes (259 existing tests plus the new ones), and flake8 reports `0`.

- [ ] **Step 6: Commit**

```bash
git add mmtwfs/scripts/nightreport.py mmtwfs/tests/test_nightreport.py pyproject.toml
git commit -m "add the nightreport console script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Real-data validation and default thresholds (manual; not CI)

**Files:**
- Possibly modify: `mmtwfs/nightlog.py` (`SETTLE_DEFAULTS` only)
- Modify: `docs/superpowers/specs/2026-10-14-nightly-wfs-report-design.md` (record the decisions)

- [ ] **Step 1: Run on the 2026 Samsung nights**

Run: `nightreport --forcedir -n 4 /Volumes/Samsung_4TB/wfsdat 2>&1 | tail -20`
Expected: exit 0, and every night directory gets a `nightreport_summary.json`. Every night with an error is a bug.
Fix each one with a failing test first, under superpowers:test-driven-development.

- [ ] **Step 2: Spot-check 20261003 against the log**

- Compare the clears in `nightreport_events.csv` with `grep 'Clearing forces and' wfs.log`. There should be 3, at
  local 20:26, 23:42, and the third one in the log.
- Check that the 04:00 local `Bad force data` send appears as `m1_partial` with "unexpected".
- Check that the visit list matches the 8 targets seen earlier: HD168624 through WNTR24okvms_2_JzJ.
- Read `m1_check_max_diff` in the summary.

- [ ] **Step 3: Spot-check 20261005 and a Binospec night (20260816 or 20260817)**

- For 20261005, confirm the summary warning about log lines outside the window. Its log runs to 2026-10-07.
- For the Binospec night, confirm that the frames have times and that visits split on targets.

- [ ] **Step 4: Choose the default thresholds**

For MMIRS and Binospec separately, tabulate how many visits reach `focus_settle_thresh` and `focus_settle_conv`, plus
the median `first_focus_unc` and median `focus_unc` over all ok frames.
- If the median `focus_unc` is at or above 5 µm, F = 10 µm is within 2σ of the noise. Propose F ≈ 2 × median
  `focus_unc`.
- Present the table and the proposal to the user. **Don't change `SETTLE_DEFAULTS` without the user's answer.**

- [ ] **Step 5: Record the results**

Add a "Validation (YYYY-MM-DD)" section to the spec with the nights run, the errors found and fixed, the
`m1_check_max_diff` values, and the threshold decision.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-10-14-nightly-wfs-report-design.md mmtwfs/nightlog.py
git commit -m "record nightreport validation on 2026 nights

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
