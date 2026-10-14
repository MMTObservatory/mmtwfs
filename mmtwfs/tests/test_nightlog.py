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
2026-10-03 01:44:51,078 - Telescope - INFO - Using command, /mmt/scripts/cell_send_forces \
/home/mmtop/wfsdat/mmirs_wfs_0325.fits.forces, to apply forces...
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
