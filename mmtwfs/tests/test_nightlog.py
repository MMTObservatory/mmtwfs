# mmtwfs/tests/test_nightlog.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits

from mmtwfs.nightlog import (
    build_visits, EVENT_COLUMNS, first_run, FRAME_COLUMNS, frame_wfs, gain_at, get_wfs, m1_frame_metrics,
    m1_resid_rms, night_window, parse_log, read_frames, SEND_COLUMNS, SETTLE_DEFAULTS, _table, visit_settle,
)
from mmtwfs.zernike import ZernikeVector

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


OUTPUT_LINE = ("2026-10-03T08:44:38,mmirs,{name},12.0,1.2,188.6,56.7,12,11.5,12.3,244.4,167.9,-559.2,2188.5,"
               "9052.3,{focus},5.5,0.4,256.9,232.5,4.9,5.4,2.4,0.9,0.08,16.3,1387.4,2220.4,full,1.05")


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
