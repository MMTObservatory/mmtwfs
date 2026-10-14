# mmtwfs/tests/test_nightreport.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib.resources
import json
import shutil
from unittest.mock import patch

import pytest

from mmtwfs.nightlog import load_night
from mmtwfs.scripts.nightreport import main, night_dirs, summarize, write_report

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
    assert summary["n_sends"]["m1"] == 8
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
