# mmtwfs/tests/test_nightreport.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib.resources
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
