# mmtwfs/tests/test_nightreport_site.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import json

import pytest

from mmtwfs.scripts.nightreport_site import NightError, collect, read_night


def make_night(root, night, wfs=("mmirs",), n_frames=10, iq=1.234, html="<html>report</html>", summary=True):
    d = root / night
    d.mkdir(parents=True)
    if html is not None:
        (d / "nightreport.html").write_text(html)
    if summary is True:
        summary = {"night": night, "wfs": list(wfs), "n_frames": n_frames, "median_delivered_iq": iq}
    if summary is not None:
        (d / "nightreport_summary.json").write_text(summary if isinstance(summary, str) else json.dumps(summary))
    return d


def test_read_night_entry(tmp_path):
    entry = read_night(make_night(tmp_path, "20261003"))
    assert entry == {"night": "20261003", "wfs": ["mmirs"], "n_frames": 10, "median_delivered_iq": 1.23}


def test_read_night_no_frames_null_iq(tmp_path):
    entry = read_night(make_night(tmp_path, "20261003", wfs=(), n_frames=0, iq=None))
    assert entry["wfs"] == []
    assert entry["n_frames"] == 0
    assert entry["median_delivered_iq"] is None


def test_read_night_uses_directory_name(tmp_path):
    d = make_night(tmp_path, "20261003", summary={"night": "20990101", "wfs": [], "n_frames": 0})
    assert read_night(d)["night"] == "20261003"


def test_read_night_neither_file(tmp_path):
    assert read_night(make_night(tmp_path, "20261003", html=None, summary=None)) is None


@pytest.mark.parametrize("kwargs", [dict(html=None), dict(summary=None), dict(summary="{not json"),
                                    dict(summary="[1, 2]")])
def test_read_night_problems(tmp_path, kwargs):
    with pytest.raises(NightError):
        read_night(make_night(tmp_path, "20261003", **kwargs))


@pytest.mark.parametrize("missing", ["wfs", "n_frames"])
def test_summary_missing_field_is_skipped(tmp_path, missing):
    summary = {"night": "20261003", "wfs": ["f5"], "n_frames": 3, "median_delivered_iq": 1.0}
    del summary[missing]
    with pytest.raises(NightError):
        read_night(make_night(tmp_path, "20261003", summary=summary))


def test_collect_sorted_and_counts(tmp_path):
    make_night(tmp_path, "20261005")
    make_night(tmp_path, "20261003")
    make_night(tmp_path, "20261004", summary="{bad")
    make_night(tmp_path, "20261006", html=None, summary=None)
    entries, skipped = collect([tmp_path])
    assert [e["night"] for e in entries] == ["20261003", "20261005"]
    assert skipped == 1


def test_duplicate_dirs_give_one_entry(tmp_path):
    make_night(tmp_path, "20261003")
    entries, skipped = collect([tmp_path, tmp_path / "20261003"])
    assert len(entries) == 1
    assert skipped == 0
