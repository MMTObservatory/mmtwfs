# mmtwfs/tests/test_nightreport_site.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import json
import os

import pytest

from mmtwfs.scripts.nightreport import render_readme
from mmtwfs.scripts.nightreport_site import TEMPLATE, NightError, collect, main, read_night, sync_report


MTIME = 1790000000


def make_night(root, night, wfs=("mmirs",), n_frames=10, iq=1.234, html="<html>report</html>", summary=True):
    d = root / night
    d.mkdir(parents=True)
    if html is not None:
        (d / "nightreport.html").write_text(html)
        os.utime(d / "nightreport.html", (MTIME, MTIME))
    if summary is True:
        summary = {"night": night, "wfs": list(wfs), "n_frames": n_frames, "median_delivered_iq": iq}
    if summary is not None:
        (d / "nightreport_summary.json").write_text(summary if isinstance(summary, str) else json.dumps(summary))
    return d


def test_read_night_entry(tmp_path):
    entry = read_night(make_night(tmp_path, "20261003"))
    assert entry == {"night": "20261003", "wfs": ["mmirs"], "n_frames": 10, "median_delivered_iq": 1.23, "mtime": MTIME}


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


BAD_VALUES = [("wfs", None), ("wfs", 3), ("wfs", "mmirs"), ("wfs", [1]), ("n_frames", None), ("n_frames", True),
              ("n_frames", "3"), ("n_frames", 1.5), ("median_delivered_iq", "1.2x"), ("median_delivered_iq", [1]),
              ("median_delivered_iq", True), ("median_delivered_iq", "NaN"), ("median_delivered_iq", "Infinity")]


@pytest.mark.parametrize("field,value", BAD_VALUES)
def test_summary_wrong_type_is_skipped(tmp_path, field, value):
    summary = {"night": "20261003", "wfs": ["f5"], "n_frames": 3, "median_delivered_iq": 1.0}
    summary[field] = value
    text = json.dumps(summary)
    if value in ("NaN", "Infinity"):  # the bare tokens Python's json writes and reads, not strings
        text = text.replace(f'"{value}"', value)
    with pytest.raises(NightError):
        read_night(make_night(tmp_path, "20261003", summary=text))


def test_wrong_type_night_skipped_rest_written(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    make_night(archive, "20261003")
    make_night(archive, "20261004", summary='{"wfs": ["f5"], "n_frames": 3, "median_delivered_iq": NaN}')
    assert main([str(archive), "-o", str(out)]) == 1
    assert [e["night"] for e in json.loads((out / "nights.json").read_text())] == ["20261003"]


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


def test_same_night_in_two_places_gives_one_entry(tmp_path):
    make_night(tmp_path / "a", "20261003", n_frames=1)
    make_night(tmp_path / "b", "20261003", n_frames=2)
    entries, skipped = collect([tmp_path / "a", tmp_path / "b"])
    assert [e["n_frames"] for e in entries] == [1]
    assert skipped == 0


def build(archive, out):
    return main([str(archive), "-o", str(out)])


def test_build_site_outputs(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    make_night(archive, "20261003", html="<html>three</html>")
    make_night(archive, "20261004", wfs=(), n_frames=0, iq=None, html="<html>four</html>")
    assert build(archive, out) == 0
    assert json.loads((out / "nights.json").read_text()) == [
        {"night": "20261003", "wfs": ["mmirs"], "n_frames": 10, "median_delivered_iq": 1.23, "mtime": MTIME},
        {"night": "20261004", "wfs": [], "n_frames": 0, "median_delivered_iq": None, "mtime": MTIME},
    ]
    for night in ("20261003", "20261004"):
        assert (out / "nights" / f"{night}.html").read_bytes() == (archive / night / "nightreport.html").read_bytes()
    index = (out / "index.html").read_text()
    assert index == TEMPLATE.read_text()
    assert "nights.json" in index
    assert (out / "readme.html").read_text() == render_readme()
    assert 'href="readme.html"' in index
    names = sorted(p.name for p in out.rglob("*"))
    assert names == ["20261003.html", "20261004.html", "index.html", "nights", "nights.json", "readme.html"]


def test_rerun_leaves_copies_untouched(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    src = make_night(archive, "20261003") / "nightreport.html"
    build(archive, out)
    copy = out / "nights" / "20261003.html"
    before = copy.stat()
    assert build(archive, out) == 0
    after = copy.stat()
    assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)
    assert sync_report(src, copy) is False


def test_changed_report_is_recopied(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    src = make_night(archive, "20261003") / "nightreport.html"
    build(archive, out)
    src.write_text("<html>new and longer report</html>")
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10))
    assert build(archive, out) == 0
    copy = out / "nights" / "20261003.html"
    assert copy.read_bytes() == src.read_bytes()
    assert copy.stat().st_mtime_ns == src.stat().st_mtime_ns


def test_regenerated_report_changes_manifest_mtime(tmp_path):
    # the page loads nights/N.html?v=<mtime>, so a regenerated report gets a new URL and isn't served from cache
    archive, out = tmp_path / "archive", tmp_path / "site"
    src = make_night(archive, "20261003") / "nightreport.html"
    build(archive, out)
    os.utime(src, (MTIME + 3600, MTIME + 3600))
    build(archive, out)
    assert json.loads((out / "nights.json").read_text())[0]["mtime"] == MTIME + 3600


def test_template_versions_report_urls_and_revalidates_manifest():
    page = TEMPLATE.read_text()
    assert "?v=${" in page
    assert 'fetch("nights.json", {cache: "no-cache"})' in page


def test_same_size_newer_report_is_recopied(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    src = make_night(archive, "20261003", html="<html>aaaa</html>") / "nightreport.html"
    build(archive, out)
    src.write_text("<html>bbbb</html>")
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10))
    build(archive, out)
    assert (out / "nights" / "20261003.html").read_text() == "<html>bbbb</html>"


def test_prune_only_touches_night_html(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    make_night(archive, "20261003")
    b = make_night(archive, "20261004")
    build(archive, out)
    (out / "notes.txt").write_text("keep")
    (out / "nights" / "readme.txt").write_text("keep")
    for f in b.iterdir():
        f.unlink()
    b.rmdir()
    assert build(archive, out) == 0
    assert not (out / "nights" / "20261004.html").exists()
    assert (out / "nights" / "20261003.html").exists()
    assert (out / "notes.txt").exists()
    assert (out / "nights" / "readme.txt").exists()


def test_skipped_night_exit_code_and_prune(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    make_night(archive, "20261003")
    b = make_night(archive, "20261004")
    build(archive, out)
    (b / "nightreport_summary.json").write_text("{bad")
    assert build(archive, out) == 1
    assert [e["night"] for e in json.loads((out / "nights.json").read_text())] == ["20261003"]
    assert not (out / "nights" / "20261004.html").exists()


def test_no_nights_writes_nothing(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "site"
    archive.mkdir()
    assert build(archive, out) == 1
    assert not out.exists()


def test_outdir_created(tmp_path):
    archive, out = tmp_path / "archive", tmp_path / "a" / "b"
    make_night(archive, "20261003")
    assert build(archive, out) == 0
    assert (out / "nights").is_dir()
