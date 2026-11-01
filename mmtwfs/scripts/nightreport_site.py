# mmtwfs/scripts/nightreport_site.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Collect the per-night reports written by nightreport into a static website: a month calendar page (index.html), a
manifest of the nights (nights.json), and a copy of each night's report (nights/YYYYMMDD.html).
"""

import argparse
import importlib.resources
import json
import logging
import math
import os
import shutil
from pathlib import Path

import coloredlogs

from mmtwfs.scripts.nightreport import night_dirs

log = logging.getLogger("NightReportSite")
coloredlogs.install(level="INFO", logger=log)

REPORT = "nightreport.html"
SUMMARY = "nightreport_summary.json"
TEMPLATE = importlib.resources.files("mmtwfs") / "data" / "nightreport_site" / "index.html"


class NightError(Exception):
    """A night that has to be skipped with a warning."""


def read_night(path):
    """
    Manifest entry for one night directory, or None if nightreport hasn't run on it. Raises NightError if only one
    of the report and summary exists or the summary can't be used.
    """
    has_report, has_summary = (path / REPORT).is_file(), (path / SUMMARY).is_file()
    if not (has_report or has_summary):
        return None
    if not has_summary:
        raise NightError(f"{path.name}: {REPORT} but no {SUMMARY}")
    if not has_report:
        raise NightError(f"{path.name}: {SUMMARY} but no {REPORT}")
    try:
        summary = json.loads((path / SUMMARY).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise NightError(f"{path.name}: can't read {SUMMARY}: {e}") from e
    if not isinstance(summary, dict):
        raise NightError(f"{path.name}: {SUMMARY} is not a JSON object")
    missing = [k for k in ("wfs", "n_frames") if k not in summary]
    if missing:
        raise NightError(f"{path.name}: {SUMMARY} has no {', '.join(missing)}")
    wfs, n_frames, iq = summary["wfs"], summary["n_frames"], summary.get("median_delivered_iq")
    if not (isinstance(wfs, list) and all(isinstance(w, str) for w in wfs)):
        raise NightError(f"{path.name}: wfs in {SUMMARY} is not a list of names: {wfs!r}")
    if not isinstance(n_frames, int) or isinstance(n_frames, bool):
        raise NightError(f"{path.name}: n_frames in {SUMMARY} is not an integer: {n_frames!r}")
    if iq is not None and (not isinstance(iq, (int, float)) or isinstance(iq, bool) or not math.isfinite(iq)):
        raise NightError(f"{path.name}: median_delivered_iq in {SUMMARY} is not a finite number: {iq!r}")
    return {
        "night": path.name,
        "wfs": wfs,
        "n_frames": n_frames,
        "median_delivered_iq": None if iq is None else round(float(iq), 2),
        # the page loads nights/N.html?v=<mtime>, so a regenerated report isn't shown from the browser's cache
        "mtime": int((path / REPORT).stat().st_mtime),
    }


def unique_nights(dirs):
    """Night name -> directory for every night under dirs. A name found in two different places keeps the first."""
    nights = {}
    for path in night_dirs(dirs):
        if path.name not in nights:
            nights[path.name] = path
        elif path.resolve() != nights[path.name].resolve():
            log.warning(f"Ignoring {path}: night {path.name} already found at {nights[path.name]}")
    return nights


def collect(dirs):
    """Manifest entries for every night under dirs, sorted by night, and the number of nights skipped."""
    entries, skipped = [], 0
    for path in unique_nights(dirs).values():
        try:
            entry = read_night(path)
        except NightError as e:
            log.warning(f"Skipping {e}")
            skipped += 1
            continue
        if entry is not None:
            entries.append(entry)
    return sorted(entries, key=lambda e: e["night"]), skipped


def _tmp(path):
    # dot-prefixed, so a half-written file is neither served under its real name nor matched by the *.html prune
    return path.with_name(f".{path.name}.tmp")


def _atomic_write(path, text):
    tmp = _tmp(path)
    tmp.write_text(text)
    os.replace(tmp, path)


def sync_report(src, dst):
    """Copy src to dst, keeping its mtime, unless dst already has the same size and mtime. Returns whether it copied."""
    s = src.stat()
    try:
        d = dst.stat()
        if (d.st_size, d.st_mtime_ns) == (s.st_size, s.st_mtime_ns):
            return False
    except FileNotFoundError:
        pass
    tmp = _tmp(dst)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


def build_site(dirs, outdir):
    """Write the site for every night under dirs into outdir. Returns the exit code."""
    entries, skipped = collect(dirs)
    if not entries:
        log.error(f"No night reports found under {', '.join(map(str, dirs))}")
        return 1
    outdir = Path(outdir)
    nightsdir = outdir / "nights"
    nightsdir.mkdir(parents=True, exist_ok=True)
    sources = unique_nights(dirs)
    copied = sum(sync_report(sources[e["night"]] / REPORT, nightsdir / f"{e['night']}.html") for e in entries)
    _atomic_write(outdir / "nights.json", json.dumps(entries, indent=1, allow_nan=False))
    _atomic_write(outdir / "index.html", TEMPLATE.read_text())
    keep = {e["night"] for e in entries}
    pruned = 0
    for page in nightsdir.glob("*.html"):
        if page.stem not in keep:
            page.unlink()
            pruned += 1
    log.info(f"{len(entries)} nights in {outdir}: {copied} copied, {pruned} pruned, {skipped} skipped")
    return 1 if skipped else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Static website of the nightreport pages, with a month calendar")
    parser.add_argument("dirs", nargs="+", metavar="DIR",
                        help="night directory (YYYYMMDD) or a parent containing night directories")
    parser.add_argument("-o", "--outdir", type=Path, required=True, help="directory to write the site into")
    args = parser.parse_args(argv)
    return build_site(args.dirs, args.outdir)
