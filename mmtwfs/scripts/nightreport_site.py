# mmtwfs/scripts/nightreport_site.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Collect the per-night reports written by nightreport into a static website: a month calendar page (index.html), a
manifest of the nights (nights.json), and a copy of each night's report (nights/YYYYMMDD.html).
"""

import json
import logging

import coloredlogs

from mmtwfs.scripts.nightreport import night_dirs

log = logging.getLogger("NightReportSite")
coloredlogs.install(level="INFO", logger=log)

REPORT = "nightreport.html"
SUMMARY = "nightreport_summary.json"


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
    iq = summary.get("median_delivered_iq")
    return {
        "night": path.name,
        "wfs": list(summary["wfs"]),
        "n_frames": summary["n_frames"],
        "median_delivered_iq": None if iq is None else round(float(iq), 2),
    }


def collect(dirs):
    """Manifest entries for every night under dirs, sorted by night, and the number of nights skipped."""
    entries, skipped, seen = [], 0, set()
    for path in night_dirs(dirs):
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        try:
            entry = read_night(path)
        except NightError as e:
            log.warning(f"Skipping {e}")
            skipped += 1
            continue
        if entry is not None:
            entries.append(entry)
    return sorted(entries, key=lambda e: e["night"]), skipped
