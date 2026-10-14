# mmtwfs/scripts/nightreport.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Nightly WFS report: render a night built by mmtwfs.nightlog.load_night into a self-contained HTML page plus CSV and
JSON data files in the night's directory.
"""

import argparse
import base64
import concurrent.futures
import html
import io
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402

from mmtwfs.nightlog import FIT_TERMS, get_mmt, load_night, SETTLE_DEFAULTS  # noqa: E402
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
