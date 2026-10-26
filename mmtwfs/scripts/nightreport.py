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
from matplotlib.dates import date2num, DateFormatter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table, vstack  # noqa: E402

from mmtwfs.nightlog import FIT_TERMS, get_mmt, load_night, SETTLE_DEFAULTS  # noqa: E402
from mmtwfs.version import version  # noqa: E402
from mmtwfs.zernike import ZernikeVector  # noqa: E402

log = logging.getLogger("NightReport")

VISIT_TABLE_COLUMNS = [
    ("catid", "{}"), ("start", "{}"), ("gap", "{:.0f}"), ("t_first_ok", "{:.0f}"), ("focus_settle_thresh", "{:.0f}"),
    ("focus_settle_conv", "{:.0f}"), ("iq_settle_conv", "{:.0f}"), ("coma_settle_thresh", "{:.0f}"),
    ("settle_last_send", "{:.0f}"), ("first_focus", "{:.1f}"), ("first_cc_x", "{:.2f}"), ("first_cc_y", "{:.2f}"),
    ("first_iq", "{:.2f}"), ("n_sends_settle", "{}"), ("net_focus_settle", "{:.1f}"), ("n_sends_visit", "{}"),
    ("net_focus_visit", "{:.1f}"), ("m1_sent_rms_visit", "{:.1f}"), ("flags", "{}"),
]
EVENT_TABLE_COLUMNS = [("flag", "{}"), ("time", "{}"), ("kind", "{}"), ("file", "{}"), ("detail", "{}")]
EVENT_KINDS = {
    "clear_m1": "M1 forces and M2 spherical-focus offsets cleared",
    "clear_m2": "M2 hexapod WFS offsets reset to 0",
    "m1_gain": "M1 gain set",
    "m2_gain": "M2 gain set",
    "wfs_select": "WFS selected in wfssrv",
    "connect": "WFS connection state reported by wfssrv",
    "offset": "telescope RA/Dec offset (RAOFF/DECOFF header, arcsec) changed within a visit, e.g. a dither; "
              "detail is the new offset",
    "validity": "wfssrv judged the wavefront fit unusable for some corrections",
    "error": "error logged by the Telescope or Secondary",
    "spher_ignored": "frames whose spherical aberration was too large to correct",
    "m1_blocked_focus": "pending focus over the M1 gate, so wfssrv would not send M1 forces",
    "m1_partial": "the cell applied only part of the requested M1 forces",
    "missing_forces": "M1 forces were sent but the .forces file isn't in the archive",
}
# one font-size step up from matplotlib's defaults, for figures viewed scaled down in a browser
PLOT_RC = {"font.size": 12, "legend.fontsize": 11, "axes.titlesize": 13}
LEGEND_RIGHT = dict(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)


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
        return {str(k): _clean(v) for k, v in value.items()}
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
        "median_zenith_seeing": _median(ok["vlt_seeing"]),
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


def _clear_handles(events):
    handles = []
    if np.any(events["kind"] == "clear_m1"):
        handles.append(Line2D([], [], color="C3", ls="--", lw=1, label="M1 forces cleared"))
    if np.any(events["kind"] == "clear_m2"):
        handles.append(Line2D([], [], color="C4", ls=":", lw=1, label="M2 WFS offsets cleared"))
    return handles


def _flag_labels(flag_times, frame_times):
    """
    Number the flags 1..N, merging runs closer than 3% of the night into one "first-last" label so they don't
    overprint. Times are matplotlib date numbers; annotate skips unit conversion. Returns (time, label) pairs.
    """
    close = 0.03 * (frame_times.max() - frame_times.min())
    labels, first = [], 0
    for n in range(1, len(flag_times) + 1):
        if n == len(flag_times) or flag_times[n] - flag_times[first] > close:
            labels.append((flag_times[first], str(first + 1) if n == first + 1 else f"{first + 1}\u2013{n}"))
            first = n
    return labels


def _twin_steps(ax, table, left, right, left_label, right_label):
    """Step plots of table columns against two y axes, (column, legend label) pairs per side, with one legend."""
    right_ax = ax.twinx()
    for n, (side, col, label) in enumerate([(ax, c, lbl) for c, lbl in left] + [(right_ax, c, lbl) for c, lbl in right]):
        side.step(table["time"], table[col], where="post", color=f"C{n}", label=label)
    ax.set_ylabel(left_label)
    right_ax.set_ylabel(right_label)
    # one legend for both axes, shifted to clear the right-hand axis labels
    ax.legend(handles=ax.get_lines() + right_ax.get_lines(), **{**LEGEND_RIGHT, "bbox_to_anchor": (1.1, 1.0)})


def timeline_figure(night):
    f, ok = night.frames, night.frames[night.frames["status"] == "ok"]
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(12, 21), gridspec_kw=dict(height_ratios=[1.6] + [1] * 6))
    pts = dict(marker="o", ls="none", ms=5, mew=0, alpha=0.7)
    ax = axes[0]
    ax.plot(f["time"], f["el"], ".", ms=4, color="0.4", label="frame elevation")
    for v in night.visits:
        ax.axvspan(v["start"], v["end"], color=f"C{v['visit'] % 2}", alpha=0.15)
        ax.text(v["start"], 1.02, v["catid"], transform=ax.get_xaxis_transform(), fontsize=10, rotation=30)
    handles = ax.get_legend_handles_labels()[0] + _clear_handles(night.events)
    if len(night.flags):
        # flags sit near the top of the panel, numbered to match the legend and the Events table
        xform = ax.get_xaxis_transform()
        ax.plot(night.flags["time"], np.full(len(night.flags), 0.9), "v", color="C3", transform=xform)
        for t, label in _flag_labels(date2num(night.flags["time"]), date2num(f["time"])):
            ax.annotate(label, (t, 0.9), xycoords=xform, xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=10, color="C3")
        handles.append(Line2D([], [], marker="v", ls="none", color="C3", label="flag (number, UTC, kind):"))
        handles += [Line2D([], [], ls="none", label=f"{n}  {str(fl['time'])[11:16]}  {fl['kind']}")
                    for n, fl in enumerate(night.flags, 1)]
    ax.legend(handles=handles, ncol=2 if len(handles) > 14 else 1, **LEGEND_RIGHT)
    ax.set_ylabel("EL (deg)")
    ax = axes[1]
    ax.axhline(0, color="0.7", lw=0.8)
    ax.errorbar(ok["time"], ok["focus"], yerr=ok["focus_unc"], fmt="o", ms=4, mew=0, lw=0.8, color="C0",
                label="measured focus error")
    focus_sends = night.sends["time"][night.sends["kind"] == "focus"]
    ax.plot(focus_sends, np.full(len(focus_sends), 0.04), "|", ms=14, mew=1.5, color="C1",
            transform=ax.get_xaxis_transform(), label="focus correction sent")
    ax.set_ylabel("focus error ($\\mu$m)")
    ax = axes[2]
    ax.plot(ok["time"], ok["cc_x_err"], **pts, label="CC x")
    ax.plot(ok["time"], ok["cc_y_err"], **pts, label="CC y")
    ax.set_ylabel("coma (arcsec)")
    ax = axes[3]
    ax.plot(ok["time"], ok["raw_vlt_seeing"], **pts, label="raw seeing")
    ax.plot(ok["time"], ok["vlt_seeing"], **pts, label="zenith seeing")
    ax.plot(ok["time"], ok["delivered_iq"], **pts, label="delivered IQ")
    ax.set_ylabel("arcsec")
    _twin_steps(axes[4], night.m2, [("wfs_z", "wfs_z ($\\mu$m)"), ("m1spherical_z", "m1spherical_z ($\\mu$m)")],
                [("wfs_cc_x", "wfs_cc_x (arcsec)"), ("wfs_cc_y", "wfs_cc_y (arcsec)")],
                "cumulative M2 focus\n($\\mu$m)", "cumulative M2 coma\n(arcsec)")
    _twin_steps(axes[5], night.m1, [("frms", "force RMS (N)")], [("surf_rms", "surface RMS (nm)")],
                "cumulative M1\nforce RMS (N)", "cumulative M1\nsurface RMS (nm)")
    ax = axes[6]
    zv = ZernikeVector()
    for term in FIT_TERMS:
        if len(night.m1) and np.max(np.abs(night.m1[term])) > 10:
            ax.step(night.m1["time"], night.m1[term], where="post", label=f"{term} {zv.shortlabel(term).strip()}")
    ax.set_ylabel("cumulative M1\nwavefront (nm)")
    ax.set_xlabel("UTC")
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    for ax in axes:
        _mark_clears(ax, night.events)
    for ax in axes[1:]:
        if ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
            ax.legend(**LEGEND_RIGHT)
    return fig


def surface_figures(night):
    figs = []
    mmt = get_mmt()
    for time, label, vec in night.m1_maps:
        fig = mmt.plot_force_influence(Table([mmt.actcoor["act_id"], vec], names=["actuator", "force"]))
        ax = fig.axes[0]
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.06 * (hi - lo))  # headroom for the outer actuator labels, which sit above their points
        ax.set_title(f"{label}, {time} UTC")
        figs.append(fig)
    return figs


def slew_figure(night):
    v = night.visits
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(v["d_el"], v["first_focus"], "o")
    axes[0, 0].set(xlabel="ΔEL (deg)", ylabel="first-frame focus ($\\mu$m)")
    axes[0, 1].plot(v["el"], v["first_focus"], "o")
    axes[0, 1].set(xlabel="EL (deg)", ylabel="first-frame focus ($\\mu$m)")
    axes[1, 0].plot(v["d_el"], v["focus_settle_thresh"], "o")
    axes[1, 0].set(xlabel="ΔEL (deg)", ylabel="focus settle (s)")
    x = np.arange(len(v))
    axes[1, 1].bar(x - 0.2, v["gap"], 0.4, label="gap")
    axes[1, 1].bar(x + 0.2, v["t_first_ok"], 0.4, label="time to first good frame")
    axes[1, 1].set(xlabel="visit", ylabel="s")
    axes[1, 1].legend(**LEGEND_RIGHT)
    return fig


def _cell(value, fmt):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return html.escape(fmt.format(value))


def _summary_value(value):
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(_summary_value(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_summary_value(v)}" for k, v in value.items()) or "none"
    return "" if value is None else str(value)


def _html_table(table, columns):
    head = "".join(f"<th>{name}</th>" for name, _ in columns)
    rows = "".join("<tr>" + "".join(f"<td>{_cell(r[name], fmt)}</td>" for name, fmt in columns) + "</tr>"
                   for r in table)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


CSS = """body{font-family:system-ui,sans-serif;margin:16px;color:#222;background:#fff}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}
th{background:#f0f0f0}.warn{color:#a40}img{max-width:100%}
.scroll{overflow:auto;max-height:70vh;max-width:100%;border:1px solid #ccc}
.scroll th{position:sticky;top:0}dt{font-weight:bold;float:left;clear:left;width:11em}dd{margin-left:11em}"""


def render_html(night, summary):
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>WFS night {summary['night']}</title>"
             f"<style>{CSS}</style></head><body><h1>WFS night {summary['night']}</h1>"]
    parts.append("<h2>Summary</h2><table>" + "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(_summary_value(val))}</td></tr>"
        for k, val in summary.items() if k not in ("warnings", "params", "counts")) + "</table>")
    parts += [f"<p class='warn'>{html.escape(w)}</p>" for w in night.warnings]
    if len(night.frames) == 0:
        parts.append("<p>No WFS frames in this night directory.</p></body></html>")
        return "".join(parts)
    with plt.rc_context(PLOT_RC):
        parts.append("<h2>Night timeline</h2>" + _img(timeline_figure(night)))
        parts.append("<h2>M1 surface</h2>")
        maps = surface_figures(night)
        parts.append("".join(_img(fig) for fig in maps) if maps else "<p>No M1 forces applied.</p>")
        parts.append("<h2>Slew response</h2>" + _img(slew_figure(night)))
    parts.append("<p>Gaps are as seen by the WFS: slew, acquisition, and WFS restart together.</p>")
    parts.append("<h2>Visits</h2><p>Times in seconds from each visit's first frame.</p>"
                 + "<div class='scroll'>" + _html_table(night.visits, VISIT_TABLE_COLUMNS) + "</div>")
    events = night.events[[k not in ("validity", "error") for k in night.events["kind"]]]  # listed again as flags
    events["flag"] = ""
    flags = Table(night.flags, copy=True)
    flags.remove_column("visit")
    flags["flag"] = [str(n) for n in range(1, len(flags) + 1)]
    combined = vstack([events, flags]) if len(flags) else events
    combined.sort("time")
    kinds = "".join(f"<dt>{k}</dt><dd>{html.escape(desc)}</dd>" for k, desc in EVENT_KINDS.items()
                    if k in set(combined["kind"]))
    parts.append("<h2>Events</h2><p>Rows with a flag number are flags, numbered as in the timeline.</p>"
                 f"<dl>{kinds}</dl>" + _html_table(combined, EVENT_TABLE_COLUMNS))
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
