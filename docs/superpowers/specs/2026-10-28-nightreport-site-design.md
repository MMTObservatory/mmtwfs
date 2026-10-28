# Night report website: design

Date: 2026-10-28
Status: draft for review

## Purpose

Collect the per-night reports written by `nightreport` into one static website. A calendar at the top selects the
night, and that night's report is shown below it. The site is for internal use and is served as a plain static
directory from an observatory web server.

## What exists

- Each night directory `YYYYMMDD` in the archive holds `nightreport.html`, `nightreport_summary.json`, and five CSVs.
  The directory date is the morning of the night (the night window runs from local noon on YYYYMMDD−1 to local noon on
  YYYYMMDD).
- `nightreport.html` is self-contained: the plots are base64 PNGs, and the CSS is inline.
- In the local 2026 archive there are 280 reports, 123 MB in all, the largest 1.7 MB. 121 of those nights have no WFS
  frames, and their summaries have `n_frames` 0 and an empty `wfs` list.
- Summary fields the site uses: `night`, `wfs` (a list of WFS keys such as `newf9`, `f5`, `mmirs`, `binospec`),
  `n_frames`, and `median_delivered_iq` (arcsec, or null).

## Output

A directory that can be copied, tarred, or synced to the web server as it is:

```
index.html        calendar page, with inline CSS and JS
nights.json       manifest of the nights
nights/YYYYMMDD.html   each night's nightreport.html, copied unchanged
```

There is no server-side code and the page makes no requests outside the directory, so it works on an isolated
network. Deployment is, for example, `rsync -a --delete site/ server:/path/`.

`nights.json` is a list sorted by night:

```json
[{"night": "20261003", "wfs": ["mmirs"], "n_frames": 580, "median_delivered_iq": 2.2}, ...]
```

`median_delivered_iq` is rounded to two decimals, or null.

## Page

- **Calendar, one month at a time:**
  - The controls are a previous-month button, a month pulldown, a year pulldown, and a next-month button.
  - The year pulldown lists the years present in `nights.json`, so it shows only 2026 until other years are added.
  - The prev/next buttons cross year boundaries, and they stop at the first and last months that have nights.
  - Weeks run Sunday to Saturday.
- **Day tiles:**
  - A night with frames is coloured by its WFS: F/9 (`newf9`, `f9`, `oldf9`), F/5, MMIRS, Binospec, "mixed" when
    the list has more than one display group, and "other" for any key not in that map. The tile shows the day
    number and, when it is not null, the median delivered IQ in small text, e.g. `1.2″`.
  - A night with no frames is drawn dimmed but can still be selected, so its report (with its warnings) can be
    seen.
  - A date with no report is blank and cannot be selected.
  - The selected night is outlined.
  - A legend under the calendar shows the WFS colours.
- **Report:** the selected night's `nights/YYYYMMDD.html` loads in a full-width iframe below the calendar. Its height
  follows its content, so the page has a single scrollbar. This works because the report has the same origin.
- **Selection and URL:**
  - The selected night is kept in the URL hash (`#20261003`), so a night can be linked to and the back button
    works.
  - On load the page opens the night in the hash. If the hash is missing or not a night in the manifest, it opens
    the most recent night with frames.
  - The calendar shows the selected night's month.
- **Keyboard:** the left and right arrow keys step to the previous or next night with frames, and the calendar
  follows into another month or year when needed.
- **Theme:** light and dark follow `prefers-color-scheme`. The report itself stays as `nightreport` draws it.
- The layout must work on a narrow screen without horizontal page scrolling.

## Build script

A new console script, `nightreport_site`, in `mmtwfs/scripts/nightreport_site.py`, registered in `pyproject.toml`:

```
nightreport_site DIR [DIR ...] -o OUTDIR
```

- `DIR` is a night directory or a parent of night directories, found with `nightreport.night_dirs()`. In normal use
  it is the whole archive, so the manifest always covers everything.
- For each night that has both `nightreport_summary.json` and `nightreport.html`:
  - It adds the manifest entry from the summary.
  - It copies the report to `OUTDIR/nights/YYYYMMDD.html` when the destination is missing or differs in size or
    modification time from the source. The copy keeps the source's modification time, so reruns copy only changed
    reports.
- Then it writes `nights.json` and `index.html`, and removes any `OUTDIR/nights/*.html` that is not in the manifest.
- `index.html` comes from a template file kept in the package (`mmtwfs/data/nightreport_site/index.html`), copied as
  it is, because all the data it needs comes from `nights.json`.
- All writes go to a temporary name followed by `os.replace`, so a site that is served during a rebuild never shows a
  half-written file.
- Logging uses coloredlogs, set up the same way as in `nightreport`.

## Errors

- A night directory that has neither file is skipped silently, because `nightreport` has not run on it yet.
- A night with only one of the two files, or with a summary that cannot be read or parsed, is skipped with a logged
  warning. Its old copy in `nights/` is removed by the prune step.
- The exit code is 1 if any night was skipped with a warning, and 0 otherwise. The rest of the site is still
  written.
- If no nights are found, the script logs an error, writes nothing, and exits with 1.

## Testing

Unit tests in `mmtwfs/tests/test_nightreport_site.py` build a small archive in `tmp_path` with hand-written summaries
and reports:

- manifest contents, ordering, and rounding, including a night with no frames and a null IQ
- reports are copied byte for byte, and a rerun leaves unchanged copies untouched (modification time is kept)
- a changed source report is copied again
- reports that are no longer in the manifest are pruned
- skipped nights: missing html, missing summary, bad JSON, and the exit codes for each
- no nights found
- `index.html` is written and references `nights.json`

The calendar's JS is checked by hand against the real 2026 archive: month and year navigation, the arrow keys, the
hash, and the dimmed and blank days.

## Out of scope

- The per-night CSVs and download links.
- Building the reports themselves (that is still `nightreport`).
- Any server configuration or access control on the web server.
