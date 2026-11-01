# Night report website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `nightreport_site` console script that collects every night's `nightreport.html` into a static site with
a month calendar page.

**Architecture:** A Python build script (`mmtwfs/scripts/nightreport_site.py`) reads each night's
`nightreport_summary.json` into a `nights.json` manifest, mirrors the reports into `OUTDIR/nights/`, and copies a
fixed `index.html` template from package data. All behaviour on the page (calendar, selection, iframe) is plain JS
in that template, driven by `nights.json`.

**Tech Stack:** Python ≥3.13 stdlib (argparse, json, shutil, os, importlib.resources), coloredlogs, pytest; vanilla
HTML/CSS/JS with no external requests.

**Spec:** `docs/superpowers/specs/2026-10-28-nightreport-site-design.md`

## Global Constraints

- flake8 max line length 127; file header comments as in `mmtwfs/scripts/nightreport.py`.
- Output layout: `OUTDIR/index.html`, `OUTDIR/nights.json`, `OUTDIR/nights/YYYYMMDD.html`.
- `nights.json`: list sorted by night of `{"night", "wfs", "n_frames", "median_delivered_iq"}`; IQ rounded to two
  decimals, or null.
- The page makes no requests outside its own directory (no CDN fonts or scripts).
- Every write goes to a temporary name in the same directory, then `os.replace`.
- Copy a report only when the destination is missing or differs in size or mtime; the copy keeps the source mtime.
- Exit 1 if any night was skipped with a warning; exit 1 and write nothing if no nights are found; else 0.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **The same night reached twice** (e.g. `nightreport_site ~/MMT/wfsdat ~/MMT/wfsdat/20261003 -o site`): one manifest
  entry, no error. Test: `test_duplicate_dirs_give_one_entry` (Task 1).
- **A summary whose JSON parses but lacks a field the site needs** (`wfs`, `n_frames`), or is not an object: skipped
  with a warning like bad JSON, not a crash. Test: `test_summary_missing_field_is_skipped` (Task 1).
- **`OUTDIR` that does not exist yet, or exists with stray files**: created; only `nights/*.html` not in the manifest
  are removed, nothing else in `OUTDIR` is touched. Test: `test_prune_only_touches_night_html` (Task 2).
- **The report grows after the iframe's load event** (window resized narrower, so tables wrap): the iframe height
  follows. Checked by hand in Task 4 (resize the window).
- **A hash that is not a night** (`#foo`, `#20990101`, a date with no report): falls back to the latest night with
  frames and rewrites the hash. Checked by hand in Task 4.

---

### Task 1: Read nights into manifest entries

**Files:**
- Create: `mmtwfs/scripts/nightreport_site.py`
- Test: `mmtwfs/tests/test_nightreport_site.py`

**Interfaces:**
- Consumes: `mmtwfs.scripts.nightreport.night_dirs(paths) -> list[Path]` (sorted, may contain duplicates).
- Produces:
  - `class NightError(Exception)` — a night that has to be skipped with a warning.
  - `read_night(path: Path) -> dict | None` — manifest entry for one night dir; `None` when neither
    `nightreport.html` nor `nightreport_summary.json` exists; raises `NightError` when only one exists, the summary
    can't be read/parsed, isn't a JSON object, or lacks `wfs` or `n_frames`. The entry's `night` is the directory
    name (not the summary's field).
  - `collect(dirs: list) -> tuple[list[dict], int]` — entries for every night under `dirs`, deduplicated by
    resolved path and sorted by night, plus the count of nights skipped with a warning (each logged with
    `log.warning`).
  - `log = logging.getLogger("NightReportSite")`, set up with `coloredlogs.install(level="INFO", logger=log)`.

- [ ] **Step 1: Write the failing tests**

Shared helper in the test file (Task 2 reuses it):

```python
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
```

Tests:
- `test_read_night_entry`: `read_night(make_night(tmp_path, "20261003"))` `==
  {"night": "20261003", "wfs": ["mmirs"], "n_frames": 10, "median_delivered_iq": 1.23}`.
- `test_read_night_no_frames_null_iq`: `wfs=(), n_frames=0, iq=None` → entry has `wfs == []`, `n_frames == 0`,
  `median_delivered_iq is None`.
- `test_read_night_neither_file`: `make_night(..., html=None, summary=None)` → `read_night` returns `None`.
- `test_read_night_problems` parametrized over `dict(html=None)`, `dict(summary=None)`, `dict(summary="{not json")`,
  `dict(summary="[1, 2]")` → `pytest.raises(NightError)`.
- `test_summary_missing_field_is_skipped` parametrized over dropping `"wfs"` and `"n_frames"` → `NightError`.
- `test_collect_sorted_and_counts`: nights `20261005`, `20261003` (good), `20261004` with `summary="{bad"`, and
  `20261006` with neither file → entries' nights `== ["20261003", "20261005"]`, skipped `== 1`.
- `test_duplicate_dirs_give_one_entry`: `collect([tmp_path, tmp_path / "20261003"])` → one entry, skipped `== 0`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest mmtwfs/tests/test_nightreport_site.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'mmtwfs.scripts.nightreport_site'`.

- [ ] **Step 3: Implement `NightError`, `read_night`, `collect` in `mmtwfs/scripts/nightreport_site.py`**

IQ: `None if iq is None else round(float(iq), 2)`. Catch `OSError`, `json.JSONDecodeError`, `UnicodeDecodeError`
and re-raise as `NightError` with the night and reason in the message.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest mmtwfs/tests/test_nightreport_site.py -v && flake8 mmtwfs --count --max-line-length=127`
Expected: all pass, flake8 prints `0`.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/scripts/nightreport_site.py mmtwfs/tests/test_nightreport_site.py
git commit -m "nightreport_site: read night summaries into manifest entries"
```

### Task 2: Build the site directory and the console script

**Files:**
- Modify: `mmtwfs/scripts/nightreport_site.py`
- Create: `mmtwfs/data/nightreport_site/index.html` (placeholder for now: a minimal page whose script fetches
  `nights.json`; Task 3 replaces it)
- Modify: `pyproject.toml` (`[project.scripts]`: `nightreport_site = "mmtwfs.scripts.nightreport_site:main"`)
- Modify: `CLAUDE.md` (add `nightreport`, `nightreport_site` to the Console Scripts list)
- Test: `mmtwfs/tests/test_nightreport_site.py`

**Interfaces:**
- Consumes: `collect`, `log` from Task 1.
- Produces:
  - `TEMPLATE = importlib.resources.files("mmtwfs") / "data" / "nightreport_site" / "index.html"`
  - `sync_report(src: Path, dst: Path) -> bool` — copy if missing or size/mtime differ (`shutil.copy2` to a temp
    name, then `os.replace`); returns whether it copied.
  - `build_site(dirs: list, outdir: Path) -> int` — exit code. No entries → `log.error`, create nothing, return 1.
    Else create `outdir/nights/`, sync each report, write `nights.json` (`json.dumps(entries, indent=1)`) and
    `index.html` (template text) atomically, unlink `outdir/nights/*.html` whose stem is not in the manifest, log a
    one-line total (nights, copied, pruned, skipped), return `1 if skipped else 0`.
  - `main(argv=None) -> int` — argparse: positional `dirs` (`nargs="+"`, `metavar="DIR"`, same help as
    `nightreport`), required `-o/--outdir` (`type=Path`); returns `build_site(args.dirs, args.outdir)`.

- [ ] **Step 1: Write the failing tests**

- `test_build_site_outputs`: two good nights → `main([str(archive), "-o", str(out)]) == 0`; `json.loads(nights.json)`
  equals the two entries; each `out/nights/N.html` bytes equal the source; `(out / "index.html").read_text() ==
  TEMPLATE.read_text()` and contains `"nights.json"`.
- `test_rerun_leaves_copies_untouched`: build, record the copy's `st_mtime_ns` and `st_ino`, build again → both
  unchanged; `sync_report(src, copy)` returns `False`.
- `test_changed_report_is_recopied`: build, rewrite the source with different content and `os.utime` it forward by
  10 s, build → copy bytes equal the new source and `copy.stat().st_mtime_ns == src.stat().st_mtime_ns`.
- `test_prune_only_touches_night_html`: build with nights A and B, also create `out/notes.txt` and
  `out/nights/readme.txt`; remove B's dir from the archive; build → `out/nights/B.html` gone, A kept, both txt
  files kept.
- `test_skipped_night_exit_code_and_prune`: build with A and B; then replace B's summary with `"{bad"`; build →
  returns `1`, `nights.json` lists only A, `out/nights/B.html` removed.
- `test_no_nights_writes_nothing`: empty archive → `main([...]) == 1` and `not out.exists()`.
- `test_outdir_created`: `out = tmp_path / "a" / "b"` (not existing) → build returns 0 and `out/nights` exists.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest mmtwfs/tests/test_nightreport_site.py -v`
Expected: the new tests fail with `ImportError` for `main` / `sync_report` / `TEMPLATE`; Task 1's tests still pass.

- [ ] **Step 3: Implement `TEMPLATE`, `sync_report`, `build_site`, `main`; add the placeholder template, the
  `pyproject.toml` entry, and the CLAUDE.md line**

Temp names start with `.` and end with `.tmp` (e.g. `nights/.20261003.html.tmp`) so the `*.html` prune never sees
them.

- [ ] **Step 4: Run the tests, reinstall, and smoke-test the entry point**

Run: `pytest mmtwfs/tests/test_nightreport_site.py -v && flake8 mmtwfs --count --max-line-length=127 && pip install -e . -q && nightreport_site --help`
Expected: all pass, flake8 `0`, help text shows `DIR` and `-o OUTDIR`.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/scripts/nightreport_site.py mmtwfs/tests/test_nightreport_site.py mmtwfs/data/nightreport_site/index.html pyproject.toml CLAUDE.md
git commit -m "nightreport_site: build the static site directory"
```

### Task 3: Calendar page template

**Files:**
- Modify: `mmtwfs/data/nightreport_site/index.html` (full page: inline `<style>` and `<script>`)

**Interfaces:**
- Consumes: `nights.json` (format in Global Constraints) and `nights/YYYYMMDD.html`, both relative to the page.
- Produces: nothing other code depends on.

Page requirements (all from the spec's **Page** section; this list fixes the choices it leaves open):

- `<title>MMT WFS night reports</title>`; header line with the title and the selected night as `YYYY-MM-DD`.
- Controls row: `‹` button, `<select>` of month names, `<select>` of years present in the manifest, `›` button. The
  buttons are disabled at the first/last month that has any night; they cross year boundaries.
- Grid: 7 columns, header `Sun … Sat`, rows for the weeks of the month. Use local-date arithmetic on `Date(y, m, d)`
  only (no timezone conversions; nights are plain date strings).
- Tile states: night with frames → background from its colour group, day number, IQ as `1.23″` when not null;
  `n_frames == 0` → neutral background at reduced opacity, still a button; no report → empty, not focusable;
  selected → 2–3 px outline in the text colour. Tiles are `<button>`s with `title` like
  `2026-10-03 · MMIRS · 580 frames · 2.20″`.
- Colour groups: `newf9`, `f9`, `oldf9` → F/9; `f5` → F/5; `mmirs` → MMIRS; `binospec` → Binospec; more than one
  distinct group → mixed; any other key → other. Six colours defined as CSS custom properties, with dark-mode values
  under `@media (prefers-color-scheme: dark)`, all keeping readable text contrast. Legend under the grid.
- Report: `<iframe id="report" src="nights/N.html">`, `width: 100%`, no border. Height = the iframe document's
  `documentElement.scrollHeight`, set on the iframe `load` event and kept current with a `ResizeObserver` on the
  iframe's `documentElement` (covers window resizes and late layout).
- Selection: `location.hash` holds `#YYYYMMDD`. On load and on `hashchange`, if the hash is a night in the manifest
  select it, else select the latest night with `n_frames > 0` (or the latest night if none have frames) and
  `history.replaceState` the hash. Clicking a tile sets `location.hash` (so back works). Changing month/year controls
  only changes the month shown, not the selection.
- Keys: `ArrowLeft`/`ArrowRight` on the page (ignore when focus is in a `<select>`) select the previous/next night
  with `n_frames > 0`; the calendar jumps to its month.
- If `fetch("nights.json")` fails, show a one-line message in place of the calendar.
- Narrow screens: grid tiles shrink (`grid-template-columns: repeat(7, minmax(0, 1fr))`), 16 px side padding, no
  horizontal page scroll at 360 px width.

- [ ] **Step 1: Write the template**

- [ ] **Step 2: Check the template still satisfies Task 2's tests**

Run: `pytest mmtwfs/tests/test_nightreport_site.py -v`
Expected: all pass (`index.html` still references `nights.json`).

- [ ] **Step 3: Commit**

```bash
git add mmtwfs/data/nightreport_site/index.html
git commit -m "nightreport_site: calendar page"
```

### Task 4: Build from the real archive and check the page by hand

**Files:** none changed unless a fix is needed (fixes go in the file that owns the bug, with a test where Python).

`SCRATCH=/tmp/claude-1000/-home-tim-MMT-mmtwfs/95bd6a6f-ab22-48d7-918a-e1e5ad90fe6e/scratchpad`

- [ ] **Step 1: Build the site from the 2026 archive into the scratchpad**

Run: `nightreport_site ~/MMT/wfsdat -o $SCRATCH/site; echo exit=$?; ls $SCRATCH/site/nights | wc -l`
Expected: exit 0 and 280 files (or exit 1 with a warning naming each skipped night; investigate any).

- [ ] **Step 2: Rerun and confirm nothing is copied**

Expected: the summary line reports 0 copied, 0 pruned.

- [ ] **Step 3: Serve it and check in a browser**

Run: `python -m http.server -d $SCRATCH/site 8765` (the iframe height needs same-origin access, which `file://`
pages may not get).
Check: opens on the latest night with frames; month/year pulldowns and ‹ › (disabled at Jan 2026 / last month);
dimmed zero-frame days open their report; blank days are inert; arrow keys cross a month boundary; `#20261003`
deep link and the back button; `#foo` falls back; iframe height follows when the window is narrowed; dark mode; a
360 px wide window has no horizontal scroll; the browser's network panel shows only same-origin requests.

- [ ] **Step 4: Run the full test suite and flake8**

Run: `pytest mmtwfs -q && flake8 mmtwfs --count --max-line-length=127`
Expected: all pass, `0`.

- [ ] **Step 5: Commit any fixes, update the spec status line to "approved, implemented"**
