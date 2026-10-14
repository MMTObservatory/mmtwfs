# Nightly WFS report: design

Date: 2026-10-14
Status: draft for review

## Purpose

Produce a report for each night from the wfssrv log, the archived WFS frames, and the reanalyze outputs. The report is
for a morning-after review of active-optics performance. It answers questions that are hard to get at today:

- When were the primary mirror forces cleared during the night?
- What were the cumulative M1 force corrections, and what effect did they have on the mirror surface?
- What were the cumulative secondary (M2) corrections?
- When did slews happen, and how long did it take to get from one target to the next?
- How long did it take to settle on each target? Focus is the headline metric, because it has the biggest effect on
  delivered IQ.
- What corrections were needed after each slew?

**Scope:** this spec is Phase 1, an offline report. Phase 2 will bring parts of it into the wfssrv analysis interface
and gets its own spec. It will reuse the parser and accounting code defined here, so that code must not depend on
writing the report.

## Usage context

- **Who runs it, and where:**
  - The user runs it by hand on archived nights, in the same way they run `reanalyze`.
  - In production it runs from cron, after the previous night's data has been archived and reanalyze has run on it.
- **How data reaches the archive:** every morning, cron rotates `/home/mmtop/wfsdat` into
  `/<archive>/wfsdat/YYYYMMDD`. **The night is whatever that directory holds**, restricted to a **night window** of
  local noon on YYYYMMDD−1 to local noon on YYYYMMDD (MST).
  - Normally each `wfs.log` runs from about 07:30 local on the previous day to about 07:30 on the directory date.
  - Some archived logs extend past their night because rotation was skipped: 20261005 runs to 2026-10-07, and 20250203
    runs to 2025-07.
  - Log lines outside the window are counted and ignored, and the summary reports how many.
- **M1 starts each night cleared:** the mirror is lowered and powered down at the end of each night, and it comes back
  up in a cleared-forces state. So the cumulative M1 state starts at zero in every night directory, and nothing carries
  over from the previous night. Clears outside wfssrv (the cell GUI, or a reboot) are known to happen but are rare.
  They are invisible here and are accepted as a limitation.
- **How the WFSs are used:**
  - Binospec and MMIRS run continuously, in parallel with science. They are the main use case for these reports.
  - F/9 is deployed and run between science observations. It's supported, but with a different settle definition.
- **Cron requirements:** it must run non-interactively, with no network, Redis, or hardware connections, and repeated
  runs must produce the same result.

## Inputs

These are all files in a night directory.

**`wfs.log`.** Python logging lines from wfssrv, formatted
`YYYY-MM-DD HH:MM:SS,mmm - <logger> - <LEVEL> - <message>`.
- **Time:** timestamps are Arizona local time (MST, a fixed UTC−7 with no DST). They are converted to UTC.
- **Format stability:** the message formats are stable from 2025 on. Archived logs from 2022–2023 are nearly empty and
  record no corrections, so they aren't a target.
- **Noise:** more than 95% of the lines are `tornado.access` polling, which is dropped before pattern matching.
- **Names:** online frame paths look like `/home/mmtop/wfsdat/<name>.fits`. Frames are matched to the archive by
  basename.

**FITS frames** (`*.fits`). This includes failed frames. Files with no `DATE-OBS`, such as `*.coadded_spot.fits`, are
skipped. These header keys are used:

| Key | Use |
|---|---|
| `DATE-OBS` | frame time (UTC) |
| `WFSNAME` | which WFS took the frame |
| `EXPTIME` | exposure time |
| `CATID`, `CATRA`, `CATDEC` | target identity, used to split the night into visits |
| `RA`, `DEC`, `RAOFF`, `DECOFF` | fallback target position, and offset events |
| `AZ`, `EL` | slew distance, ΔAZ, ΔEL |
| `TRANSX`, `TRANSY`, `TRANSZ`, `TILTX`, `TILTY` | hexapod total position, plotted for context |

**Reanalyze outputs**, when present:
- `<stem>.output`: one CSV line with the columns in `mmtwfs.scripts.reanalyze.CSV_COLUMNS`.
- `<stem>.reanalyze.zernike`.
- `<stem>.failed`.

**Online wfssrv outputs:**
- `<name>.fits.forces`: 104 lines of actuator index and force in N, with the gain already applied. It is written
  **only for frames whose forces were sent**, and each send is an increment that the cell adds to its existing forces.
- `<name>.fits.masked.zernike`: the Zernike vector actually used for the proposed M1 forces, after errorbar shrinkage
  and the term mask. It is written for every frame analyzed online.

## Architecture

The logic lives in a new module, `mmtwfs/nightlog.py`, which builds the data and does no report rendering.
`mmtwfs/scripts/nightreport.py` is the console script and the renderer. A new `nightreport` entry in
`[project.scripts]` exposes it.

```
night dir ──► parse_log()      ─► sends, events, analyses
          ──► read_frames()    ─► frames
                  │
                  ▼
          build_visits(frames)                  ─► visits
          settle(visits, frames, sends, params)  ─► per-visit settle and correction metrics
          m1_accounting(sends, events)           ─► M1 snapshots (forces, surface RMS, Zernike terms)
          m2_accounting(sends, events)           ─► M2 snapshots (wfs z, CC x/y, m1spherical z)
          flags(analyses, frames, events)        ─► flagged events
                  │
                  ▼
          Night (dataclass that holds all of the above)
                  │
                  ▼
nightreport.py: render_html(night), write_csvs(night), write_summary(night)
```

`load_night(path, params=SETTLE_DEFAULTS) -> Night` is the one entry point, and Phase 2 will call it from wfssrv.
Every table is an `astropy.table.Table` with UTC `Time` columns.

## Data model

### `frames`: one row per FITS frame

- **From the header:** `time`, `file`, `wfs`, `exptime`, `catid`, `catra`, `catdec`, `ra`, `dec`, `raoff`, `decoff`,
  `az`, `el`, `transx`, `transy`, `transz`, `tiltx`, `tilty`.
- **From reanalyze `.output`, if present:** `focus`, `focerr`, `cc_x_err`, `cc_y_err`, `wavefront_rms`,
  `residual_rms`, `raw_vlt_seeing`, `delivered_iq`, `method`.
- **`status`:** `ok` (an `.output` exists), `failed` (a `.failed` exists), or `unanalyzed`.
- **`spher`:** whether this frame was analyzed online with `spher=true`, taken from the `/analyze` request in the log.
  It is null if the frame wasn't analyzed online.
- **`m1_resid_rms`:** the RMS of the M1-actionable Zernike vector (nm).
- **`m1_force_rms`:** the RMS of the proposed M1 forces (N).
- **Sources for those two M1 fields:** the frame's `.masked.zernike` if it exists. Otherwise they are recomputed with
  `wfs.calculate_primary(zv, mask=spher_mask)` from `.reanalyze.zernike`. The gain is the M1 gain in effect at that
  time, from the log, or the WFS default if the log doesn't say.
- **`visit`:** the index of the visit the frame belongs to.

### `sends`: one row per correction sent

Columns are `time`, `kind`, `file`, `value`, `value2`, `forces_file`, `fmax`, `fmin`, `applied`. The kinds are:

| kind | Log source | value(s) |
|---|---|---|
| `m1` | `Sending forces from <file>.forces` | `forces_file`; `fmax`/`fmin` come from `Maximum force` / `Minimum force` |
| `m1focus` | `Adjusting M1 focus by N um` | µm sent to the `m1spherical` z channel |
| `focus` | `Sending N um focus to secondary` | µm sent to the `wfs` z channel |
| `coma` | `Sending X arcsec/Y arcsec CC_X/CC_Y to secondary` | `value` = CC_X in ″, `value2` = CC_Y in ″ |

For `m1`, `applied` is `full` if the next cell reply is `...forces successfully applied in full.`, `partial` or `error`
for the other replies, and `unknown` if no reply is found.

**Matching `file`:** a send line doesn't name its frame, except for `m1`. So `file` is the most recently analyzed frame
before the send.

### `events`

These are non-send events. Columns are `time`, `kind`, `detail`, `file`. The kinds are:

| kind | Source |
|---|---|
| `clear_m1` | `GET /clearm1`, and the corresponding `Clearing forces...` lines |
| `clear_m2` | `GET /clearm2` |
| `clear_all` | `GET /clear` |
| `m1_gain`, `m2_gain` | `Setting M1 gain to N`, `Setting M2 gain to N` |
| `wfs_select` | `WFSsrv - INFO - Setting <wfs>`, logged when a WFS is selected (including at server start). Gains go back to that WFS's defaults. |
| `error` | `Telescope - ERROR` and `Secondary - ERROR` lines, such as cell `Bad force data` |
| `validity` | `only focus corrections valid` and `wavefront fit too poor` |

### `analyses`

One row per online `/analyze` request, with columns `time`, `file`, `spher`, `connect`.

### `visits`

A visit is a run of consecutive frames (in time order) that share a target key. The key is
(`round(catra, 4 decimal places of hours)`, `round(catdec, 4 decimal places of degrees)`, `catid`). If the catalog keys
are missing, the fallback is the target position (`ra`, `dec` minus `raoff`, `decoff`), and a new visit starts when the
separation exceeds `params["slew_sep"]` (default 60″).

Columns:
- `visit`, `catid`, `start`, `end`, `n_frames`, `n_ok`, `n_failed`
- `slew_sep`, `d_az`, `d_el`, `el`
- `gap`: from the previous visit's last frame to this visit's first frame
- `t_first_ok`: from visit start to the first `ok` frame
- the settle metrics below
- the correction metrics below
- `flags`

The first visit of the night has a null `gap` and null slew fields.

**Offsets:** changes in `raoff`/`decoff` within a visit become `offset` rows in `events`, not new visits.

**Caveat on `gap`:** it's what the WFS can see. It includes the slew, acquisition, and restarting the WFS. Pure slew
time is not available from these data, and the report states this.

## Settle definitions

All times are measured from the visit's first frame (`start`). Each metric can be `null` ("not reached").

**Binospec and MMIRS:**
- **`focus_settle_thresh` (definition 1):** the first of K consecutive `ok` frames with |`focus`| < F. Nothing has to
  be sent for this. `focerr` at that frame is reported alongside, so you can tell whether F is below the noise.
- **`focus_settle_conv` (definition 3):** the plateau is the median of |`focus`| over the last half of the visit's `ok`
  frames. Settling is the first of K consecutive `ok` frames with |`focus`| ≤ plateau + F.
- **`iq_settle_conv`:** the same plateau test on `delivered_iq`. Settling is within max(`iq_abs`, `iq_frac` × plateau)
  of the plateau.
- **`coma_settle_thresh`:** the first of K consecutive `ok` frames with |`cc_x_err`| < C and |`cc_y_err`| < C.

**F/9 (definition 2):**
- **`settle_last_send`:** the time of the last send of any kind during the visit. If there are no sends, it is the
  time of the first `ok` frame.
- Definitions 1 and 3 are still computed and stored, but they're marked as not meaningful for F/9, because the frames
  are sparse.

**M1** has no settle time, because the set of terms feeding the forces changes from frame to frame with the errorbars.
Instead, each visit reports:
- `m1_resid_first` and `m1_resid_last`: `m1_resid_rms` on the visit's first and last `ok` frames.
- `m1_force_first`: `m1_force_rms` on the first `ok` frame.

**Parameters**, in `SETTLE_DEFAULTS`. Each can be overridden from the CLI.

| name | default | meaning |
|---|---|---|
| `focus_thresh` (F) | 10 µm | focus settle threshold |
| `coma_thresh` (C) | 2″ | coma settle threshold |
| `consecutive` (K) | 2 | frames that must consecutively meet a threshold |
| `iq_abs` | 0.05″ | absolute IQ plateau tolerance |
| `iq_frac` | 0.05 | fractional IQ plateau tolerance |
| `slew_sep` | 60″ | fallback visit-split separation |

These defaults are first guesses. Before merging, they get checked against 20261005 (MMIRS) and a Binospec night.

## Corrections after slewing

Per visit:

**First good frame**, meaning what the slew left behind: `first_focus`, `first_focerr`, `first_cc_x`, `first_cc_y`,
`first_wavefront_rms`, `first_iq`, from the first `ok` frame.

**Sends**, summed over two windows: [`start`, settle] using `focus_settle_thresh` (or the visit end if it's null), and
the whole visit. The `_settle` and `_visit` columns give each window. For each window:
- `n_sends`
- `net_focus` (µm, `wfs` z)
- `net_m1focus` (µm)
- `net_cc_x`, `net_cc_y` (″)
- `m1_sent_rms`: the RMS of the summed `.forces` vectors sent in the window (N)
- `m1_sent_surf_rms`: that sum's surface RMS in nm, computed with `act2surf`

## Cumulative accounting

### M1

The state is a 104-element force vector, starting at zero for each night.
- **Sends:** each `m1` send adds the vector from its `forces_file`. A send whose `applied` isn't `full` is still added,
  and also flagged.
- **Clears:** `clear_m1` and `clear_all` reset the vector to zero.
- **A send with no forces file:** a missing `forces_file` is flagged, and the state doesn't change.
- **Snapshots:** after every change, a snapshot records:
  - `time`, and the max, min, and RMS of the force (N)
  - `surf_rms`: cumulative force @ `MMT.load_act2surf()` gives the surface at the BCV nodes (nm). This is its RMS.
  - **Zernike terms:** a least-squares fit of Z05, Z06, and Z09–Z22 to that surface. The fit uses
    `nodecoor["bcv_rho"]` and `nodecoor["bcv_phi"]` with the same normalization `bending_forces` uses.
  - **Units:** the coefficients are reported as wavefront nm (2 × surface), so they can be compared directly with the
    WFS-measured Zernikes. Plot labels say "wavefront".
- **Actuator mapping:** the index-to-actuator-ID mapping for `.forces` rows is the same one `bend_mirror` and
  `to_rcell` use.

**Consistency check:** within one span between clears, the fitted terms should roughly equal
−Σ(gain × sent `.masked.zernike`) over the terms M1 can hold. The integration test requires every term larger than
50 nm to agree within 25% of its amplitude. If the real-data validation shows that's too tight or too loose, the
tolerance is revisited before merge, and the decision is recorded. The report also shows the size of the mismatch as a
diagnostic.

### M2

Running sums per channel, each starting at zero:
- `wfs_z` (µm), which is the sum of `focus` sends
- `wfs_cc_x` and `wfs_cc_y` (″), which are the sums of `coma` sends
- `m1spherical_z` (µm), which is the sum of `m1focus` sends

Resets:
- `clear_m1` resets `m1spherical_z`.
- `clear_m2` resets `wfs_*`.
- `clear_all` resets everything.

A snapshot is taken after every change. The header `TRANSZ`, `TILTX`, and `TILTY` from the frames are plotted alongside
as total hexapod position, which includes the elevation and temperature LUTs. There's no reconciliation between them
and the commanded sums.

## Flagged events

| flag | Detection | Reported detail |
|---|---|---|
| `spher_ignored` | spans of consecutive `analyses` with `spher=true` | span start and end; Z11 being ignored, from the frames' Zernikes; force max and RMS of `calculate_primary` with only the Z22 mask, i.e. the forces a full correction including spherical would have needed |
| `m1_blocked_focus` | `ok` frames with \|`focus`\| > 150 µm (the wfssrv gate) | file, focus |
| `validity` | `validity` events | message |
| `m1_partial` | `m1` sends whose `applied` isn't `full` | file, reply |
| `missing_forces` | `m1` send with no `.forces` file in the archive | file |
| `error` | `error` events | message |
| `wfs_reselect` | a `wfs_select` after the night's first frame | time and WFS. Gains reset to the defaults here. M1 and M2 accounting continue, because the hardware state persists. |

Flags appear on the visit they fall in, in the events table, and as markers on the timeline.

## Outputs

These are all written to the night directory. They're regenerated in place on each run.

**`nightreport.html`** is self-contained, with matplotlib figures inlined as base64 PNG. It has no JavaScript and no
external resources. Sections:
1. **Summary:**
   - night, WFS(s), and UTC span
   - frames: ok, failed, unanalyzed
   - number of visits, clears by kind, and sends by kind
   - medians: `raw_vlt_seeing`, `delivered_iq`, `focus_settle_thresh`, `focus_settle_conv`, `gap`, `t_first_ok`
   - flag counts
   - warnings: reanalyze outputs missing, `wfs.log` missing, WFS reselected mid-night, log lines outside the night
     window
2. **Night timeline:** panels stacked on a shared UTC axis.
   - **Panels, top to bottom:**
     1. EL, with the visits drawn as labeled bands
     2. focus with `focerr` errorbars, plus ticks for focus sends
     3. CC x/y
     4. raw VLT seeing and delivered IQ
     5. cumulative M2 channels
     6. cumulative M1 force RMS and surface RMS
     7. cumulative M1 Zernike terms (wavefront nm)
   - **Overlays:** clears as vertical lines on every panel, and flags as markers.
3. **M1 surface maps** of the cumulative forces, drawn just before each clear and at the end of the night, in the style
   of `MMT.plot_force_influence`.
4. **Slew response:** `first_focus` against `d_el` and against `el`; `focus_settle_thresh` against `d_el`; and `gap` and
   `t_first_ok` per visit.
5. **Visits table:** one row per visit with the columns above.
6. **Events table:** clears, gain changes, offsets, flags, and errors.

**Data files:**
- `nightreport_frames.csv`
- `nightreport_visits.csv`
- `nightreport_events.csv`: sends and events combined, with a `kind` column
- `nightreport_m1.csv` and `nightreport_m2.csv`: the snapshots
- `nightreport_summary.json`: the summary scalars, plus the mmtwfs version and the parameters used

## CLI

```
nightreport [--forcedir] [--focus-thresh 10] [--coma-thresh 2] [--consecutive 2]
            [--iq-abs 0.05] [--iq-frac 0.05] [--nproc N] DIR [DIR ...]
```

- **Arguments:** each `DIR` is a night directory, or a parent containing `YYYYMMDD` night directories. The argument
  handling matches `reanalyze`.
- **Skipping:** a night that already has `nightreport_summary.json` is skipped unless `--forcedir` is given.
- **Errors:**
  - An error on one night is logged with its traceback and doesn't stop the other nights.
  - The exit status is 1 if any night errored, and 0 otherwise.
  - Nights with no frames or no log are not errors.
- **Degraded nights:**
  - **No frames:** a stub report is written that says so.
  - **No `wfs.log`:** a report is written from the frames only, with visits, settle times, and seeing. Sends, clears,
    and accounting are omitted, and a warning is shown.
  - **No reanalyze outputs:** a report is written with slews, sends, clears, and accounting. The frame measurements and
    settle times are missing, and a warning is shown.

## Error handling

- **Parser:**
  - Unrecognized log lines are counted, and the count goes into the summary.
  - Malformed lines, such as truncated ones at log rotation, are skipped and counted.
- **Frames:** unreadable FITS files are skipped with a warning and counted.
- **Recomputing M1 values:** any failure gives a null for that frame, with a log warning. The night still completes.
- **Partial results:** each output file is written to a temp name and renamed into place. An interrupted run never
  leaves a half-written report that `--forcedir`-less reruns would then skip.

## Testing

**Parser:**
- Table-driven unit tests with real log lines copied from 2025 and 2026 logs, one per pattern: sends of each kind,
  clears, gains, validity, cell errors, spherical-ignore `/analyze` requests, and polling noise.
- Timezone conversion and the night window, including a log that runs past the night, like 20261005.

**Visits:** synthetic frame tables covering a target change, an offset within a target, and the fallback with no
`CATRA`.

**Settle:** synthetic sequences covering settling, never settling, a single good frame among bad ones (with K=2, it
must not count), the F/9 last-send definition, and a visit with no `ok` frames.

**Accounting:**
- Sends interleaved with each clear kind, checking that the resets are per channel.
- A missing forces file.
- A surface Zernike fit on synthetic forces from `bending_forces(ZernikeVector(Z05=…))`, which should recover Z05.

**Integration:**
- A trimmed real night under `mmtwfs/data/test_data/nightreport/`: about 20 frames spanning two visits. It includes
  their `.output`, `.forces`, and `.masked.zernike` files, plus a trimmed `wfs.log` with a clear and sends of each kind.
- Assertions on the visits, settle times, accounting snapshots, the consistency check, and the files produced.
  Running it twice must give identical CSVs.

**CLI:**
- Exit codes, `--forcedir` skipping, and a parent directory with multiple nights.
- A night with no log and a night with no reanalyze outputs.

**Validation before merge, done by hand, not in CI:** run on the 2026 Samsung_4TB nights. Spot-check 20261003 (with its
cell `Bad force data` error) and 20261005 against the log. Choose the default thresholds from what that shows.

## Out of scope

- wfssrv panels (Phase 2).
- Trending across many nights. `nightreport_summary.json` is designed so a later aggregator can collect it.
- Pure slew time, and any reconciliation of commanded M2 offsets against the header hexapod positions.
- Logs from before 2025.
