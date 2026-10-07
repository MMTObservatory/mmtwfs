# Poor-seeing WFS analysis: background modeling and periodicity focus fallback

Date: 2026-10-06
Status: approved design, pending implementation plan

## Motivation

In the MMIRS run of 2026-10-01..05, 1707 of 2697 WFS frames failed analysis. Vignetting explains only a few
percent of them. A random sample of failures shows:

- many frames with a clearly visible lenslet grid whose spots are too fuzzy or too low-contrast for spot finding
  and centroiding (e.g. 20261005/0245, 20261003/0298, 20261002/0386);
- faint frames where the grid is still periodic (20261002/0073, 20261005/0367);
- strong large-scale halos and gradients under the pupil (e.g. 20261003/0519 and the successful 20261001/0328) that
  the current small-box `Background2D` (20x20 px for MMIRS) handles poorly. In bad seeing the spot mask covers most
  of the pupil and the background fit absorbs spot light, lowering contrast.

When spots blur together, centroiding fails, but the grid period is still measurable. Pure defocus rescales the
spot grid, so the period gives a focus correction even when nothing else can be measured.

## Goals

1. Improve background removal so the normal pipeline succeeds on more poor-seeing frames, without degrading
   frames that already succeed.
2. When the normal pipeline fails but the grid period is measurable, produce a focus-only correction that wfssrv
   can apply live to M2.
3. Implement both in shared code so every WFS (F/9, F/5, Binospec, MMIRS, FLWO) can use them; tune and validate on
   MMIRS first.
4. Fix the slicing typo in `get_apertures`.

## Non-goals

- Using the grid's anisotropy (x/y scale difference, shear) to correct astigmatism. It is logged only.
- Any M1, coma or recenter correction from the fallback.
- Changes to vignetting analysis (see `~/MMT/mmirs_vignetting`).

## Design

### 1. Background model

New module `mmtwfs/background.py` with two functions:

- `pupil_background(data, pupil_mask, box=64, order=3)`
  - Uses only pixels outside the pupil footprint and inside the central obscuration (from the WFS pupil mask
    already used for centering, dilated by a few pixels so spot wings are excluded).
  - Estimates the large-scale halo with a coarse `Background2D` (box about 64 px), then fills in across the pupil
    with a sigma-clipped 2D polynomial (order 2-3) fit to the unmasked background.
  - Returns the background image; the caller subtracts it.
- `pedestal(data, pitch, smooth=None)`
  - Grey opening (`scipy.ndimage.grey_opening`) with a square footprint of about one lenslet pitch in pixels,
    followed by Gaussian smoothing (sigma about pitch/2). This estimates the diffuse light floor between spots.
  - Returns the pedestal image; the caller subtracts it inside the pupil.

`process_image()` in `WFS` and the subclasses that override it (`NewF9`, `F5` (inherited by `Binospec`), `MMIRS`)
keep their trimming and cosmic-ray cleaning, then dispatch on configuration:

| Config key  | Values                                  | Meaning                                         |
|-------------|-----------------------------------------|-------------------------------------------------|
| `bkg_method` | `"background2d"` (current) or `"pupil"` | which background model to use                   |
| `bkg_box`    | int, px                                 | box size for the coarse background              |
| `bkg_pedestal` | bool                                  | subtract the inter-spot pedestal after the halo |

The defaults keep `"background2d"` for every WFS except where validation shows no regression. The plan is to
switch MMIRS first. The current per-class `Background2D` parameters move into config so the old path is preserved
exactly. New keys get their defaults in the base `WFS` class (falling back when a config block lacks them), so the
frozen legacy F/9 config block is not edited.

`process_image()` needs the pupil mask and the lenslet pitch in pixels. The pupil mask comes from
`self.pupil_mask(hdr=hdr)` (already header-aware for MMIRS) and the pitch from the mode's reference
(`xspacing`, `yspacing`). The mode is resolved before background subtraction so the right reference is used.

### 2. Grid period measurement

New function `measure_grid_period(data, center, radius, ref_spacing, search=0.2, pad=4)` in `mmtwfs/wfs.py`:

1. Crop a square around the pupil center with half-width `radius` (pixels).
2. Apply an annular Tukey-tapered window (outer radius `radius`, inner radius from `pup_inner`) to suppress
   the pupil edge and central obscuration.
3. Zero-pad by `pad` and take the 2D power spectrum.
4. Search an annulus of `+/- search` fractional width around the reference fundamental frequency `1/ref_spacing`
   for the strongest peak, then the strongest peak at least 60 degrees from it (and from its conjugate). This
   handles both square and hexagonal lenslet arrays.
5. Refine each peak to sub-pixel accuracy with a 3x3 log-paraboloid (Gaussian) fit. Keep the fit covariance,
   with the noise level estimated from the robust scatter of the power in the search annulus.
6. Build the 2x2 grid frequency matrix and derive:
   - `scale`: mean scale relative to the reference (reference spacing divided by measured spacing, averaged
     over the two vectors);
   - `scale_err`: uncertainty on `scale`,
     `sqrt((period_err_factor * sigma_fit)**2 + period_err_floor**2)`, where `sigma_fit` is propagated from the
     peak-fit covariances. `period_err_factor` (default 1.0) and `period_err_floor` (default 0.0) are config keys
     calibrated in validation step (a); the floor accounts for aberrations other than defocus (coma,
     astigmatism) that the pure-scale model ignores;
   - `xscale`, `yscale`, `rotation` (diagnostics);
   - `snr`: peak power divided by the robust (median/MAD) power in the search annulus.
7. Return a dict, or `None` if either peak's SNR is below `period_snr_thresh` (config, default about 10; tuned
   during validation).

### 3. Conversion to focus

To keep reference aberrations, rotation and sign conventions identical to the full analysis, the scale is not
converted with a hand-coded formula in production code. Instead:

- At the reference aperture positions (`ref.masked_apertures` after `adjust_center`/`apply_pupil`), synthesize
  the slope field of a pure scale change: `dx = (scale - 1) * x`, `dy = (scale - 1) * y` in pixels, relative to
  the pupil center.
- Fit it with the same machinery as `fit_wavefront` (`slope_diff` with `-tiltfactor` scaling and the reference
  Zernike initialisation), but with only Z02, Z03 and Z04 free. Tilts should come out as zero and serve as a
  sanity check.
- Apply the same rotation and reference subtraction that `fit_wavefront` applies, producing a ZernikeVector
  with only Z04 populated (plus reference terms as they normally are).
- Feed it to the unchanged `calculate_focus()`.

Uncertainty propagation:

- Fitting noiseless synthetic slopes gives a Z04 `stderr` of about zero (lmfit scales the covariance by the
  reduced chi-squared), or `None`. Because `calculate_focus()` scales the correction by
  `1 - frac_error("Z04")`, that would apply a noisy fallback measurement at full gain.
- Z04 is linear in `scale`, so compute `k = dZ04/dscale` by running the synthetic-slope fit for a unit scale
  perturbation. This stays well defined when `scale` is near 1.
- Before rotation and reference subtraction, set `errorbars["Z04"] = |k| * scale_err` on the raw fit vector,
  replacing the fit's own Z04 `stderr`. The existing ZernikeVector machinery then carries it to
  `calculate_focus()`: `rotate()` passes error bars through, subtracting the reference adds them in
  quadrature, and `denormalize()` rescales them.
- As a result, `calculate_focus()` automatically down-weights poorly measured corrections and returns zero when
  `scale_err` is comparable to the measured defocus.

Analytic cross-check, used only in tests: the fringe Z4 coefficient is approximately
`tiltfactor * (scale - 1) * R / 4`, where `R = pup_size / 2`. For MMIRS (`tiltfactor` about 3207 nm/px,
R = 172.5 px) this is about `1.4e5 * (scale - 1)` nm, so a scale error of 1e-3 is about 140 nm of Z4, roughly
4 um of M2 focus.

### 4. Result contract and wfssrv integration

mmtwfs:

- In `WFS.measure_slopes`, when `get_slopes` raises `WFSAnalysisFailed`, try the fallback. It uses the pupil
  center from `center_pupil` if that succeeded, otherwise the reference pupil location.
- If the fallback succeeds, return the existing failure-shaped dict (`slopes=None`, `figures`, `mode`) plus:
  - `focus_only = True`
  - `method = "periodicity"`
  - `grid`: the dict from `measure_grid_period`
  - `zernike`: the Z04-only ZernikeVector
  - `pending_focus`: `calculate_focus()` result, after applying `m2_gain_periodicity` and clipping to
    `+/- periodicity_focus_max`
  - `figures["periodicity"]`: power spectrum with the search annulus and the detected peaks (when plotting)
- If the fallback also fails, the return value is exactly as today. Existing callers that check
  `results["slopes"] is None` are unaffected.
- New config keys (base `WFS` defaults, overridable per WFS):
  - `periodicity_fallback`: bool, default True
  - `period_snr_thresh`: float
  - `m2_gain_periodicity`: default 0.5. This is applied on top of the uncertainty-based down-weighting in
    `calculate_focus()`, so it partly double-counts. It stays at 0.5 for the first runs and is revisited with
    real data; it may end up at 1.0 or be removed.
  - `periodicity_focus_max`: default 300 um
  - `period_err_factor`, `period_err_floor`: uncertainty calibration (section 2)
- `reanalyze` records `method` and the fallback focus in its output so archived data can be compared.

wfssrv (separate PR, after the mmtwfs PR merges):

- In the `results["slopes"] is None` branch, if `results.get("focus_only")`:
  - set `pending_focus` from `results["pending_focus"]` and `has_pending_focus = True`;
  - leave `has_pending_m1`, `has_pending_coma` and `has_pending_recenter` False;
  - log a warning naming the degraded mode and the measured scale and SNR;
  - save the `.zernike` file with a comment recording `method = periodicity`;
  - show the periodicity figure in the slopes panel.

### 5. `get_apertures` slicing fix

`mmtwfs/wfs.py` `get_apertures` computes the background statistics from
`data[ycen - 50:ycen + 50, xcen - 50:ycen + 50]`. The column slice should end at `xcen + 50`. Fix it, and add a
regression test with an off-diagonal center where the two slices differ. Because this changes the noise estimate
used for spot detection and S/N, its effect is measured separately in the validation (step b below) before the
background change is layered on top.

## Testing

Unit tests in `mmtwfs/tests/`:

- `measure_grid_period` on a synthetic SH image (square and hexagonal grids) at known scale, with Gaussian blur
  up to about the pitch/2, Poisson noise and an added halo: recovers `scale` to within 2e-4 on sharp images and
  1e-3 on blurred ones; returns `None` on pure noise.
- The fit-based Z4 from section 3 matches the analytic formula to within 2%.
- On synthetic images with many noise realisations, the scatter of `scale` matches `scale_err` to within 30%
  (with factor 1, floor 0). The Z04 error bar survives rotation, reference subtraction and `denormalize()`, and
  `calculate_focus()` returns zero when `scale_err` is larger than `|scale - 1|`.
- `pupil_background` and `pedestal` on synthetic data: spot centroids shift by less than 0.05 px; the halo is
  removed to within the noise.
- `measure_slopes` on a test frame where `get_slopes` fails returns `focus_only=True` with a finite
  `pending_focus`. Use an existing failing test image if there is one; otherwise blur an existing good test
  image.
- The `get_apertures` regression test from section 5.

Offline validation on the October 2026 MMIRS run (scripts kept in `~/MMT/mmirs_vignetting`, not shipped):

a. On the frames that already succeed (about 990), compare periodicity-derived focus with full-fit focus.
   Target: scatter of 10 um or less around the median difference (the median is reported, not corrected). This sets `period_snr_thresh`. It also calibrates the uncertainty:
   choose `period_err_factor` and `period_err_floor` so that the pulls
   `(fallback focus - full-fit focus - median difference) / sigma` have unit width across the SNR range, with sigma combining the
   fallback uncertainty and the full fit's Z04 error bar in quadrature.
b. Compare old and new background, and the slicing fix alone, on success rate and on Zernike agreement for
   frames that succeed both ways. The new background must not change Zernikes beyond the existing fit scatter.
c. Count how many of the 1707 failures now give a full result, a focus-only result, or neither, broken down by
   failure category (vignetted, faint, fuzzy).

## Rollout

1. mmtwfs PR on branch `poor-seeing-analysis`: background module, period measurement, fallback, slicing fix,
   tests, config defaults (MMIRS `bkg_method = "pupil"` only if validation passes).
2. wfssrv PR consuming the new result keys.
3. After an observing run with the fallback live, review the logs and decide whether to switch other WFSs to
   the new background.

## Amendments (from prototyping during planning, 2026-10-06)

Throwaway prototypes on synthetic grids and on blurred copies of `mmirs_wfs_0150.fits` changed these details.
Where they conflict with the sections above, these take precedence.

- **Module layout.** The period estimator lives in a new `mmtwfs/period.py` (`measure_grid_period`, `grid_scale`,
  `plot_periodicity`) rather than in `wfs.py`. Diagnostics are per-vector `scales` and a `rotation`, not
  `xscale`/`yscale`, since hexagonal grids have no x/y vectors.
- **High-pass before the FFT.** `measure_grid_period` subtracts a Gaussian-smoothed copy (sigma = one pitch) before
  windowing. Without it, a halo 30x brighter than the spot peaks biased the scale by 1e-2. With it the bias is
  2e-3 or less.
- **Reference grid measured the same way.** The scale is the ratio of reference to measured grid frequency, both
  from `measure_grid_period`, so window effects cancel. `SH_Reference` keeps its image-frame pupil center
  (`img_xcen`, `img_ycen`) for this.
- **Pupil background.** It uses box-median binning (`bkg_box` default 16 px) plus a sigma-clipped polynomial of
  order 4 (`bkg_order`). Order 4 interpolated broad halos (sigma 250 to 400 px) to within the noise; order 3 left
  residuals 4 to 10 times larger.
- **`period_snr_thresh` default is 20.** Pure noise gives peak SNR of about 7; real MMIRS frames give about 1000.
- **The propagated `scale_err` is conservative** by about 3x, because zero-padded bins are correlated. The test
  pins the empirical-scatter / error ratio to between 0.2 and 1.0. `period_err_factor` (default 1.0) is calibrated
  on real data.
- **No scale offset.** The FFT focus differs from the full fit by a few um (about 6 um on the test frame), because
  the full fit separates spherical aberration from focus and the mean grid scale doesn't. That difference is not
  corrected. Spherical and focus are strongly coupled and are corrected together by focus moves and M1 bending
  anyway, and in seeing bad enough to need the fallback a few um of focus doesn't matter. Validation step (a)
  reports the median difference as a diagnostic and calibrates the uncertainty from the scatter around it. Blur also
  moves both estimates by about 1e-3 in scale through edge-spot centroid shifts. A harmonic-ratio blur correction
  was tried and rejected because it over-corrected synthetic data.
- **Pedestal test.** It checks for a flat residual (std < 0.1) rather than centroid shifts directly. A flat
  residual cannot move centroids.
- **Background helper.** Each `process_image` override keeps its own `Background2D` parameters, passed to a shared
  `subtract_background2d` helper. They are not moved into config.
- **wfssrv output.** wfssrv saves fallback wavefronts as `<file>.periodicity.zernike` instead of adding a comment to
  `.zernike`. `reanalyze` does the same and adds a trailing `method` column. Cached `.output` lines from older runs
  are upgraded with `method = full`.
- **Stale pending corrections.** wfssrv clears pending M1, coma and recenter flags when it applies a focus-only
  result, so nothing pending from an earlier image is applied with it.

## Validation outcome (October 2026 MMIRS run, 2697 frames)

- The `get_apertures` slicing fix alone lost 31 full fits: the +/-50 px noise box is larger than the MMIRS central
  obscuration (40 px radius), so on blurry frames the inner ring of spots inflated the noise estimate. The box
  half-width is now `pup_inner / sqrt(2)`, so it fits inside the obscuration. Result: 1015 full fits vs 990 on
  `main`, none lost, identical Zernikes on the common frames.
- The pupil background reached the same 1015 but shifted Z11 by about -32 nm, so MMIRS stays on `background2d`.
- Fallback vs full fit on 963 frames with both: focus scatter 3.8 um, median offset 0. `period_err_factor` = 1.48
  for MMIRS, no floor.
- Fallback-only frames repeat to 6 um (SNR > 1000) and 10 um (300-1000) but about 28 um below SNR 300, where the
  propagated errors are 1.5-2.4x too small. `period_snr_thresh` default is 250, which keeps the F/9 frosted-donut
  frame (SNR 262).
- 1672 of the 1707 frames that failed on `main` now get a focus-only correction.

## Validation on F/9, F/5 and Binospec (214, 370 and 2778 frames, 2022-2026)

- Noise-box fix: no full fits lost on any WFS; F/9 +5, Binospec +24, F/5 unchanged. Zernikes unchanged on common
  frames (MAD 0).
- Fallback Z04 vs full-fit Z04 slope: F/9 0.997 (hex grid, -12000 to +4300 nm), F/5 0.983, Binospec 0.84 (small
  focus range). Scatter about 215, 100 and 95 nm.
- The fallback underestimates extreme defocus (F/9 scale 1.08: -17200 vs -28800 nm), well beyond the 300 um clip.
- In-focus F/9 blue frames sit at grid scale ~0.93 because of the reference's ~8300 nm Z04. A clip centered on
  the scale-1 reference offset was tried and reverted: it cut real -100 to -320 um corrections to -61 um. The clip
  stays at +/-300 um around zero correction.
- Newly recovered as focus-only: F/9 8, F/5 40, Binospec 6 frames.
