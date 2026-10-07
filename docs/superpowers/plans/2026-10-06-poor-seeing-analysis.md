# Poor-Seeing WFS Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover more poor-seeing WFS frames by improving background removal, and give a live focus-only correction
from the spot-grid period when spots are too blurred to centroid.

**Architecture:** Two new small modules hold the numerics:
- `mmtwfs/background.py`: pupil-aware halo model plus inter-spot pedestal.
- `mmtwfs/period.py`: FFT grid-period estimator.

`WFS` in `mmtwfs/wfs.py` gains config defaults and four methods: `prepare_reference`, `find_pupil_center`,
`subtract_pupil_background` and `reference_grid`, plus `focus_from_scale` and `periodicity_focus`. `measure_slopes`
calls the fallback when `get_slopes` raises `WFSAnalysisFailed` and returns the existing failure dict extended with
`focus_only` keys. The `reanalyze` script records the method, and a separate wfssrv PR applies focus-only results
to M2.

**Tech Stack:** Python 3.13+, numpy, scipy.ndimage, astropy.modeling, lmfit, photutils, pytest

**Spec:** `docs/superpowers/specs/2026-10-06-poor-seeing-analysis-design.md`. The amendments in its final section were
made while writing this plan and take precedence over the earlier sections.

## Global Constraints

- Run Python and tests with `/Users/tim/conda/envs/mmtwfs/bin/python` (`-m pytest ...`; for tox use `-m tox -e py314`).
  The base-env `tox` is broken.
- Flake8 max line length 127 (`/Users/tim/conda/envs/mmtwfs/bin/python -m flake8 mmtwfs --max-line-length=127`).
- Never edit the legacy `"f9"` block in `mmtwfs/config.py`. New config keys get defaults as `WFS` class attributes.
- With default config, every existing analysis must behave exactly as before: `bkg_method` defaults to
  `"background2d"` everywhere. The fallback only adds keys to results that already have `slopes = None`.
- The fallback never produces M1, coma or recenter corrections.
- Branch: `poor-seeing-analysis` in `~/MMT/mmtwfs` (already created; holds the spec). The wfssrv work goes on its own
  branch from `master` in `~/MMT/wfssrv`.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The fallback must never crash `measure_slopes`.** Any exception inside the fallback (odd header, pupil centring
   failure, degenerate spectrum) must be logged, and the plain failure result returned. Pinned by
   `test_periodicity_fallback_swallows_errors` (Task 6).
2. **Pure noise or a blank frame must not produce a focus correction.** Pinned by
   `test_measure_grid_period_noise_returns_none` (Task 3) and `test_periodicity_no_grid` (Task 6).
3. **Huge or poorly measured defocus must not drive M2 far.**
   - When the error bar is at least as large as Z04, the correction is zero.
   - Large scales are clipped to `periodicity_focus_max`.
   - Pinned by `test_focus_from_scale_large_error_is_zero` and `test_focus_from_scale_clipped` (Task 5).
4. **Cached `.output` lines in archives written by the old `reanalyze` have one fewer column.** Mixing them with
   new lines must not misalign the CSV. Pinned by `test_upgrade_cached_line` (Task 7).
5. **A pupil centred near the image edge** (MMIRS pupils move with probe position) must not make the cutout index out
   of bounds. Pinned by `test_measure_grid_period_off_edge` (Task 3).

---

## File Structure

| File | Responsibility |
|---|---|
| `mmtwfs/background.py` (new) | `pupil_footprint`, `pupil_background`, `pedestal`: background models that don't need spot detection |
| `mmtwfs/period.py` (new) | `annular_window`, `measure_grid_period`, `grid_scale`, `plot_periodicity`: grid period from the power spectrum |
| `mmtwfs/wfs.py` (modify) | slicing fix; `subtract_background2d` helper; `SH_Reference` image center; `WFS` config defaults and new methods; fallback in `measure_slopes`; `process_image` overrides call the helper |
| `mmtwfs/scripts/reanalyze.py` (modify) | `method` column, focus-only rows, `upgrade_cached_line` |
| `mmtwfs/tests/synthetic.py` (new) | synthetic SH image and halo generators shared by tests |
| `mmtwfs/tests/test_background.py`, `test_period.py`, `test_reanalyze.py` (new) | unit tests |
| `mmtwfs/tests/test_wfs.py` (modify) | slicing regression, integration tests for background and fallback |
| `mmtwfs/config.py` (modify, Task 9 only) | calibrated MMIRS values |
| `~/MMT/mmirs_vignetting/seeing/validate.py`, `calibrate.py` (new, not in repo) | October validation and calibration |
| `~/MMT/wfssrv/wfssrv/wfssrv.py` (modify, Task 10) | apply focus-only results |

---

### Task 1: Fix `get_apertures` background-region slicing

**Files:**
- Modify: `mmtwfs/wfs.py:366` (inside `get_apertures`)
- Test: `mmtwfs/tests/test_wfs.py`

**Interfaces:**
- Consumes: nothing new
- Produces: nothing new (bug fix)

- [ ] **Step 1: Write the failing test.** Add `import pytest` to the imports at the top of `mmtwfs/tests/test_wfs.py`,
  add `get_apertures` to the `from mmtwfs.wfs import ...` line, and append:

```python
def test_get_apertures_background_region():
    # quiet region left of x=150, very noisy to the right. with cen=(100, 400) the 100x100 stats box is
    # entirely in the quiet region; the old typo (xcen - 50:ycen + 50) made it run to x=450.
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 1.0, (512, 512))
    data[:, 150:] = rng.normal(0.0, 100.0, (512, 362))
    captured = {}

    def fake_wfsfind(data, fwhm=7.0, threshold=5.0, plot=True, ap_radius=5.0, std=None):
        captured["std"] = std
        raise RuntimeError("stop after background stats")

    with patch("mmtwfs.wfs.wfsfind", side_effect=fake_wfsfind):
        with pytest.raises(RuntimeError):
            get_apertures(data, 20.0, cen=(100, 400))
    assert captured["std"] < 2.0
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py::test_get_apertures_background_region -v`
Expected: FAIL on `assert captured["std"] < 2.0` (std is about 90).

- [ ] **Step 3: Fix the slice.** In `get_apertures`, change

```python
            data[ycen - 50:ycen + 50, xcen - 50:ycen + 50], sigma=3.0, maxiters=None
```

to

```python
            data[ycen - 50:ycen + 50, xcen - 50:xcen + 50], sigma=3.0, maxiters=None
```

- [ ] **Step 4: Run the new test and the full WFS suite**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py -v`
Expected: all PASS. If an existing analysis test's Zernike value moves out of range, stop and report the
old and new values. The noise estimate changed, and the user decides whether to recalibrate the test.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/wfs.py mmtwfs/tests/test_wfs.py
git commit -m "fix get_apertures background box using ycen for the x upper bound

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Background module

**Files:**
- Create: `mmtwfs/background.py`
- Create: `mmtwfs/tests/synthetic.py`
- Test: `mmtwfs/tests/test_background.py`

**Interfaces:**
- Produces:
  - `pupil_footprint(shape, center, outer, inner=0.0, margin=0.0) -> np.ndarray[bool]`: True on (padded) pupil pixels
  - `pupil_background(data, footprint, box=16, order=4) -> np.ndarray`: smooth background image, same shape as data;
    raises `ValueError` if too few background samples
  - `pedestal(data, pitch, presmooth=1.5) -> np.ndarray`: inter-spot floor image
  - test helpers `make_sh_image(...) -> (image, positions)` and `gaussian_halo(shape, center, width, amplitude, tilt=0.0)`

- [ ] **Step 1: Write the synthetic-image helper** `mmtwfs/tests/synthetic.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Synthetic Shack-Hartmann images for testing background and grid-period code.
"""

import numpy as np


def make_sh_image(
    shape=(512, 512),
    center=(256.0, 256.0),
    radius=172.5,
    inner=40.0,
    spacing=22.7,
    angle=0.0,
    hexgrid=False,
    sigma=2.0,
    flux=2000.0,
):
    """
    Render gaussian spots on a square or hexagonal grid inside an annular pupil.

    Returns
    -------
    image : 2D np.ndarray
    positions : (N, 2) np.ndarray of (x, y) spot centers
    """
    if hexgrid:
        basis = spacing * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    else:
        basis = spacing * np.array([[1.0, 0.0], [0.0, 1.0]])
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    n = int(2 * radius / spacing) + 3
    ij = np.array([(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)], dtype=float)
    pts = ij @ basis @ rot.T
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(r < radius) & (r > inner)] + np.asarray(center)

    image = np.zeros(shape)
    h = int(4 * sigma) + 1
    norm = flux / (2.0 * np.pi * sigma**2)
    for x, y in pts:
        x0, y0 = int(round(x)), int(round(y))
        yy, xx = np.mgrid[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]
        image[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1] += norm * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        )
    return image, pts


def gaussian_halo(shape, center, width, amplitude, tilt=0.0):
    """
    Broad gaussian glow plus a linear gradient in x, like scattered light around a WFS pupil.
    """
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    r2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return amplitude * np.exp(-r2 / (2.0 * width**2)) + tilt * xx
```

- [ ] **Step 2: Write the failing tests** `mmtwfs/tests/test_background.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from scipy import ndimage

from mmtwfs.background import pupil_footprint, pupil_background, pedestal
from mmtwfs.tests.synthetic import make_sh_image, gaussian_halo

SHAPE = (512, 512)
CENTER = (256.0, 256.0)
RADIUS = 172.5
INNER = 40.0
PITCH = 22.7


def test_pupil_footprint():
    fp = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER, margin=5.0)
    assert fp[256, 356]
    assert not fp[256, 256]  # central obscuration
    assert fp[256, 256 + 176]  # within the outer margin
    assert not fp[256, 256 + 180]  # beyond pupil + margin
    assert fp[256, 256 + 36]  # inner edge pulled in by the margin to r = 35
    assert not fp[256, 256 + 34]


@pytest.mark.parametrize("width, tol", [(400.0, 1.0), (250.0, 3.0)])
def test_pupil_background_halo(width, tol):
    rng = np.random.default_rng(3)
    spots, _ = make_sh_image(sigma=4.0)
    halo = gaussian_halo(SHAPE, CENTER, width, 300.0, tilt=0.2)
    img = spots + halo + rng.normal(0.0, 5.0, SHAPE)
    fp = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER, margin=PITCH)
    bkg = pupil_background(img, fp, box=16, order=4)
    inside = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER)
    resid = (bkg - halo)[inside]
    assert np.std(resid) < tol
    assert abs(np.mean(resid)) < 3.0 * tol


def test_pupil_background_too_few_samples():
    with pytest.raises(ValueError):
        pupil_background(np.zeros(SHAPE), np.ones(SHAPE, dtype=bool))


@pytest.mark.parametrize("sigma, max_offset", [(2.0, 0.5), (4.0, 0.5), (6.0, 2.0)])
def test_pedestal_flat_residual(sigma, max_offset):
    # a smooth floor under the spots must come out, leaving a flat residual so centroids can't shift.
    # for blurry spots the opening also takes the overlapping wings, hence the larger allowed offset.
    spots, _ = make_sh_image(sigma=sigma)
    inside = pupil_footprint(SHAPE, CENTER, RADIUS, inner=INNER)
    floor = 50.0 * ndimage.gaussian_filter(inside.astype(float), 10.0)
    cleaned = spots + floor - pedestal(spots + floor, PITCH)
    deep = pupil_footprint(SHAPE, CENTER, RADIUS - 2 * PITCH, inner=INNER + 2 * PITCH)
    resid = (cleaned - spots)[deep]
    assert abs(np.median(resid)) < max_offset
    assert np.std(resid) < 0.1
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_background.py -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'mmtwfs.background'`

- [ ] **Step 4: Implement** `mmtwfs/background.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Background models for Shack-Hartmann WFS images that don't depend on detecting spots. In poor seeing the spots
blend together, spot masks cover most of the pupil, and small-box background estimators end up absorbing spot
light. These models instead fit the scattered-light halo from pixels outside the pupil and remove the diffuse
floor between spots with a morphological opening.
"""

import warnings

import numpy as np
from scipy import ndimage

from astropy.modeling.models import Polynomial2D
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from astropy.stats import sigma_clip

__all__ = ["pupil_footprint", "pupil_background", "pedestal"]


def pupil_footprint(shape, center, outer, inner=0.0, margin=0.0):
    """
    Boolean image that is True where the pupil (padded by margin on both edges) illuminates the detector.

    Parameters
    ----------
    shape : tuple
        Image shape (ny, nx)
    center : list-like
        Pupil center (x, y) in pixels
    outer, inner : float
        Outer radius and central obscuration radius in pixels
    margin : float
        Padding added outside the outer edge and inside the inner edge
    """
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    r = np.hypot(xx - center[0], yy - center[1])
    return (r < outer + margin) & (r > max(inner - margin, 0.0))


def pupil_background(data, footprint, box=16, order=4):
    """
    Model large-scale background (scattered light halo, gradients) using only pixels outside the footprint,
    i.e. outside the pupil and inside the central obscuration, and interpolate it across the pupil.

    The image is binned into box x box blocks; blocks that are at least half background contribute their
    median to a sigma-clipped 2D polynomial fit.

    Parameters
    ----------
    data : 2D np.ndarray
    footprint : 2D bool np.ndarray
        True on pixels to exclude (the padded pupil), e.g. from `pupil_footprint`
    box : int
        Block size in pixels
    order : int
        Polynomial order

    Returns
    -------
    background : 2D np.ndarray, same shape as data
    """
    ny, nx = data.shape
    nby, nbx = ny // box, nx // box
    blocks = np.where(footprint, np.nan, data)[: nby * box, : nbx * box].reshape(nby, box, nbx, box)
    with warnings.catch_warnings():
        # all-NaN blocks inside the pupil are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(blocks, axis=(1, 3))
    frac = np.isfinite(blocks).mean(axis=(1, 3))
    yb, xb = (np.mgrid[:nby, :nbx] + 0.5) * box - 0.5
    good = (frac > 0.5) & np.isfinite(med)

    nterms = (order + 1) * (order + 2) // 2
    if good.sum() < 2 * nterms:
        raise ValueError(
            f"Only {good.sum()} background blocks available for an order {order} fit ({2 * nterms} needed)."
        )

    # fit in normalized coordinates to keep the polynomial well conditioned
    def norm(x, n):
        return 2.0 * x / (n - 1) - 1.0

    fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, niter=3, sigma=3.0)
    model, _ = fitter(Polynomial2D(order), norm(xb[good], nx), norm(yb[good], ny), med[good])

    yy, xx = np.mgrid[:ny, :nx]
    return model(norm(xx, nx), norm(yy, ny))


def pedestal(data, pitch, presmooth=1.5):
    """
    Estimate the diffuse light floor between WFS spots with a grey opening whose footprint is about one lenslet
    pitch, so every window reaches the gap between spots. A light pre-smoothing keeps the opening from locking onto
    negative noise excursions, and the result is smoothed by half a pitch.

    Parameters
    ----------
    data : 2D np.ndarray
    pitch : float
        Lenslet pitch (spot spacing) in pixels
    presmooth : float
        Sigma in pixels of gaussian smoothing applied before the opening

    Returns
    -------
    floor : 2D np.ndarray, same shape as data
    """
    size = int(np.ceil(pitch)) | 1  # odd so the footprint is centered
    floor = ndimage.grey_opening(ndimage.gaussian_filter(data, presmooth), size=(size, size))
    return ndimage.gaussian_filter(floor, pitch / 2.0)
```

- [ ] **Step 5: Run tests and confirm they pass**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_background.py -v`
Expected: 7 PASS.

- [ ] **Step 6: Commit**

```bash
git add mmtwfs/background.py mmtwfs/tests/synthetic.py mmtwfs/tests/test_background.py
git commit -m "add pupil-aware background and inter-spot pedestal models

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Grid-period estimator

**Files:**
- Create: `mmtwfs/period.py`
- Test: `mmtwfs/tests/test_period.py`

**Interfaces:**
- Consumes: `make_sh_image`, `gaussian_halo` from `mmtwfs/tests/synthetic.py`
- Produces:
  - `measure_grid_period(data, center, radius, ref_spacing, inner=0.0, search=0.2, pad=4, snr_thresh=20.0, min_sep=60.0) -> dict | None`
    with keys:
    - `freqs`: (2, 2) array of (fx, fy) in cycles/px
    - `freq_errs`: (2,) radial 1-sigma
    - `snr`: (2,)
    - `spacing`: (2,) px
    - `angle`: (2,) deg, in [0, 180)
    - `power`: 2D array, fftshifted
    - `freq_axis`: 1D array
    - `annulus`: (fmin, fmax)
  - `grid_scale(meas, ref) -> dict` with keys:
    - `scale`: measured spacing divided by reference spacing, mean of the two vectors
    - `scale_err_fit`
    - `scales`: (2,)
    - `rotation`: deg
  - `plot_periodicity(meas) -> matplotlib Figure`

- [ ] **Step 1: Write the failing tests** `mmtwfs/tests/test_period.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
import matplotlib.pyplot as plt

from mmtwfs.period import measure_grid_period, grid_scale, plot_periodicity
from mmtwfs.tests.synthetic import make_sh_image, gaussian_halo

CENTER = (256.0, 256.0)
RADIUS = 172.5
INNER = 40.0
PITCH = 22.7


def _scale(img, hexgrid=False, center=CENTER):
    ref, _ = make_sh_image(sigma=2.0, hexgrid=hexgrid)
    r = measure_grid_period(ref, CENTER, RADIUS, PITCH, inner=INNER, snr_thresh=0.0)
    m = measure_grid_period(img, center, RADIUS, PITCH, inner=INNER)
    return grid_scale(m, r)


@pytest.mark.parametrize("hexgrid", [False, True])
@pytest.mark.parametrize("truth", [0.97, 1.0, 1.03])
@pytest.mark.parametrize("sigma, tol", [(2.0, 2e-4), (6.0, 1e-3)])
def test_measure_grid_period_scale(hexgrid, truth, sigma, tol):
    rng = np.random.default_rng(1)
    img, _ = make_sh_image(spacing=PITCH * truth, sigma=sigma, hexgrid=hexgrid)
    img = img + rng.normal(0.0, 5.0, img.shape)
    g = _scale(img, hexgrid=hexgrid)
    assert abs(g["scale"] - truth) < tol
    assert abs(g["rotation"]) < 0.5


def test_measure_grid_period_rotated_hex():
    rng = np.random.default_rng(4)
    img, _ = make_sh_image(hexgrid=True, angle=10.0, sigma=3.0)
    ref, _ = make_sh_image(hexgrid=True, angle=10.0, sigma=2.0)
    r = measure_grid_period(ref, CENTER, RADIUS, PITCH, inner=INNER, snr_thresh=0.0)
    m = measure_grid_period(img + rng.normal(0.0, 5.0, img.shape), CENTER, RADIUS, PITCH, inner=INNER)
    assert abs(grid_scale(m, r)["scale"] - 1.0) < 5e-4


def test_measure_grid_period_halo():
    # a halo 30x brighter than the spot peaks; the built-in high-pass keeps it out of the search annulus
    rng = np.random.default_rng(5)
    img, _ = make_sh_image(spacing=PITCH * 1.03, sigma=6.0)
    img = img + gaussian_halo(img.shape, CENTER, 150.0, 300.0) + rng.normal(0.0, 5.0, img.shape)
    g = _scale(img)
    assert abs(g["scale"] - 1.03) < 3e-3


def test_measure_grid_period_noise_returns_none():
    rng = np.random.default_rng(6)
    noise = rng.normal(0.0, 5.0, (512, 512))
    assert measure_grid_period(noise, CENTER, RADIUS, PITCH, inner=INNER) is None


def test_measure_grid_period_off_edge():
    # pupil hanging 60 px off the left edge must not raise, and still find the grid
    rng = np.random.default_rng(7)
    center = (RADIUS - 60.0, 256.0)
    img, _ = make_sh_image(shape=(512, 640), center=(RADIUS + 68.0, 256.0), sigma=3.0)
    img = img[:, 128:] + rng.normal(0.0, 5.0, (512, 512))
    g = _scale(img, center=center)
    assert abs(g["scale"] - 1.0) < 2e-3


def test_scale_err_is_conservative():
    # the propagated error ignores correlations between zero-padded bins, so it overestimates the true
    # scatter. pin it between 1x and 5x; period_err_factor calibrates it on real data.
    rng = np.random.default_rng(8)
    ref, _ = make_sh_image(sigma=2.0)
    r = measure_grid_period(ref, CENTER, RADIUS, PITCH, inner=INNER, snr_thresh=0.0)
    img0, _ = make_sh_image(spacing=PITCH * 1.004, sigma=2.0)
    scales, errs = [], []
    for _ in range(40):
        m = measure_grid_period(img0 + rng.normal(0.0, 30.0, img0.shape), CENTER, RADIUS, PITCH, inner=INNER)
        g = grid_scale(m, r)
        scales.append(g["scale"])
        errs.append(g["scale_err_fit"])
    ratio = np.std(scales) / np.median(errs)
    assert 0.2 < ratio < 1.0


def test_plot_periodicity():
    img, _ = make_sh_image(sigma=3.0)
    m = measure_grid_period(img, CENTER, RADIUS, PITCH, inner=INNER)
    fig = plot_periodicity(m)
    assert fig.get_label() == "Grid Periodicity"
    plt.close("all")
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_period.py -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'mmtwfs.period'`

- [ ] **Step 3: Implement** `mmtwfs/period.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Measure the period of a Shack-Hartmann spot grid from the power spectrum of the pupil image. This works when the
spots are too blurred to detect and centroid individually. Defocus rescales the grid, so the scale relative to the
reference grid gives a focus correction.
"""

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt

__all__ = ["annular_window", "measure_grid_period", "grid_scale", "plot_periodicity"]


def annular_window(n, radius, inner=0.0, taper=10.0):
    """
    n x n window that is 1 inside an annulus and rolls off to 0 with raised-cosine edges of width taper.
    """
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.0
    r = np.hypot(xx, yy)
    w = np.ones_like(r)
    t0 = radius - taper
    w[r >= radius] = 0.0
    edge = (r > t0) & (r < radius)
    w[edge] = 0.5 * (1.0 + np.cos(np.pi * (r[edge] - t0) / taper))
    if inner > 0.0:
        w[r <= inner] = 0.0
        edge = (r > inner) & (r < inner + taper)
        w[edge] *= 0.5 * (1.0 - np.cos(np.pi * (r[edge] - inner) / taper))
    return w


def _cutout(data, center, half):
    """
    2*half square cutout centered on center, zero-filled where it hangs off the image.
    """
    n = 2 * half
    x0 = int(round(center[0])) - half
    y0 = int(round(center[1])) - half
    sub = np.zeros((n, n))
    ys = slice(max(y0, 0), min(y0 + n, data.shape[0]))
    xs = slice(max(x0, 0), min(x0 + n, data.shape[1]))
    if ys.start < ys.stop and xs.start < xs.stop:
        sub[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = data[ys, xs]
    return sub


def _refine_peak(power, n, iy, ix, noise):
    """
    Sub-bin peak position from 3-point log-parabola (i.e. gaussian) interpolation along each axis. The power of a
    signal peak plus noise has variance of about 2 * P_signal * P_noise, so ln(P) has sigma sqrt(2 P_noise / P);
    that is propagated through the interpolation formula.

    Returns (fx, fy, radial_sigma) in cycles/pixel, or None if this isn't a local maximum.
    """
    patch = np.maximum(power[iy - 1:iy + 2, ix - 1:ix + 2], np.finfo(float).tiny)
    lnp = np.log(patch)
    slnp = np.sqrt(np.minimum(2.0 * noise / patch, 1.0))
    offsets, variances = [], []
    for (lm, l0, lp), (sm, s0, sp) in (
        ((lnp[1, 0], lnp[1, 1], lnp[1, 2]), (slnp[1, 0], slnp[1, 1], slnp[1, 2])),
        ((lnp[0, 1], lnp[1, 1], lnp[2, 1]), (slnp[0, 1], slnp[1, 1], slnp[2, 1])),
    ):
        d = lm - 2.0 * l0 + lp
        if d >= 0.0:
            return None
        num = lm - lp
        offsets.append(num / (2.0 * d))
        variances.append(
            ((d - num) / (2.0 * d**2) * sm) ** 2
            + ((-d - num) / (2.0 * d**2) * sp) ** 2
            + (num / d**2 * s0) ** 2
        )
    fx = (ix - n // 2 + offsets[0]) / n
    fy = (iy - n // 2 + offsets[1]) / n
    sfx = np.sqrt(variances[0]) / n
    sfy = np.sqrt(variances[1]) / n
    sigma = np.hypot(fx * sfx, fy * sfy) / np.hypot(fx, fy)
    return fx, fy, sigma


def measure_grid_period(
    data, center, radius, ref_spacing, inner=0.0, search=0.2, pad=4, snr_thresh=20.0, min_sep=60.0
):
    """
    Measure the two fundamental grid frequencies of a WFS spot pattern from the power spectrum of the pupil.

    Parameters
    ----------
    data : 2D np.ndarray
        WFS image
    center : list-like
        Pupil center (x, y) in pixels
    radius : float
        Pupil radius in pixels
    ref_spacing : float or list-like
        Expected spot spacing in pixels (mean is used); sets the search annulus
    inner : float
        Central obscuration radius in pixels
    search : float
        Half-width of the search annulus as a fraction of the expected frequency
    pad : int
        Zero-padding factor for the FFT
    snr_thresh : float
        Minimum peak power / noise power for both peaks; below this, return None
    min_sep : float
        Minimum angle in degrees between the two grid vectors (60 handles hexagonal and square grids)

    Returns
    -------
    dict or None
    """
    spacing = float(np.mean(ref_spacing))
    half = int(np.ceil(radius))
    sub = _cutout(np.asarray(data, dtype=float), center, half)

    # high-pass: anything smoother than the grid (halo, pedestal) would otherwise leak through the window
    # into the search annulus and bias the peaks
    sub = sub - ndimage.gaussian_filter(sub, spacing)
    w = annular_window(2 * half, radius, inner=inner, taper=spacing)
    sub = (sub - np.sum(sub * w) / np.sum(w)) * w

    n = pad * 2 * half
    power = np.abs(np.fft.fftshift(np.fft.fft2(sub, s=(n, n)))) ** 2
    f = (np.arange(n) - n // 2) / n
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    f0 = 1.0 / spacing
    fmin, fmax = f0 * (1.0 - search), f0 * (1.0 + search)
    annulus = (fr > fmin) & (fr < fmax)
    upper = (fy > 0) | ((fy == 0) & (fx > 0))  # one of each +/- conjugate pair
    candidates = annulus & upper

    # noise power is exponentially distributed, so median = ln(2) * mean
    noise = np.median(power[annulus]) / np.log(2.0)
    if not noise > 0.0:
        return None

    angle = np.degrees(np.arctan2(fy, fx)) % 180.0
    first = np.unravel_index(np.argmax(np.where(candidates, power, -1.0)), power.shape)
    dang = np.abs(angle - angle[first])
    dang = np.minimum(dang, 180.0 - dang)
    second_ok = candidates & (dang >= min_sep)
    if not second_ok.any():
        return None
    second = np.unravel_index(np.argmax(np.where(second_ok, power, -1.0)), power.shape)

    freqs, ferrs, snrs = [], [], []
    for iy, ix in (first, second):
        snr = power[iy, ix] / noise
        refined = _refine_peak(power, n, iy, ix, noise)
        if snr < snr_thresh or refined is None:
            return None
        freqs.append(refined[:2])
        ferrs.append(refined[2])
        snrs.append(snr)

    freqs = np.array(freqs)
    return {
        "freqs": freqs,
        "freq_errs": np.array(ferrs),
        "snr": np.array(snrs),
        "spacing": 1.0 / np.hypot(freqs[:, 0], freqs[:, 1]),
        "angle": np.degrees(np.arctan2(freqs[:, 1], freqs[:, 0])) % 180.0,
        "power": power,
        "freq_axis": f,
        "annulus": (fmin, fmax),
    }


def grid_scale(meas, ref):
    """
    Compare measured grid frequencies with the reference grid's. Each measured vector is matched to the reference
    vector closest in angle. Scale is measured spacing / reference spacing, the same convention as the grid fit in
    `~mmtwfs.wfs.get_slopes`.

    Returns
    -------
    dict with scale, scale_err_fit, scales (per vector), rotation (deg)
    """
    fm = np.hypot(meas["freqs"][:, 0], meas["freqs"][:, 1])
    fr = np.hypot(ref["freqs"][:, 0], ref["freqs"][:, 1])
    scales, errs, rots = [], [], []
    for i in range(len(fm)):
        d = np.abs(ref["angle"] - meas["angle"][i])
        d = np.minimum(d, 180.0 - d)
        j = int(np.argmin(d))
        s = fr[j] / fm[i]
        scales.append(s)
        errs.append(s * np.hypot(meas["freq_errs"][i] / fm[i], ref["freq_errs"][j] / fr[j]))
        rots.append((meas["angle"][i] - ref["angle"][j] + 90.0) % 180.0 - 90.0)
    scales = np.array(scales)
    errs = np.array(errs)
    return {
        "scale": float(scales.mean()),
        "scale_err_fit": float(np.sqrt(np.sum(errs**2)) / len(errs)),
        "scales": scales,
        "rotation": float(np.mean(rots)),
    }


def plot_periodicity(meas):
    """
    Show the power spectrum around the search annulus with the detected grid peaks.
    """
    f = meas["freq_axis"]
    fmin, fmax = meas["annulus"]
    lim = 1.3 * fmax
    keep = np.abs(f) <= lim
    fig, ax = plt.subplots()
    fig.set_label("Grid Periodicity")
    ax.imshow(
        np.log10(meas["power"][np.ix_(keep, keep)] + np.finfo(float).tiny),
        origin="lower",
        cmap="Greys",
        extent=(f[keep][0], f[keep][-1], f[keep][0], f[keep][-1]),
    )
    t = np.linspace(0, 2 * np.pi, 361)
    for rad in (fmin, fmax):
        ax.plot(rad * np.cos(t), rad * np.sin(t), color="blue", lw=0.8)
    for (fx, fy), snr, sp in zip(meas["freqs"], meas["snr"], meas["spacing"]):
        ax.scatter([fx, -fx], [fy, -fy], facecolors="none", edgecolors="red", s=80)
        ax.annotate(f"{sp:.3f} px (SNR {snr:.0f})", (fx, fy), color="red", fontsize=8)
    ax.set_xlabel("f$_x$ (cycles/pixel)")
    ax.set_ylabel("f$_y$ (cycles/pixel)")
    ax.set_title("Grid period (focus-only fallback)")
    return fig
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_period.py -v`
Expected: all PASS. A prototype of this code measured, at noise sigma 5:
- scale errors of 1e-4 or less for sharp spots and 6e-4 or less for sigma = 6 px;
- 2e-3 or less with the halo;
- an std/err ratio of 0.31 to 0.39.

If a tolerance fails, report the measured value; don't loosen it silently.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/period.py mmtwfs/tests/test_period.py
git commit -m "add FFT grid-period estimator for blurred WFS spot patterns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: WFS config defaults and background integration

**Files:**
- Modify: `mmtwfs/wfs.py`:
  - imports
  - `SH_Reference.__init__` (around line 818)
  - `WFS` class body (line 913)
  - `WFS.process_image` (1163)
  - `WFS.measure_slopes` (1202)
  - `NewF9.process_image` (1665)
  - `F5.process_image` (1701)
  - `MMIRS.process_image` (2204)
- Test: `mmtwfs/tests/test_wfs.py`

**Interfaces:**
- Consumes: `pupil_footprint`, `pupil_background`, `pedestal` (Task 2)
- Produces:
  - module function `subtract_background2d(data, box_size, filter_size, npixels=5, dilate_size=11) -> np.ndarray`
  - `SH_Reference.img_xcen`, `SH_Reference.img_ycen`: pupil center in the reference image's own pixels, never
    changed by `adjust_center`
  - `SH_Reference.grid = None`: cache slot used by Task 5
  - `WFS` class attributes (defaults; config overrides):
    - `bkg_method = "background2d"`
    - `bkg_box = 16`
    - `bkg_order = 4`
    - `pedestal = True`
    - `periodicity_fallback = True`
    - `period_snr_thresh = 20.0`
    - `period_err_factor = 1.0`
    - `period_err_floor = 0.0`
    - `period_scale_offset = 0.0`
    - `m2_gain_periodicity = 0.5`
    - `periodicity_focus_max = 300.0 * u.um`
  - `WFS.prepare_reference(mode, hdr=None) -> SH_Reference`: centers the reference and applies the pupil, as
    `measure_slopes` already does
  - `WFS.find_pupil_center(data, pup_mask) -> (xcen, ycen)`: never raises; falls back to `cor_coords`
  - `WFS.subtract_pupil_background(data, mode, center) -> np.ndarray`

- [ ] **Step 1: Write the failing tests.** Append to `mmtwfs/tests/test_wfs.py`:

```python
def test_wfs_poor_seeing_defaults():
    for s in mmtwfs_config["wfs"]:
        wfs = WFSFactory(wfs=s)
        assert wfs.bkg_method == "background2d"
        assert wfs.periodicity_fallback
        assert wfs.m2_gain_periodicity == 0.5
    plt.close("all")


def test_reference_image_center_is_fixed():
    mmirs = WFSFactory(wfs="mmirs")
    ref = mmirs.modes["mmirs2"]["reference"]
    x0, y0 = ref.img_xcen, ref.img_ycen
    ref.adjust_center(x0 + 10.0, y0 - 5.0)
    assert (ref.img_xcen, ref.img_ycen) == (x0, y0)
    assert ref.xcen == x0 + 10.0


def test_find_pupil_center_falls_back():
    mmirs = WFSFactory(wfs="mmirs")
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    data, hdr = mmirs.process_image(test_file)
    xc, yc = mmirs.find_pupil_center(data, mmirs.pupil_mask(hdr=hdr))
    assert np.hypot(xc - mmirs.cor_coords[0], yc - mmirs.cor_coords[1]) < mmirs.cen_tol
    xc, yc = mmirs.find_pupil_center(np.zeros((10, 10)), mmirs.pupil_mask(hdr=hdr))
    assert (xc, yc) == tuple(mmirs.cor_coords)


def test_mmirs_analysis_pupil_background():
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    mmirs = WFSFactory(wfs="mmirs", config={"bkg_method": "pupil"})
    zresults = _analyze_image(mmirs, test_file)
    testval = int(zresults["zernike"]["Z10"].value)
    # same window as test_mmirs_analysis: the new background must not change a good frame's wavefront
    assert (testval > 388) & (testval < 408)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py -k "defaults or image_center or find_pupil or pupil_background" -v`
Expected: FAIL with `AttributeError` (`bkg_method`, `img_xcen`, `find_pupil_center`).

- [ ] **Step 3: Implement.**

(a) Imports at the top of `mmtwfs/wfs.py`, after `from mmtwfs.photometry import make_spot_mask`:

```python
from mmtwfs.background import pupil_footprint, pupil_background, pedestal
```

(b) Module helper, placed just above `def wfs_norm(`:

```python
def subtract_background2d(data, box_size, filter_size, npixels=5, dilate_size=11):
    """
    Small-box mode-estimator background with WFS spots masked out. This is the original background
    method; it works well when spots are well separated.
    """
    bkg_estimator = ModeEstimatorBackground()
    mask = make_spot_mask(data, nsigma=2, npixels=npixels, dilate_size=dilate_size)
    bkg = Background2D(data, box_size, filter_size=filter_size, bkg_estimator=bkg_estimator, mask=mask)
    return data - bkg.background
```

(c) Replace the background block in each `process_image`. The parameters are copied exactly from the current code:
- `WFS.process_image`: replace the five lines from `# calculate the background and subtract it` through
  `data -= bkg.background` with:

```python
        # calculate the background and subtract it. with bkg_method = "pupil" this is done later in
        # measure_slopes() once the pupil center is known.
        if self.bkg_method == "background2d":
            data = subtract_background2d(data, (10, 10), (5, 5))
```

- `NewF9.process_image`, the same replacement using:

```python
        if self.bkg_method == "background2d":
            data = subtract_background2d(data, (50, 50), (15, 15), npixels=7, dilate_size=13)
```

- `F5.process_image`:

```python
        if self.bkg_method == "background2d":
            data = subtract_background2d(data, (20, 20), (11, 11))
```

- `MMIRS.process_image`:

```python
        if self.bkg_method == "background2d":
            data = subtract_background2d(data, (20, 20), (7, 7))
```

Before replacing, confirm each block's `npixels` and `dilate_size`; the values above match the code as of commit
`2f6cbf4`. Remove the `ModeEstimatorBackground`/`Background2D` imports only if nothing else uses them; `grep` first.

(d) `SH_Reference.__init__`: right after `self.ycen = self.apertures["ycentroid"].mean()` add:

```python
        # pupil center in the reference image's own pixels. adjust_center() moves xcen/ycen to wherever the pupil
        # is on the science frame, but measuring the reference grid needs to know where it is in self.data.
        self.img_xcen = self.xcen
        self.img_ycen = self.ycen
        # grid frequencies measured from self.data; filled in by WFS.reference_grid()
        self.grid = None
```

(e) `WFS` class attributes, inserted between the class docstring and `def __init__`:

```python
    # defaults for poor-seeing handling. these are class attributes so that config blocks (including the frozen
    # legacy f9 one) don't need to define them; any config can override them.
    bkg_method = "background2d"  # or "pupil"
    bkg_box = 16  # block size in pixels for the pupil background fit
    bkg_order = 4  # polynomial order of the pupil background fit
    pedestal = True  # with bkg_method = "pupil", also remove the floor between spots
    periodicity_fallback = True  # focus-only correction from the grid period when spot analysis fails
    period_snr_thresh = 20.0
    period_err_factor = 1.0  # calibration of the propagated grid-scale error
    period_err_floor = 0.0  # systematic grid-scale error added in quadrature
    period_scale_offset = 0.0  # calibrated offset added to the measured grid scale
    m2_gain_periodicity = 0.5  # extra gain on fallback focus corrections
    periodicity_focus_max = 300.0 * u.um
```

(f) New `WFS` methods, placed right after `get_mode` (before `process_image`):

```python
    def prepare_reference(self, mode, hdr=None):
        """
        Center the mode's reference apertures on the expected pupil position and apply the pupil mask.
        """
        ref = self.modes[mode]["reference"]
        xcen, ycen = self.ref_pupil_location(mode, hdr=hdr)
        ref.adjust_center(xcen, ycen)
        ref.apply_pupil(self.pup_inner, self.pup_size / 2.0)
        return ref

    def find_pupil_center(self, data, pup_mask):
        """
        Locate the pupil with center_pupil(). Never raises: falls back to the nominal center (cor_coords) if
        centering fails or lands more than cen_tol away from it.
        """
        try:
            xcen, ycen, _ = center_pupil(
                data, pup_mask, threshold=self.cen_thresh, sigma=self.cen_sigma, plot=False
            )
        except Exception as e:
            log.warning(f"Pupil centering failed, using nominal center: {e}")
            return tuple(self.cor_coords)
        if np.hypot(xcen - self.cor_coords[0], ycen - self.cor_coords[1]) > self.cen_tol:
            return tuple(self.cor_coords)
        return xcen, ycen

    def subtract_pupil_background(self, data, mode, center):
        """
        Remove the scattered-light halo using pixels outside the pupil and, optionally, the diffuse floor
        between spots. See `mmtwfs.background`.
        """
        ref = self.modes[mode]["reference"]
        pitch = np.mean([ref.xspacing, ref.yspacing])
        footprint = pupil_footprint(
            data.shape, center, self.pup_size / 2.0, inner=self.pup_inner, margin=pitch
        )
        data = data - pupil_background(data, footprint, box=self.bkg_box, order=self.bkg_order)
        if self.pedestal:
            data = data - pedestal(data, pitch)
        return data
```

(g) `measure_slopes`: replace

```python
        # get adjusted reference center position and update the reference
        xcen, ycen = self.ref_pupil_location(mode, hdr=hdr)
        self.modes[mode]["reference"].adjust_center(xcen, ycen)

        # apply pupil to the reference
        self.modes[mode]["reference"].apply_pupil(self.pup_inner, self.pup_size / 2.0)
```

with

```python
        self.prepare_reference(mode, hdr=hdr)

        # pupil center for the pupil background and the periodicity fallback; only computed when needed
        center = None
        if self.bkg_method == "pupil":
            center = self.find_pupil_center(data, pup_mask)
            data = self.subtract_pupil_background(data, mode, center)
```

`pup_mask = self.pupil_mask(hdr=hdr)` is defined just above that block. Keep it before the new block.

- [ ] **Step 4: Run the new tests and the whole WFS suite**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py -v`
Expected: all PASS. If `test_mmirs_analysis_pupil_background` fails, stop and report the Z10 value and full Zernike
difference from the legacy method. That is a validation finding for the user, not something to tune away here.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/wfs.py mmtwfs/tests/test_wfs.py
git commit -m "add pupil background option and poor-seeing config defaults to WFS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Grid scale to focus conversion with uncertainty

**Files:**
- Modify: `mmtwfs/wfs.py` (imports; new `WFS` methods after `subtract_pupil_background`)
- Test: `mmtwfs/tests/test_wfs.py`

**Interfaces:**
- Consumes:
  - `measure_grid_period` (Task 3)
  - `prepare_reference`
  - `SH_Reference.img_xcen`/`img_ycen`/`grid` (Task 4)
  - existing `make_init_pars`, `slope_diff`, `calculate_focus`, `reference_aberrations`
- Produces:
  - `WFS.reference_grid(mode) -> dict | None`: cached `measure_grid_period` result for the reference image
  - `WFS.focus_from_scale(scale, scale_err, mode, rotator, hdr=None) -> (ZernikeVector, Quantity[um])`. Needs
    `prepare_reference(mode, hdr)` to have been called first, as `measure_slopes` does. Returns the rotated,
    reference-subtracted ZernikeVector with `errorbars["Z04"]` set, and the focus after `m2_gain_periodicity` and
    `periodicity_focus_max` clipping.

- [ ] **Step 1: Write the failing tests.** Add `import astropy.units as u` to the imports in
  `mmtwfs/tests/test_wfs.py` and append:

```python
def _mmirs_ready(config={}):
    mmirs = WFSFactory(wfs="mmirs", config=config)
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    data, hdr = check_wfsdata(test_file, header=True)
    mode = mmirs.get_mode(hdr)
    mmirs.prepare_reference(mode, hdr=hdr)
    return mmirs, mode, hdr


def test_focus_from_scale_matches_analytic():
    # pure defocus: slope = (s - 1) * r, and d(fringe Z04)/dr = 4 Z04 r, so Z04 = -tiltfactor * (s - 1) * R / 4
    mmirs, mode, hdr = _mmirs_ready()
    zref = mmirs.reference_aberrations(mode, hdr=hdr)
    k = -mmirs.tiltfactor * (mmirs.pup_size / 2.0) / 4.0
    zv, focus = mmirs.focus_from_scale(1.01, 1e-4, mode, 0.0 * u.deg, hdr=hdr)
    expected = k * 0.01 - zref["Z04"].value
    assert np.isclose(zv["Z04"].value, expected, rtol=0.02)
    assert np.isclose(zv.errorbars["Z04"].value, abs(k) * 1e-4, rtol=0.02)


def test_focus_from_scale_gain():
    mmirs, mode, hdr = _mmirs_ready()
    zv, focus = mmirs.focus_from_scale(0.99, 1e-5, mode, 0.0 * u.deg, hdr=hdr)
    assert np.isclose(focus.value, 0.5 * mmirs.calculate_focus(zv.copy()).value, atol=0.02)


def test_focus_from_scale_large_error_is_zero():
    mmirs, mode, hdr = _mmirs_ready()
    zv, focus = mmirs.focus_from_scale(0.99, 1.0, mode, 0.0 * u.deg, hdr=hdr)
    assert focus.value == 0.0


def test_focus_from_scale_clipped():
    mmirs, mode, hdr = _mmirs_ready({"m2_gain_periodicity": 1.0})
    zv, focus = mmirs.focus_from_scale(0.8, 1e-5, mode, 0.0 * u.deg, hdr=hdr)
    assert abs(focus.to_value(u.um)) == 300.0


def test_reference_grid_cached():
    mmirs, mode, hdr = _mmirs_ready()
    g = mmirs.reference_grid(mode)
    assert g is not None
    ref = mmirs.modes[mode]["reference"]
    assert np.allclose(g["spacing"], np.mean([ref.xspacing, ref.yspacing]), rtol=0.02)
    assert mmirs.reference_grid(mode) is g
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py -k "focus_from_scale or reference_grid" -v`
Expected: FAIL with `AttributeError: 'MMIRS' object has no attribute 'focus_from_scale'`.

- [ ] **Step 3: Implement.** Add to the imports in `mmtwfs/wfs.py`:

```python
from mmtwfs.period import measure_grid_period, grid_scale, plot_periodicity
```

Then add these methods to `WFS`, after `subtract_pupil_background`:

```python
    def reference_grid(self, mode):
        """
        Measure, once per reference, the grid frequencies of the mode's reference image with the same method used
        on science frames, so window and sampling effects cancel in the ratio.
        """
        ref = self.modes[mode]["reference"]
        if ref.grid is None:
            ref.grid = measure_grid_period(
                ref.data - np.median(ref.data),
                (ref.img_xcen, ref.img_ycen),
                self.pup_size / 2.0,
                np.mean([ref.xspacing, ref.yspacing]),
                inner=self.pup_inner,
                snr_thresh=0.0,
            )
        return ref.grid

    def focus_from_scale(self, scale, scale_err, mode, rotator, hdr=None):
        """
        Convert a grid scale (measured spacing / reference spacing) into a focus-only wavefront and M2 focus
        correction. The slope field of a pure scale change is synthesized at the reference aperture positions and
        fit with the same machinery as fit_wavefront(), so reference aberrations, rotation, and sign conventions
        match the full analysis. Requires prepare_reference(mode, hdr) to have been called.

        The fit of noiseless synthetic slopes has ~zero formal error, so the Z04 error bar is set from scale_err
        instead. calculate_focus() scales corrections by (1 - frac_error), so poorly measured scales are
        automatically down-weighted.

        Returns
        -------
        zv : ZernikeVector
            Rotated, reference-subtracted wavefront with Z04 error bar
        focus : `~astropy.units.Quantity`
            M2 focus correction after m2_gain_periodicity and periodicity_focus_max clipping
        """
        ref = self.modes[mode]["reference"]
        x = np.asarray(ref.masked_apertures["xcentroid"])
        y = np.asarray(ref.masked_apertures["ycentroid"])
        coords = ref.pup_coords(self.pup_size / 2.0)

        # the fit is linear in the slopes, so fit a unit scale change once and scale the coefficients
        unit_slopes = -self.tiltfactor * np.array([x, y])
        params = make_init_pars(nmodes=3, modestart=2)
        unit = ZernikeVector(coeffs=lmfit.minimize(slope_diff, params, args=(coords, unit_slopes)))
        ds = scale - 1.0
        raw = ZernikeVector(
            Z02=unit["Z02"].value * ds,
            Z03=unit["Z03"].value * ds,
            Z04=unit["Z04"].value * ds,
            errorbars={"Z04": np.abs(unit["Z04"].value) * scale_err},
        )

        raw.rotate(angle=-(self.rotation - rotator))
        zv = raw - self.reference_aberrations(mode, hdr=hdr)

        focus = self.m2_gain_periodicity * self.calculate_focus(zv.copy())
        fmax = self.periodicity_focus_max.to_value(u.um)
        focus = np.clip(focus.to_value(u.um), -fmax, fmax) * u.um
        return zv, focus
```

Check that `reference_aberrations` accepts `hdr`: the base version takes `**kwargs` and F5's takes `hdr=None`, so
`hdr=hdr` works for both.

- [ ] **Step 4: Run tests and confirm they pass**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py -k "focus_from_scale or reference_grid" -v`
Expected: 5 PASS. A pre-plan check with the real fit gave `unit["Z04"]` = -138328.5 nm, matching
-tiltfactor * 172.5 / 4 exactly.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/wfs.py mmtwfs/tests/test_wfs.py
git commit -m "convert grid scale to focus-only wavefront with propagated uncertainty

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Periodicity fallback in `measure_slopes`

**Files:**
- Modify: `mmtwfs/wfs.py`: new `WFS.periodicity_focus` after `focus_from_scale`, and the
  `except WFSAnalysisFailed` branch of `measure_slopes` (around line 1284)
- Test: `mmtwfs/tests/test_wfs.py`

**Interfaces:**
- Consumes: `reference_grid`, `focus_from_scale`, `find_pupil_center` (Tasks 4–5); `measure_grid_period`,
  `grid_scale`, `plot_periodicity` (Task 3)
- Produces:
  - `WFS.periodicity_focus(data, mode, center, rotator, hdr=None, plot=True) -> dict | None` with keys `grid`,
    `zernike`, `pending_focus`, `figure`. `grid` has:
    - `scale`: offset applied
    - `scale_raw`
    - `scale_err`: factor and floor applied
    - `scale_err_fit`
    - `scales`
    - `rotation`
    - `snr`
    - `center`
  - On failure with a measurable grid, `measure_slopes` returns the existing dict (`slopes=None`, `figures`, `mode`)
    plus:
    - `focus_only=True`
    - `method="periodicity"`
    - `grid`
    - `zernike`
    - `pending_focus`
    - `figures["periodicity"]`

- [ ] **Step 1: Write the failing tests.** Add `from scipy import ndimage` and `from astropy.io import fits` to the
  imports in `mmtwfs/tests/test_wfs.py`, then append:

```python
def _blurred_mmirs(tmp_path, sigma):
    # gaussian blur of a good MMIRS frame: sigma >= 5 makes the spot analysis fail but leaves the grid visible
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    with fits.open(test_file) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header.copy()
    out = tmp_path / f"mmirs_blur{sigma}.fits"
    # the original header has lower-case keywords (ActualX) that need fixing to write
    fits.writeto(out, ndimage.gaussian_filter(data, sigma).astype(np.float32), hdr, output_verify="silentfix")
    return out


def test_periodicity_fallback(tmp_path):
    mmirs = WFSFactory(wfs="mmirs", config={"m2_gain_periodicity": 1.0})
    results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 6), plot=True)
    assert results["slopes"] is None
    assert results["focus_only"]
    assert results["method"] == "periodicity"
    # full analysis of the unblurred frame gives about -45 um; the fallback on this blur gave -43.5 um in a prototype
    assert -60.0 < results["pending_focus"].to_value(u.um) < -30.0
    assert results["grid"]["scale_err"] > 0.0
    assert results["figures"]["periodicity"].get_label() == "Grid Periodicity"
    plt.close("all")


def test_periodicity_matches_full_fit(tmp_path):
    # sigma = 3 blur still passes the spot analysis, so both answers are available on the same frame
    mmirs = WFSFactory(wfs="mmirs", config={"m2_gain_periodicity": 1.0})
    results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 3), plot=False)
    assert results["slopes"] is not None
    full = mmirs.calculate_focus(mmirs.fit_wavefront(results, plot=False)["zernike"])
    fb = mmirs.periodicity_focus(
        results["data"], results["mode"], (results["xcen"], results["ycen"]), results["rotator"],
        hdr=results["header"], plot=False,
    )
    assert abs(fb["pending_focus"].to_value(u.um) - full.to_value(u.um)) < 10.0
    plt.close("all")


def test_periodicity_fallback_disabled(tmp_path):
    mmirs = WFSFactory(wfs="mmirs", config={"periodicity_fallback": False})
    results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 6), plot=False)
    assert results["slopes"] is None
    assert "focus_only" not in results


def test_periodicity_no_grid(tmp_path):
    test_file = WFS_DATA_DIR / "test_data" / "mmirs_wfs_0150.fits"
    with fits.open(test_file) as hdul:
        hdr = hdul[0].header.copy()
        shape = hdul[0].data.shape
    noise = np.random.default_rng(9).normal(1250.0, 5.0, shape).astype(np.float32)
    out = tmp_path / "mmirs_noise.fits"
    fits.writeto(out, noise, hdr, output_verify="silentfix")
    mmirs = WFSFactory(wfs="mmirs")
    try:
        results = mmirs.measure_slopes(out, plot=False)
    except WFSAnalysisFailed:
        return  # also acceptable: no correction either way
    assert results["slopes"] is None
    assert not results.get("focus_only", False)


def test_periodicity_fallback_swallows_errors(tmp_path):
    mmirs = WFSFactory(wfs="mmirs")
    with patch.object(mmirs, "periodicity_focus", side_effect=RuntimeError("boom")):
        results = mmirs.measure_slopes(_blurred_mmirs(tmp_path, 6), plot=False)
    assert results["slopes"] is None
    assert "focus_only" not in results
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py -k periodicity -v`
Expected: `test_periodicity_fallback` and `test_periodicity_matches_full_fit` FAIL (KeyError `focus_only` /
AttributeError `periodicity_focus`). `test_periodicity_fallback_swallows_errors` FAILS on `patch.object` because the
attribute doesn't exist. The disabled and no-grid tests may already pass.

- [ ] **Step 3: Implement `periodicity_focus`.** Add it to `WFS` after `focus_from_scale`:

```python
    def periodicity_focus(self, data, mode, center, rotator, hdr=None, plot=True):
        """
        Focus-only fallback for frames whose spots are visible but too blurred to centroid: measure the grid
        period from the power spectrum, compare with the reference grid, and convert the scale change into a
        focus correction. Requires prepare_reference(mode, hdr). Returns None if the grid isn't detected.
        """
        ref = self.modes[mode]["reference"]
        ref_grid = self.reference_grid(mode)
        meas = measure_grid_period(
            data,
            center,
            self.pup_size / 2.0,
            np.mean([ref.xspacing, ref.yspacing]),
            inner=self.pup_inner,
            snr_thresh=self.period_snr_thresh,
        )
        if meas is None or ref_grid is None:
            return None

        grid = grid_scale(meas, ref_grid)
        grid["scale_raw"] = grid["scale"]
        grid["scale"] = grid["scale_raw"] + self.period_scale_offset
        grid["scale_err"] = float(np.hypot(self.period_err_factor * grid["scale_err_fit"], self.period_err_floor))
        grid["snr"] = meas["snr"]
        grid["center"] = tuple(center)

        zv, focus = self.focus_from_scale(grid["scale"], grid["scale_err"], mode, rotator, hdr=hdr)
        fig = plot_periodicity(meas) if plot else None
        return {"grid": grid, "zernike": zv, "pending_focus": focus, "figure": fig}
```

- [ ] **Step 4: Call it from `measure_slopes`.** In the `except WFSAnalysisFailed as e:` branch, replace the final
  `return results` (after `results["figures"]["slopes"] = slope_fig`) with:

```python
            if self.periodicity_fallback:
                # this must never turn an analysis failure into an exception
                try:
                    if center is None:
                        center = self.find_pupil_center(data, pup_mask)
                    fallback = self.periodicity_focus(data, mode, center, rotator, hdr=hdr, plot=plot)
                except Exception as fe:
                    log.warning(f"Periodicity fallback failed: {fe}")
                    fallback = None
                if fallback is not None:
                    grid = fallback["grid"]
                    log.warning(
                        f"Using focus-only periodicity fallback: scale = {grid['scale']:.5f} +/- "
                        f"{grid['scale_err']:.5f}, SNR = {grid['snr'].min():.0f}, focus = {fallback['pending_focus']}"
                    )
                    results["focus_only"] = True
                    results["method"] = "periodicity"
                    results["grid"] = grid
                    results["zernike"] = fallback["zernike"]
                    results["pending_focus"] = fallback["pending_focus"]
                    results["figures"]["periodicity"] = fallback["figure"]
            return results
```

`center` exists because Task 4 defined it before the `try`. `pup_mask` and `rotator` are defined earlier in
`measure_slopes`.

- [ ] **Step 5: Run the periodicity tests, then the full suite**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py -k periodicity -v`
Expected: 5 PASS.
Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs -v`
Expected: all PASS. `test_too_few_spots` (`mmirs_bogus.fits`) and `test_frosted_donut` still only assert
`slopes is None`, so they pass whether or not the fallback fires. Record in the task report whether each one
produced `focus_only`.

- [ ] **Step 6: Commit**

```bash
git add mmtwfs/wfs.py mmtwfs/tests/test_wfs.py
git commit -m "add focus-only periodicity fallback when spot analysis fails

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `reanalyze` records the analysis method

**Files:**
- Modify: `mmtwfs/scripts/reanalyze.py` (`process_image` around lines 212–302, `csv_header` around 353)
- Test: `mmtwfs/tests/test_reanalyze.py` (new)

**Interfaces:**
- Consumes: `focus_only`, `pending_focus`, `grid`, `zernike` result keys (Task 6)
- Produces:
  - CSV gains a trailing `method` column (`full` or `periodicity`)
  - `upgrade_cached_line(line) -> str`

- [ ] **Step 1: Write the failing test** `mmtwfs/tests/test_reanalyze.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from mmtwfs.scripts.reanalyze import upgrade_cached_line, CSV_HEADER


def test_upgrade_cached_line():
    ncols = len(CSV_HEADER.strip().split(","))
    old = ",".join(["1"] * (ncols - 1)) + "\n"
    new = upgrade_cached_line(old)
    assert new.endswith(",full\n")
    assert len(new.strip().split(",")) == ncols
    # lines that already have the method column pass through untouched
    assert upgrade_cached_line(new) == new
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_reanalyze.py -v`
Expected: FAIL with `ImportError: cannot import name 'upgrade_cached_line'`.

- [ ] **Step 3: Implement.** In `mmtwfs/scripts/reanalyze.py`:

(a) At module level, after the imports, add:

```python
CSV_HEADER = "time,wfs,file,exptime,airmass,az,el,osst,outt,chamt,tiltx,tilty,"\
    "transx,transy,focus,focerr,cc_x_err,cc_y_err,xcen,ycen,seeing,raw_seeing,"\
    "vlt_seeing,raw_vlt_seeing,ellipticity,fwhm,wavefront_rms,residual_rms,method\n"


def upgrade_cached_line(line):
    """
    .output files written before the method column existed are one field short; they were all full analyses.
    """
    if len(line.strip().split(",")) == len(CSV_HEADER.strip().split(",")) - 1:
        return line.rstrip("\n") + ",full\n"
    return line
```

(b) In `main()`, replace the local `csv_header = ...` assignment with `csv_header = CSV_HEADER`.

(c) In `process_image`, where cached output is returned, change `return lines[0]` to
`return upgrade_cached_line(lines[0])`.

(d) In the full-analysis line, change the final `f"{zresults['residual_rms'].value}\n"` to
`f"{zresults['residual_rms'].value},full\n"`.

(e) Replace the final `else:` branch (`failed.touch()  # mark this file as failed` / `return None`) with:

```python
    elif results.get('focus_only', False):
        nan = np.nan
        focerr = results['pending_focus']
        xcen, ycen = results['grid']['center']
        line = f"{obstime},{wfskey},{f.name},{exptime},{airmass},{az},{el},{osst},{outt}," \
            f"{chamt},{tiltx},{tilty},{transx},{transy},{focus},{focerr.value},{nan},{nan}," \
            f"{xcen},{ycen},{nan},{nan},{nan},{nan},{nan},{nan},{nan},{nan},periodicity\n"
        results['zernike'].save(filename=f.parent / (f.stem + ".periodicity.zernike"))
        with open(outfile, 'w') as fp:
            fp.write(line)
        return line
    else:
        failed.touch()  # mark this file as failed
        return None
```

- [ ] **Step 4: Run the test and a smoke run of the script on the test data**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_reanalyze.py -v`
Expected: PASS.
Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m flake8 mmtwfs/scripts/reanalyze.py --max-line-length=127`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add mmtwfs/scripts/reanalyze.py mmtwfs/tests/test_reanalyze.py
git commit -m "record analysis method in reanalyze output, including focus-only rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Validate on the October 2026 MMIRS run

Offline. The scripts live in `~/MMT/mmirs_vignetting/seeing/`, not in the repo. The output is numbers for Task 9 and a
short report for the user.

**Files:**
- Create: `~/MMT/mmirs_vignetting/seeing/validate.py`
- Create: `~/MMT/mmirs_vignetting/seeing/calibrate.py`

**Interfaces:**
- Consumes: `~/MMT/mmirs_vignetting/runO.ecsv` (columns `night`, `file`, `failed`, `cam`),
  `~/MMT/mmirs_vignetting/rawvig_oct.ecsv` (`night`, `file`, `mm`, `bright`); raw frames at
  `/Volumes/Samsung_4TB/wfsdat/<night>/<file>`
- Produces: `val_<label>.ecsv` per run, `calibration.txt`

- [ ] **Step 1: Write** `~/MMT/mmirs_vignetting/seeing/validate.py`:

```python
"""
Run the WFS analysis on every October 2026 MMIRS frame and record full-fit and periodicity-fallback results.

usage: python validate.py <label> [bkg_method]
Run with PYTHONPATH pointing at the mmtwfs checkout to test (e.g. a worktree of main for the baseline).
"""
import sys
import warnings
import logging
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import astropy.units as u
from astropy.table import Table

from mmtwfs.wfs import WFSFactory

DATA = "/Volumes/Samsung_4TB/wfsdat"
label = sys.argv[1]
bkg = sys.argv[2] if len(sys.argv) > 2 else "background2d"
_wfs = None


def init():
    global _wfs
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    # unity extra gain and uncalibrated errors so calibrate.py sees raw values
    _wfs = WFSFactory(wfs="mmirs", plot=False, config={"bkg_method": bkg, "m2_gain_periodicity": 1.0})


def fb_row(fb):
    g = fb["grid"]
    return dict(fb_ok=True, fb_focus=fb["pending_focus"].to_value(u.um), fb_z04=fb["zernike"]["Z04"].value,
                fb_z04_err=fb["zernike"].errorbars["Z04"].value, scale=g["scale_raw"],
                scale_err_fit=g["scale_err_fit"], snr=float(np.min(g["snr"])))


def run(args):
    night, fname = args
    row = dict(night=night, file=fname, full_ok=False, full_focus=np.nan, full_z04=np.nan, full_z04_err=np.nan,
               fb_ok=False, fb_focus=np.nan, fb_z04=np.nan, fb_z04_err=np.nan, scale=np.nan,
               scale_err_fit=np.nan, snr=np.nan, error="")
    row.update({f"Z{i:02d}": np.nan for i in range(5, 12)})
    try:
        res = _wfs.measure_slopes(f"{DATA}/{night}/{fname}", plot=False)
        if res["slopes"] is not None:
            z = _wfs.fit_wavefront(res, plot=False)["zernike"]
            row.update(full_ok=True, full_focus=_wfs.calculate_focus(z.copy()).to_value(u.um),
                       full_z04=z["Z04"].value, full_z04_err=z.errorbars.get("Z04", 0 * u.nm).value)
            row.update({f"Z{i:02d}": z[f"Z{i:02d}"].value for i in range(5, 12)})
            if hasattr(_wfs, "periodicity_focus"):
                fb = _wfs.periodicity_focus(res["data"], res["mode"], (res["xcen"], res["ycen"]),
                                            res["rotator"], hdr=res["header"], plot=False)
                if fb is not None:
                    row.update(fb_row(fb))
        elif res.get("focus_only", False):
            row.update(fb_row({"grid": res["grid"], "zernike": res["zernike"],
                               "pending_focus": res["pending_focus"]}))
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


if __name__ == "__main__":
    t = Table.read("/Users/tim/MMT/mmirs_vignetting/runO.ecsv")
    jobs = [(str(n), str(f)) for n, f in zip(t["night"], t["file"])]
    with Pool(8, initializer=init) as pool:
        rows = pool.map(run, jobs, chunksize=8)
    out = Table(rows=rows)
    out.write(f"val_{label}.ecsv", overwrite=True)
    print(label, "full ok:", out["full_ok"].sum(), "fallback ok:", out["fb_ok"].sum(), "of", len(out))
```

- [ ] **Step 2: Run the three configurations.**
  1. A baseline from `main`, using a worktree.
  2. The branch with the legacy background (this includes the slicing fix).
  3. The branch with the pupil background.

```bash
cd /Users/tim/MMT/mmtwfs && git worktree add ../mmtwfs-main main
mkdir -p ~/MMT/mmirs_vignetting/seeing && cd ~/MMT/mmirs_vignetting/seeing
PYTHONPATH=/Users/tim/MMT/mmtwfs-main /Users/tim/conda/envs/mmtwfs/bin/python validate.py main
PYTHONPATH=/Users/tim/MMT/mmtwfs /Users/tim/conda/envs/mmtwfs/bin/python validate.py branch_bkg2d background2d
PYTHONPATH=/Users/tim/MMT/mmtwfs /Users/tim/conda/envs/mmtwfs/bin/python validate.py branch_pupil pupil
```

Run each in the background; each takes on the order of 15–30 minutes with 8 workers. Expected: three files
`val_main.ecsv`, `val_branch_bkg2d.ecsv`, `val_branch_pupil.ecsv`, each with 2697 rows. If `error` is non-empty for
more than about 1% of rows, look at the messages before going on.

- [ ] **Step 3: Write** `~/MMT/mmirs_vignetting/seeing/calibrate.py`:

```python
"""
Calibrate the periodicity fallback against full fits and compare background methods.
"""
import numpy as np
from astropy.table import Table, join

TILT = 3207.6173854022736  # nm/px, MMIRS tiltfactor
K = -TILT * 172.5 / 4.0    # d(Z04)/d(scale), nm

main = Table.read("val_main.ecsv")
b2d = Table.read("val_branch_bkg2d.ecsv")
pup = Table.read("val_branch_pupil.ecsv")
vig = Table.read("/Users/tim/MMT/mmirs_vignetting/rawvig_oct.ecsv")["night", "file", "mm", "bright"]
lines = []


def say(s=""):
    print(s)
    lines.append(s)


# (a) calibration on frames where both full fit and fallback exist (legacy background, branch code)
both = b2d[b2d["full_ok"] & b2d["fb_ok"]]
d = both["full_z04"] - both["fb_z04"]  # nm
offset_nm = np.median(d)
say(f"(a) {len(both)} frames with both answers")
say(f"    median(full - fallback) Z04 = {offset_nm:.1f} nm -> period_scale_offset = {offset_nm / K:+.2e}")
sig_fit = np.abs(K) * both["scale_err_fit"]
r2 = (d - offset_nm) ** 2 - both["full_z04_err"] ** 2
# robust linear fit r2 ~ a * sig_fit^2 + b over sig_fit quintiles (medians of r2 / 0.455 for chi2_1)
q = np.quantile(sig_fit, np.linspace(0, 1, 6))
xs, ys = [], []
for lo, hi in zip(q[:-1], q[1:]):
    m = (sig_fit >= lo) & (sig_fit <= hi)
    xs.append(np.median(sig_fit[m] ** 2))
    ys.append(np.median(r2[m]) / 0.455)
a, b = np.polyfit(xs, ys, 1)
a, b = max(a, 0.0), max(b, 0.0)
factor, floor = np.sqrt(a), np.sqrt(b) / np.abs(K)
say(f"    period_err_factor = {factor:.3f}, period_err_floor = {floor:.2e}")
total = np.hypot(np.hypot(factor * sig_fit, floor * np.abs(K)), both["full_z04_err"])
pulls = (d - offset_nm) / total
say(f"    pull width (1.4826*MAD) = {1.4826 * np.median(np.abs(pulls - np.median(pulls))):.2f} (target 1)")
dfoc = both["full_focus"] - both["fb_focus"]
say(f"    focus full - fallback: median {np.median(dfoc):+.1f} um, 1.4826*MAD "
    f"{1.4826 * np.median(np.abs(dfoc - np.median(dfoc))):.1f} um (target <= 10 um after offset)")
for lo, hi in ((20, 100), (100, 300), (300, 1000), (1000, np.inf)):
    m = (both["snr"] >= lo) & (both["snr"] < hi)
    if m.sum() > 5:
        say(f"    SNR {lo}-{hi}: n={m.sum()} focus scatter "
            f"{1.4826 * np.median(np.abs(dfoc[m] - np.median(dfoc[m]))):.1f} um")

# (b) background and slicing-fix comparison
say()
for name, t in (("main", main), ("branch bkg2d (slicing fix)", b2d), ("branch pupil", pup)):
    say(f"(b) {name}: full ok {t['full_ok'].sum()}, fallback ok {t['fb_ok'].sum()} of {len(t)}")
for name, t in (("branch bkg2d vs main", (main, b2d)), ("pupil vs bkg2d", (b2d, pup))):
    j = join(t[0], t[1], keys=["night", "file"], table_names=["a", "b"])
    j = j[j["full_ok_a"] & j["full_ok_b"]]
    for z in ["Z04"] + [f"Z{i:02d}" for i in range(5, 12)]:
        col = "full_z04" if z == "Z04" else z
        dz = j[f"{col}_b"] - j[f"{col}_a"]
        say(f"    {name} {z}: n={len(j)} median diff {np.median(dz):+.1f} nm, "
            f"1.4826*MAD {1.4826 * np.median(np.abs(dz - np.median(dz))):.1f} nm")

# (c) recovery of previously failed frames, by category
say()
j = join(join(main["night", "file", "full_ok"], pup, keys=["night", "file"], table_names=["main", "new"]),
         vig, keys=["night", "file"], join_type="left")
failed = j[~j["full_ok_main"]]
faint_cut = np.nanpercentile(j["bright"], 25)
cat = np.where(failed["mm"] < 0.5, "vignetted", np.where(failed["bright"] < faint_cut, "faint", "fuzzy"))
for c in ("vignetted", "faint", "fuzzy"):
    m = cat == c
    say(f"(c) {c}: {m.sum()} failed in main -> full {failed['full_ok_new'][m].sum()}, "
        f"focus-only {(~failed['full_ok_new'][m] & failed['fb_ok'][m]).sum()}, "
        f"neither {(~failed['full_ok_new'][m] & ~failed['fb_ok'][m]).sum()}")

open("calibration.txt", "w").write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run it**

Run: `cd ~/MMT/mmirs_vignetting/seeing && /Users/tim/conda/envs/mmtwfs/bin/python calibrate.py`
Expected: `calibration.txt` with sections (a)–(c). The pass criteria come from the spec:
- (a) fallback-vs-full focus scatter of 10 µm or less after the offset, and pull width close to 1;
- (b) pupil vs bkg2d Zernike median differences within the existing fit scatter (1.4826*MAD), and the pupil
  method's full-ok count at least that of bkg2d;
- (c) a recovery count reported per category.

- [ ] **Step 5: Report to the user.** Paste `calibration.txt` and recommend values for Task 9. This includes
  whether MMIRS should switch to `bkg_method = "pupil"` and whether `period_snr_thresh` should rise if low-SNR bins
  scatter by more than 10 µm. **Wait for their decision before Task 9.**

---

### Task 9: Apply calibrated MMIRS configuration

**Files:**
- Modify: `mmtwfs/config.py` (the `"mmirs"` block, after `"nzern": 21,`)
- Test: `mmtwfs/tests/test_wfs.py`

**Interfaces:**
- Consumes: the numbers from `calibration.txt` and the user's decision (Task 8)

- [ ] **Step 1: Write the failing test,** filling in the approved values for `<factor>`, `<floor>` and `<offset>`.
  Delete the `bkg_method` assertion if the user decided against switching:

```python
def test_mmirs_poor_seeing_config():
    mmirs = WFSFactory(wfs="mmirs")
    assert mmirs.bkg_method == "pupil"
    assert mmirs.period_err_factor == <factor>
    assert mmirs.period_err_floor == <floor>
    assert mmirs.period_scale_offset == <offset>
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs/tests/test_wfs.py::test_mmirs_poor_seeing_config -v`
Expected: FAIL on the first differing attribute.

- [ ] **Step 3: Add the keys to the `"mmirs"` config block** after `"nzern": 21,`, with a comment naming the
  calibration source:

```python
            # poor-seeing handling, calibrated on the 2026-10-01..05 run (see docs/superpowers/specs/
            # 2026-10-06-poor-seeing-analysis-design.md)
            "bkg_method": "pupil",
            "period_err_factor": <factor>,
            "period_err_floor": <floor>,
            "period_scale_offset": <offset>,
```

- [ ] **Step 4: Run the full suite**

Run: `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest mmtwfs -v` and
`/Users/tim/conda/envs/mmtwfs/bin/python -m flake8 mmtwfs --count --max-line-length=127`
Expected: all PASS, flake8 count 0. If switching MMIRS to `"pupil"` changes `test_mmirs_analysis`, the earlier
`test_mmirs_analysis_pupil_background` result already showed the new value; keep the same window.

- [ ] **Step 5: Commit and open the PR**

```bash
git add mmtwfs/config.py mmtwfs/tests/test_wfs.py
git commit -m "calibrate MMIRS periodicity fallback and switch to pupil background

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin poor-seeing-analysis
gh pr create --title "Poor-seeing analysis: pupil background and periodicity focus fallback" --body "$(cat <<'EOF'
Improves WFS analysis in poor seeing.

- `mmtwfs/background.py`: pupil-aware halo model + inter-spot pedestal (opt-in per WFS via `bkg_method`)
- `mmtwfs/period.py`: FFT grid-period estimator
- `measure_slopes` falls back to a focus-only correction (`focus_only`, `pending_focus`, ...) when spot analysis
  fails but the grid period is measurable; the Z04 error bar carries the period uncertainty so `calculate_focus`
  down-weights noisy measurements
- fixes the `get_apertures` background box (`xcen - 50:ycen + 50`)
- `reanalyze` records the analysis method
- MMIRS calibrated on 2026-10-01..05 (see `calibration.txt` summary below)

<paste calibration.txt>

Spec: docs/superpowers/specs/2026-10-06-poor-seeing-analysis-design.md
Plan: docs/superpowers/plans/2026-10-06-poor-seeing-analysis.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Then remove the baseline worktree: `git worktree remove ../mmtwfs-main`.

---

### Task 10: wfssrv applies focus-only results

Do this only after the mmtwfs PR is merged and installed (`pip install --no-deps -e /Users/tim/MMT/mmtwfs` in the
mmtwfs env already points at the checkout).

**Files:**
- Modify: `~/MMT/wfssrv/wfssrv/wfssrv.py`: the `else:` branch of `if results["slopes"] is not None:` in the analyze
  handler (around line 430)

**Interfaces:**
- Consumes: `focus_only`, `grid`, `zernike`, `pending_focus`, `figures["periodicity"]` from `measure_slopes`

- [ ] **Step 1: Branch from master** (the checkout is currently on `js-tests`):

```bash
cd ~/MMT/wfssrv && git status --short && git switch master && git pull && git switch -c periodicity-focus
```

If `git status --short` shows local changes, stop and ask the user before switching.

- [ ] **Step 2: Implement.** Replace

```python
                else:
                    log.error(f"Wavefront measurement failed: {filename}")
                    figures = create_default_figures()
                    figures["slopes"] = results["figures"]["slopes"]
                    self.application.refresh_figures(figures=figures)
```

with

```python
                elif results.get("focus_only", False):
                    grid = results["grid"]
                    log.warning(
                        f"{filename}: spots too blurred for full analysis; using focus-only correction from the "
                        f"grid period (scale = {grid['scale']:.5f} +/- {grid['scale_err']:.5f}, "
                        f"SNR = {grid['snr'].min():.0f})"
                    )
                    # only focus is valid. clear anything left pending from an earlier image.
                    self.application.has_pending_m1 = False
                    self.application.has_pending_coma = False
                    self.application.has_pending_recenter = False
                    self.application.pending_focus = results["pending_focus"]
                    self.application.has_pending_focus = True
                    zvec = results["zernike"]
                    self.application.wavefront_fit = zvec.copy()
                    zvec.save(filename=self.application.datadir / (filename + ".periodicity.zernike"))
                    figures = create_default_figures()
                    figures["slopes"] = results["figures"].get("periodicity") or results["figures"]["slopes"]
                    self.application.refresh_figures(figures=figures)
                else:
                    log.error(f"Wavefront measurement failed: {filename}")
                    figures = create_default_figures()
                    figures["slopes"] = results["figures"]["slopes"]
                    self.application.refresh_figures(figures=figures)
```

- [ ] **Step 3: Test.** Run the existing tests with `/Users/tim/conda/envs/mmtwfs/bin/python -m pytest wfssrv -v`
  (expected: PASS). Then exercise it locally:
  1. Copy a blurred frame into a scratch data dir. Use `_blurred_mmirs` logic, or a real failed October frame such
     as `20261005/mmirs_wfs_0245.fits`.
  2. Start `WFSROOT=<scratch dir> /Users/tim/conda/envs/mmtwfs/bin/python -m wfssrv.wfssrv`.
  3. Call `/analyze?fitsfile=<file>&connect=false`. Use `connect=false`: the default Redis host is production.
  4. Confirm all of these:
     - the log shows the focus-only warning;
     - the slopes panel shows the periodicity figure;
     - `<file>.periodicity.zernike` exists;
     - the M1, coma and recenter buttons are not enabled.

- [ ] **Step 4: Commit and open the PR**

```bash
git add wfssrv/wfssrv.py
git commit -m "apply focus-only corrections from the mmtwfs periodicity fallback

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin periodicity-focus
gh pr create --title "Apply focus-only periodicity corrections" --body "$(cat <<'EOF'
When mmtwfs can't centroid spots but measures the grid period (`results["focus_only"]`), offer the focus
correction only. M1/coma/recenter pending flags are cleared, the periodicity spectrum is shown in the slopes panel,
and the wavefront is saved as `.periodicity.zernike`.

Requires mmtwfs with the poor-seeing-analysis changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
