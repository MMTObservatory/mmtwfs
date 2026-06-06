# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mmtwfs` is a Python package for wavefront sensing and active optics management at the MMT Observatory. It analyzes Shack-Hartmann wavefront sensor (WFS) data to measure optical aberrations and calculate corrections for the telescope's adaptive optics systems.

## Common Commands

### Testing
# tox envlist defines py313 and py314 (requires-python is >=3.13);
# there is no py312 environment. Default to py314; use py313 if needed.
```bash
# Run all tests (uses pytest)
tox -e py314

# Run tests with all optional dependencies
tox -e py314-alldeps

# Run tests with development dependencies (nightly wheels of core deps)
tox -e py314-devdeps

# Run tests with coverage
tox -e py314-cov

# Run specific test file
pytest mmtwfs/tests/test_wfs.py

# Run specific test
pytest mmtwfs/tests/test_wfs.py::test_function_name
```

### Code Quality
```bash
# Check code style with flake8 (max line length: 127)
tox -e codestyle

# Manually run flake8
flake8 mmtwfs --count --max-line-length=127
```

### Documentation
```bash
# Build documentation
tox -e build_docs

# Check documentation links
tox -e linkcheck

# Generate HTML coverage report
tox -e cov_report
```

### Installation
```bash
# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with all optional dependencies
pip install -e ".[test,docs,extra]"
```

## Architecture

### Core Components

The package uses a factory pattern to instantiate telescope-specific and WFS-specific classes:

1. **WFS Classes** (`mmtwfs/wfs.py`): Wavefront sensor implementations
   - `WFS`: Base class with common configuration and analysis methods
   - `F9`, `F5`: MMT secondary-specific WFS configurations
   - `Binospec`, `MMIRS`: Instrument-specific WFS implementations
   - `FLWO12`: FLWO 1.2m telescope WFS
   - `WFSFactory()`: Factory function to instantiate the correct WFS subclass

2. **Telescope Classes** (`mmtwfs/telescope.py`): Telescope optical models
   - `Telescope`: Base class defining telescope parameters
   - `MMT`: MMT 6.5m telescope with primary mirror control
   - `FLWO12`, `FLWO15`: FLWO telescope configurations
   - `TelescopeFactory()`: Factory function to instantiate the correct telescope
   - Uses POPPY library for optical modeling and PSF calculations

3. **Secondary Mirror Classes** (`mmtwfs/secondary.py`): Secondary mirror control
   - `Secondary`: Base configuration class
   - `F5`, `F9`: MMT secondary mirrors with hexapod control
   - `MMTSecondary`: Mixin providing hexapod communication methods
   - Methods: `focus()`, `cc()` (center of curvature), `zc()` (zero-coma), `correct_coma()`, `recenter()`
   - `SecondaryFactory()`: Factory function to instantiate the correct secondary

4. **Zernike Analysis** (`mmtwfs/zernike.py`): Wavefront decomposition
   - `ZernikeVector`: Dictionary-like class for managing Zernike coefficients
   - Functions for calculating Zernike polynomials and their derivatives
   - Converts between Noll indexing and (n, m) indexing
   - `zernike_slopes()`: Calculate expected spot positions from Zernike coefficients

5. **Configuration** (`mmtwfs/config.py`): Central configuration repository
   - `mmtwfs_config`: Dictionary containing telescope, secondary, and WFS parameters
   - Includes optical prescriptions, detector parameters, reference files
   - `merge_config()`: Utility to merge configuration dictionaries
   - `recursive_subclasses()`: Utility to find all subclasses of a given class

6. **MMT Primary Cell** (`mmtwfs/mmtcell.py`): Primary mirror cell control
   - `Cell`: Async class for communicating with the MMT primary mirror cell
   - Manages actuator forces to correct primary mirror figure

7. **F/9 Topbox Control** (`mmtwfs/f9topbox.py`): F/9 instrument support
   - `CompMirror`: Controls the F/9 comparison mirror (in/out)

8. **Supporting modules**
   - `mmtwfs/photometry.py`: Spot photometry helpers for SH images (e.g. `make_spot_mask()`)
   - `mmtwfs/utils.py`: `srvlookup()` for DNS SRV resolution of hardware hostnames/ports
   - `mmtwfs/custom_exceptions.py`: Package-specific exception types

### Console Scripts

Entry points defined in `pyproject.toml` (`[project.scripts]`), installed on `pip install`:
- `reanalyze`: Batch re-analysis of WFS data (`mmtwfs/scripts/reanalyze.py`)
- `fix_mmtwfs_csvs`, `fix_mmirs_exposure_time`, `rename_mmirs_files`: Data-file maintenance utilities

### Data Flow

1. **Image Acquisition**: WFS images (FITS files) are loaded from disk
2. **Reference Setup**: A reference image is analyzed to locate WFS spots (`SH_Reference` class)
3. **Spot Detection**: Science images are analyzed to find spot centroids (`wfsfind()`, uses `photutils.DAOStarFinder`)
4. **Spot Matching**: Science spots are matched to reference spots using k-d trees
5. **Slope Calculation**: Spot displacements are converted to wavefront slopes
6. **Zernike Fitting**: Slopes are fit to Zernike polynomials using least-squares
7. **Correction Calculation**: Zernike coefficients are used to calculate:
   - Secondary hexapod corrections (focus, coma, tilt)
   - Primary mirror actuator forces (higher-order aberrations)
8. **Command Application**: Corrections are sent to telescope systems via network sockets

### Key Design Patterns

- **Factory Pattern**: `WFSFactory()`, `TelescopeFactory()`, `SecondaryFactory()` dynamically instantiate subclasses
- **Configuration Hierarchy**: Base configuration from `mmtwfs_config` can be overridden per-instance
- **Mixin Classes**: `MMTSecondary` adds hexapod control to secondary mirror classes
- **Reference Images**: Each WFS mode has a reference image defining nominal spot positions
- **Units**: Extensive use of `astropy.units` for dimensional analysis

### Important Data Files

Located in `mmtwfs/data/`:
- `Surf2ActTEL_*.bin`: Influence matrices mapping actuator forces to surface displacement
- `actuator_coordinates.dat`: Positions of primary mirror force actuators
- `bcv_node_coordinates.dat`: Finite element model node coordinates
- `*zernfield*.tab`: Zernike-to-hexapod correction mappings
- `ref_images/`: Reference WFS images for different configurations

### Testing Structure

- Tests are in `mmtwfs/tests/`
- Test data is in `mmtwfs/data/test_data/`
- Tests use `pytest` with `pytest-astropy` plugin
- Some tests use `pytest-benchmark` for performance testing

## Python Requirements

- Minimum Python version: 3.13 (specified in `pyproject.toml`)
- Uses `setuptools_scm` for version management (version written to `mmtwfs/version.py`)

## Important Notes

- This package interfaces with real telescope hardware via network sockets
- Set `connected=False` during testing to avoid sending commands
- WFS analysis can be slow for large images; consider using smaller test datasets
- The package uses Noll indexing for Zernike polynomials (starts at j=1 for piston)
- Coordinate systems vary by instrument; check flip/rotation parameters in config
