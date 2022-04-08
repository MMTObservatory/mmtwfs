2.1.0 (2022-04-08)
------------------

General
^^^^^^^

Significant performance enhancements:

- Use KD-tree for matching observed and reference aperture coordinates
- Implement caching and reuse of zernike polynomial terms when calculating set of slopes

Overall improvement of about a factor of two in performance when doing a full analysis of a typical Shack-Hartmann image.


2.0.2 (2022-04-06)
------------------

Bug Fixes
^^^^^^^^^

- Bump ``python_requires`` to 3.8
- Convert README to ``.rst`` format
- Move dependencies from ``all`` to ``install_requires`` so ``pip`` installs from PyPI pull in everything that's needed.


2.0.0 (2022-04-05)
------------------

General
^^^^^^^

- The minimum supported Python is now 3.8.
- The minimum supported Astropy is now 5.0.
- The minimum supported Numpy is now 1.18.

New Functionality
^^^^^^^^^^^^^^^^^

- Command-line script for batch re-analysis of archived data.
- Support for MMIRS vignetting model to predict visible spots.
- Support for calculating uncertainties in wavefront model fits.
- Use lenslet optics to calculate spot sizes from diffraction instead of using reference images.
- Fix recentering calculations for Hecto.
- Add support for FLWO engineering wavefront sensors.
- Add configuration for building a container image and automate it in CI.

Bug Fixes
^^^^^^^^^

- Improve reliability of pupil registration.
- Improve reliability in marginal seeing conditions.
- Fix effective wavelengths to be based on more accurate QE and filter response curves.
- Update reference images for MMIRS.
- Various CI fixes, improvements, and eventual migration to GitHub Actions.
- Remove ``astropy_helpers`` and update/modernize package configuration.
- Fix bug in ``pup_mask`` that was tripped over by ``scipy >= 1.4``.
- Improve background substraction in WFS images.
- Fix DNS resolver to match API changes in upstream package.


1.0.0 (2018-05-29)
------------------

General
^^^^^^^

- Initial Release