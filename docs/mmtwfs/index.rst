Wavefront Sensor Analysis Software for the MMTO (`mmtwfs`)
==========================================================

Introduction
------------

.. note::
    `mmtwfs` works only with astropy version 2.0 or later and python 3.6 or later.

The `mmtwfs` package provides:

+ A telescope class, `~mmtwfs.telescope.Telescope`, that encompasses telescope and primary mirror system configuration and
  functionality.

+ A secondary class, `~mmtwfs.secondary.Secondary`, that encompasses secondary mirror system configuration and functionality.

+ A WFS class, `~mmtwfs.wfs.WFS`, that encompasses wavefront sensor configuration, functionality, wavefront analysis, and
  calculation of corrections to be applied.

+ A factory method, `~mmtwfs.wfs.WFSFactory`, that builds and configures a subclass of `~mmtwfs.wfs.WFS` based on passed keyword
  arguments and configuration data.

+ A set of functions that perform the image analysis of Shack-Hartmann wavefront sensor frames required to measure motions of
  the spot positions and convert these motions into wavefront slopes.

+ A set of functions for manipulating Zernike polynomials and fitting them to wavefront slopes.

+ A class, `~mmtwfs.zernike.ZernikeVector`, to help facilitate the manipulation and visualization of sets of Zernike
  polynomial coefficients.

Getting Started
---------------

A ``WFS`` object can be created using the `~mmtwfs.wfs.WFSFactory` method. The ``wfs`` keyword is used to specify which ``WFS``
subclass to construct. The default configuration for each supported WFS is defined in `~mmtwfs.config.mmt_config`:

    >>> from mmtwfs.wfs import WFSFactory
    >>> bino_wfs = WFSFactory(wfs="binospec")

To analyze a Shack-Hartmann image taken by the wavefront sensor and measure the wavefront slopes, use the
`~mmtwfs.wfs.WFS.measure_slopes()` method:

    >>> slope_results = bino_wfs.measure_slopes("bino_sh_data.fits")

To fit a model to these slopes, pass `slope_results` to the `~mmtwfs.wfs.WFS.fit_wavefront()` method:

    >>> wavefront_results = bino_wfs.fit_wavefront(slope_results)

Using `mmtwfs`
--------------

.. toctree::
    :maxdepth: 1

    usage.rst
    tips_tricks_caveats.rst
    api_docs.rst

.. _GitHub repo: https://github.com/MMTObservatory/mmtwfs
