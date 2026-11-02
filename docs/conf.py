# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime
import sys
from importlib.metadata import metadata

try:
    from sphinx_astropy.conf.v3 import *  # noqa: F401,F403
except ImportError:
    print("ERROR: the documentation requires the sphinx-astropy package to be installed")
    sys.exit(1)

_meta = metadata("mmtwfs")

project = _meta["Name"]
author = _meta["Author-email"].split(" <")[0].strip('"') if _meta["Author-email"] else _meta["Author"]
copyright = f"{datetime.datetime.now().year}, {author}"
release = _meta["Version"]
version = release.split("-", 1)[0]

GITHUB_PROJECT = "MMTObservatory/mmtwfs"

highlight_language = "python3"
exclude_patterns.append("_templates")  # noqa: F405
exclude_patterns.append("superpowers")  # noqa: F405  design notes and plans, not user documentation
rst_epilog += "\n"  # noqa: F405

# -- HTML output -------------------------------------------------------------
# sphinx-astropy's conf.v3 selects the pydata-based "astropy-unified" theme; we
# keep its extension/intersphinx machinery but render with the Read the Docs
# theme instead, so override the theme (and its options) here.
html_theme = "sphinx_rtd_theme"

html_title = f"{project} v{release}"
html_static_path = ["_static"]

# The trimmed grey MMT logo sits in the RTD sidebar header.
html_logo = "_static/mmt_logo.png"

# MMT favicon (from mmto.org) in place of the theme's astropy default.
html_favicon = "_static/mmt_favicon.ico"

html_theme_options = {
    "logo_only": False,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_external_links": True,
}

# Wire up the RTD theme's "Edit on GitHub" link.
html_context = {
    "display_github": True,
    "github_user": GITHUB_PROJECT.split("/")[0],
    "github_repo": GITHUB_PROJECT.split("/")[1],
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Resolving issue number to links in changelog -----------------------------
github_issues_url = f"https://github.com/{GITHUB_PROJECT}/issues/"
