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
