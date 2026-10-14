# mmtwfs/tests/test_outputs.py
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math

import pytest

from mmtwfs.outputs import CSV_COLUMNS, CSV_HEADER, read_output

LINE = ("2026-10-05T01:54:18,mmirs,mmirs_wfs_0001.fits,12.0,1.1959,188.66,56.74,12,11.53,12.3,244.43,167.91,"
        "-559.21,2188.5,9052.31,-0.0,5.511,0.0,256.99,232.50,4.886,5.439,2.442,2.719,0.083,16.34,1387.4,2220.4,"
        "full,2.805")


def test_columns_match_header():
    assert CSV_COLUMNS == CSV_HEADER.strip().split(",")
    assert CSV_COLUMNS[-2:] == ["method", "delivered_iq"]


def test_read_output(tmp_path):
    p = tmp_path / "a.output"
    p.write_text(LINE + "\n")
    row = read_output(p)
    assert row["file"] == "mmirs_wfs_0001.fits"
    assert row["method"] == "full"
    # "focus" is the hexapod focus position (TRANSZ); "focerr" is the measured focus correction
    assert row["focus"] == pytest.approx(9052.31)
    assert row["focerr"] == pytest.approx(-0.0)
    assert row["delivered_iq"] == pytest.approx(2.805)
    assert isinstance(row["raw_vlt_seeing"], float)


def test_read_output_pads_old_lines(tmp_path):
    p = tmp_path / "old.output"
    p.write_text(",".join(LINE.split(",")[:-2]) + "\n")
    row = read_output(p)
    assert row["method"] == "full"
    assert math.isnan(row["delivered_iq"])


def test_read_output_empty_and_bad(tmp_path):
    empty = tmp_path / "e.output"
    empty.write_text("")
    assert read_output(empty) is None
    bad = tmp_path / "b.output"
    bad.write_text("1,2,3\n")
    with pytest.raises(ValueError):
        read_output(bad)
