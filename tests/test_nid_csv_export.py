"""A .nid curve reads as Nanosurf's own CSV export of it shows it."""

# Copyright (C) Richard J. Sheridan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Each export is one curve of the .nid: a header line of names and SI units, then 1024
# rows of tip-sample separation and force for the forward (extend) and backward
# (retract) halves, printed to five significant digits. FWD runs in time order, as the
# reader's extend does; BWD runs far end first, the reader's retract reversed. A forward
# half shorter than 1024 points is padded at the end with rows whose force is empty.
# Nanosurf's R counts rows from the bottom and both R and C count from 1, so the curve
# R, C is the reader's (rows - R, C - 1).

import re
from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile

NID = "20251216_141819_PVDF2_Film_U_00002.nid"
CSV_DIR = "PVDF2_Film_U_00002/CSV"
# reader (0, 0), the opposite corner (9, 9), and (0, 1), the shortest extend
CSVS = [
    "PVDF2_Film_U_00002_R10_C01.csv",
    "PVDF2_Film_U_00002_R01_C10.csv",
    "PVDF2_Film_U_00002_R10_C02.csv",
]
# at R10_C01 every separation in each column is one value, which no curve explains
SEPARATION_CSVS = [name for name in CSVS if not name.endswith("_R10_C01.csv")]
HEADER = [
    "Tip-Sample-Sep FWD [m]",
    "Force FWD [N]",
    "Tip-Sample-Sep BWD [m]",
    "Force BWD [N]",
]
CSV_ROWS = 1024


def read_export(real_data_file, name):
    """The export's columns: FWD separation and force, BWD separation and force."""
    path = Path(real_data_file(f"{CSV_DIR}/{name}"))
    with path.open() as f:
        assert f.readline().rstrip("\n").split(",") == HEADER
    # empty fields read as NaN
    rows = np.genfromtxt(path, delimiter=",", skip_header=1)
    assert rows.shape == (CSV_ROWS, len(HEADER))
    return rows.T


def read_curve(real_data_file, name):
    """The reader's spring constant and curve at the pixel the export's name gives."""
    m = re.fullmatch(r".*_R(\d+)_C(\d+)\.csv", name)
    assert m is not None
    r, c = int(m[1]), int(m[2])
    with open_fvfile(Path(real_data_file(NID))) as fvfile:
        (volume,) = fvfile.volumes
        rows = volume.shape[0]
        (z_ext, z_ret), (d_ext, d_ret) = volume.get_curve(rows - r, c - 1)
        k = fvfile.k
    z_ext, z_ret, d_ext, d_ret = (
        np.asarray(a, np.float64) for a in (z_ext, z_ret, d_ext, d_ret)
    )
    return k, (z_ext, z_ret), (d_ext, d_ret)


def assert_within_last_digit(reader, export):
    """reader is within one unit of the fifth significant digit of each export value."""
    magnitude = np.abs(export)
    # none of the exports prints a zero; one would get its column's smallest unit
    magnitude = np.where(magnitude == 0, np.min(magnitude[magnitude > 0]), magnitude)
    unit = 10.0 ** (np.floor(np.log10(magnitude)) - 4)
    np.testing.assert_allclose((reader - export) / unit, 0, rtol=0, atol=1)


@pytest.mark.slow
@pytest.mark.parametrize("name", CSVS)
def test_nid_csv_export_pads_the_forward_half_past_the_reader_extend(
    real_data_file, name
):
    """FWD force is present on the first rows, as many as the reader's extend has
    points, and empty after; BWD force is present on every row, the retract's length."""
    _, force_fwd, _, force_bwd = read_export(real_data_file, name)
    _, (z_ext, z_ret), _ = read_curve(real_data_file, name)

    np.testing.assert_array_equal(
        np.isnan(force_fwd), np.arange(CSV_ROWS) >= len(z_ext)
    )
    assert not np.isnan(force_bwd).any()
    assert len(z_ret) == CSV_ROWS


@pytest.mark.slow
@pytest.mark.parametrize("name", CSVS)
def test_nid_force_matches_the_nanosurf_csv_export(real_data_file, name):
    """The export's force is k times the reader's deflection, on both halves."""
    _, force_fwd, _, force_bwd = read_export(real_data_file, name)
    k, _, (d_ext, d_ret) = read_curve(real_data_file, name)
    n = len(d_ext)

    # k in N/m, deflection in nm
    assert_within_last_digit(k * d_ext * 1e-9, force_fwd[:n])
    assert_within_last_digit(k * d_ret[::-1] * 1e-9, force_bwd)


@pytest.mark.slow
@pytest.mark.parametrize("name", SEPARATION_CSVS)
def test_nid_separation_matches_the_nanosurf_csv_export(real_data_file, name):
    """The export's tip-sample separation is the reader's z less its deflection, on both
    halves."""
    sep_fwd, _, sep_bwd, _ = read_export(real_data_file, name)
    _, (z_ext, z_ret), (d_ext, d_ret) = read_curve(real_data_file, name)
    n = len(z_ext)

    # z and deflection in nm
    assert_within_last_digit((z_ext - d_ext) * 1e-9, sep_fwd[:n])
    assert_within_last_digit((z_ret - d_ret)[::-1] * 1e-9, sep_bwd)
