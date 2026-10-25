"""A .pfc curve reads as Nanoscope Analysis's own ASCII export of it shows it."""

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

# The export is one retract half of record 261 of onekhz.001.pfc, which the reader puts
# at row 8, column 5: a header line of units, then 128 rows of z and deflection in nm,
# listed far end first.

from pathlib import Path

import numpy as np
import pytest

from magic_afm.data_readers import SUFFIX_FVFILE_MAP

PFC = "PS_PCL_SAA/QNM/onekhz.001.pfc"
EXPORT = "onekhz.001.pfc-record261-retract-ascii.txt"
PIXEL = (8, 5)
# Nanoscope Analysis cuts the PeakForce period into halves two samples later than the
# reader, so its retract starts two samples into the reader's and, the curve being one
# period, ends on the first two samples of the reader's extend.
CUT_LAG = 2
# Nanoscope Analysis's z starts from 0 at the bottom of the stroke; the reader's swings
# by the peak force amplitude about the Height image value.
PEAK_FORCE_AMPLITUDE_NM = 75.0


@pytest.mark.slow
def test_pfc_curve_matches_the_nanoscope_analysis_ascii_export(real_data_file):
    """z and deflection of one .pfc retract equal the vendor's ASCII export of it."""
    export = Path(real_data_file(EXPORT))
    with export.open() as f:
        assert f.readline().split() == ["nm", "nm"]
    rows = np.loadtxt(export, skiprows=1)
    assert rows.shape == (128, 2)
    # reversed, the rows run forward in time as the reader's halves do
    export_z, export_d = rows[::-1].T

    path = Path(real_data_file(PFC))
    fvfile_cls, opener = SUFFIX_FVFILE_MAP[path.suffix.lower()]
    with opener(path) as data:
        fvfile = fvfile_cls.parse(data)
        (volume,) = fvfile.volumes
        (z_ext, z_ret), (d_ext, d_ret) = volume.get_curve(*PIXEL)
        height = fvfile.images["Height"]
        assert height.units == "m"
        height_nm = float(height.get_image()[PIXEL]) * 1e9

    n = len(z_ret)
    assert len(z_ext) == len(d_ext) == len(d_ret) == n == len(export_z)
    z = np.concatenate([z_ext, z_ret])
    d = np.concatenate([d_ext, d_ret])
    window = (n + CUT_LAG + np.arange(n)) % (2 * n)

    # Nanoscope Analysis scales positive codes by 1/32767 of full scale where the
    # reader scales every code by 1/32768, 1.6e-4 nm at this curve's largest deflection
    np.testing.assert_allclose(d[window], export_d, rtol=0, atol=2e-4)
    # the export prints 7 significant digits, 5e-5 nm at 100 nm and above
    np.testing.assert_allclose(
        z[window] - height_nm + PEAK_FORCE_AMPLITUDE_NM, export_z, rtol=0, atol=1e-4
    )
