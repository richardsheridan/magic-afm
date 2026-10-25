"""Real ARDF files read the same as the ARH5 files Asylum's converter made from them."""

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

# Both formats store the same float32 values and both readers scale them to nm the same
# way, so curves and images agree exactly, and so does every metadata value, each parsed
# from the same note text.

from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile

# each ARDF and its conversion, under PDMS_SAA/ffm/: both directions kept,
# and the retrace alone
STEMS = ["both0001", "rtrace0001"]
METADATA = ("k", "defl_sens", "sync_dist", "scansize", "t_step", "trace")


def assert_exactly_equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


def open_pair(real_data_file, stem):
    """The ARDF and the ARH5 file of stem, each opened the way the CLI opens it."""
    ardf = Path(real_data_file(f"PDMS_SAA/ffm/{stem}.ARDF"))
    arh5 = Path(real_data_file(f"PDMS_SAA/ffm/{stem}.h5"))
    return open_fvfile(ardf), open_fvfile(arh5)


def volumes_by_name(fvfile) -> dict:
    names = [v.name for v in fvfile.volumes]
    assert len(set(names)) == len(names), names
    return {v.name: v for v in fvfile.volumes}


@pytest.mark.slow
@pytest.mark.parametrize("stem", STEMS)
def test_ardf_and_its_arh5_conversion_hold_the_same_curves(real_data_file, stem):
    """Volumes of the same name have the same shape, curves and curve indices."""
    ardf_cm, arh5_cm = open_pair(real_data_file, stem)
    with ardf_cm as ardf, arh5_cm as arh5:
        ardf_volumes, arh5_volumes = volumes_by_name(ardf), volumes_by_name(arh5)
        assert sorted(ardf_volumes) == sorted(arh5_volumes)
        for name, ardf_volume in ardf_volumes.items():
            arh5_volume = arh5_volumes[name]
            assert ardf_volume.shape == arh5_volume.shape, name
            # each file iterates in its own order
            ardf_indices = sorted(map(tuple, ardf_volume.iter_indices()))
            arh5_indices = sorted(map(tuple, arh5_volume.iter_indices()))
            assert ardf_indices == arh5_indices == list(np.ndindex(ardf_volume.shape))
            for r, c in ardf_indices:
                ardf_curve = ardf_volume.get_curve(r, c)
                arh5_curve = arh5_volume.get_curve(r, c)
                for ardf_part, arh5_part in zip(ardf_curve, arh5_curve, strict=True):
                    for ardf_half, arh5_half in zip(ardf_part, arh5_part, strict=True):
                        assert ardf_half.dtype == arh5_half.dtype == np.float32
                        assert ardf_half.shape == arh5_half.shape, (name, r, c)
                        assert_exactly_equal(ardf_half, arh5_half)
            ardf_all, arh5_all = (
                ardf_volume.get_all_curves(),
                arh5_volume.get_all_curves(),
            )
            assert ardf_all.dtype == arh5_all.dtype == np.float32
            assert ardf_all.shape == arh5_all.shape, name
            assert_exactly_equal(ardf_all, arh5_all)


@pytest.mark.slow
@pytest.mark.parametrize("stem", STEMS)
def test_ardf_and_its_arh5_conversion_declare_the_same_metadata(real_data_file, stem):
    """k, deflection sensitivity, sync distance, scan size, t_step and trace."""
    ardf_cm, arh5_cm = open_pair(real_data_file, stem)
    with ardf_cm as ardf, arh5_cm as arh5:
        for attr in METADATA:
            assert getattr(ardf, attr) == getattr(arh5, attr), attr


@pytest.mark.slow
@pytest.mark.parametrize("stem", STEMS)
def test_ardf_and_its_arh5_conversion_hold_the_same_images(real_data_file, stem):
    """The same image names, each with the same shape, units and values."""
    ardf_cm, arh5_cm = open_pair(real_data_file, stem)
    with ardf_cm as ardf, arh5_cm as arh5:
        assert ardf.images
        assert sorted(ardf.images) == sorted(arh5.images)
        for name, ardf_image in ardf.images.items():
            arh5_image = arh5.images[name]
            assert ardf_image.shape == arh5_image.shape, name
            assert ardf_image.units == arh5_image.units, name
            ardf_values, arh5_values = ardf_image.get_image(), arh5_image.get_image()
            assert ardf_values.dtype == arh5_values.dtype == np.float32, name
            assert_exactly_equal(ardf_values, arh5_values)
