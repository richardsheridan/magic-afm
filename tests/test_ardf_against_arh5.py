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

# each ARDF and its conversion, by corpus relpath without the suffix: two fast maps, one
# with both directions kept and one with the retrace alone, and a slow force map
STEMS = ["PDMS_SAA/ffm/both0001", "PDMS_SAA/ffm/rtrace0001", "AC160s00"]
STEM_IDS = [Path(stem).name for stem in STEMS]
METADATA = ("k", "defl_sens", "sync_dist", "scansize", "t_step", "trace")
# images the ARH5 file holds and the ARDF does not; every other stem has none
ARH5_ONLY_IMAGES = {
    "AC160s00": {
        "HeaterHumidity",
        "HeaterTemperature",
        "Seconds",
        "TriggerTime",
        "XLVDT",
        "YLVDT",
    },
}
# AC160s00 is a slow map whose extends stop at the trigger, so their lengths vary; the
# lengths below are those of Asylum's own .ibw exports of its curves, (4, 5) being the
# one pixel that never triggered
AC160S00_SHAPE = (5, 6)
AC160S00_EXTEND_POINTS = {(4, 4): 339, (2, 0): 535, (4, 5): 496}
AC160S00_RETRACT_POINTS = 496


def assert_exactly_equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


def open_pair(real_data_file, stem):
    """The ARDF and the ARH5 file of stem, each opened the way the CLI opens it."""
    ardf = Path(real_data_file(f"{stem}.ARDF"))
    arh5 = Path(real_data_file(f"{stem}.h5"))
    return open_fvfile(ardf), open_fvfile(arh5)


def volumes_by_name(fvfile) -> dict:
    names = [v.name for v in fvfile.volumes]
    assert len(set(names)) == len(names), names
    return {v.name: v for v in fvfile.volumes}


@pytest.mark.slow
@pytest.mark.parametrize("stem", STEMS, ids=STEM_IDS)
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
@pytest.mark.parametrize("stem", STEMS, ids=STEM_IDS)
def test_ardf_and_its_arh5_conversion_declare_the_same_metadata(real_data_file, stem):
    """k, deflection sensitivity, sync distance, scan size, t_step and trace."""
    ardf_cm, arh5_cm = open_pair(real_data_file, stem)
    with ardf_cm as ardf, arh5_cm as arh5:
        for attr in METADATA:
            assert getattr(ardf, attr) == getattr(arh5, attr), attr


@pytest.mark.slow
@pytest.mark.parametrize("stem", STEMS, ids=STEM_IDS)
def test_ardf_and_its_arh5_conversion_hold_the_same_images(real_data_file, stem):
    """Every ARDF image is in the ARH5 file with the same shape, units and values.

    The ARH5 file holds no other image but those ARH5_ONLY_IMAGES names for its stem.
    """
    ardf_cm, arh5_cm = open_pair(real_data_file, stem)
    with ardf_cm as ardf, arh5_cm as arh5:
        assert ardf.images
        assert set(ardf.images) <= set(arh5.images)
        assert set(arh5.images) - set(ardf.images) == ARH5_ONLY_IMAGES.get(stem, set())
        for name, ardf_image in ardf.images.items():
            arh5_image = arh5.images[name]
            assert ardf_image.shape == arh5_image.shape, name
            assert ardf_image.units == arh5_image.units, name
            ardf_values, arh5_values = ardf_image.get_image(), arh5_image.get_image()
            assert ardf_values.dtype == arh5_values.dtype == np.float32, name
            assert_exactly_equal(ardf_values, arh5_values)


@pytest.mark.slow
@pytest.mark.parametrize("suffix", [".ARDF", ".h5"])
def test_ac160s00_curves_have_the_lengths_of_the_asylum_ibw_exports(
    real_data_file, suffix
):
    """Each reader gives AC160s00 the extend and retract lengths of the .ibw exports."""
    path = Path(real_data_file(f"AC160s00{suffix}"))
    with open_fvfile(path) as fvfile:
        (volume,) = fvfile.volumes
        assert volume.shape == AC160S00_SHAPE
        for r, c in np.ndindex(volume.shape):
            (z_ext, z_ret), (d_ext, d_ret) = volume.get_curve(r, c)
            assert len(z_ext) == len(d_ext), (r, c)
            assert len(z_ret) == len(d_ret) == AC160S00_RETRACT_POINTS, (r, c)
            if (r, c) in AC160S00_EXTEND_POINTS:
                assert len(z_ext) == AC160S00_EXTEND_POINTS[r, c], (r, c)
