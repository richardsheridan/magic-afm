"""Reader images declare their units, and the GUI labels its colourbar with them."""

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

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import attrs
import h5py
import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.ardf import build_ardf
from synthetic_files.arh5 import build_arh5_ffm, build_arh5_force_map
from synthetic_files.nanoscope import build_pfc, build_spm
from synthetic_files.nid import build_nid

BOTH_ARDF = "PDMS_SAA/ffm/both0001.ARDF"
BOTH_H5 = "PDMS_SAA/ffm/both0001.h5"
ASYLUM_UNITS = {
    "HeightTrace": "m",
    "AdhesionTrace": "N",
    "ForceTrace": "N",
    "YoungsJKRTrace": "Pa",
    "ZSensorTrace": "m",
}
# (relpath, image name, the unit its values are in)
REAL_UNITS = [
    *((BOTH_ARDF, name, unit) for name, unit in ASYLUM_UNITS.items()),
    *((BOTH_H5, name, unit) for name, unit in ASYLUM_UNITS.items()),
    ("PS_PCL_SAA/QNM/onekhz.000.pfc", "Height", "m"),
    ("A1_OCSOH-PMMA21k_300421.0_300425.spm", "Height Sensor", "m"),
    ("20251216_141819_PVDF2_Film_U_00002.nid", "MapHeight", "m"),
    ("20251216_141819_PVDF2_Film_U_00002.nid", "MaxRetAdhesionForce", "N"),
]
SLOW_MAP_BUILDERS = {
    "ardf-slow-map": partial(build_ardf, slow_map=True),
    "arh5-force-map": build_arh5_force_map,
}
# (case, image name, the unit its values are in), named as the real slow map names them
SLOW_MAP_UNITS = [
    ("ardf-slow-map", "MapHeight", "m"),
    ("ardf-slow-map", "MapAdhesion", "N"),
    ("arh5-force-map", "MapHeight", "m"),
    ("arh5-force-map", "MapAdhesion", "N"),
    ("arh5-force-map", "TriggerTime", "s"),
    ("arh5-force-map", "Seconds", "s"),
    ("arh5-force-map", "XLVDT", "m"),
    ("arh5-force-map", "YLVDT", "m"),
    ("arh5-force-map", "HeaterTemperature", "V"),
    ("arh5-force-map", "HeaterHumidity", "V"),
]
BUILDERS = {
    "nid": build_nid,
    "arh5-ffm-both": partial(build_arh5_ffm, capture="both"),
    "arh5-force-map": build_arh5_force_map,
    "pfc": build_pfc,
    "spm": build_spm,
    "ardf-both": partial(build_ardf, capture="both"),
}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("relpath", "name", "unit"),
    REAL_UNITS,
    ids=[f"{Path(relpath).name}-{name}" for relpath, name, _ in REAL_UNITS],
)
def test_real_image_declares_the_unit_of_its_values(
    real_data_file, relpath, name, unit
):
    with open_fvfile(Path(real_data_file(relpath))) as fvfile:
        assert fvfile.images[name].units == unit


@pytest.mark.parametrize(
    ("case", "name", "unit"),
    SLOW_MAP_UNITS,
    ids=[f"{case}-{name}" for case, name, _ in SLOW_MAP_UNITS],
)
def test_asylum_slow_map_image_declares_the_unit_of_its_values(
    tmp_path, case, name, unit
):
    synthetic = SLOW_MAP_BUILDERS[case](tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        assert fvfile.images[name].units == unit


@pytest.mark.parametrize(
    ("name", "unit"),
    [("MapZSensor", "m"), ("MapForce", "N"), ("MapYoungsJKR", "Pa")],
)
def test_asylum_map_prefix_takes_the_unit_of_the_name_it_prefixes(tmp_path, name, unit):
    """A slow-map image named Map plus a known quantity is in that quantity's unit."""
    synthetic = build_arh5_force_map(tmp_path)
    with h5py.File(synthetic.path, "r+") as f:
        f["Image"].create_dataset(name, data=np.zeros((3, 4), np.float32))
    with open_fvfile(synthetic.path) as fvfile:
        assert fvfile.images[name].units == unit


@dataclass(frozen=True)
class DeclaredImage:
    """An Image of zeros that declares whatever unit it is given."""

    name: str
    shape: tuple[int, int]
    units: str

    def get_image(self) -> np.ndarray:
        return np.zeros(self.shape, np.float32)


@pytest.mark.parametrize("case", BUILDERS)
def test_gui_labels_a_reader_image_with_its_declared_unit(tmp_path, case):
    from magic_afm.gui import _impl as gui

    synthetic = BUILDERS[case](tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        opened = gui.GUIFVFile(fvfile)
        for name, image in fvfile.images.items():
            assert opened.get_image_units(name) == image.units, name


@pytest.mark.parametrize(
    ("name", "unit"),
    [("Height", "nm"), ("AdhesionTrace", "pN"), ("Phase", "°")],
)
def test_gui_labels_an_image_with_its_declared_unit_not_one_known_by_name(
    tmp_path, name, unit
):
    """The unit comes from the image, whatever its name would suggest."""
    from magic_afm.gui import _impl as gui

    synthetic = build_pfc(tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        image = DeclaredImage(name, next(iter(fvfile.images.values())).shape, unit)
        opened = gui.GUIFVFile(attrs.evolve(fvfile, images={name: image}))
        assert opened.get_image_units(name) == unit


def test_gui_labels_a_calculated_map_with_the_unit_it_was_added_with(tmp_path):
    from magic_afm.gui import _impl as gui

    synthetic = build_pfc(tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        opened = gui.GUIFVFile(fvfile)
        name = "CalcRetUnitsCheckTrace"
        opened.add_image(image_name=name, units="Pa", image=np.zeros((2, 2)))
        assert opened.get_image_units(name) == "Pa"
