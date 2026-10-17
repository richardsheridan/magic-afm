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
"""Naming, filtering and scaling of exported property maps, shared by GUI and CLI."""

import math
from collections.abc import Iterable

import numpy as np

from . import calculation
from .calculation import FitFix, FitMode
from .data_readers import Volume

DIRECTIONS = frozenset({"Trace", "Retrace"})

# declared SI unit: (factor, shift) from fit units (nm, nN, GPa, ln(nm))
SI_FROM_FIT_UNITS = {
    None: (1.0, 0.0),
    "rad": (1.0, 0.0),
    "samples": (1.0, 0.0),
    "m": (1e-9, 0.0),
    "N": (1e-9, 0.0),
    "Pa": (1e9, 0.0),
    "ln(m)": (1.0, math.log(1e-9)),
}


def mode_tag(fit_mode) -> str:
    if fit_mode == FitMode.EXTEND:
        return "Ext"
    elif fit_mode == FitMode.RETRACT:
        return "Ret"
    return "Both"


def trace_suffix(volumes: Iterable[Volume], selected: Volume) -> str:
    """The selected volume's name if the file holds both directions, else empty."""
    return selected.name if DIRECTIONS <= {v.name for v in volumes} else ""


def export_maps(
    property_map, parms_map, parms_err_map, fit_fix, k_sens
) -> list[tuple[str, str | None, np.ndarray]]:
    """(name, units, array) for each map to export, in export units, unflipped."""
    out = []
    for names_units, map_, err_str in zip(
        [
            calculation.PROPERTY_UNITS_DICT,
            calculation.PARMS_UNITS_DICT,
            calculation.PARMS_UNITS_DICT,
        ],
        [property_map, parms_map, parms_err_map],
        ["", "", "Err"],
    ):
        for name in names_units:
            units = names_units[name]
            factor, shift = SI_FROM_FIT_UNITS[units]
            if err_str:
                shift = 0.0
            name2 = name
            # skip writing fittable params that are fixed
            if name.upper() in FitFix.__members__ and fit_fix & FitFix[name.upper()]:
                continue
            if name == "M":
                if not fit_fix & FitFix.RADIUS:
                    continue
                name2 = "IndentationModulus"
            if name == "li_pha" and fit_fix & FitFix.LI_AMP:
                continue
            if name == "fc":
                name2 = "AdhesionForce"
            if name == "SensIndMod_k" and not k_sens:
                continue
            array = map_[name].squeeze() * factor + shift
            out.append((name2 + err_str, units, array))
    return out


def map_name(mode: str, name: str, suffix: str) -> str:
    return f"{mode}{name}{suffix}"


def export_file_name(root: str, map_name: str, ext: str) -> str:
    """{root}_{map_name}{ext}, ext including its dot."""
    return f"{root}_{map_name}{ext}"


def options_file_name(root: str) -> str:
    return f"{root}_options.json"
