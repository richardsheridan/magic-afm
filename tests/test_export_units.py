"""Exported maps are in the SI unit calculation's units dicts declare for them."""

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

# The fit works in nm, nN and GPa; the exports are in m, N and Pa. The expected units
# and factors are written out here rather than read from PROPERTY_UNITS_DICT and
# PARMS_UNITS_DICT, so that a wrong declaration fails. Only the fields whose
# declaration is settled are listed: lj_scale, vd, li_per, li_amp, li_pha, drag and
# SumSquaredError are not checked here.

import numpy as np
import pytest
import test_goldens as t2
from synthetic_files import open_fvfile
from synthetic_files.nid import build_nid
from test_cli import export_name, invoke_cli

from magic_afm import _export, calculation
from magic_afm.calculation import FitFix, FitMode

# field: (SI unit, factor from fit units)
PROPERTY_SI = {
    "Deflection": ("m", 1e-9),
    "Indentation": ("m", 1e-9),
    "ContactRadius": ("m", 1e-9),
    "TrueHeight": ("m", 1e-9),
    "IndentationRatio": (None, 1.0),
    "SensIndMod_k": (None, 1.0),
}
PARMS_SI = {
    "radius": ("m", 1e-9),
    "tau": (None, 1.0),
    "M": ("Pa", 1e9),
    "fc": ("N", 1e-9),
    "delta_shift": ("m", 1e-9),
    "force_shift": ("N", 1e-9),
}
EXPORTED_AS = {"M": "IndentationModulus", "fc": "AdhesionForce"}

FREE_RADIUS = FitFix.DEFAULTS & ~FitFix.RADIUS
# case: (fit_fix, the PARMS_SI fields it exports)
FIT_FIX_CASES = {
    "defaults": (FitFix.DEFAULTS, ("M", "fc", "delta_shift", "force_shift")),
    "free_radius_and_tau": (
        FREE_RADIUS & ~FitFix.TAU,
        ("radius", "tau", "fc", "delta_shift", "force_shift"),
    ),
}
SHAPE = (2, 3)


def filled(dtype, offset):
    """A SHAPE map of dtype, each field and pixel holding a different value."""
    a = np.empty(SHAPE, dtype)
    for i, name in enumerate(dtype.names):
        a[name] = offset + 10 * i + np.arange(a.size).reshape(SHAPE) / 8
    return a


@pytest.mark.parametrize("case", FIT_FIX_CASES)
def test_export_maps_writes_values_and_errors_in_the_declared_si_unit(case):
    fit_fix, parms = FIT_FIX_CASES[case]
    property_map = filled(calculation.PROPERTY_DTYPE, 1.0)
    parms_map = filled(calculation.PARMS_DTYPE, 2.0)
    parms_err_map = filled(calculation.PARMS_DTYPE, 3.0)
    maps = _export.export_maps(property_map, parms_map, parms_err_map, fit_fix, True)
    exported = {name: (units, array) for name, units, array in maps}

    expected = {}
    for field, (unit, factor) in PROPERTY_SI.items():
        expected[field] = (unit, factor * property_map[field])
    for field in parms:
        unit, factor = PARMS_SI[field]
        name = EXPORTED_AS.get(field, field)
        expected[name] = (unit, factor * parms_map[field])
        expected[name + "Err"] = (unit, factor * parms_err_map[field])
    assert expected.keys() <= exported.keys()
    for name, (unit, want) in expected.items():
        units, array = exported[name]
        assert units == unit, name
        np.testing.assert_allclose(array, want, rtol=1e-6, atol=0, err_msg=name)


def fit_demo(fit_fix, **options):
    """Fit the GUI Demo curve at pixel (0, 0) as the CLI does, as 1 x 1 maps."""
    curve = t2.demo_curve(seed=(0, 0))
    options = {**t2.DEFAULT_OPTIONS, **options, "fit_fix": fit_fix}
    with np.errstate(all="ignore"):
        _, properties, parms, parms_err = calculation.fit_curve_properties(
            t2.preprocess(((0, 0), curve), FitMode.RETRACT),
            fit_mode=FitMode.RETRACT,
            k=t2.DEMO_K,
            **options,
        )
    assert properties is not None, "the demo curve does not fit"
    maps = [np.reshape(m, (1, 1)) for m in (properties, parms, parms_err)]
    return {
        name: array.item()
        for name, _, array in _export.export_maps(*maps, fit_fix, True)
    }


def test_export_maps_of_a_known_curve_are_its_model_parameters_in_si():
    """The Demo curve is made with M 0.1 GPa, fc 2 nN and radius 20 nm."""
    with_radius_fixed = fit_demo(FitFix.DEFAULTS)
    with_radius_free = fit_demo(FREE_RADIUS, M=t2.DEMO_PARMS["M"])
    np.testing.assert_allclose(
        with_radius_fixed["IndentationModulus"], 1e8, rtol=0.1, atol=0
    )
    np.testing.assert_allclose(
        with_radius_fixed["AdhesionForce"], 2e-9, rtol=0.1, atol=0
    )
    np.testing.assert_allclose(with_radius_free["radius"], 2e-8, rtol=0.1, atol=0)


def test_cli_writes_a_fitted_radius_and_its_error_in_metres(tmp_path):
    (tmp_path / "in").mkdir()
    synthetic = build_nid(tmp_path / "in")
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        want = {name: np.full(volume.shape, np.nan) for name in ("parms", "parms_err")}
        options = {**t2.DEFAULT_OPTIONS, "fit_fix": FREE_RADIUS}
        for rc_curve in volume.iter_curves():
            rc = rc_curve[0]
            with np.errstate(all="ignore"):
                _, _, parms, parms_err = calculation.fit_curve_properties(
                    t2.preprocess(rc_curve, FitMode.RETRACT),
                    fit_mode=FitMode.RETRACT,
                    k=fvfile.k,
                    **options,
                )
            if parms is not None:
                want["parms"][rc] = parms["radius"].item()
                want["parms_err"][rc] = parms_err["radius"].item()
    assert np.isfinite(want["parms"]).any(), "no curve fits"
    out = tmp_path / "out"
    result = invoke_cli(synthetic.path, out, ["--stop-on-error", "--fit-radius"])
    assert result.exit_code == 0, result.output
    stem = synthetic.path.stem
    for name, group in (("radius", "parms"), ("radiusErr", "parms_err")):
        # the CLI exports rows reversed relative to the reader
        actual = np.load(out / export_name(stem, name)).astype(np.float64)[::-1]
        np.testing.assert_allclose(
            actual, 1e-9 * want[group], rtol=1e-4, atol=0, equal_nan=True, err_msg=name
        )
