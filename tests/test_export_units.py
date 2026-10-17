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
# PARMS_UNITS_DICT, so that a wrong declaration fails. Three fields are special:
# SumSquaredError is a dimensionless weighted chi-squared, the fit's implicit weight
# of 1 being 1/(1 nm)²; lj_scale is ln(L / 1 nm) in the fit and ln(L / 1 m) in the
# export, a shift and no factor, its Err unshifted; drag is in resampled sample
# intervals and exported as fitted (its conversion to seconds is deferred to M2).

import math

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
    "SumSquaredError": (None, 1.0),
}
# field: (SI unit, factor from fit units, shift added to the value, not to its Err)
LN_NM_IN_M = math.log(1e-9)
PARMS_SI = {
    "radius": ("m", 1e-9, 0.0),
    "tau": (None, 1.0, 0.0),
    "M": ("Pa", 1e9, 0.0),
    "fc": ("N", 1e-9, 0.0),
    "delta_shift": ("m", 1e-9, 0.0),
    "force_shift": ("N", 1e-9, 0.0),
    "lj_scale": ("ln(m)", 1.0, LN_NM_IN_M),
    "vd": (None, 1.0, 0.0),
    "li_per": ("m", 1e-9, 0.0),
    "li_amp": ("m", 1e-9, 0.0),
    "li_pha": ("rad", 1.0, 0.0),
    "drag": ("samples", 1.0, 0.0),
}
EXPORTED_AS = {"M": "IndentationModulus", "fc": "AdhesionForce"}

FREE_RADIUS = FitFix.DEFAULTS & ~FitFix.RADIUS
FREE_ARTIFACTS = ~(FitFix.VD | FitFix.LI_PER | FitFix.LI_AMP | FitFix.DRAG)
# case: (fit_fix, the PARMS_SI fields it exports)
FIT_FIX_CASES = {
    "defaults": (
        FitFix.DEFAULTS,
        ("M", "fc", "delta_shift", "force_shift", "lj_scale"),
    ),
    "free_radius_and_tau": (
        FREE_RADIUS & ~FitFix.TAU,
        ("radius", "tau", "fc", "delta_shift", "force_shift", "lj_scale"),
    ),
    "free_artifacts": (
        FitFix.DEFAULTS & FREE_ARTIFACTS,
        ("M", "fc", "delta_shift", "force_shift", "lj_scale")
        + ("vd", "li_per", "li_amp", "li_pha", "drag"),
    ),
}
SHAPE = (2, 3)


def filled(dtype, offset):
    """A SHAPE map of dtype, each field and pixel holding a different value."""
    a = np.empty(SHAPE, dtype)
    for i, name in enumerate(dtype.names):
        a[name] = offset + 10 * i + np.arange(a.size).reshape(SHAPE) / 8
    return a


def expected_maps(parms):
    """map name: (group, field, unit, factor, shift) for every map a case exports."""
    out = {f: ("properties", f, u, k, 0.0) for f, (u, k) in PROPERTY_SI.items()}
    for field in parms:
        unit, factor, shift = PARMS_SI[field]
        name = EXPORTED_AS.get(field, field)
        out[name] = ("parms", field, unit, factor, shift)
        out[name + "Err"] = ("parms_err", field, unit, factor, 0.0)
    return out


# exported map: the first FIT_FIX_CASES case that exports it
MAP_CASES = {}
for _case, (_, _parms) in FIT_FIX_CASES.items():
    for _name in expected_maps(_parms):
        MAP_CASES.setdefault(_name, _case)


@pytest.mark.parametrize("name", MAP_CASES)
def test_export_maps_writes_a_map_in_its_declared_si_unit(name):
    fit_fix, parms = FIT_FIX_CASES[MAP_CASES[name]]
    group, field, unit, factor, shift = expected_maps(parms)[name]
    fit_maps = {
        "properties": filled(calculation.PROPERTY_DTYPE, 1.0),
        "parms": filled(calculation.PARMS_DTYPE, 2.0),
        "parms_err": filled(calculation.PARMS_DTYPE, 3.0),
    }
    maps = _export.export_maps(*fit_maps.values(), fit_fix, True)
    exported = {n: (units, array) for n, units, array in maps}
    assert name in exported
    units, array = exported[name]
    assert units == unit
    want = factor * fit_maps[group][field] + shift
    np.testing.assert_allclose(array, want, rtol=1e-6, atol=0)


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


# field: (fit_fix, CLI arguments); lj_scale is free at the defaults
CLI_CASES = {
    "radius": (FREE_RADIUS, ("--fit-radius",)),
    "lj_scale": (FitFix.DEFAULTS, ("--fit-lj-scale",)),
}


@pytest.mark.parametrize("field", CLI_CASES)
def test_cli_writes_a_free_parameter_and_its_error_in_its_si_unit(tmp_path, field):
    fit_fix, args = CLI_CASES[field]
    _, factor, shift = PARMS_SI[field]
    (tmp_path / "in").mkdir()
    synthetic = build_nid(tmp_path / "in")
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        fitted = {name: np.full(volume.shape, np.nan) for name in ("parms", "err")}
        options = {**t2.DEFAULT_OPTIONS, "fit_fix": fit_fix}
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
                fitted["parms"][rc] = parms[field].item()
                fitted["err"][rc] = parms_err[field].item()
    assert np.isfinite(fitted["parms"]).any(), "no curve fits"
    want = {
        field: factor * fitted["parms"] + shift,
        field + "Err": factor * fitted["err"],
    }
    out = tmp_path / "out"
    result = invoke_cli(synthetic.path, out, ["--stop-on-error", *args])
    assert result.exit_code == 0, result.output
    stem = synthetic.path.stem
    for name, expected in want.items():
        # the CLI exports rows reversed relative to the reader
        actual = np.load(out / export_name(stem, name)).astype(np.float64)[::-1]
        np.testing.assert_allclose(
            actual, expected, rtol=1e-4, atol=0, equal_nan=True, err_msg=name
        )
