"""fit_force_curve takes only the arguments it uses; its callers still fit the same."""

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

import inspect

import numpy as np
import pytest
import test_goldens as t2
from attrs import asdict
from synthetic_files import open_fvfile
from synthetic_files.nid import build_nid
from test_cli import DEFAULT_MAPS, EXPORT_FIELDS, export_name, invoke_cli, map_tolerance

from magic_afm import calculation
from magic_afm.calculation import FitFix, FitMode

# Keys the CLI and GUI carry in their option dicts that fit_force_curve does not use,
# and one that nothing uses.
NOT_FIT_ARGUMENTS = (
    "bogus",
    "defl_sens",
    "sync_dist",
    "trace",
    "k_sens",
    "disp_kind",
    "fix_radius",
)


def fit_args(mode):
    """z, d and the keyword arguments of a valid fit of a Demo curve in mode."""
    curve = t2.demo_curve(**t2.SYNTHETIC_CASES["demo_r0_c0"])
    z, d, split, _ = t2.preprocess(((0, 0), curve), mode)
    kwargs = dict(k=t2.DEMO_K, fit_mode=mode, split=split, **t2.FIT_OPTIONS)
    return z, d, kwargs


def gui_options(mode, k):
    """The GUI's options at its defaults, as test_stopped_scans.run_gui_fit sets them."""
    from magic_afm.gui import _impl as gui

    return gui.ForceCurveOptions(
        fit_mode=mode,
        fit_fix=FitFix.DEFAULTS,
        disp_kind=gui.DispKind.zd,
        k=k,
        k_sens=True,
        defl_sens=5.0,
        sync_dist=0,
        radius=20.0,
        tau=0.0,
        M=1.0,
        lj_scale=2.0,
        vd=0.0,
        li_per=0.0,
        li_amp=0.0,
        drag=0.0,
        trace=None,
    )


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("name", NOT_FIT_ARGUMENTS)
def test_fit_force_curve_rejects_a_keyword_it_does_not_use(name, mode):
    z, d, kwargs = fit_args(mode)
    with pytest.raises(TypeError, match=name):
        calculation.fit_force_curve(z, d, **kwargs, **{name: 1})


def test_fit_force_curve_rejects_a_misspelt_parameter_in_place_of_the_real_one():
    z, d, kwargs = fit_args(FitMode.RETRACT)
    kwargs["radus"] = kwargs.pop("radius")
    with pytest.raises(TypeError, match="radus"):
        calculation.fit_force_curve(z, d, **kwargs)


def test_fit_force_curve_rejects_a_misspelt_optional_parameter():
    z, d, kwargs = fit_args(FitMode.RETRACT)
    with pytest.raises(TypeError, match="nan_on_eror"):
        calculation.fit_force_curve(z, d, nan_on_eror=True, **kwargs)


def test_fit_force_curve_names_split_and_takes_no_var_keyword():
    parameters = inspect.signature(calculation.fit_force_curve).parameters
    assert "split" in parameters
    kinds = [p.kind for p in parameters.values()]
    assert inspect.Parameter.VAR_KEYWORD not in kinds


def test_cli_run_on_a_synthetic_file_matches_a_direct_fit(tmp_path):
    """Every map a default CLI run writes equals a fit of the reader's curves."""
    (tmp_path / "in").mkdir()
    synthetic = build_nid(tmp_path / "in")
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        want = {
            group: {name: np.full(volume.shape, np.nan) for name in names}
            for group, names in t2.FIELDS.items()
        }
        for rc_curve in volume.iter_curves():
            pre = t2.preprocess(rc_curve, FitMode.RETRACT)
            fitted = t2.fit_one(pre, FitMode.RETRACT, fvfile.k)
            for group, values in fitted.items():
                for name, value in values.items():
                    want[group][name][rc_curve[0]] = value
    assert np.isfinite(want["parms"]["M"]).any(), "no curve fits"
    out = tmp_path / "out"
    result = invoke_cli(synthetic.path, out, ["--stop-on-error"])
    assert result.exit_code == 0, result.output
    stem = synthetic.path.stem
    for m in DEFAULT_MAPS:
        # the CLI exports rows reversed relative to the reader
        actual = np.load(out / export_name(stem, m)).astype(np.float64)[::-1]
        group, field, factor = EXPORT_FIELDS[m]
        rtol, atol = map_tolerance(m)
        np.testing.assert_allclose(
            actual,
            factor * want[group][field],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=m,
        )


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
def test_fit_curve_properties_with_the_gui_map_options_matches_the_fit_options(mode):
    """The dict the GUI map run passes fits the same as the fit options alone."""
    curve = t2.demo_curve(**t2.SYNTHETIC_CASES["demo_r0_c0"])
    pre = t2.preprocess(((0, 0), curve), mode)
    options = asdict(gui_options(mode, t2.DEMO_K))
    del options["disp_kind"]
    with np.errstate(all="ignore"):
        actual = calculation.fit_curve_properties(pre, **options)
        want = calculation.fit_curve_properties(
            pre, fit_mode=mode, k=t2.DEMO_K, **t2.DEFAULT_OPTIONS
        )
    for a, w in zip(actual[1:], want[1:], strict=True):
        np.testing.assert_allclose(a.item(), w.item(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("mode", t2.MODES, ids=lambda m: m.name)
def test_gui_readout_fit_matches_a_direct_fit(mode):
    """The GUI's force-curve readout fits as fit_force_curve does on the same curve."""
    from magic_afm.gui import _impl as gui

    (zext, zret), (dext, dret) = t2.demo_curve(**t2.SYNTHETIC_CASES["demo_r0_c0"])
    # under RESAMPLE_NPTS points, so the readout does not resample
    zxr, dxr = (zext[::2], zret[::2]), (dext[::2], dret[::2])
    assert sum(map(len, zxr)) <= calculation.RESAMPLE_NPTS
    options = gui_options(mode, t2.DEMO_K)
    fvparams = gui.ForceVolumeParams(
        k=t2.DEMO_K, defl_sens=5.0, sync_dist=0, trace=None
    )
    if mode == FitMode.EXTEND:
        z, d, split = zxr[0], dxr[0], None
    elif mode == FitMode.RETRACT:
        z, d, split = zxr[1], dxr[1], None
    else:
        z, d, split = np.concatenate(zxr), np.concatenate(dxr), len(zxr[0])
    with np.errstate(all="ignore"):
        data = gui.calculate_force_data(zxr, dxr, 1e-3, options, fvparams)
        parms, parms_err, sse, _ = calculation.fit_force_curve(
            z, d, k=t2.DEMO_K, fit_mode=mode, split=split, **t2.FIT_OPTIONS
        )
    np.testing.assert_allclose(data.parms.item(), parms.item(), rtol=0, atol=0)
    np.testing.assert_allclose(
        data.parms_err.item(), parms_err.item(), rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(data.sse, sse, rtol=0, atol=0)


@pytest.mark.parametrize("given", ["missing", "None"])
def test_fit_force_curve_in_both_mode_requires_split(given):
    """BOTH without a split would fit the whole curve twice; it raises instead."""
    z, d, kwargs = fit_args(FitMode.BOTH)
    if given == "missing":
        del kwargs["split"]
    else:
        kwargs["split"] = None
    with pytest.raises(ValueError, match="split"):
        calculation.fit_force_curve(z, d, **kwargs)


@pytest.mark.parametrize(
    "mode", [FitMode.EXTEND, FitMode.RETRACT], ids=lambda m: m.name
)
def test_fit_force_curve_in_one_segment_mode_needs_no_split(mode):
    z, d, kwargs = fit_args(mode)
    del kwargs["split"]
    with np.errstate(all="ignore"):
        parms, *_ = calculation.fit_force_curve(z, d, **kwargs)
    assert np.all(np.isfinite(parms.item()))
