"""The GUI's single-curve readout matches the map path, checked without a Tk window."""

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

import io

import numpy as np
import pytest
from attrs import asdict

from magic_afm import calculation
from magic_afm._options import dump_options, load_options
from magic_afm.calculation import FitFix, FitMode

FIT_MODES = [FitMode.EXTEND, FitMode.RETRACT, FitMode.BOTH]
PIXELS = [(0, 0), (3, 5)]
# Demo curves have 1000 points: the default resamples them, 2048 leaves them be
RESAMPLE_NPTS = [calculation.RESAMPLE_NPTS, 2048]
FIT_OPTIONS = dict(
    fit_fix=FitFix.DEFAULTS,
    radius=20.0,
    tau=0.0,
    M=1.0,
    lj_scale=2.0,
    vd=0.0,
    li_per=0.0,
    li_amp=0.0,
    drag=0.0,
)


def gui_options(fvfile, **overrides):
    """The GUI's options for fvfile at its own k, defl_sens, sync_dist and trace."""
    from magic_afm.gui import _impl as gui

    initial = fvfile.initial_parameters
    options = dict(
        fit_mode=FitMode.RETRACT,
        disp_kind=gui.DispKind.zd,
        k=initial.k,
        k_sens=True,
        defl_sens=initial.defl_sens,
        sync_dist=initial.sync_dist,
        trace=initial.trace,
        **FIT_OPTIONS,
    )
    return gui.ForceCurveOptions(**{**options, **overrides})


def map_pixel(fvfile, rc, options):
    """(properties, parms, parms_err) of pixel rc as the GUI's property map fits it."""
    curve = fvfile.get_curve(*rc, options.trace, options.sync_dist)
    z_d_s_rc = calculation.prepare_force_curve(
        (rc, curve),
        s_ratio=options.defl_sens / fvfile.initial_parameters.defl_sens,
        fit_mode=options.fit_mode,
        resample_npts=options.resample_npts,
    )
    fit_options = asdict(options)
    del fit_options["disp_kind"]
    with np.errstate(all="ignore"):
        _, properties, parms, parms_err = calculation.fit_curve_properties(
            z_d_s_rc, **fit_options
        )
    return properties, parms, parms_err


def readout(fvfile, rc, options):
    """The GUI's single-curve readout of pixel rc."""
    from magic_afm.gui import _impl as gui

    curve = fvfile.get_curve(*rc, options.trace, options.sync_dist)
    with np.errstate(all="ignore"):
        return gui.calculate_force_data(
            *curve, fvfile.fvfile.t_step, options, fvfile.initial_parameters
        )


def readout_properties(data):
    """The readout's derived quantities under the map's property names."""
    return {
        "Deflection": data.defl,
        "Indentation": data.ind,
        "ContactRadius": data.a_c,
        "TrueHeight": -data.z_tru,
        "IndentationRatio": data.defl / data.ind,
        "SensIndMod_k": data.sens,
        "SumSquaredError": data.sse,
    }


def as_float32(values):
    return np.array([np.asarray(v).item() for v in values], dtype=np.float32)


@pytest.mark.parametrize("resample_npts", RESAMPLE_NPTS)
@pytest.mark.parametrize("s_ratio", [1.0, 1.1])
@pytest.mark.parametrize("k_sens", [True, False], ids=["k_sens", "no_k_sens"])
@pytest.mark.parametrize("fit_mode", FIT_MODES, ids=lambda m: m.name.lower())
@pytest.mark.parametrize("rc", PIXELS, ids=str)
def test_readout_reports_the_map_pixel_fit_and_properties(
    rc, fit_mode, k_sens, s_ratio, resample_npts
):
    """The readout's parms, errors and properties equal the map's for that pixel."""
    from magic_afm.gui import _impl as gui

    demo = gui.DemoForceVolumeFile()
    options = gui_options(
        demo,
        fit_mode=fit_mode,
        k_sens=k_sens,
        defl_sens=demo.initial_parameters.defl_sens * s_ratio,
        resample_npts=resample_npts,
    )
    properties, parms, parms_err = map_pixel(demo, rc, options)
    assert properties is not None, "the map failed to fit this Demo curve"
    data = readout(demo, rc, options)

    names = calculation.PROPERTY_DTYPE.names
    got = readout_properties(data)
    np.testing.assert_allclose(
        as_float32(got[name] for name in names),
        as_float32(properties.item()),
        rtol=0,
        atol=0,
        err_msg=f"properties {names}",
    )
    for group, got_parms, expected in (
        ("parms", data.parms, parms),
        ("parms_err", data.parms_err, parms_err),
    ):
        np.testing.assert_allclose(
            as_float32(got_parms.item()),
            as_float32(expected.item()),
            rtol=0,
            atol=0,
            err_msg=f"{group} {calculation.PARMS_DTYPE.names}",
        )


@pytest.mark.parametrize("fit_mode", FIT_MODES, ids=lambda m: m.name.lower())
def test_readout_raises_on_an_unfittable_curve_the_map_writes_as_nan(fit_mode):
    """An all-zero curve is NaN in the map, but the readout raises the fit's error."""
    from magic_afm.gui import _impl as gui

    class ZeroCurveFile(gui.DemoForceVolumeFile):
        def get_curve(self, r, c, trace=None, sync_dist=None):
            zeros = np.zeros(500, dtype=np.float32)
            return (zeros, zeros.copy()), (zeros.copy(), zeros.copy())

    fvfile = ZeroCurveFile()
    options = gui_options(fvfile, fit_mode=fit_mode)
    assert map_pixel(fvfile, (0, 0), options) == (None, None, None)
    with pytest.raises(ValueError):
        readout(fvfile, (0, 0), options)


ROUND_TRIP_OPTIONS = {
    "set": dict(
        fit_mode=FitMode.BOTH,
        fit_fix=FitFix.RADIUS | FitFix.LJ_SCALE | FitFix.DRAG,
        k=0.462,
        k_sens=False,
        defl_sens=55.471,
        sync_dist=68.0125,
        radius=12.5,
        tau=0.35,
        M=10 ** (6.5 - 9),
        lj_scale=-1.5,
        vd=0.125,
        li_per=640.0,
        li_amp=0.3,
        drag=1e-3,
        trace=1,
        resample_npts=1024,
    ),
    "null": dict(sync_dist=None, trace=None),
}


@pytest.mark.parametrize("case", ROUND_TRIP_OPTIONS)
def test_gui_options_round_trip_through_the_options_json(case):
    """ForceCurveOptions written as the GUI writes them read back equal."""
    from magic_afm.gui import _impl as gui

    options = gui_options(gui.DemoForceVolumeFile(), **ROUND_TRIP_OPTIONS[case])
    # as AsyncNavigationToolbar2Tk._aexport_calculations writes them
    written = asdict(options)
    del written["disp_kind"]
    loaded = load_options(io.StringIO(dump_options(written)))
    assert gui.ForceCurveOptions(**loaded, disp_kind=options.disp_kind) == options
