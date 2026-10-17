"""The GUI's Demo file stands in for a reader's file, checked without a Tk window."""

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

import contextlib
from itertools import islice
from unittest.mock import MagicMock

import numpy as np
import pytest
import trio
import trio_parallel
from test_cli import DEFAULT_MAPS
from test_stopped_scans import RecordingProgressBar

from magic_afm.calculation import FitFix, FitMode

# objects other than a reader's parsed file that the GUI puts in GUIFVFile.fvfile, by
# their name in magic_afm.gui._impl
STAND_INS = {"demo": "DemoStub"}


def test_demo_file_reports_initial_parameters_with_zero_sync_dist_and_no_trace():
    """Demo starts at k 10, defl_sens 5, sync_dist 0 and trace None."""
    from magic_afm.gui import _impl as gui

    expected = gui.ForceVolumeParams(k=10.0, defl_sens=5.0, sync_dist=0, trace=None)
    assert gui.DemoForceVolumeFile().initial_parameters == expected


@pytest.mark.parametrize("member", ["sync_dist", "trace"])
@pytest.mark.parametrize("stand_in", STAND_INS)
def test_gui_stand_in_file_exposes_fvfile_protocol_member(stand_in, member):
    """A GUI stand-in for a reader's file has the FVFile protocol's member."""
    from magic_afm.gui import _impl as gui

    assert hasattr(getattr(gui, STAND_INS[stand_in])(), member)


def assert_halves_of_float32(curve):
    assert type(curve) is tuple and len(curve) == 2, type(curve)
    for part in curve:
        assert type(part) is tuple and len(part) == 2, type(part)
        for half in part:
            assert isinstance(half, np.ndarray), type(half)
            assert half.ndim == 1, half.shape
            assert half.dtype == np.float32, half.dtype


def test_demo_curves_are_extend_and_retract_halves_of_float32():
    """Demo curves are ((z_ext, z_ret), (d_ext, d_ret)) of 1-D float32 arrays."""
    from magic_afm.gui import _impl as gui

    demo = gui.DemoForceVolumeFile()
    assert_halves_of_float32(demo.get_curve(3, 5))
    for (r, c), curve in islice(demo.iter_curves(), 3):
        assert_halves_of_float32(curve)
        want = demo.get_curve(r, c)
        for part, want_part in zip(curve, want, strict=True):
            for half, want_half in zip(part, want_part, strict=True):
                np.testing.assert_array_equal(half, want_half)


def test_demo_property_maps_are_added_finite_and_named_without_direction(monkeypatch):
    """Fitting the Demo adds every default map as a finite CalcExt image, no suffix."""
    from magic_afm.gui import _impl as gui

    shape = (4, 4)
    monkeypatch.setattr(gui.DemoForceVolumeFile, "shape", shape)
    monkeypatch.setattr(gui, "tqdm_tk", RecordingProgressBar)
    monkeypatch.setattr(RecordingProgressBar, "instances", [])
    monkeypatch.setattr(trio_parallel, "run_sync", trio.to_thread.run_sync)

    opened_fvol = gui.DemoForceVolumeFile()
    params = opened_fvol.initial_parameters
    controller = gui.ForceVolumeController(MagicMock(), opened_fvol)
    controller.axesimage = MagicMock(**{"get_size.return_value": shape})
    controller.spinner_scope = contextlib.nullcontext
    # the GUI's option defaults, fitting the extend half as the GUI smoke test does
    options = gui.ForceCurveOptions(
        fit_mode=FitMode.EXTEND,
        fit_fix=FitFix.DEFAULTS,
        disp_kind=gui.DispKind.zd,
        k=params.k,
        k_sens=True,
        defl_sens=params.defl_sens,
        sync_dist=params.sync_dist,
        radius=20.0,
        tau=0.0,
        M=10 ** (6.0 - 9),
        lj_scale=2.0,
        vd=0.0,
        li_per=0.0,
        li_amp=0.0,
        drag=0.0,
        trace=params.trace,
    )
    with np.errstate(all="ignore"):
        trio.run(controller.calc_prop_map_response, options)

    calc = {name for name in opened_fvol.image_names if name.startswith("Calc")}
    assert calc == {"CalcExt" + m for m in DEFAULT_MAPS}
    not_finite = {
        name: int((~np.isfinite(opened_fvol.get_image(name))).sum()) for name in calc
    }
    assert not_finite == dict.fromkeys(calc, 0)
