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

from itertools import islice

import numpy as np
import pytest

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
