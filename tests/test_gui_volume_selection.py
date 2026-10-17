"""The GUI picks a volume by direction name, checked without a Tk window."""

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

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.ardf import build_ardf
from synthetic_files.arh5 import build_arh5_ffm, build_arh5_force_map
from synthetic_files.nanoscope import build_pfc, build_spm
from synthetic_files.nid import build_nid

from magic_afm._options import TraceChoice

DOE4_RELPATH = "test_cases/DOE4_20nN_80nm.0_00012.spm"

DIRECTED_BUILDERS = {
    f"{fmt}-{capture}": partial(builder, capture=capture)
    for fmt, builder in (("ardf", build_ardf), ("arh5", build_arh5_ffm))
    for capture in ("trace", "retrace", "both")
}
UNDIRECTED_BUILDERS = {
    "pfc": build_pfc,
    "spm": build_spm,
    "nid": build_nid,
    "arh5-force-map": build_arh5_force_map,
}
DIRECTIONS = {TraceChoice.TRACE: "Trace", TraceChoice.RETRACE: "Retrace"}
CAPTURED = {"trace": [TraceChoice.TRACE], "retrace": [TraceChoice.RETRACE]}
CAPTURED["both"] = CAPTURED["trace"] + CAPTURED["retrace"]
# (case, trace) for every direction each directed file holds
HELD = [
    (case, trace)
    for case in DIRECTED_BUILDERS
    for trace in CAPTURED[case.split("-")[1]]
]


def trace_id(value):
    """Test ids that name the direction instead of its integer value."""
    return value.name.lower() if isinstance(value, TraceChoice) else None


def assert_same_curve(curve, expected, rtol=0.0):
    """Asylum files store float32 metres, read in nm to one float32 rounding."""
    for part, expected_part in zip(curve, expected, strict=True):
        for half, expected_half in zip(part, expected_part, strict=True):
            np.testing.assert_allclose(
                half, expected_half, rtol=rtol, atol=0, equal_nan=True
            )


def gui_curve(fvfile, trace):
    """The GUI's curve (0, 0) for trace, at the file's own sync_dist."""
    from magic_afm.gui import _impl as gui

    return gui.GUIFVFile(fvfile).get_curve(0, 0, trace, fvfile.sync_dist)


@pytest.mark.parametrize(("case", "trace"), HELD, ids=trace_id)
def test_gui_selects_the_volume_named_for_the_requested_direction(
    tmp_path, case, trace
):
    """TRACE gets the volume named Trace and RETRACE the one named Retrace."""
    synthetic = DIRECTED_BUILDERS[case](tmp_path)
    named = {v.name: v for v in synthetic.volumes}
    with open_fvfile(synthetic.path) as fvfile:
        curve = gui_curve(fvfile, trace)
    assert_same_curve(curve, named[DIRECTIONS[trace]].curve(0, 0), rtol=1e-7)


@pytest.mark.parametrize(
    ("case", "trace"),
    [
        ("ardf-retrace", TraceChoice.TRACE),
        ("arh5-retrace", TraceChoice.TRACE),
        ("ardf-trace", TraceChoice.RETRACE),
        ("arh5-trace", TraceChoice.RETRACE),
    ],
    ids=trace_id,
)
def test_gui_fails_for_a_direction_the_file_lacks(tmp_path, case, trace):
    """A directed file without the requested direction does not give another one."""
    synthetic = DIRECTED_BUILDERS[case](tmp_path)
    with open_fvfile(synthetic.path) as fvfile, pytest.raises(Exception):
        gui_curve(fvfile, trace)


@pytest.mark.parametrize("case", [*DIRECTED_BUILDERS, *UNDIRECTED_BUILDERS])
def test_gui_selects_the_first_volume_when_trace_is_none(tmp_path, case):
    builder = {**DIRECTED_BUILDERS, **UNDIRECTED_BUILDERS}[case]
    synthetic = builder(tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        curve = gui_curve(fvfile, None)
        expected = fvfile.volumes[0].get_curve(0, 0)
    assert_same_curve(curve, expected)


@pytest.mark.parametrize("trace", DIRECTIONS, ids=trace_id)
@pytest.mark.parametrize("case", UNDIRECTED_BUILDERS)
def test_gui_ignores_trace_on_a_file_without_a_direction(tmp_path, case, trace):
    """A file with no direction gives its first volume whatever trace asks."""
    synthetic = UNDIRECTED_BUILDERS[case](tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        curve = gui_curve(fvfile, trace)
        expected = fvfile.volumes[0].get_curve(0, 0)
    assert_same_curve(curve, expected)


@pytest.mark.slow
@pytest.mark.parametrize("trace", DIRECTIONS, ids=trace_id)
def test_gui_ignores_trace_on_two_undirected_volumes(real_data_file, trace):
    """DOE4's two Deflection Error volumes have no direction: the first is used."""
    with open_fvfile(Path(real_data_file(DOE4_RELPATH))) as fvfile:
        curve = gui_curve(fvfile, trace)
        expected = fvfile.volumes[0].get_curve(0, 0)
    assert_same_curve(curve, expected)
