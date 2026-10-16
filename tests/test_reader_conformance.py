"""Every reader's file object meets the FVFile protocol, checked on synthetic files."""

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

import typing
from functools import partial

import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.ardf import build_ardf
from synthetic_files.arh5 import build_arh5_ffm, build_arh5_force_map
from synthetic_files.nanoscope import build_pfc, build_spm
from synthetic_files.nid import build_nid

from magic_afm.data_readers import FVFile

# builder, volumes the file holds, and the trace the file should report: 1 (the first
# volume) for an ARDF holding both a trace and a retrace volume, else None
BUILDERS = {
    "nid": (build_nid, 1, None),
    "arh5-ffm-trace": (partial(build_arh5_ffm, capture="trace"), 1, None),
    # the ARH5 reader does not report which of its volumes is the trace
    "arh5-ffm-both": (partial(build_arh5_ffm, capture="both"), 2, None),
    "arh5-force-map": (build_arh5_force_map, 1, None),
    "pfc": (build_pfc, 1, None),
    "spm": (build_spm, 1, None),
    "ardf-trace": (partial(build_ardf, capture="trace"), 1, None),
    "ardf-both": (partial(build_ardf, capture="both"), 2, 1),
}


@pytest.mark.parametrize("case", BUILDERS)
def test_file_declares_sync_dist_as_the_file_states_it_or_none(tmp_path, case):
    """sync_dist reads directly: the .pfc's lag in points, None for other formats."""
    builder, _, _ = BUILDERS[case]
    synthetic = builder(tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        sync_dist = fvfile.sync_dist
    if synthetic.sync_dist is None:
        assert sync_dist is None
    else:
        np.testing.assert_allclose(sync_dist, synthetic.sync_dist, rtol=1e-12)


@pytest.mark.parametrize("case", BUILDERS)
def test_file_declares_trace_as_one_for_two_volume_ardf_else_none(tmp_path, case):
    """trace reads directly: 1 for an ARDF with trace and retrace volumes, else None."""
    builder, expected_volumes, expected = BUILDERS[case]
    synthetic = builder(tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        trace = fvfile.trace
        n_volumes = len(fvfile.volumes)
    assert n_volumes == expected_volumes
    assert trace == expected


def test_fvfile_protocol_declares_sync_dist_and_trace():
    """FVFile annotates sync_dist as float | None and trace as int | None."""
    hints = typing.get_type_hints(FVFile)
    assert hints["sync_dist"] == float | None
    assert hints["trace"] == int | None


@pytest.mark.parametrize("case", BUILDERS)
def test_get_all_curves_returns_one_float32_array_z_d_by_grid_by_halves(tmp_path, case):
    """Each volume's get_all_curves is one float32 (2, rows, cols, 2, n) array."""
    builder, _, _ = BUILDERS[case]
    synthetic = builder(tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        for volume in fvfile.volumes:
            result = volume.get_all_curves()
            assert isinstance(result, np.ndarray), type(result)
            assert result.dtype == np.float32
            assert result.shape[:4] == (2, *volume.shape, 2)
