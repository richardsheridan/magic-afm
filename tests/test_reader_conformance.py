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

from magic_afm.data_readers import FVFile, Image

# builder, volumes the file holds, and the trace the file should report: 1 for a file
# holding both a trace and a retrace volume, else None
BUILDERS = {
    "nid": (build_nid, 1, None),
    "arh5-ffm-trace": (partial(build_arh5_ffm, capture="trace"), 1, None),
    "arh5-ffm-retrace": (partial(build_arh5_ffm, capture="retrace"), 1, None),
    "arh5-ffm-both": (partial(build_arh5_ffm, capture="both"), 2, 1),
    "arh5-force-map": (build_arh5_force_map, 1, None),
    "pfc": (build_pfc, 1, None),
    "spm": (build_spm, 1, None),
    "ardf-trace": (partial(build_ardf, capture="trace"), 1, None),
    "ardf-retrace": (partial(build_ardf, capture="retrace"), 1, None),
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
def test_file_declares_trace_as_one_when_it_holds_trace_and_retrace_else_none(
    tmp_path, case
):
    """trace reads directly: 1 for a file with trace and retrace volumes, else None."""
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


def test_image_protocol_declares_units_as_str():
    """Image annotates units as str, the unit of the values get_image returns."""
    assert typing.get_type_hints(Image)["units"] is str


@pytest.mark.parametrize("case", BUILDERS)
def test_every_image_declares_a_nonempty_str_unit(tmp_path, case):
    """Every reader's images each declare a unit, whether or not the file states it."""
    builder, _, _ = BUILDERS[case]
    synthetic = builder(tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        units = {name: image.units for name, image in fvfile.images.items()}
    assert units
    for name, unit in units.items():
        assert isinstance(unit, str) and unit, (name, unit)


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


# halves of unequal length, on the readers whose files can hold them, and the ARDF
# slow-map reader, which a stopped scan goes through
RAGGED_NID_EXT = 64 - np.arange(15).reshape(5, 3) % 5 * 4
RAGGED_ARH5_EXT = 40 - np.arange(12).reshape(3, 4) % 3 * 2
RAGGED_ARH5_RET = 40 - np.arange(12).reshape(3, 4) % 4
CURVE_BUILDERS = {case: builder for case, (builder, _, _) in BUILDERS.items()} | {
    "nid-longer-extend": partial(build_nid, shape=(3, 5), points=(48, 40)),
    "nid-ragged": partial(
        build_nid, shape=(5, 3), points=(64, 40), ext_points=RAGGED_NID_EXT
    ),
    "nid-stopped": partial(build_nid, shape=(3, 5), points=(48, 40), curves_taken=7),
    "arh5-force-map-ragged": partial(
        build_arh5_force_map,
        shape=(3, 4),
        points=(40, 40),
        ext_points=RAGGED_ARH5_EXT,
        ret_points=RAGGED_ARH5_RET,
    ),
    "ardf-stopped": partial(build_ardf, shape=(4, 4), lines_taken=3),
}


def acquired_mask(synthetic_volume):
    if synthetic_volume.acquired is None:
        return np.ones(synthetic_volume.shape, dtype=bool)
    return synthetic_volume.acquired


def assert_halves_of_float32(curve, lengths=None):
    """curve is ((z_ext, z_ret), (d_ext, d_ret)) of 1-D float32 arrays."""
    assert type(curve) is tuple and len(curve) == 2, type(curve)
    for part in curve:
        assert type(part) is tuple and len(part) == 2, type(part)
        for half in part:
            assert isinstance(half, np.ndarray), type(half)
            assert half.ndim == 1, half.shape
            assert half.dtype == np.float32, half.dtype
    if lengths is not None:
        assert [[len(half) for half in part] for part in curve] == [lengths] * 2


@pytest.mark.parametrize("case", CURVE_BUILDERS)
def test_get_curve_returns_extend_and_retract_halves_each_as_long_as_stored(
    tmp_path, case
):
    """get_curve gives ((z_ext, z_ret), (d_ext, d_ret)), each half as stored."""
    synthetic = CURVE_BUILDERS[case](tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, synthetic.volumes, strict=True):
            acquired = acquired_mask(expected)
            for r, c in np.ndindex(volume.shape):
                lengths = None
                if acquired[r, c]:
                    lengths = [expected.ext_points[r, c], expected.ret_points[r, c]]
                assert_halves_of_float32(volume.get_curve(r, c), lengths)


@pytest.mark.parametrize("case", CURVE_BUILDERS)
def test_iter_curves_yields_index_and_the_curve_get_curve_returns(tmp_path, case):
    """iter_curves yields (index, curve) with get_curve's structure and values."""
    synthetic = CURVE_BUILDERS[case](tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        for volume in fvfile.volumes:
            for (r, c), curve in volume.iter_curves():
                want = volume.get_curve(r, c)
                assert_halves_of_float32(curve, [len(half) for half in want[0]])
                for part, want_part in zip(curve, want, strict=True):
                    for half, want_half in zip(part, want_part, strict=True):
                        np.testing.assert_array_equal(half, want_half)


@pytest.mark.parametrize("case", CURVE_BUILDERS)
def test_get_all_curves_equals_get_curve_cut_to_the_shortest_half(tmp_path, case):
    """get_all_curves holds the points nearest the surface, NaN where unacquired."""
    synthetic = CURVE_BUILDERS[case](tmp_path)
    with open_fvfile(synthetic.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, synthetic.volumes, strict=True):
            acquired = acquired_mask(expected)
            n = min(
                expected.ext_points[acquired].min(), expected.ret_points[acquired].min()
            )
            z, d = volume.get_all_curves()
            assert z.shape == d.shape == (*volume.shape, 2, n)
            for r, c in np.ndindex(volume.shape):
                if not acquired[r, c]:
                    assert np.isnan(z[r, c]).all() and np.isnan(d[r, c]).all()
                    continue
                (zext, zret), (dext, dret) = volume.get_curve(r, c)
                for got, ext, ret in ((z, zext, zret), (d, dext, dret)):
                    np.testing.assert_allclose(
                        got[r, c, 0], ext[len(ext) - n :], rtol=1e-6, atol=0
                    )
                    np.testing.assert_allclose(got[r, c, 1], ret[:n], rtol=1e-6, atol=0)
