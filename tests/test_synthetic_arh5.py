"""Synthetic Asylum ARH5 files read back through the public reader surface."""

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

# The files store float32 metres and the reader multiplies by 1e9 in float32, so curves
# match the stored values in nm to one float32 rounding (rtol 1e-7). Images are returned
# as stored and match exactly. Metadata parsed from decimal text match to double
# rounding.

from pathlib import Path

import h5py
import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.arh5 import build_arh5_ffm, build_arh5_force_map

from magic_afm.calculation import FitMode, prepare_force_curve

ARH5_RELPATH = "test_cases/PDMS_SAA/ffm/rtrace0001.h5"

# grid (rows, cols), points per half curve, captured directions
FFM_CASES = {
    "square-16-both": ((4, 4), 16, "both"),
    "square-100-both": ((4, 4), 100, "both"),
    "wide-16-both": ((3, 5), 16, "both"),
    "wide-100-both": ((3, 5), 100, "both"),
    "tall-16-trace": ((5, 3), 16, "trace"),
    "tall-16-retrace": ((5, 3), 16, "retrace"),
}

RAGGED_EXT = 40 - np.arange(12).reshape(3, 4) % 3 * 2
RAGGED_RET = 40 - np.arange(12).reshape(3, 4) % 4
RAGGED_RET_EVEN = 40 - np.arange(12).reshape(3, 4) % 3
# grid, extend and retract points, other builder arguments
FORCE_MAP_CASES = {
    "square-32": ((4, 4), (32, 32), {}),
    "wide-20": ((3, 5), (20, 20), {}),
    "ragged": (
        (3, 4),
        (40, 40),
        {"ext_points": RAGGED_EXT, "ret_points": RAGGED_RET},
    ),
    # shortest halves 36 and 38, an even total that a split in the middle gets wrong
    "ragged-even": (
        (3, 4),
        (40, 40),
        {"ext_points": RAGGED_EXT, "ret_points": RAGGED_RET_EVEN},
    ),
}
# Cases with every curve as long as the shortest of its halves. When the shortest extend
# and retract halves differ, as in "ragged", t_step counts only the shortest halves'
# points.
UNIFORM_FORCE_MAP_CASES = ["square-32", "wide-20"]


def assert_float32_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0)


def assert_same_curve(curve, expected):
    (zext, zret), (dext, dret) = curve
    (ezext, ezret), (edext, edret) = expected
    assert_float32_close(zext, ezext)
    assert_float32_close(zret, ezret)
    assert_float32_close(dext, edext)
    assert_float32_close(dret, edret)


def half_shapes(curve):
    """Shapes of ((zext, zret), (dext, dret)), or of whatever curve holds instead."""
    return [[np.shape(half) for half in part] for part in curve]


def assert_metadata(fvfile, synthetic):
    assert [v.shape for v in fvfile.volumes] == [v.shape for v in synthetic.volumes]
    np.testing.assert_allclose(fvfile.k, synthetic.k, rtol=1e-12)
    np.testing.assert_allclose(fvfile.defl_sens, synthetic.defl_sens, rtol=1e-12)
    np.testing.assert_allclose(fvfile.scansize, synthetic.scansize, rtol=1e-12)


def assert_images(fvfile, synthetic):
    assert sorted(fvfile.images) == sorted(synthetic.images)
    for name, image in fvfile.images.items():
        assert image.shape == synthetic.images[name].shape
        np.testing.assert_array_equal(image.get_image(), synthetic.images[name])


@pytest.fixture(params=list(FFM_CASES))
def ffm(request, tmp_path):
    shape, points, capture = FFM_CASES[request.param]
    return build_arh5_ffm(tmp_path, shape, points, capture=capture)


def test_arh5_ffm_metadata_round_trips(ffm):
    with open_fvfile(ffm.path) as fvfile:
        assert_metadata(fvfile, ffm)
        np.testing.assert_allclose(fvfile.t_step, ffm.t_step, rtol=1e-12)


# a retrace-only capture is left out: the reader calls its one volume Trace
@pytest.mark.parametrize("capture", ["trace", "both"])
def test_arh5_ffm_volumes_are_named_by_direction(tmp_path, capture):
    synthetic = build_arh5_ffm(tmp_path, capture=capture)
    with open_fvfile(synthetic.path) as fvfile:
        names = [v.name for v in fvfile.volumes]
    assert names == [v.name for v in synthetic.volumes]


def test_arh5_ffm_get_curve_returns_the_stored_curves(ffm):
    with open_fvfile(ffm.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, ffm.volumes, strict=True):
            rows, cols = expected.shape
            for r in range(rows):
                for c in range(cols):
                    assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))


def test_arh5_ffm_iter_curves_goes_row_by_row(ffm):
    with open_fvfile(ffm.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, ffm.volumes, strict=True):
            assert list(volume.iter_indices()) == list(expected.order)
            seen = []
            for (r, c), curve in volume.iter_curves():
                seen.append((r, c))
                assert_same_curve(curve, expected.curve(r, c))
            assert seen == list(expected.order)


def test_arh5_ffm_get_all_curves_returns_every_curve(ffm):
    with open_fvfile(ffm.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, ffm.volumes, strict=True):
            z, d = volume.get_all_curves()
            assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
            assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


def test_arh5_ffm_images_round_trip_exactly(ffm):
    with open_fvfile(ffm.path) as fvfile:
        assert_images(fvfile, ffm)


@pytest.fixture(params=list(FORCE_MAP_CASES))
def force_map(request, tmp_path):
    shape, points, kwargs = FORCE_MAP_CASES[request.param]
    return build_arh5_force_map(tmp_path, shape, points, **kwargs)


def test_arh5_force_map_metadata_round_trips(force_map):
    with open_fvfile(force_map.path) as fvfile:
        assert [v.name for v in fvfile.volumes] == ["FMAP"]
        assert_metadata(fvfile, force_map)


@pytest.mark.parametrize("case", UNIFORM_FORCE_MAP_CASES)
def test_arh5_force_map_t_step_round_trips(tmp_path, case):
    shape, points, kwargs = FORCE_MAP_CASES[case]
    synthetic = build_arh5_force_map(tmp_path, shape, points, **kwargs)
    with open_fvfile(synthetic.path) as fvfile:
        np.testing.assert_allclose(fvfile.t_step, synthetic.t_step, rtol=1e-12)


def test_arh5_force_map_indices_cover_the_grid_once(force_map):
    with open_fvfile(force_map.path) as fvfile:
        indices = list(fvfile.volumes[0].iter_indices())
    assert sorted(indices) == list(np.ndindex(force_map.volumes[0].shape))


def test_arh5_force_map_get_curve_returns_each_half_as_stored(force_map):
    expected = force_map.volumes[0]
    rows, cols = expected.shape
    with open_fvfile(force_map.path) as fvfile:
        volume = fvfile.volumes[0]
        for r in range(rows):
            for c in range(cols):
                assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))


def test_arh5_force_map_iter_curves_yields_what_get_curve_returns(force_map):
    with open_fvfile(force_map.path) as fvfile:
        volume = fvfile.volumes[0]
        seen = []
        for (r, c), curve in volume.iter_curves():
            seen.append((r, c))
            want = volume.get_curve(r, c)
            assert half_shapes(curve) == half_shapes(want), (r, c)
            for part, want_part in zip(curve, want, strict=True):
                for half, want_half in zip(part, want_part, strict=True):
                    np.testing.assert_array_equal(half, want_half)
    assert sorted(seen) == list(np.ndindex(force_map.volumes[0].shape))


# Left out: fit mode both, and curves long enough to be resampled, which with halves of
# different lengths prepare_force_curve cannot join into one array.
@pytest.mark.parametrize(
    "fit_mode", [FitMode.EXTEND, FitMode.RETRACT], ids=lambda m: m.name.lower()
)
def test_arh5_force_map_iter_curves_feed_prepare_force_curve(force_map, fit_mode):
    """The CLI passes each yielded curve to prepare_force_curve; it takes one half."""
    expected = force_map.volumes[0]
    half = 0 if fit_mode == FitMode.EXTEND else 1
    with open_fvfile(force_map.path) as fvfile:
        for rc, curve in fvfile.volumes[0].iter_curves():
            z, d, split, out_rc = prepare_force_curve((rc, curve), fit_mode, 1.0)
            assert out_rc == rc
            assert split is None
            (ezext, ezret), (edext, edret) = expected.curve(*rc)
            assert_float32_close(z, (ezext, ezret)[half])
            assert_float32_close(d, (edext, edret)[half])


def test_arh5_force_map_get_all_curves_cuts_both_halves_to_the_shortest(force_map):
    """Each half is cut to the shortest extend or retract half on the map."""
    expected = force_map.volumes[0]
    n = int(min(expected.ext_points.min(), expected.ret_points.min()))
    rows, cols = expected.shape
    with open_fvfile(force_map.path) as fvfile:
        curves = fvfile.volumes[0].get_all_curves()
    assert isinstance(curves, np.ndarray), type(curves)
    assert curves.dtype == np.float32
    assert curves.shape == (2, rows, cols, 2, n)
    z, d = curves
    for r in range(rows):
        for c in range(cols):
            (ezext, ezret), (edext, edret) = expected.curve(r, c, n, n)
            assert_float32_close(z[r, c], np.stack([ezext, ezret]))
            assert_float32_close(d[r, c], np.stack([edext, edret]))


def test_arh5_force_map_images_round_trip_exactly(force_map):
    with open_fvfile(force_map.path) as fvfile:
        assert_images(fvfile, force_map)


def attribute_types(obj):
    """Each attribute's type class, character set, shape and, if strings, value."""
    types = {}
    for name in obj.attrs:
        attr = obj.attrs.get_id(name)
        dtype = attr.get_type()
        is_string = isinstance(dtype, h5py.h5t.TypeStringID)
        types[name] = (
            dtype.get_class(),
            dtype.get_cset() if is_string else None,
            attr.shape,
            obj.attrs[name].tolist() if is_string and attr.shape else None,
        )
    return types


def h5_tree(h5file):
    """Every group and dataset with its shape, dtype, chunks and attribute types."""
    tree = {"/": attribute_types(h5file)}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            tree[name] = (obj.shape, obj.dtype.str, obj.chunks, attribute_types(obj))
        else:
            tree[name] = attribute_types(obj)

    h5file.visititems(visit)
    return tree


def structure(fvfile):
    """What a reader makes of an ARH5 file, without the values."""
    return {
        "class": type(fvfile),
        "volumes": [(type(v), v.shape) for v in fvfile.volumes],
        "volume names": [v.name for v in fvfile.volumes],
        "curve": [
            tuple(np.shape(half) for half in v.get_curve(0, 0)) for v in fvfile.volumes
        ],
        "all curves": [
            tuple(np.shape(a) for a in v.get_all_curves()) for v in fvfile.volumes
        ],
        "images": {
            name: (image.shape, image.get_image().dtype)
            for name, image in fvfile.images.items()
        },
    }


@pytest.mark.slow
def test_synthetic_arh5_has_the_structure_of_the_real_file(real_data_file, tmp_path):
    with h5py.File(Path(real_data_file(ARH5_RELPATH)), "r") as real_h5:
        real_tree = h5_tree(real_h5)
    with open_fvfile(Path(real_data_file(ARH5_RELPATH))) as real:
        real_structure = structure(real)
        real_note = real.headers
        shape = real.volumes[0].shape
        points = real.volumes[0].get_curve(0, 0)[0].shape[-1]
    # the real file kept only the retrace
    synthetic = build_arh5_ffm(tmp_path, shape, points, capture="retrace")
    with open_fvfile(synthetic.path) as fvfile:
        assert structure(fvfile) == real_structure
        # every note key the builder writes is one the real file has
        assert set(fvfile.headers) <= set(real_note), set(fvfile.headers) - set(
            real_note
        )
    with h5py.File(synthetic.path, "r") as h5:
        assert h5_tree(h5) == real_tree
