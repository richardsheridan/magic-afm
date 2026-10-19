"""Readers on files whose size matters: long headers, 4 GiB offsets, large grids."""

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

# Tolerances follow the per-format tests: ARDF and ARH5 store float32 metres, which the
# readers scale to nm in float32 (rtol 1e-7); NID stores scaled integers, decoded in
# float32 (rtol 1e-6 and an atol of 1e-6 of the largest magnitude).

import os
import struct
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.ardf import build_ardf
from synthetic_files.arh5 import build_arh5_force_map
from synthetic_files.nid import build_nid
from test_synthetic_ardf import (
    assert_images_where_held,
    assert_metadata,
    assert_same_curve,
)

from magic_afm.data_readers import _SEARCH_CHUNK_BYTES, NIDFile, _find_in_buffer


def assert_nid_close(actual, expected):
    expected = np.asarray(expected)
    atol = 1e-6 * np.nanmax(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=atol, equal_nan=True)


def stack_curves(volume) -> np.ndarray:
    """A synthetic volume's curves as (2, rows, cols, 2, n), NaN where not held."""
    return np.stack(
        [
            np.stack([volume.zext, volume.zret], axis=-2),
            np.stack([volume.dext, volume.dret], axis=-2),
        ]
    )


#################################################
# A header ending at the search window's edge
#################################################

# NanoscopeFile.parse and NIDFile.parse find the end of their header with
# _find_in_buffer. The CLI and the GUI open both formats with SUFFIX_FVFILE_MAP's
# opener, an mmap, which _find_in_buffer searches with the mmap's own find; only a
# buffer with no find of its own, such as a memoryview handed to parse directly, is
# searched in overlapping windows. The Nanoscope marker is one byte and cannot straddle
# a window's end; the NID marker #! can.
WINDOW = _SEARCH_CHUNK_BYTES
# a map large enough that its header alone runs to most of the first window
NID_GRID = (28, 28)
MARKER_OFFSETS = {
    "inside-the-first-window": WINDOW - 2,
    "straddling-its-end": WINDOW - 1,
    "just-past-it": WINDOW,
}


def assert_reads_known_nid(fvfile, synthetic):
    expected = synthetic.volumes[0]
    (volume,) = fvfile.volumes
    assert volume.shape == expected.shape
    np.testing.assert_allclose(fvfile.k, synthetic.k, rtol=1e-12)
    np.testing.assert_allclose(fvfile.defl_sens, synthetic.defl_sens, rtol=1e-12)
    np.testing.assert_allclose(fvfile.t_step, synthetic.t_step, rtol=1e-12)
    np.testing.assert_allclose(fvfile.scansize, synthetic.scansize, rtol=1e-12)
    # the datasets start just past the marker, so a misplaced marker garbles them
    assert_nid_close(volume.get_all_curves(), stack_curves(expected))
    assert sorted(fvfile.images) == sorted(synthetic.images)
    for name, image in fvfile.images.items():
        assert image.units == synthetic.image_units[name]
        assert_nid_close(image.get_image(), synthetic.images[name])


def build_nid_with_marker_at(directory, offset):
    synthetic = build_nid(directory, NID_GRID, (16, 16), header_bytes=offset)
    with open(synthetic.path, "rb") as f:
        head = f.read(offset + 2)
    assert head.find(b"#!") == offset
    return synthetic


@pytest.mark.slow
@pytest.mark.parametrize("offset", MARKER_OFFSETS.values(), ids=MARKER_OFFSETS)
def test_nid_header_ending_at_the_search_window_reads_through_the_cli_opener(
    tmp_path, offset
):
    """The CLI's mmap finds #! at the first window's end and reads the datasets."""
    synthetic = build_nid_with_marker_at(tmp_path, offset)
    with open_fvfile(synthetic.path) as fvfile:
        assert_reads_known_nid(fvfile, synthetic)


@pytest.mark.slow
@pytest.mark.parametrize("offset", MARKER_OFFSETS.values(), ids=MARKER_OFFSETS)
def test_nid_header_ending_at_the_search_window_reads_from_a_memoryview(
    tmp_path, offset
):
    """parse on a buffer without find, which no caller in the package passes."""
    synthetic = build_nid_with_marker_at(tmp_path, offset)
    with memoryview(synthetic.path.read_bytes()) as data:
        assert_reads_known_nid(NIDFile.parse(data), synthetic)


def window_ends(sub: bytes) -> tuple[int, int]:
    """Where the first two windows of a search for sub end."""
    # each window starts len(sub) - 1 bytes before the previous one ends
    step = WINDOW - len(sub) + 1
    return WINDOW, step + WINDOW


@pytest.mark.slow
@pytest.mark.parametrize("sub", [b"\x1a", b"#!", b"END"], ids=["1", "2", "3"])
def test_find_in_buffer_finds_a_marker_across_a_window_end_as_bytes_find_does(sub):
    """A buffer without a find is searched in windows overlapping by len(sub) - 1."""
    data = bytearray(2 * WINDOW + 64)
    for end in window_ends(sub):
        # from wholly inside the window, through straddling its end, to wholly past it
        for start in range(end - len(sub) - 1, end + 2):
            data[start : start + len(sub)] = sub
            with memoryview(data) as view:
                found = _find_in_buffer(view, sub)
            assert found == bytes(data).find(sub) == start, (end, start)
            data[start : start + len(sub)] = bytes(len(sub))


@pytest.mark.slow
@pytest.mark.parametrize("sub", [b"#!", b"END"], ids=["2", "3"])
def test_find_in_buffer_misses_a_straddling_marker_cut_by_end(sub):
    """A match running past end is not found, even where a window would hold it."""
    data = bytearray(2 * WINDOW)
    start = WINDOW - 1
    data[start : start + len(sub)] = sub
    with memoryview(data) as view:
        cut = _find_in_buffer(view, sub, 0, start + len(sub) - 1)
        whole = _find_in_buffer(view, sub, 0, start + len(sub))
    assert cut == bytes(data).find(sub, 0, start + len(sub) - 1) == -1
    assert whole == bytes(data).find(sub, 0, start + len(sub)) == start


#################################################
# ARDF offsets past 4 GiB, in sparse files
#################################################

FOUR_GIB = 1 << 32
# a hole this long must take far less disk than its length for a file to count as sparse
PROBE_HOLE = 64 << 20
ARDF_TOC_ENTRY = struct.Struct("<LL4sLQ")
ARDF_TEXT_TOC_ENTRY = struct.Struct("<LL4sLQQ")


def allocated_bytes(path: Path) -> int:
    return os.stat(path).st_blocks * 512


@pytest.fixture
def sparse_dir(tmp_path):
    """tmp_path, where a file written with a hole is sparse, emptied afterwards."""
    if not hasattr(os.stat_result, "st_blocks"):
        pytest.skip(
            "no allocated size (st_blocks) on this platform, as on Windows, so a"
            " sparse file cannot be confirmed before writing 4 GiB"
        )
    probe = tmp_path / "sparse-probe"
    with open(probe, "wb") as f:
        f.seek(PROBE_HOLE)
        f.write(b"\1")
    sparse = allocated_bytes(probe) < PROBE_HOLE // 16
    probe.unlink()
    if not sparse:
        pytest.skip(f"files in {tmp_path} are not sparse: a hole takes up disk")
    try:
        yield tmp_path
    finally:
        for path in tmp_path.iterdir():
            path.unlink()


def first_level_pointers(path: Path) -> list[int]:
    """The offsets the file table and the file text table point at."""
    with open(path, "rb") as f:
        head = f.read(4096)
    pointers = []
    # the file table follows the 16 byte ARDF header; the text table follows it
    offset = 16
    for entry in (ARDF_TOC_ENTRY, ARDF_TEXT_TOC_ENTRY):
        size, slots, stride = struct.unpack_from("<QLL", head, offset + 16)
        assert stride == entry.size
        for slot in range(slots):
            *_, pointer = entry.unpack_from(head, offset + 32 + slot * stride)
            if pointer:
                pointers.append(pointer)
        offset += size
    return pointers


# builder arguments: grid (lines, points per line), points per half curve and so on
PAST_4_GIB_CASES = {
    "complete-trace-from-line-0": {
        "shape": (6, 5),
        "points": 24,
        "capture": "trace",
        "scan_down": True,
    },
    "complete-both-from-last-line": {
        "shape": (5, 6),
        "points": 24,
        "capture": "both",
        "scan_down": False,
    },
    "stopped-trace-from-last-line": {
        "shape": (6, 5),
        "points": 24,
        "scan_down": False,
        "lines_taken": 4,
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("case", PAST_4_GIB_CASES)
def test_ardf_with_everything_past_4_gib_reads_the_known_curves_and_images(
    sparse_dir, case
):
    """64-bit pointers past 4 GiB, through the strided and the per-curve readers."""
    synthetic = build_ardf(sparse_dir, **PAST_4_GIB_CASES[case], gap=FOUR_GIB)
    size = synthetic.path.stat().st_size
    if allocated_bytes(synthetic.path) > PROBE_HOLE:
        pytest.skip(f"the hole in a {size} byte file was not left sparse")
    assert size > FOUR_GIB
    pointers = first_level_pointers(synthetic.path)
    assert pointers and min(pointers) > FOUR_GIB, pointers

    stopped = synthetic.volumes[0].acquired is not None
    with open_fvfile(synthetic.path) as fvfile:
        assert_metadata(fvfile, synthetic)
        # a complete scan is read as one strided view, a stopped one curve by curve
        names = ["FMAP"] if stopped else [v.name for v in synthetic.volumes]
        assert [v.name for v in fvfile.volumes] == names
        for volume, expected in zip(fvfile.volumes, synthetic.volumes, strict=True):
            for r, c in np.ndindex(volume.shape):
                curve = volume.get_curve(r, c)
                if expected.acquired is None or expected.acquired[r, c]:
                    assert_same_curve(curve, expected.curve(r, c))
                else:
                    for part in curve:
                        for half in part:
                            assert np.shape(half) == (expected.ext_points[r, c],)
                            assert np.isnan(half).all()
            seen = []
            for rc, curve in volume.iter_curves():
                seen.append(rc)
                assert_same_curve(curve, expected.curve(*rc))
            assert seen == list(expected.order)
            all_curves = volume.get_all_curves()
            assert all_curves.dtype == np.float32
            np.testing.assert_allclose(
                all_curves, stack_curves(expected), rtol=1e-7, atol=0, equal_nan=True
            )
        assert_images_where_held(fvfile, synthetic)
        if stopped:
            missing = ~synthetic.volumes[0].acquired
            for image in fvfile.images.values():
                assert np.isnan(image.get_image()[missing]).all()


#################################################
# Curves of ragged length on a large grid
#################################################

RAGGED_GRID = (64, 48)
# every extend half 30 to 64 points and every retract half 30 to 48, but for the
# shortest of each, at different pixels away from the edges of the grid
RAGGED_EXT = np.random.default_rng(7).integers(30, 65, RAGGED_GRID)
RAGGED_EXT[23, 17] = 21
RAGGED_RET = np.random.default_rng(8).integers(30, 49, RAGGED_GRID)
RAGGED_RET[41, 30] = 26
RAGGED_CASES = {
    # a map stopped part way along its last row
    "nid": partial(
        build_nid,
        shape=RAGGED_GRID,
        points=(64, 48),
        ext_points=RAGGED_EXT,
        ret_points=RAGGED_RET,
        curves_taken=RAGGED_GRID[0] * RAGGED_GRID[1] - 29,
    ),
    "arh5-force-map": partial(
        build_arh5_force_map,
        shape=RAGGED_GRID,
        points=(64, 48),
        ext_points=RAGGED_EXT,
        ret_points=RAGGED_RET,
    ),
    # curves of one length, read curve by curve, with lines never taken
    "ardf-stopped": partial(build_ardf, shape=RAGGED_GRID, points=24, lines_taken=37),
    # a slow map, its curves read by their own segments, with lines never taken
    "ardf-slow-map": partial(
        build_ardf,
        shape=RAGGED_GRID,
        points=48,
        slow_map=True,
        ext_points=RAGGED_EXT,
        ret_points=RAGGED_RET,
        lines_taken=37,
    ),
}


def acquired_mask(volume) -> np.ndarray:
    if volume.acquired is None:
        return np.ones(volume.shape, dtype=bool)
    return volume.acquired


def padded(expected, curves) -> np.ndarray:
    """(index, curve) pairs as (2, rows, cols, 2, n) arrays shaped like expected's.

    Each half fills its own length from the start, the rest NaN, as the synthetic volume
    holds them.
    """
    n = max(expected.zext.shape[-1], expected.zret.shape[-1])
    out = np.full((2, *expected.shape, 2, n), np.nan)
    for (r, c), ((zext, zret), (dext, dret)) in curves:
        for i, half in enumerate((zext, zret)):
            out[0, r, c, i, : len(half)] = half
        for i, half in enumerate((dext, dret)):
            out[1, r, c, i, : len(half)] = half
    return out


def expected_padded(expected) -> np.ndarray:
    n = max(expected.zext.shape[-1], expected.zret.shape[-1])
    out = np.full((2, *expected.shape, 2, n), np.nan)
    for i, half in enumerate((expected.zext, expected.zret)):
        out[0, :, :, i, : half.shape[-1]] = half
    for i, half in enumerate((expected.dext, expected.dret)):
        out[1, :, :, i, : half.shape[-1]] = half
    return out


@pytest.mark.slow
@pytest.mark.parametrize("case", RAGGED_CASES)
def test_get_curve_on_a_large_ragged_grid_returns_each_curve_as_stored(tmp_path, case):
    """Every acquired curve comes back at its own length, with its own values."""
    synthetic = RAGGED_CASES[case](tmp_path)
    expected = synthetic.volumes[0]
    acquired = acquired_mask(expected)
    with open_fvfile(synthetic.path) as fvfile:
        (volume,) = fvfile.volumes
        got = padded(
            expected,
            (((r, c), volume.get_curve(r, c)) for r, c in zip(*np.nonzero(acquired))),
        )
    assert_nid_close(got, expected_padded(expected))


@pytest.mark.slow
@pytest.mark.parametrize("case", RAGGED_CASES)
def test_iter_curves_on_a_large_ragged_grid_yields_each_acquired_curve_once(
    tmp_path, case
):
    """iter_curves visits each acquired pixel once, with what get_curve returns."""
    synthetic = RAGGED_CASES[case](tmp_path)
    expected = synthetic.volumes[0]
    acquired = acquired_mask(expected)
    with open_fvfile(synthetic.path) as fvfile:
        (volume,) = fvfile.volumes
        yielded = list(volume.iter_curves())
        indices = [rc for rc, _ in yielded]
        if expected.order is not None:
            assert indices == list(expected.order)
        assert sorted(indices) == [tuple(rc) for rc in np.argwhere(acquired).tolist()]
        from_iter = padded(expected, yielded)
        from_get = padded(expected, ((rc, volume.get_curve(*rc)) for rc in indices))
    np.testing.assert_array_equal(from_iter, from_get)


@pytest.mark.slow
@pytest.mark.parametrize("case", RAGGED_CASES)
def test_get_all_curves_on_a_large_ragged_grid_is_get_curve_cut_to_the_shortest(
    tmp_path, case
):
    """Halves cut to the shortest acquired half, nearest the surface; NaN elsewhere."""
    synthetic = RAGGED_CASES[case](tmp_path)
    expected = synthetic.volumes[0]
    acquired = acquired_mask(expected)
    n = int(
        min(expected.ext_points[acquired].min(), expected.ret_points[acquired].min())
    )
    with open_fvfile(synthetic.path) as fvfile:
        (volume,) = fvfile.volumes
        all_curves = volume.get_all_curves()
        cut = np.full((2, *volume.shape, 2, n), np.nan, dtype=np.float32)
        for r, c in zip(*np.nonzero(acquired)):
            (zext, zret), (dext, dret) = volume.get_curve(r, c)
            cut[0, r, c] = zext[len(zext) - n :], zret[:n]
            cut[1, r, c] = dext[len(dext) - n :], dret[:n]
    assert all_curves.dtype == np.float32
    assert all_curves.shape == (2, *expected.shape, 2, n)
    np.testing.assert_allclose(all_curves, cut, rtol=1e-6, atol=0, equal_nan=True)
