"""Synthetic Asylum ARDF files read back through the public reader surface."""

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

# The files store float32 in each channel's declared unit (metres unless a test says
# otherwise) and the reader multiplies by one factor to nm in float32, so curves match
# the stored values in nm to one float32 rounding (rtol 1e-7). Images are returned as
# stored and match exactly. Metadata parsed from decimal text match to double rounding;
# the time step is stored as a double and matches exactly.

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.ardf import build_ardf

COMPLETE_RELPATH = "test_cases/PDMS_SAA/ffm/both0001.ARDF"
STOPPED_RELPATH = "test_cases/PDMS_SAA/ffm/hard0001.ARDF"

# grid (lines, points per line), points per half curve, captured directions, ScanDown
# (True takes line 0 first, False the last line)
COMPLETE_CASES = {
    "trace-16-from-last-line": ((4, 4), 16, "trace", False),
    "trace-100-from-line-0": ((6, 6), 100, "trace", True),
    "both-16-from-line-0": ((4, 4), 16, "both", True),
    "both-100-from-last-line": ((6, 6), 100, "both", False),
    "retrace-16-from-line-0": ((4, 4), 16, "retrace", True),
    "trace-16-tall-from-line-0": ((6, 4), 16, "trace", True),
    "retrace-16-wide-from-last-line": ((4, 6), 16, "retrace", False),
    "both-16-tall-from-last-line": ((6, 4), 16, "both", False),
    "both-16-wide-from-line-0": ((4, 6), 16, "both", True),
}

# grid, points per half curve, ScanDown, lines taken before the scan stopped
STOPPED_CASES = {
    "stopped-from-line-0": ((8, 8), 16, True, 5),
    "stopped-from-last-line": ((8, 8), 16, False, 5),
    "stopped-tall-from-line-0": ((8, 6), 16, True, 5),
    "stopped-wide-from-last-line": ((6, 8), 16, False, 4),
}
QUANTITIES = ("Height", "ZSensor", "Adhesion", "Force", "YoungsJKR")
DIRECTIONS = {"trace": ["Trace"], "retrace": ["Retrace"], "both": ["Trace", "Retrace"]}


def assert_float32_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0, equal_nan=True)


def assert_same_curve(curve, expected):
    (zext, zret), (dext, dret) = curve
    (ezext, ezret), (edext, edret) = expected
    assert_float32_close(zext, ezext)
    assert_float32_close(zret, ezret)
    assert_float32_close(dext, edext)
    assert_float32_close(dret, edret)


def assert_nan_curve_like(curve, acquired_curve, points):
    """Each half of curve is all NaN, shaped and typed as the acquired curve's."""
    (zext, zret), (dext, dret) = curve
    (azext, azret), (adext, adret) = acquired_curve
    for half, acquired_half in zip(
        (zext, zret, dext, dret), (azext, azret, adext, adret), strict=True
    ):
        assert np.shape(half) == np.shape(acquired_half) == (points,)
        assert np.asarray(half).dtype == np.asarray(acquired_half).dtype
        assert np.isnan(half).all()


def assert_metadata(fvfile, synthetic):
    assert [v.shape for v in fvfile.volumes] == [v.shape for v in synthetic.volumes]
    np.testing.assert_allclose(fvfile.k, synthetic.k, rtol=1e-12)
    np.testing.assert_allclose(fvfile.defl_sens, synthetic.defl_sens, rtol=1e-12)
    np.testing.assert_allclose(fvfile.scansize, synthetic.scansize, rtol=1e-12)
    assert fvfile.t_step == synthetic.t_step
    assert fvfile.trace == (1 if len(synthetic.volumes) > 1 else None)
    # the format stores no lag between drive and deflection
    assert getattr(fvfile, "sync_dist", None) is None


def assert_images_where_held(fvfile, synthetic):
    """Names in file order, units, and values wherever the file holds one."""
    assert list(fvfile.images) == list(synthetic.images)
    for name, image in fvfile.images.items():
        expected = synthetic.images[name]
        assert image.shape == expected.shape
        assert image.units == synthetic.image_units[name]
        values = image.get_image()
        assert values.shape == image.shape
        assert values.dtype == np.float32
        held = synthetic.volumes[0].acquired
        held = np.ones(expected.shape, bool) if held is None else held
        np.testing.assert_array_equal(values[held], expected[held])


def build_case(tmp_path, case):
    if case in COMPLETE_CASES:
        shape, points, capture, scan_down = COMPLETE_CASES[case]
        return build_ardf(tmp_path, shape, points, capture=capture, scan_down=scan_down)
    shape, points, scan_down, lines_taken = STOPPED_CASES[case]
    return build_ardf(
        tmp_path, shape, points, scan_down=scan_down, lines_taken=lines_taken
    )


@pytest.fixture(params=list(COMPLETE_CASES))
def complete(request, tmp_path):
    return build_case(tmp_path, request.param)


@pytest.fixture(params=list(STOPPED_CASES))
def stopped(request, tmp_path):
    return build_case(tmp_path, request.param)


def test_ardf_metadata_round_trips(complete):
    with open_fvfile(complete.path) as fvfile:
        assert_metadata(fvfile, complete)


def test_ardf_volumes_are_named_by_direction(complete):
    with open_fvfile(complete.path) as fvfile:
        names = [v.name for v in fvfile.volumes]
    assert names == [v.name for v in complete.volumes]


def test_ardf_get_curve_returns_the_stored_curves(complete):
    with open_fvfile(complete.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, complete.volumes, strict=True):
            for r, c in np.ndindex(volume.shape):
                assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))


def test_ardf_iter_curves_follows_the_order_on_disk(complete):
    with open_fvfile(complete.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, complete.volumes, strict=True):
            assert list(volume.iter_indices()) == list(expected.order)
            seen = []
            for (r, c), curve in volume.iter_curves():
                seen.append((r, c))
                assert_same_curve(curve, expected.curve(r, c))
            assert seen == list(expected.order)


def test_ardf_get_all_curves_returns_every_curve(complete):
    with open_fvfile(complete.path) as fvfile:
        for volume, expected in zip(fvfile.volumes, complete.volumes, strict=True):
            z, d = volume.get_all_curves()
            assert z.shape[:2] == d.shape[:2] == volume.shape
            assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
            assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


@pytest.mark.parametrize(
    "channel_units",
    [{"Raw": "nm", "Defl": "nm", "Drive": "nm"}, {"Raw": "nm"}, {"Defl": "nm"}],
    ids=lambda u: "-".join(f"{k}={v}" for k, v in u.items()),
)
@pytest.mark.parametrize("lines_taken", [None, 2], ids=["complete", "stopped"])
def test_ardf_curves_in_other_units_read_back_the_same_in_nm(
    tmp_path, channel_units, lines_taken
):
    """Curves come back in nm from whatever length unit each channel declares."""
    (tmp_path / "m").mkdir()
    (tmp_path / "other").mkdir()
    kwargs = {"lines_taken": lines_taken}
    metres = build_ardf(tmp_path / "m", **kwargs)
    other = build_ardf(tmp_path / "other", channel_units=channel_units, **kwargs)
    with open_fvfile(metres.path) as fvfile:
        metres_z, metres_d = fvfile.volumes[0].get_all_curves()
    with open_fvfile(other.path) as fvfile:
        volume, expected = fvfile.volumes[0], other.volumes[0]
        for (r, c), curve in volume.iter_curves():
            assert_same_curve(curve, expected.curve(r, c))
            assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))
        z, d = volume.get_all_curves()
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))
    # each file holds its own float32 rounding of the same curves
    np.testing.assert_allclose(z, metres_z, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(d, metres_d, rtol=1e-6, equal_nan=True)


def test_ardf_images_round_trip_exactly(complete):
    with open_fvfile(complete.path) as fvfile:
        assert_images_where_held(fvfile, complete)


@pytest.mark.parametrize("capture", list(DIRECTIONS))
def test_ardf_images_are_named_by_quantity_then_direction(tmp_path, capture):
    synthetic = build_ardf(tmp_path, capture=capture)
    with open_fvfile(synthetic.path) as fvfile:
        names = list(fvfile.images)
    assert names == [q + d for q in QUANTITIES for d in DIRECTIONS[capture]]


def test_ardf_stopped_scan_metadata_round_trips(stopped):
    with open_fvfile(stopped.path) as fvfile:
        assert_metadata(fvfile, stopped)


def test_ardf_stopped_scan_is_read_curve_by_curve_as_fmap(stopped):
    with open_fvfile(stopped.path) as fvfile:
        assert [v.name for v in fvfile.volumes] == ["FMAP"]


def test_ardf_stopped_scan_yields_exactly_the_acquired_curves(stopped):
    expected = stopped.volumes[0]
    with open_fvfile(stopped.path) as fvfile:
        volume = fvfile.volumes[0]
        indices = list(volume.iter_indices())
        assert indices == list(expected.order)
        acquired = [tuple(rc) for rc in np.argwhere(expected.acquired).tolist()]
        assert sorted(indices) == acquired
        seen = []
        for (r, c), curve in volume.iter_curves():
            seen.append((r, c))
            assert_same_curve(curve, expected.curve(r, c))
        assert seen == indices


def test_ardf_stopped_scan_get_curve_returns_the_acquired_curves(stopped):
    expected = stopped.volumes[0]
    with open_fvfile(stopped.path) as fvfile:
        volume = fvfile.volumes[0]
        for r, c in np.argwhere(expected.acquired).tolist():
            assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))


def test_ardf_stopped_scan_get_all_curves_is_nan_where_never_acquired(stopped):
    expected = stopped.volumes[0]
    with open_fvfile(stopped.path) as fvfile:
        z, d = fvfile.volumes[0].get_all_curves()
        assert z.shape[:2] == d.shape[:2] == fvfile.volumes[0].shape
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


def test_ardf_stopped_scan_get_curve_off_the_scan_is_nan_of_the_curve_length(stopped):
    expected = stopped.volumes[0]
    points = int(expected.ext_points.max())
    with open_fvfile(stopped.path) as fvfile:
        volume = fvfile.volumes[0]
        acquired_curve = volume.get_curve(*expected.order[0])
        for r, c in np.argwhere(~expected.acquired).tolist():
            assert_nan_curve_like(volume.get_curve(r, c), acquired_curve, points)


def test_ardf_stopped_scan_images_round_trip_on_the_lines_taken(stopped):
    with open_fvfile(stopped.path) as fvfile:
        assert_images_where_held(fvfile, stopped)


def test_ardf_stopped_scan_images_are_nan_off_the_scan(stopped):
    missing = ~stopped.volumes[0].acquired
    with open_fvfile(stopped.path) as fvfile:
        for image in fvfile.images.values():
            assert np.isnan(image.get_image()[missing]).all()


#################################################
# The file layout, read without the reader
#################################################

HEADER = struct.Struct("<LL4sL")
TABLE = struct.Struct("<QLL")
TABLE_NAMES = {b"FTOC", b"TTOC", b"IMAG", b"VOLM", b"VTOC", b"IBOX"}
# fields after the header, by section name and size; "p" marks a file offset
FIELDS = {
    (b"IMAG", 24): ("<Q", "p"),
    (b"VOLM", 24): ("<Q", "p"),
    (b"NEXT", 24): ("<Q", "p"),
    (b"THMB", 24): ("<Q", "p"),
    (b"NSET", 24): ("<Q", "-"),
    (b"TOFF", 32): ("<QQ", "-p"),
    (b"VOFF", 40): ("<LLQQ", "---p"),
    b"IDEF": ("<LLQQdd32s32s32s32s", ""),
    b"VDEF": ("<LL24sddd32s32s32s32sQ", ""),
    b"VCHN": ("<32s32s", ""),
    b"XDEF": ("<LL", ""),
    b"VSET": ("<LLLLQQ", "----pp"),
    b"VNAM": ("<LLLL32s", ""),
    b"VDAT": ("<10L", ""),
    b"THMB": ("<LLLL", ""),
    b"TEXT": ("<LL", ""),
}


def sections(data: bytes) -> list[tuple[int, bytes, int, int]]:
    """Every section from start to end as (offset, name, size, flags).

    Tables are stepped slot by slot, an unused slot as a section named b"" so that
    no byte goes unaccounted for. Every CRC is checked.
    """
    found = []
    offset = 0

    def check(offset):
        crc, size, name, flags = HEADER.unpack_from(data, offset)
        assert size >= HEADER.size, (offset, name, size)
        assert crc == zlib.crc32(data[offset + 4 : offset + size]), (offset, name)
        found.append((offset, name, size, flags))
        return name, size

    while offset < len(data):
        name, size = check(offset)
        if name in TABLE_NAMES and size == HEADER.size + TABLE.size:
            table_size, slots, stride = TABLE.unpack_from(data, offset + HEADER.size)
            assert table_size == size + slots * stride, (offset, name)
            for slot in range(offset + size, offset + table_size, stride):
                if data[slot : slot + stride] == bytes(stride):
                    found.append((slot, b"", stride, 0))
                else:
                    assert check(slot)[1] == stride, (slot, name)
            offset += table_size
        else:
            offset += size
    assert offset == len(data)
    return found


def layout(data: bytes):
    """Each section's name, size, flags and fields, offsets as section numbers.

    A text's size depends on its text, so it is left out; the keys of each text
    come back separately, in file order.
    """
    found = sections(data)
    number = {offset: i for i, (offset, *_) in enumerate(found)}
    described, notes = [], []
    for offset, name, size, flags in found:
        if name in TABLE_NAMES and size == HEADER.size + TABLE.size:
            fmt, kinds = "<QLL", ""
        else:
            fmt, kinds = FIELDS.get((name, size), FIELDS.get(name, ("", "")))
        values = struct.unpack_from(fmt, data, offset + HEADER.size)
        fields = []
        for i, value in enumerate(values):
            if i < len(kinds) and kinds[i] == "p":
                value = number[value] if value else None
            elif isinstance(value, bytes):
                value = value.rstrip(b"\0")
            fields.append(value)
        if name == b"XDEF":
            start = offset + HEADER.size + 8
            fields.append(data[start : start + values[1]])
        if name == b"TEXT":
            start = offset + HEADER.size + 8
            text = data[start : start + values[1]].decode("windows-1252")
            notes.append({line.split(":", 1)[0] for line in text.split("\r") if line})
            fields, size = fields[:1], None
        described.append((name, size, flags, tuple(fields)))
    return described, notes


def structure(fvfile):
    """What a reader makes of an ARDF file, without the values."""
    first = [next(iter(v.iter_indices())) for v in fvfile.volumes]
    return {
        "class": type(fvfile),
        "volumes": [(type(v), v.shape) for v in fvfile.volumes],
        "volume names": [v.name for v in fvfile.volumes],
        "trace": fvfile.trace,
        "curves": [sum(1 for _ in v.iter_indices()) for v in fvfile.volumes],
        "curve": [
            [tuple(map(np.shape, part)) for part in v.get_curve(*rc)]
            for v, rc in zip(fvfile.volumes, first, strict=True)
        ],
        "all curves": [
            tuple(np.shape(a) for a in v.get_all_curves()) for v in fvfile.volumes
        ],
        "images": {
            name: (image.shape, image.units, image.get_image().dtype)
            for name, image in fvfile.images.items()
        },
        "metadata": {
            name: type(getattr(fvfile, name))
            for name in ("k", "defl_sens", "t_step", "scansize")
        },
    }


@pytest.mark.parametrize("case", [*COMPLETE_CASES, *STOPPED_CASES])
def test_ardf_every_section_has_a_valid_crc(tmp_path, case):
    # the reader checks most CRCs, but not those of the VTOC header, the image data, the
    # thumbnails or the image texts, and for a complete scan only the first curve's VSET
    # and VNAM, none of its VDAT sections
    sections(build_case(tmp_path, case).path.read_bytes())


def assert_same_layout(synthetic_path: Path, real_path: Path):
    # reading the layout checks every CRC of the real file too
    real_layout, real_notes = layout(real_path.read_bytes())
    synthetic_layout, synthetic_notes = layout(synthetic_path.read_bytes())
    assert synthetic_layout == real_layout
    # every note key the builder writes is one the real file has
    for keys, real_keys in zip(synthetic_notes, real_notes, strict=True):
        assert keys <= real_keys, keys - real_keys
    with open_fvfile(real_path) as real, open_fvfile(synthetic_path) as fvfile:
        assert structure(fvfile) == structure(real)
        assert set(fvfile.headers) <= set(real.headers)


@pytest.mark.slow
def test_synthetic_complete_ardf_has_the_layout_of_the_real_file(
    real_data_file, tmp_path
):
    # both directions, 32 x 32, 1000 + 1000 points, taken from the last line
    synthetic = build_ardf(tmp_path, (32, 32), 1000, capture="both", scan_down=False)
    assert_same_layout(synthetic.path, Path(real_data_file(COMPLETE_RELPATH)))


@pytest.mark.slow
def test_synthetic_stopped_ardf_has_the_layout_of_the_real_file(
    real_data_file, tmp_path
):
    # trace only, 32 x 32, 1000 + 1000 points, stopped after 19 lines from line 0
    synthetic = build_ardf(tmp_path, (32, 32), 1000, scan_down=True, lines_taken=19)
    assert_same_layout(synthetic.path, Path(real_data_file(STOPPED_RELPATH)))
