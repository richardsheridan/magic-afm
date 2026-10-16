"""Synthetic Nanosurf NID files read back through the public reader surface."""

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

# The files store curves as scaled integers and the reader decodes them in float32, so
# curves and maps match the builder's float64 decoding to float32 rounding: rtol 1e-6
# with an atol of 1e-6 of the array's largest magnitude. Metadata parsed from decimal
# text match to double rounding.

from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.nid import build_nid

NID_RELPATH = "test_cases/20251216_141819_PVDF2_Film_U_00002.nid"

# forward rows of a force-triggered map stop early; backward rows stay full, as in the
# real file
RAGGED_EXT = 64 - np.arange(15).reshape(5, 3) % 5 * 4

# grid (rows, cols), points per curve (extend, retract), other builder arguments
CASES = {
    "square-16": ((4, 4), (16, 16), {}),
    "square-100": ((4, 4), (100, 100), {}),
    "wide-16": ((3, 5), (16, 16), {}),
    "wide-100": ((3, 5), (100, 100), {}),
    "tall-ragged": ((5, 3), (64, 64), {"ext_points": RAGGED_EXT}),
    "longer-retract": ((3, 5), (40, 48), {}),
    "longer-extend": ((3, 5), (48, 40), {}),
    "tall-ragged-longer-extend": ((5, 3), (64, 40), {"ext_points": RAGGED_EXT}),
    "16-bit": ((4, 4), (32, 32), {"save_bits": 16}),
}
# forward rows of 44 to 56 points, in rows of 64
SHORT_EXT = 44 + np.arange(16).reshape(4, 4) % 4 * 4
# grid, points per curve (extend, retract), curves taken before the map stopped, other
# builder arguments; the last case's forward rows are ragged and all stop short of the
# row width
STOPPED_CASES = {
    "stopped-mid-row": ((4, 4), (16, 16), 6, {}),
    "stopped-at-row-end": ((3, 5), (16, 16), 10, {}),
    "stopped-longer-retract": ((3, 5), (40, 48), 7, {}),
    "stopped-longer-extend": ((3, 5), (48, 40), 7, {}),
    "stopped-short-forward-rows": ((4, 4), (64, 64), 6, {"ext_points": SHORT_EXT}),
}


def assert_float32_close(actual, expected):
    expected = np.asarray(expected)
    atol = 1e-6 * np.nanmax(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=atol)


@pytest.fixture(params=list(CASES))
def synthetic(request, tmp_path):
    shape, points, kwargs = CASES[request.param]
    return build_nid(tmp_path, shape, points, **kwargs)


@pytest.fixture(params=list(STOPPED_CASES))
def stopped(request, tmp_path):
    shape, points, taken, kwargs = STOPPED_CASES[request.param]
    return build_nid(tmp_path, shape, points, curves_taken=taken, **kwargs), points


def test_nid_metadata_round_trips(synthetic):
    with open_fvfile(synthetic.path) as fvfile:
        assert [v.name for v in fvfile.volumes] == ["SpecMap"]
        assert fvfile.volumes[0].shape == synthetic.volumes[0].shape
        np.testing.assert_allclose(fvfile.k, synthetic.k, rtol=1e-12)
        np.testing.assert_allclose(fvfile.defl_sens, synthetic.defl_sens, rtol=1e-12)
        np.testing.assert_allclose(fvfile.t_step, synthetic.t_step, rtol=1e-12)
        np.testing.assert_allclose(fvfile.scansize, synthetic.scansize, rtol=1e-12)


def test_nid_indices_come_in_acquisition_order(synthetic):
    with open_fvfile(synthetic.path) as fvfile:
        indices = list(fvfile.volumes[0].iter_indices())
    assert indices == list(synthetic.volumes[0].order)


def test_nid_get_curve_returns_the_points_nearest_the_surface(synthetic):
    expected = synthetic.volumes[0]
    rows, cols = expected.shape
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        for r in range(rows):
            for c in range(cols):
                # both halves cut to the shorter one
                n = min(expected.ext_points[r, c], expected.ret_points[r, c])
                (zext, zret), (dext, dret) = volume.get_curve(r, c)
                (ezext, ezret), (edext, edret) = expected.curve(r, c, n, n)
                assert_float32_close(zext, ezext)
                assert_float32_close(zret, ezret)
                assert_float32_close(dext, edext)
                assert_float32_close(dret, edret)


def test_nid_iter_curves_matches_the_known_curves(synthetic):
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        seen = []
        for (r, c), ((zext, zret), (dext, dret)) in fvfile.volumes[0].iter_curves():
            seen.append((r, c))
            n = min(expected.ext_points[r, c], expected.ret_points[r, c])
            (ezext, ezret), (edext, edret) = expected.curve(r, c, n, n)
            assert_float32_close(np.stack([zext, zret]), np.stack([ezext, ezret]))
            assert_float32_close(np.stack([dext, dret]), np.stack([edext, edret]))
    assert seen == list(expected.order)


def test_nid_get_all_curves_cuts_every_curve_to_the_shortest(synthetic):
    expected = synthetic.volumes[0]
    rows, cols = expected.shape
    n = min(expected.ext_points.min(), expected.ret_points.min())
    with open_fvfile(synthetic.path) as fvfile:
        z, d = fvfile.volumes[0].get_all_curves()
    assert z.shape == d.shape == (rows, cols, 2, n)
    for r in range(rows):
        for c in range(cols):
            (ezext, ezret), (edext, edret) = expected.curve(r, c, n, n)
            assert_float32_close(z[r, c], np.stack([ezext, ezret]))
            assert_float32_close(d[r, c], np.stack([edext, edret]))


def test_nid_images_round_trip_with_units(synthetic):
    with open_fvfile(synthetic.path) as fvfile:
        assert sorted(fvfile.images) == sorted(synthetic.images)
        for name, image in fvfile.images.items():
            expected = synthetic.images[name]
            assert image.shape == expected.shape
            assert image.units == synthetic.image_units[name]
            assert_float32_close(image.get_image(), expected)


def test_nid_stopped_map_yields_exactly_the_acquired_curves(stopped):
    synthetic, _ = stopped
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        assert list(volume.iter_indices()) == list(expected.order)
        acquired = [tuple(rc) for rc in np.argwhere(expected.acquired).tolist()]
        assert sorted(expected.order) == acquired
        for r, c in expected.order:
            n = min(expected.ext_points[r, c], expected.ret_points[r, c])
            (zext, zret), (dext, dret) = volume.get_curve(r, c)
            (ezext, ezret), (edext, edret) = expected.curve(r, c, n, n)
            assert_float32_close(np.stack([zext, zret]), np.stack([ezext, ezret]))
            assert_float32_close(np.stack([dext, dret]), np.stack([edext, edret]))


def test_nid_stopped_map_get_curve_off_the_map_is_nan_of_the_row_length(stopped):
    # both halves come back cut to the shorter of the forward and backward rows
    synthetic, points = stopped
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        for r, c in np.argwhere(~expected.acquired).tolist():
            (zext, zret), (dext, dret) = volume.get_curve(r, c)
            for half in (zext, zret, dext, dret):
                assert np.shape(half) == (min(points),)
                assert np.asarray(half).dtype == np.float32
                assert np.isnan(half).all()


def test_nid_stopped_map_images_are_nan_off_the_map(stopped):
    synthetic, _ = stopped
    missing = ~synthetic.volumes[0].acquired
    with open_fvfile(synthetic.path) as fvfile:
        assert sorted(fvfile.images) == sorted(synthetic.images)
        for name, image in fvfile.images.items():
            values = image.get_image()
            assert np.isnan(values[missing]).all()
            assert_float32_close(values[~missing], synthetic.images[name][~missing])


@pytest.mark.parametrize(
    "units",
    [
        {"z_unit": "µm"},
        {"z_unit": "nm"},
        {"defl_unit": "nN"},
        {"defl_unit": "m"},
        {"defl_unit": "nm"},
        {"defl_unit": "V"},
        {"k_unit": "mN/m"},
        {"z_unit": "nm", "defl_unit": "nN", "k_unit": "mN/m"},
    ],
    ids=lambda u: "-".join(f"{k}={v}" for k, v in u.items()),
)
def test_nid_curves_in_other_units_read_back_the_same_in_nm(tmp_path, units):
    """Curves come back in nm from whatever units the file declares for them."""
    shape, points = (3, 4), (32, 32)
    (tmp_path / "si").mkdir()
    (tmp_path / "other").mkdir()
    si = build_nid(tmp_path / "si", shape, points)
    other = build_nid(tmp_path / "other", shape, points, **units)
    expected = other.volumes[0]
    n = min(expected.ext_points.min(), expected.ret_points.min())
    with open_fvfile(si.path) as fvfile:
        si_z, si_d = fvfile.volumes[0].get_all_curves()
    with open_fvfile(other.path) as fvfile:
        np.testing.assert_allclose(fvfile.k, other.k, rtol=1e-12)
        z, d = fvfile.volumes[0].get_all_curves()
    for r, c in np.ndindex(shape):
        (ezext, ezret), (edext, edret) = expected.curve(r, c, n, n)
        assert_float32_close(z[r, c], np.stack([ezext, ezret]))
        assert_float32_close(d[r, c], np.stack([edext, edret]))
    assert_float32_close(z, si_z)
    assert_float32_close(d, si_d)


def section_keys(headers):
    return {
        name: set(section)
        for name, section in headers.items()
        if isinstance(section, dict) and name not in ("param", "cantilever")
    }


def curve_datasets(headers):
    """(frame, channel name, unit, save bits) of every dataset but the scan image."""
    return [
        (s["Frame"], s["Dim2Name"], s["Dim2Unit"], s["SaveBits"])
        for s in (headers[name] for name in headers["datasets"])
        if not s["Frame"].startswith("Scan")
    ]


def structure(fvfile):
    """What a reader makes of a NID file, without the values."""
    volume = fvfile.volumes[0]
    order = list(volume.iter_indices())
    return {
        "class": type(fvfile),
        "volumes": [(v.name, v.shape) for v in fvfile.volumes],
        "order": order,
        "curve lengths": [
            tuple(len(half) for half in volume.get_curve(*rc)[0]) for rc in order
        ],
        "all curves": volume.get_all_curves()[0].shape,
        "images": {
            name: (image.shape, image.units, image.get_image().dtype)
            for name, image in fvfile.images.items()
        },
        "datasets": curve_datasets(fvfile.headers),
        "spec map": fvfile.headers["spec_map"].shape,
        "k unit": fvfile.headers["cantilever"]["Prop0"].rpartition("*")[2],
        "has sensitivity": "Sensitivity" in fvfile.headers["cantilever"],
    }


def curve_points(fvfile, frame):
    """Points of each curve of a spectroscopy frame, on the map grid."""
    headers = fvfile.headers
    volume = fvfile.volumes[0]
    counts = np.zeros(volume.shape, dtype=int)
    for name, points in zip(headers["datasets"], headers["param"]["LinePoints"]):
        if headers[name]["Frame"] == frame:
            for rc, n in zip(volume.iter_indices(), points):
                counts[rc] = n
    return counts


def row_width(fvfile, frame):
    headers = fvfile.headers
    return next(
        int(headers[name]["Points"])
        for name in headers["datasets"]
        if headers[name]["Frame"] == frame
    )


@pytest.mark.slow
def test_synthetic_nid_has_the_structure_of_the_real_file(real_data_file, tmp_path):
    with open_fvfile(Path(real_data_file(NID_RELPATH))) as real:
        real_structure = structure(real)
        real_headers = real.headers
        shape = real.volumes[0].shape
        points = (row_width(real, "Spec forward"), row_width(real, "Spec backward"))
        ext_points = curve_points(real, "Spec forward")
        ret_points = curve_points(real, "Spec backward")
    synthetic = build_nid(
        tmp_path, shape, points, ext_points=ext_points, ret_points=ret_points
    )
    with open_fvfile(synthetic.path) as fvfile:
        assert structure(fvfile) == real_structure
        # every section and key the builder writes is one the real file has
        real_keys = section_keys(real_headers)
        for name, keys in section_keys(fvfile.headers).items():
            assert name in real_keys, name
            assert keys <= real_keys[name], (name, keys - real_keys[name])
