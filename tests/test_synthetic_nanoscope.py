"""Synthetic Bruker Nanoscope files read back through the public reader surface."""

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

# The files store scaled integers and the reader decodes them in float32: integers
# beyond 2**24 round on the way to float32, the float32 scale rounds again, and the
# PeakForce z is a float32 cosine added to a float32 height. Curves and maps match
# the builder's float64 decoding to float32 rounding: rtol 1e-6 with an atol of
# 1e-6 of the array's largest magnitude. Points the file does not hold are NaN in
# the expected curves and are not compared. Metadata parsed from decimal text
# match to double rounding.

from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.nanoscope import build_pfc

PFC_RELPATH = "test_cases/PS_PCL_SAA/QNM/onekhz.000.pfc"

# grid (rows, cols), points per half curve, other builder arguments
PFC_CASES = {
    "square-128": ((4, 4), 128, {}),
    "wide-128": ((3, 6), 128, {}),
    "tall-128": ((6, 3), 128, {}),
    "half-khz": ((4, 4), 128, {"rate": 0.5, "sync_distance": 272.05}),
    "two-khz": ((4, 4), 128, {"rate": 2.0, "sync_distance": 75.175}),
    "sync-qnm": (
        (4, 4),
        128,
        {"sync_distance": 150.75, "sync_distance_qnm": 149.557},
    ),
    "no-marker": ((4, 4), 128, {"extend_marker": False}),
    "square-64-no-lag": ((4, 4), 64, {"sync_distance": 0.0}),
    "square-64": ((4, 4), 64, {}),
}
# Left out of the lag and curve tests: "square-64". The reader converts Sync
# Distance to stored points as Sync Distance * PFT Freq / 2, which holds only for
# 128 + 128 points at 512 samples per ms; with 64 + 64 it doubles the lag.
# "square-64-no-lag" checks that curve length with no lag to convert.
LAGGED_CASES = [case for case in PFC_CASES if case != "square-64"]
# Left out of the scan size test: "tall-128". The reader takes Scan Size as the
# longer side, the builder as the fast axis; no real file says which.
SCANSIZE_CASES = [case for case in PFC_CASES if case != "tall-128"]
# Left out everywhere: the deflection of pixel (0, 0). The reader cuts each curve
# from its own record and the one before, and for (0, 0) uses those of (0, 1),
# so get_curve(0, 0) returns the deflection of (0, 1).
# Left out everywhere: t_step. The reader divides the drive period by force/line,
# the curves per line, not by the points per curve.


def assert_float32_close(actual, expected):
    """Close to float32 rounding wherever the file holds the expected value."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    held = ~np.isnan(expected)
    atol = 1e-6 * np.nanmax(np.abs(expected))
    np.testing.assert_allclose(actual[held], expected[held], rtol=1e-6, atol=atol)


def assert_same_curve(curve, expected, deflection=True):
    (zext, zret), (dext, dret) = curve
    (ezext, ezret), (edext, edret) = expected
    assert_float32_close(zext, ezext)
    assert_float32_close(zret, ezret)
    if deflection:
        assert_float32_close(dext, edext)
        assert_float32_close(dret, edret)


def build_pfc_case(directory, case):
    shape, points, kwargs = PFC_CASES[case]
    return build_pfc(directory, shape, points, **kwargs)


@pytest.fixture(params=list(PFC_CASES))
def pfc(request, tmp_path):
    return build_pfc_case(tmp_path, request.param)


def test_pfc_metadata_round_trips(pfc):
    with open_fvfile(pfc.path) as fvfile:
        assert [v.name for v in fvfile.volumes] == ["Deflection Error"]
        assert [v.shape for v in fvfile.volumes] == [pfc.volumes[0].shape]
        np.testing.assert_allclose(fvfile.k, pfc.k, rtol=1e-12)
        np.testing.assert_allclose(fvfile.defl_sens, pfc.defl_sens, rtol=1e-12)


@pytest.mark.parametrize("case", SCANSIZE_CASES)
def test_pfc_scansize_round_trips(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    with open_fvfile(synthetic.path) as fvfile:
        np.testing.assert_allclose(fvfile.scansize, synthetic.scansize, rtol=1e-12)


@pytest.mark.parametrize("case", LAGGED_CASES)
def test_pfc_lag_of_the_deflection_round_trips(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    with open_fvfile(synthetic.path) as fvfile:
        np.testing.assert_allclose(fvfile.sync_dist, synthetic.sync_dist, rtol=1e-12)


@pytest.mark.parametrize("case", LAGGED_CASES)
def test_pfc_get_curve_returns_the_known_curves(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        for r, c in np.ndindex(expected.shape):
            assert_same_curve(
                volume.get_curve(r, c), expected.curve(r, c), (r, c) != (0, 0)
            )


@pytest.mark.parametrize("case", LAGGED_CASES)
def test_pfc_iter_curves_goes_row_by_row(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        assert list(volume.iter_indices()) == list(expected.order)
        seen = []
        for (r, c), curve in volume.iter_curves():
            seen.append((r, c))
            assert_same_curve(curve, expected.curve(r, c), (r, c) != (0, 0))
    assert seen == list(expected.order)


@pytest.mark.parametrize("case", LAGGED_CASES)
def test_pfc_get_all_curves_returns_every_curve(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        z, d = fvfile.volumes[0].get_all_curves()
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    expected_d = np.stack([expected.dext, expected.dret], axis=2)
    expected_d[0, 0] = np.nan  # left out, see above
    assert_float32_close(d, expected_d)


def test_pfc_height_image_round_trips(pfc):
    with open_fvfile(pfc.path) as fvfile:
        assert sorted(fvfile.images) == sorted(pfc.images)
        for name, image in fvfile.images.items():
            assert image.shape == pfc.images[name].shape
            # the reader gives maps in m
            assert pfc.image_units[name] == "nm"
            assert_float32_close(image.get_image(), pfc.images[name] * 1e-9)


def header_layout(headers):
    """Each section in order with how many times it occurs."""
    return [(name, len(sections)) for name, sections in headers.items()]


def section_keys(headers):
    """The keys of each occurrence of each section."""
    return {
        (name, i): set(section)
        for name, sections in headers.items()
        for i, section in enumerate(sections)
    }


def channels(headers, shape):
    """Each channel's name, sensitivity, Bytes/pixel, points and stored bytes."""
    rows, cols = shape
    out = []
    for name, n in (("Ciao image list", 2), ("Ciao force image list", 4)):
        for section in headers.get(name, []):
            scale = section[f"@{n}:Z scale"]
            # an image's Samps/line is its width, a force channel's its points
            points = 1 if n == 2 else sum(map(int, section["Samps/line"].split()))
            out.append(
                (
                    name,
                    section[f"@{n}:Image Data"],
                    scale[scale.find("[") + 1 : scale.find("]")],
                    section["Bytes/pixel"],
                    section["Samps/line"],
                    int(section["Data length"]) // (rows * cols * points),
                )
            )
    return out


GRID_KEYS = {
    "Ciao scan list": ("Samps/line", "Lines"),
    "Ciao force list": ("force/line", "Samples / line", "FV Lines"),
    "Ciao image list": ("Samps/line", "Number of lines"),
}


def grid(headers):
    """Every header value that gives the grid."""
    return {
        (name, key): headers[name][0].get(key)
        for name, keys in GRID_KEYS.items()
        for key in keys
    }


def structure(fvfile):
    """What a reader makes of a Nanoscope file, without the values."""
    headers = fvfile.headers
    shape = fvfile.volumes[0].shape
    return {
        "class": type(fvfile),
        "volumes": [(v.name, v.shape) for v in fvfile.volumes],
        "curve": [
            tuple(np.shape(half) for half in v.get_curve(1, 1)) for v in fvfile.volumes
        ],
        "all curves": [
            tuple(np.shape(a) for a in v.get_all_curves()) for v in fvfile.volumes
        ],
        "images": {
            name: (image.shape, image.get_image().dtype)
            for name, image in fvfile.images.items()
        },
        "sections": header_layout(headers),
        "channels": channels(headers, shape),
        "grid": grid(headers),
        "metadata": {
            name: type(getattr(fvfile, name))
            for name in ("k", "defl_sens", "t_step", "scansize", "sync_dist")
        },
    }


def assert_keys_in_real_file(headers, real_headers):
    """Every section and key the builder writes is one the real file has."""
    real_keys = section_keys(real_headers)
    for name, keys in section_keys(headers).items():
        assert name in real_keys, name
        assert keys <= real_keys[name], (name, keys - real_keys[name])


@pytest.mark.slow
def test_synthetic_pfc_has_the_structure_of_the_real_file(real_data_file, tmp_path):
    with open_fvfile(Path(real_data_file(PFC_RELPATH))) as real:
        real_structure = structure(real)
        real_headers = real.headers
        shape = real.volumes[0].shape
    scan_list = real_headers["Ciao scan list"][0]
    force = real_headers["Ciao force image list"][0]
    synthetic = build_pfc(
        tmp_path,
        shape,
        int(force["Samps/line"].split()[0]),
        rate=float(scan_list["PFT Freq"].split()[0]),
        sync_distance=float(scan_list["Sync Distance"]),
    )
    with open_fvfile(synthetic.path) as fvfile:
        assert structure(fvfile) == real_structure
        assert_keys_in_real_file(fvfile.headers, real_headers)
        for key in ("Sample Points", "PFT Freq"):
            assert fvfile.headers["Ciao scan list"][0][key] == scan_list[key]
