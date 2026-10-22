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
# PeakForce z is a float32 cosine added to a float32 height. Curves and maps match the
# builder's float64 decoding to float32 rounding: rtol 1e-6 with an atol of 1e-6 of the
# array's largest magnitude. Points the file does not hold are NaN in the expected
# curves and are not compared. Metadata parsed from decimal text match to double
# rounding.

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from synthetic_files import open_fvfile
from synthetic_files.nanoscope import (
    INT16_MIN,
    INT32_MIN,
    build_pfc,
    build_spm,
    edit_nanoscope_header,
)

PFC_RELPATH = "test_cases/PS_PCL_SAA/QNM/onekhz.000.pfc"
SPM_RELPATH = "test_cases/A1_OCSOH-PMMA21k_300421.0_300425.spm"
# a force volume whose Ciao scan list grid (16 x 8) is not its 8 x 8 grid
STALE_GRID_SPM_RELPATH = "test_cases/DOE4_20nN_80nm.0_00012.spm"

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
    "square-32": ((4, 4), 32, {}),
    "sample-points-1024": ((4, 4), 128, {"sample_points": 1024}),
}
# Left out of the scan size test: "tall-128". The reader takes Scan Size as the longer
# side, the builder as the fast axis; no real file says which.
SCANSIZE_CASES = [case for case in PFC_CASES if case != "tall-128"]

# grid (rows, cols), points per half curve
SPM_CASES = {
    "square-512": ((4, 4), 512),
    "square-64": ((4, 4), 64),
    "wide-64": ((3, 6), 64),
    "tall-64": ((6, 3), 64),
}
# Left out of the scan size test: "tall-64", as for a capture.
SPM_SCANSIZE_CASES = [case for case in SPM_CASES if case != "tall-64"]


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


@pytest.mark.parametrize("case", list(PFC_CASES))
def test_pfc_lag_of_the_deflection_round_trips(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    with open_fvfile(synthetic.path) as fvfile:
        np.testing.assert_allclose(fvfile.sync_dist, synthetic.sync_dist, rtol=1e-12)
        np.testing.assert_allclose(
            fvfile.volumes[0].sync_dist, synthetic.sync_dist, rtol=1e-12
        )


def assert_time_step(fvfile, t_step, rtol=1e-12):
    """The file's time step and every volume's are t_step seconds."""
    np.testing.assert_allclose(fvfile.t_step, t_step, rtol=rtol)
    for volume in fvfile.volumes:
        step, unit = volume.step_info[2]
        assert unit == "s"
        np.testing.assert_allclose(step, t_step, rtol=rtol)


@pytest.mark.parametrize("case", list(PFC_CASES))
def test_pfc_time_step_is_one_drive_period_over_the_points_per_curve(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    with open_fvfile(synthetic.path) as fvfile:
        assert_time_step(fvfile, synthetic.t_step)


@pytest.mark.parametrize(
    "drive_frequency, t_step",
    [(0.9765625, 4.0e-6), (0.9, 1 / 900 / 256)],
    ids=["976.5625-Hz", "900-Hz"],
)
def test_pfc_time_step_is_the_written_drive_period_over_256_points(
    tmp_path, drive_frequency, t_step
):
    """The period comes from the drive frequency line, not the PFT Freq of 1 kHz."""
    synthetic = build_pfc(
        tmp_path, (4, 4), 128, rate=1.0, drive_frequency=drive_frequency
    )
    with open_fvfile(synthetic.path) as fvfile:
        assert_time_step(fvfile, t_step)


@pytest.mark.parametrize(
    "shape, points, ramp_rate",
    [((4, 4), 512, 1.00469), ((3, 6), 64, 2.0), ((6, 3), 128, 0.5)],
    ids=["square-512-1.00469-Hz", "wide-64-2-Hz", "tall-128-0.5-Hz"],
)
def test_spm_time_step_is_one_scan_rate_period_over_the_points_per_curve(
    tmp_path, shape, points, ramp_rate
):
    """One extend and retract of 2 x points samples spans 1 / Scan rate."""
    synthetic = build_spm(tmp_path, shape, points, ramp_rate=ramp_rate)
    with open_fvfile(synthetic.path) as fvfile:
        assert_time_step(fvfile, synthetic.t_step)


def _drop_drive_lines(lines):
    kept = [line for line in lines if "DriveFrequency3SoftHarmoniX:" not in line]
    assert len(kept) == len(lines) - 1
    return kept


@pytest.mark.parametrize(
    "edit", [None, _drop_drive_lines], ids=["only-group-3", "no-drive-line"]
)
def test_pfc_without_a_drive_frequency_takes_the_period_from_pft_freq(tmp_path, edit):
    """1 / (1 kHz) over 256 points."""
    synthetic = build_pfc(tmp_path, (4, 4), 128, rate=1.0, drive_frequency_line=False)
    if edit is not None:
        edit_nanoscope_header(synthetic.path, edit)
    with open_fvfile(synthetic.path) as fvfile:
        assert_time_step(fvfile, 1 / 1000 / 256)


def test_pfc_without_sample_points_takes_512_khz_over_pft_freq(tmp_path):
    """With no Sample Points the lag is Sync Distance x points x PFT Freq / 512 kHz."""
    synthetic = build_pfc(tmp_path, (4, 4), 64, rate=2.0, sync_distance=75.175)

    def drop_sample_points(lines):
        kept = [line for line in lines if not line.startswith("\\Sample Points:")]
        assert len(kept) == len(lines) - 1
        return kept

    edit_nanoscope_header(synthetic.path, drop_sample_points)
    lag = 75.175 * 128 * 2.0 / 512
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        np.testing.assert_allclose(fvfile.sync_dist, lag, rtol=1e-12)
        np.testing.assert_allclose(fvfile.volumes[0].sync_dist, lag, rtol=1e-12)
        z, d = fvfile.volumes[0].get_all_curves()
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


@pytest.mark.parametrize("case", list(PFC_CASES))
def test_pfc_get_curve_returns_the_known_curves(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        for r, c in np.ndindex(expected.shape):
            assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))


@pytest.mark.parametrize("case", list(PFC_CASES))
def test_pfc_iter_curves_goes_row_by_row(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        assert list(volume.iter_indices()) == list(expected.order)
        seen = []
        for (r, c), curve in volume.iter_curves():
            seen.append((r, c))
            assert_same_curve(curve, expected.curve(r, c))
    assert seen == list(expected.order)


@pytest.mark.parametrize("case", list(PFC_CASES))
def test_pfc_get_all_curves_returns_every_curve(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        z, d = fvfile.volumes[0].get_all_curves()
    assert np.isfinite(z).all() and np.isfinite(d).all()
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


def pfc_records(path):
    """Each pixel's deflection record as stored, and nm per LSB, without a reader.

    The INT16_MIN marker of a record, the last stored point of its first half, is
    left out, so a record holds 2 * points or 2 * points - 1 values.
    """
    data, sections = read_header(path)
    keys = dict(sections)
    scan_list, force = keys["Ciao scan list"], keys["Ciao force image list"]
    rows, cols = int(scan_list["Lines"]), int(scan_list["Samps/line"])
    points = int(force["Samps/line"].split()[0])
    count = rows * cols * 2 * points
    offset = int(force["Data offset"])
    stored = np.frombuffer(data, "<i2", count, offset).reshape(rows * cols, 2 * points)
    scale = force["@4:Z scale"]
    v_per_lsb = float(scale[scale.find("(") + 1 :].split()[0])
    nm_per_v = float(scan_list["@Sens. DeflSens"].split()[1])
    records = []
    for record in stored:
        if record[points - 1] == INT16_MIN:
            record = np.delete(record, points - 1)
        records.append(record.astype(np.int64))
    return records, v_per_lsb * nm_per_v


def assert_curves_hold_their_own_records(d, path):
    """Each curve holds every value of its own record, so none from another.

    d is (rows, cols, 2, points) in nm; a curve point off the record, where the
    marker was, may hold any value.
    """
    records, lsb = pfc_records(path)
    rows, cols = d.shape[:2]
    curves = np.asarray(d, dtype=np.float64).reshape(rows * cols, -1)
    for i, (curve, record) in enumerate(zip(curves, records, strict=True)):
        held = curve[~np.isnan(curve)]
        assert len(held) >= len(record)
        codes = np.rint(held / lsb).astype(np.int64)
        missing = Counter(record.tolist()) - Counter(codes.tolist())
        assert not missing, (np.unravel_index(i, (rows, cols)), missing)


def test_pfc_builder_writes_each_pixel_in_its_own_record(pfc):
    """The expected deflection of each pixel is its own stored record, rotated."""
    expected = pfc.volumes[0]
    d = np.stack([expected.dext, expected.dret], axis=2)
    assert_curves_hold_their_own_records(d, pfc.path)


@pytest.mark.parametrize("case", list(PFC_CASES))
def test_pfc_deflection_comes_from_its_own_record_only(tmp_path, case):
    synthetic = build_pfc_case(tmp_path, case)
    with open_fvfile(synthetic.path) as fvfile:
        _, d = fvfile.volumes[0].get_all_curves()
    assert np.isfinite(d).all()
    assert_curves_hold_their_own_records(d, synthetic.path)


def test_pfc_height_image_round_trips(pfc):
    with open_fvfile(pfc.path) as fvfile:
        assert sorted(fvfile.images) == sorted(pfc.images)
        for name, image in fvfile.images.items():
            assert image.shape == pfc.images[name].shape
            # the reader gives maps in m
            assert pfc.image_units[name] == "nm"
            assert image.units == "m"
            assert_float32_close(image.get_image(), pfc.images[name] * 1e-9)


@pytest.fixture(params=list(SPM_CASES))
def spm(request, tmp_path):
    shape, points = SPM_CASES[request.param]
    return build_spm(tmp_path, shape, points)


def test_spm_metadata_round_trips(spm):
    with open_fvfile(spm.path) as fvfile:
        assert [v.name for v in fvfile.volumes] == ["Deflection Error"]
        assert [v.shape for v in fvfile.volumes] == [spm.volumes[0].shape]
        assert fvfile.sync_dist is None
        np.testing.assert_allclose(fvfile.k, spm.k, rtol=1e-12)
        np.testing.assert_allclose(fvfile.defl_sens, spm.defl_sens, rtol=1e-12)


@pytest.mark.parametrize("case", SPM_SCANSIZE_CASES)
def test_spm_scansize_round_trips(tmp_path, case):
    synthetic = build_spm(tmp_path, *SPM_CASES[case])
    with open_fvfile(synthetic.path) as fvfile:
        np.testing.assert_allclose(fvfile.scansize, synthetic.scansize, rtol=1e-12)


def test_spm_get_curve_returns_the_known_curves(spm):
    expected = spm.volumes[0]
    with open_fvfile(spm.path) as fvfile:
        volume = fvfile.volumes[0]
        for r, c in np.ndindex(expected.shape):
            assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))


def test_spm_iter_curves_goes_row_by_row(spm):
    expected = spm.volumes[0]
    with open_fvfile(spm.path) as fvfile:
        volume = fvfile.volumes[0]
        assert list(volume.iter_indices()) == list(expected.order)
        seen = []
        for (r, c), curve in volume.iter_curves():
            seen.append((r, c))
            assert_same_curve(curve, expected.curve(r, c))
    assert seen == list(expected.order)


def test_spm_get_all_curves_returns_every_curve(spm):
    expected = spm.volumes[0]
    with open_fvfile(spm.path) as fvfile:
        z, d = fvfile.volumes[0].get_all_curves()
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


def test_spm_height_sensor_image_round_trips(spm):
    with open_fvfile(spm.path) as fvfile:
        assert sorted(fvfile.images) == sorted(spm.images)
        for name, image in fvfile.images.items():
            assert image.shape == spm.images[name].shape
            # the reader gives maps in m
            assert spm.image_units[name] == "nm"
            assert image.units == "m"
            assert_float32_close(image.get_image(), spm.images[name] * 1e-9)


# builder and points per half curve of each format, for the grid pitch cases
PITCH_BUILDERS = {"pfc": (build_pfc, 128), "spm": (build_spm, 64)}
# captures of one line, one column or one curve; kept out of the scan size cases, as
# (4, 1) is tall
THIN_GRIDS = [(1, 4), (4, 1), (1, 1)]


def grid_id(shape):
    return "x".join(map(str, shape))


@pytest.mark.parametrize("shape", [(4, 4), (3, 6), (6, 3), *THIN_GRIDS], ids=grid_id)
@pytest.mark.parametrize("fmt", PITCH_BUILDERS)
def test_volume_step_is_the_scan_size_over_the_grid(tmp_path, fmt, shape):
    """Each volume's x and y step in m is scansize over its columns and rows."""
    builder, points = PITCH_BUILDERS[fmt]
    synthetic = builder(tmp_path, shape, points)
    with open_fvfile(synthetic.path) as fvfile:
        fast, slow = fvfile.scansize
        assert fvfile.volumes
        for volume in fvfile.volumes:
            assert volume.shape == synthetic.volumes[0].shape
            rows, cols = volume.shape
            (x_step, x_unit), (y_step, y_unit) = volume.step_info[:2]
            assert (x_unit, y_unit) == ("m", "m")
            np.testing.assert_allclose(x_step, fast * 1e-9 / cols, rtol=1e-12)
            np.testing.assert_allclose(y_step, slow * 1e-9 / rows, rtol=1e-12)


@pytest.mark.parametrize("shape", THIN_GRIDS, ids=grid_id)
@pytest.mark.parametrize("fmt", PITCH_BUILDERS)
def test_capture_of_one_line_or_one_column_reads_its_curves(tmp_path, fmt, shape):
    builder, points = PITCH_BUILDERS[fmt]
    synthetic = builder(tmp_path, shape, points)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        assert [v.shape for v in fvfile.volumes] == [expected.shape]
        volume = fvfile.volumes[0]
        for r, c in np.ndindex(expected.shape):
            assert_same_curve(volume.get_curve(r, c), expected.curve(r, c))
        assert list(volume.iter_indices()) == list(expected.order)
        seen = []
        for (r, c), curve in volume.iter_curves():
            seen.append((r, c))
            assert_same_curve(curve, expected.curve(r, c))
        z, d = volume.get_all_curves()
    assert seen == list(expected.order)
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))
    if fmt == "pfc":
        assert np.isfinite(z).all() and np.isfinite(d).all()
        assert_curves_hold_their_own_records(d, synthetic.path)


# builder and height image name of each format, for files that also hold a Phase image
# on the unitless Sens. Phase soft scale of the real scan lists
PHASE_BUILDERS = {"pfc": (build_pfc, "Height"), "spm": (build_spm, "Height Sensor")}


@pytest.mark.parametrize("fmt", PHASE_BUILDERS)
def test_image_on_a_unitless_soft_scale_reads_in_volts(tmp_path, fmt):
    """Its values are the hard-scaled volts times the soft scale, declared as V."""
    builder, _ = PHASE_BUILDERS[fmt]
    synthetic = builder(tmp_path, phase_image=True)
    with open_fvfile(synthetic.path) as fvfile:
        assert sorted(fvfile.images) == sorted(synthetic.images)
        image = fvfile.images["Phase"]
        assert synthetic.image_units["Phase"] == "V"
        assert image.units == "V"
        assert_float32_close(image.get_image(), synthetic.images["Phase"])


@pytest.mark.parametrize("fmt", PHASE_BUILDERS)
def test_height_image_reads_the_same_beside_an_image_on_a_unitless_soft_scale(
    tmp_path, fmt
):
    builder, height_name = PHASE_BUILDERS[fmt]
    (tmp_path / "plain").mkdir()
    (tmp_path / "phase").mkdir()
    plain = builder(tmp_path / "plain")
    with_phase = builder(tmp_path / "phase", phase_image=True)
    with open_fvfile(plain.path) as fvfile:
        expected = fvfile.images[height_name].get_image()
    with open_fvfile(with_phase.path) as fvfile:
        image = fvfile.images[height_name]
        assert image.units == "m"
        np.testing.assert_array_equal(image.get_image(), expected)
    assert_float32_close(expected, with_phase.images[height_name] * 1e-9)


def read_header(path):
    """The file's bytes and its header as (section, {key: value}), without a reader."""
    data = path.read_bytes()
    sections = []
    for line in data[: data.index(b"\x1a")].decode("windows-1252").split("\r\n"):
        if line.startswith("\\*"):
            sections.append((line[2:], {}))
        elif line:
            key, _, value = line[1:].partition(": ")
            sections[-1][1][key] = value
    return data, sections


def test_spm_builder_marks_invalid_pixels_as_the_real_file_does(tmp_path):
    shape, points = (4, 4), 16
    synthetic = build_spm(tmp_path, shape, points, invalid_pixels=[(1, 2)])
    invalid = np.zeros(shape, dtype=bool)
    invalid[1, 2] = True
    data, sections = read_header(synthetic.path)

    def stored(keys, stored_shape):
        count = int(np.prod(stored_shape))
        offset = int(keys["Data offset"])
        return np.frombuffer(data, "<i4", count, offset).reshape(stored_shape)

    marked = []
    for name, keys in sections:
        if name == "Ciao force image list":
            values = stored(keys, (*shape, 2, points))
            is_min = values == INT32_MIN
            marked.append(is_min[:, :, 0].all(axis=-1).tolist())
            assert is_min.sum() == points * is_min[:, :, 0].all(axis=-1).sum()
        elif name == "Ciao image list":
            np.testing.assert_array_equal(stored(keys, shape) == INT32_MIN, invalid)
        if "Invalid Data Flag" in keys:
            assert keys["Invalid Data Flag"] == "None"
    # all but the first deflection channel, over the whole extend half
    none = np.zeros(shape, dtype=bool).tolist()
    assert marked == [none, invalid.tolist(), invalid.tolist(), invalid.tolist()]
    expected = synthetic.volumes[0]
    for half, nan in (
        (expected.zext, True),
        (expected.dext, False),
        (expected.zret, False),
        (expected.dret, False),
    ):
        unheld = np.broadcast_to((invalid & nan)[..., None], half.shape)
        np.testing.assert_array_equal(np.isnan(half), unheld)
    np.testing.assert_array_equal(np.isnan(synthetic.images["Height Sensor"]), invalid)


def test_spm_builder_can_write_a_stale_scan_list_grid(tmp_path):
    points = 16
    synthetic = build_spm(tmp_path, (8, 8), points, scan_list_grid=(8, 16))
    _, sections = read_header(synthetic.path)
    grid_values = {
        (name, key): keys[key]
        for name, keys in sections
        for key in GRID_KEYS.get(name, ())
    }
    assert grid_values == {
        ("Ciao scan list", "Samps/line"): "16",
        ("Ciao scan list", "Lines"): "8",
        ("Ciao force list", "force/line"): "8",
        ("Ciao force list", "Samples / line"): "8",
        ("Ciao force list", "FV Lines"): "8",
        ("Ciao image list", "Samps/line"): "8",
        ("Ciao image list", "Number of lines"): "8",
    }
    lengths = [
        int(keys["Data length"])
        for name, keys in sections
        if name == "Ciao force image list"
    ]
    assert lengths == [8 * 8 * 2 * points * 4] * 4
    assert synthetic.volumes[0].shape == (8, 8)


# grid (rows, cols) of the file, and of its Ciao scan list alone
FORCE_LIST_GRID_CASES = {
    "stale-scan-list": ((8, 8), (8, 16)),
    "transposed-scan-list": ((3, 6), (6, 3)),
    "zero-scan-list": ((4, 4), (0, 0)),
}


@pytest.mark.parametrize("case", FORCE_LIST_GRID_CASES)
def test_spm_grid_comes_from_the_force_list(tmp_path, case):
    """Rows are the force list's FV Lines and columns its force/line."""
    shape, scan_list_grid = FORCE_LIST_GRID_CASES[case]
    synthetic = build_spm(tmp_path, shape, 16, scan_list_grid=scan_list_grid)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        assert [v.shape for v in fvfile.volumes] == [shape]
        np.testing.assert_allclose(fvfile.scansize, synthetic.scansize, rtol=1e-12)
        volume = fvfile.volumes[0]
        (x_step, _), (y_step, _) = volume.step_info[:2]
        z, d = volume.get_all_curves()
    fast, slow = synthetic.scansize
    rows, cols = shape
    np.testing.assert_allclose(x_step, fast * 1e-9 / cols, rtol=1e-12)
    np.testing.assert_allclose(y_step, slow * 1e-9 / rows, rtol=1e-12)
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


# force sections in file order, unlike the A1 file's
FORCE_SECTION_CASES = {
    "repeated": ("deflection", "height sensor", "deflection 2.5 V", "height sensor"),
    "later-deflection-differs": (
        "deflection",
        "height sensor",
        "other deflection",
        "height sensor",
    ),
    "later-height-sensor-differs": (
        "deflection",
        "height sensor",
        "deflection 2.5 V",
        "other height sensor",
    ),
    "peak-force-error-first": ("peak force error", "deflection", "height sensor"),
    "height-error-first": ("height error", "deflection", "height sensor"),
    "unknown-name-first": ("adhesion", "deflection", "height sensor"),
}


@pytest.mark.parametrize("case", FORCE_SECTION_CASES)
def test_spm_pairs_the_first_deflection_error_with_the_first_height(tmp_path, case):
    """One volume, from the first of each name; sections of other names go unread."""
    sections = FORCE_SECTION_CASES[case]
    synthetic = build_spm(tmp_path, (4, 4), 16, force_sections=sections)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        assert [(v.name, v.shape) for v in fvfile.volumes] == [
            ("Deflection Error", (4, 4))
        ]
        z, d = fvfile.volumes[0].get_all_curves()
    assert_float32_close(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_float32_close(d, np.stack([expected.dext, expected.dret], axis=2))


def assert_nan_where_expected(actual, expected):
    """NaN exactly where the expected values are, close to them elsewhere."""
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    if not np.isnan(expected).all():
        assert_float32_close(actual, expected)


# samples written as INT32_MIN: channel, pixel, half (0 extend, 1 retract), point
INVALID_SAMPLES = [
    ("height sensor", (0, 1), 1, 5),
    ("height sensor", (2, 0), 0, 0),
    ("deflection", (2, 3), 0, 3),
    ("deflection", (3, 0), 1, 0),
    ("deflection", (3, 3), 0, 15),
]


@pytest.mark.parametrize(
    "kwargs",
    [{"invalid_pixels": [(1, 2)]}, {"invalid_samples": INVALID_SAMPLES}],
    ids=["invalid-pixel", "invalid-samples"],
)
def test_spm_int32_min_force_samples_read_nan(tmp_path, kwargs):
    """Each INT32_MIN sample of z or deflection, in either half, reads NaN."""
    synthetic = build_spm(tmp_path, (4, 4), 16, **kwargs)
    expected = synthetic.volumes[0]
    with open_fvfile(synthetic.path) as fvfile:
        volume = fvfile.volumes[0]
        curves = {(r, c): volume.get_curve(r, c) for r, c in np.ndindex(volume.shape)}
        z, d = volume.get_all_curves()
    for (r, c), ((zext, zret), (dext, dret)) in curves.items():
        (ezext, ezret), (edext, edret) = expected.curve(r, c)
        for actual, half in (
            (zext, ezext),
            (zret, ezret),
            (dext, edext),
            (dret, edret),
        ):
            assert_nan_where_expected(actual, half)
    assert_nan_where_expected(z, np.stack([expected.zext, expected.zret], axis=2))
    assert_nan_where_expected(d, np.stack([expected.dext, expected.dret], axis=2))


def test_spm_int32_min_image_pixels_read_nan(tmp_path):
    synthetic = build_spm(tmp_path, (4, 4), 16, invalid_pixels=[(1, 2)])
    with open_fvfile(synthetic.path) as fvfile:
        image = fvfile.images["Height Sensor"].get_image()
    assert_nan_where_expected(image, synthetic.images["Height Sensor"] * 1e-9)


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
            [tuple(map(np.shape, part)) for part in v.get_curve(1, 1)]
            for v in fvfile.volumes
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
        for key in (
            "Sample Points",
            "PFT Freq",
            "@2:DriveFrequency3SoftHarmoniX",
            "@3:DriveFrequency3SoftHarmoniX",
        ):
            assert fvfile.headers["Ciao scan list"][0][key] == scan_list[key]


@pytest.mark.slow
def test_real_pfc_time_step_is_one_drive_period_over_256_points(real_data_file):
    """1 / (976.5625 Hz) / 256 = 4 us."""
    with open_fvfile(Path(real_data_file(PFC_RELPATH))) as fvfile:
        assert_time_step(fvfile, 4.0e-6)


@pytest.mark.slow
def test_real_pfc_first_pixel_has_its_own_deflection(real_data_file):
    with open_fvfile(Path(real_data_file(PFC_RELPATH))) as fvfile:
        volume = fvfile.volumes[0]
        _, first = volume.get_curve(0, 0)
        _, second = volume.get_curve(0, 1)
    assert not np.array_equal(np.stack(first), np.stack(second))


@pytest.mark.slow
def test_real_pfc_deflection_comes_from_its_own_record_only(real_data_file):
    path = Path(real_data_file(PFC_RELPATH))
    with open_fvfile(path) as fvfile:
        _, d = fvfile.volumes[0].get_all_curves()
    assert np.isfinite(d).all()
    assert_curves_hold_their_own_records(d, path)


@pytest.mark.slow
def test_synthetic_spm_has_the_structure_of_the_real_file(real_data_file, tmp_path):
    with open_fvfile(Path(real_data_file(SPM_RELPATH))) as real:
        real_structure = structure(real)
        real_headers = real.headers
        shape = real.volumes[0].shape
    force = real_headers["Ciao force image list"][0]
    synthetic = build_spm(tmp_path, shape, int(force["Samps/line"].split()[0]))
    with open_fvfile(synthetic.path) as fvfile:
        assert structure(fvfile) == real_structure
        assert_keys_in_real_file(fvfile.headers, real_headers)


@pytest.mark.slow
@pytest.mark.parametrize(
    "relpath", [SPM_RELPATH, STALE_GRID_SPM_RELPATH], ids=["A1", "DOE4"]
)
def test_real_spm_time_step_is_one_scan_rate_period_over_1024_points(
    real_data_file, relpath
):
    """1 / (1.00469 Hz) / 1024, whatever the PFT Freq and drive frequency."""
    with open_fvfile(Path(real_data_file(relpath))) as fvfile:
        assert_time_step(fvfile, 9.720038e-4, rtol=1e-6)


@pytest.mark.slow
def test_real_spm_with_a_stale_scan_list_has_the_force_list_grid(real_data_file):
    """One 8 x 8 volume over 500 x 500 nm, steps of 62.5 nm."""
    with open_fvfile(Path(real_data_file(STALE_GRID_SPM_RELPATH))) as fvfile:
        assert [(v.name, v.shape) for v in fvfile.volumes] == [
            ("Deflection Error", (8, 8))
        ]
        np.testing.assert_allclose(fvfile.scansize, (500.0, 500.0), rtol=1e-12)
        steps = fvfile.volumes[0].step_info[:2]
    assert [unit for _, unit in steps] == ["m", "m"]
    np.testing.assert_allclose([step for step, _ in steps], 62.5e-9, rtol=1e-12)


@pytest.mark.slow
def test_real_spm_with_a_stale_scan_list_has_a_z_ramp_in_each_half(real_data_file):
    """Each half of every z spans over 10 nm and runs steadily one way."""
    with open_fvfile(Path(real_data_file(STALE_GRID_SPM_RELPATH))) as fvfile:
        z, _ = fvfile.volumes[0].get_all_curves()
    index = np.arange(z.shape[-1])
    for r, c, half in np.ndindex(z.shape[:3]):
        curve = z[r, c, half].astype(np.float64)
        assert np.ptp(curve) > 10.0, ((r, c, half), np.ptp(curve))
        assert abs(np.corrcoef(index, curve)[0, 1]) > 0.9, (r, c, half)
    assert z.shape[:3] == (8, 8, 2)


# the one pixel of the real force volume whose z extend half and image are INT32_MIN
SPM_INVALID_PIXEL = (12, 3)


@pytest.mark.slow
def test_real_spm_invalid_pixel_has_a_nan_z_extend_and_real_deflection(real_data_file):
    """Only its z extend half is NaN; its z retract and its deflection are real."""
    with open_fvfile(Path(real_data_file(SPM_RELPATH))) as fvfile:
        volume = fvfile.volumes[0]
        (zext, zret), (dext, dret) = volume.get_curve(*SPM_INVALID_PIXEL)
        z, d = volume.get_all_curves()
    assert np.isnan(zext).all()
    for half in (zret, dext, dret):
        assert np.isfinite(half).all()
    nan_z = np.zeros(z.shape, dtype=bool)
    nan_z[(*SPM_INVALID_PIXEL, 0)] = True
    np.testing.assert_array_equal(np.isnan(z), nan_z)
    assert np.isfinite(d).all()


@pytest.mark.slow
def test_real_spm_height_sensor_image_is_nan_at_its_invalid_pixel_only(
    real_data_file,
):
    with open_fvfile(Path(real_data_file(SPM_RELPATH))) as fvfile:
        image = fvfile.images["Height Sensor"].get_image()
    invalid = np.zeros(image.shape, dtype=bool)
    invalid[SPM_INVALID_PIXEL] = True
    np.testing.assert_array_equal(np.isnan(image), invalid)
