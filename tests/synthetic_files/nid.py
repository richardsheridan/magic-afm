"""Nanosurf NID files holding one spectroscopy map.

A NID file is an INI-like header of ``[section]`` lines each followed by
``key=value`` lines, with CRLF line endings and UTF-8 text, then the two bytes
``#!`` and the datasets back to back in the order the ``[DataSet]`` section lists
them (group by group, channel by channel). Each dataset is a block of ``Lines``
rows of ``Points`` little-endian signed integers of ``SaveBits`` bits, which
decode to ``int * Dim2Range / 2**SaveBits + Dim2Range / 2 + Dim2Min`` in
``Dim2Unit``.

A spectroscopy map stores one row per curve in the order the map was acquired,
a serpentine over the grid: row 0 left to right, row 1 right to left, and so on,
``Map0`` in ``[DataSet\\SpecInfos\\SpecMapTable]`` giving the x and y extents of
the outermost points and the number of points along x then y. ``Spec forward``
rows hold the extend half in time order, far from the surface to near it;
``Spec backward`` rows hold the retract half stored far to near as well, the
reverse of time order. A row holds ``LineDim{n}Points`` points from its start
and zeros after them, because a force-triggered ramp stops early. The spring
constant is ``Prop0`` of the cantilever calibration and the deflection
sensitivity is the full scale of the ``In5`` scanhead input over its 10 V.

The layout follows a real Nanosurf file (software 3.10, a 10 x 10 map of 1024
point curves): its groups, channels, section and key names, the analysis
results the software saves beside the curves, and a scan image taken before the
map. Not taken from it: the SetView and ProcessorSet sections (display and
analysis settings) and most scanhead and parameter keys, which nothing reads.
"""

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import Index, SyntheticFile, SyntheticVolume, force_curves, point_counts

LENGTH_UNITS = {"m": 1.0, "µm": 1e-6, "nm": 1e-9}
FORCE_UNITS = {"N": 1.0, "nN": 1e-9}
SPRING_CONSTANT_UNITS = {"N/m": 1.0, "mN/m": 1e-3}
# the deflection input spans +-10 V
INPUT_VOLTS = 10.0
# full scale of the Z-Axis Sensor input of the scanhead, in m
Z_SENSOR_FULL_SCALE = 1.282e-05
# full scale of the Z-Axis output, in m
Z_AXIS_FULL_SCALE = 4.92e-06

# per-curve analysis results the software saves for a map: frame, points per curve,
# what the values are (each column 0 is the result; later columns are the running
# minimum and maximum, a guess at what the real file's chart columns hold)
RESULT_FRAMES = (
    ("SlopeOut00", 2, "ext window"),
    ("SlopeOut10", 3, "ext slope"),
    ("SlopeOut20", 1, "ext force"),
    ("MaxAdhesionOut00", 3, "ret adhesion force"),
    ("MaxAdhesionOut10", 1, "ret adhesion height"),
    ("SlopeOut01", 2, "ret window"),
    ("SlopeOut11", 3, "ret slope"),
    ("SlopeOut21", 1, "ret force"),
    ("SnapInOut00", 3, "ext adhesion force"),
    ("SnapInOut10", 1, "ext adhesion height"),
)
# result frames shown as maps: the file stores adhesion as a negative force and the
# map shows its magnitude; the extend half's adhesion is the snap-in
RESULT_IMAGES = {
    "MaxAdhesionOut00": ("MaxRetAdhesionForce", -1.0),
    "MaxAdhesionOut10": ("MaxRetAdhesionHeight", 1.0),
    "SnapInOut00": ("MaxExtAdhesionForce", -1.0),
    "SnapInOut10": ("MaxExtAdhesionHeight", 1.0),
}


def _g(x: float) -> str:
    """A number as the Nanosurf software writes it: six significant digits."""
    return f"{x:g}"


def map_order(shape: Index) -> tuple[Index, ...]:
    """Pixels in acquisition order: a serpentine starting along row 0."""
    rows, cols = shape
    return tuple(
        (r, c if r % 2 == 0 else cols - 1 - c) for r in range(rows) for c in range(cols)
    )


@dataclass
class _Dataset:
    group: int
    channel: int
    keys: dict[str, str]
    codes: np.ndarray  # (lines, points) integers as stored


def _quantize(values, dim_range: float, dim_min: float, bits: int):
    """Integer codes for values and what they decode to; NaN stores as code 0."""
    scale = dim_range / 2**bits
    offset = dim_range / 2 + dim_min
    valid = ~np.isnan(values)
    codes = np.rint(np.where(valid, values - offset, 0.0) / scale)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    if codes.min() < lo or codes.max() > hi:
        raise ValueError("values outside the dataset's range", dim_min, dim_range)
    codes = codes.astype(np.int64)
    decoded = np.where(valid, codes * scale + offset, np.nan)
    return codes, decoded


def _symmetric_range(full_scale: float) -> tuple[str, str]:
    """Dim2Range and Dim2Min of a raw input of the given full scale."""
    fs = float(_g(full_scale))
    return _g(2 * fs), _g(-fs)


def _result_range(values: np.ndarray) -> tuple[str, str]:
    """Dim2Range and Dim2Min of an analysis result.

    The real file's results span twice the magnitude of the first curve's value,
    from zero or from minus the span; this takes the largest value instead, so
    that no value can fall outside.
    """
    biggest = float(np.nanmax(np.abs(values))) or 1.0
    if np.nanmax(values) <= 0:
        span = float(_g(2 * biggest))
        return _g(span), _g(-span)
    if np.nanmin(values) >= 0:
        return _g(2 * biggest), "0"
    span = float(_g(4 * biggest))
    return _g(span), _g(-span / 2)


def _rows(grid: np.ndarray, counts: np.ndarray, order, width: int, reverse: bool):
    """Per-pixel (rows, cols, n) codes laid out as stored: one row per curve."""
    out = np.zeros((len(order), width), dtype=np.int64)
    for line, (r, c) in enumerate(order):
        n = counts[r, c]
        out[line, :n] = grid[r, c, :n][::-1] if reverse else grid[r, c, :n]
    return out


def _section(name: str, keys: dict[str, str]) -> str:
    return "".join(
        [f"[{name}]\r\n", *(f"{k}={v}\r\n" for k, v in keys.items()), "\r\n"]
    )


def _line_dims(counts_by_line, mins=None, ranges=None) -> dict[str, str]:
    keys = {}
    for n, points in enumerate(counts_by_line):
        keys[f"LineDim{n}Range"] = "0" if ranges is None else _g(ranges[n])
        keys[f"LineDim{n}Min"] = "0" if mins is None else _g(mins[n])
        keys[f"LineDim{n}Points"] = str(points)
    return keys


def _save_keys(bits: int) -> dict[str, str]:
    return {
        "SaveMode": "Binary",
        "SaveBits": str(bits),
        "SaveSign": "Signed",
        "SaveOrder": "Intel",
    }


def build_nid(
    directory: Path,
    shape: Index = (3, 4),
    points: tuple[int, int] = (32, 32),
    *,
    ext_points=None,
    ret_points=None,
    seed: int = 0,
    k: float = 25.03,
    defl_sens: float = 169.0,
    scan_size: tuple[float, float] = (8000.0, 8000.0),
    mod_time: float = 1.0,
    z_unit: str = "m",
    defl_unit: str = "N",
    k_unit: str = "N/m",
    save_bits: int = 32,
    scan_shape: Index = (8, 8),
    filename: str = "synthetic.nid",
) -> SyntheticFile:
    """Write a NID spectroscopy map of shape (rows, cols) and say what it holds.

    ``points`` are the forward and backward datapoints, the width of each row;
    ``ext_points`` and ``ret_points`` (an int or a (rows, cols) array) give each
    curve's own length up to that, all full by default. ``k`` is in N/m,
    ``defl_sens`` in nm/V, ``scan_size`` in nm along x then y, ``mod_time`` the
    seconds each ramp takes. The raw channels and the spring constant are
    written in ``z_unit``, ``defl_unit`` (a force, a length or "V") and
    ``k_unit``. ``scan_shape`` is the (lines, points) of the scan image saved
    with the map.
    """
    rows, cols = shape
    if rows < 2 or cols < 2:
        raise ValueError("the map's extents need two points along each axis", shape)
    if save_bits not in (16, 32):
        raise ValueError("SaveBits is 16 or 32", save_bits)
    n_fw, n_bw = points
    ext_points = point_counts(shape, n_fw if ext_points is None else ext_points)
    ret_points = point_counts(shape, n_bw if ret_points is None else ret_points)
    if ext_points.max() > n_fw or ret_points.max() > n_bw:
        raise ValueError("a curve is longer than its row", points)
    z_scale = LENGTH_UNITS[z_unit]
    k_str = _g(k / SPRING_CONSTANT_UNITS[k_unit])
    k_si = float(k_str) * SPRING_CONSTANT_UNITS[k_unit]
    sens_full_scale = _g(defl_sens * 1e-9 * INPUT_VOLTS)
    sens = float(sens_full_scale) / INPUT_VOLTS  # m/V
    if defl_unit in FORCE_UNITS:
        defl_scale = 1e-9 * k_si / FORCE_UNITS[defl_unit]  # unit per nm
        defl_full_scale = k_si * sens * INPUT_VOLTS / FORCE_UNITS[defl_unit]
    elif defl_unit in LENGTH_UNITS:
        defl_scale = 1e-9 / LENGTH_UNITS[defl_unit]
        defl_full_scale = sens * INPUT_VOLTS / LENGTH_UNITS[defl_unit]
    elif defl_unit == "V":
        defl_scale = 1e-9 / sens
        defl_full_scale = INPUT_VOLTS
    else:
        raise ValueError("unknown deflection unit", defl_unit)

    rng = np.random.default_rng(seed)
    zext, zret, dext, dret = force_curves(
        rng, ext_points, ret_points, z_far=-5200.0, ramp=500.0
    )
    order = map_order(shape)
    lines = len(order)
    by_line = tuple(zip(*order))
    fw_by_line = ext_points[by_line]
    bw_by_line = ret_points[by_line]

    # raw curves in the file's units, quantized on the grid in time order
    z_range = _symmetric_range(Z_SENSOR_FULL_SCALE / z_scale)
    d_range = _symmetric_range(defl_full_scale)
    z_lims = float(z_range[0]), float(z_range[1])
    d_lims = float(d_range[0]), float(d_range[1])
    zext_codes, zext_file = _quantize(zext * 1e-9 / z_scale, *z_lims, save_bits)
    zret_codes, zret_file = _quantize(zret * 1e-9 / z_scale, *z_lims, save_bits)
    dext_codes, dext_file = _quantize(dext * defl_scale, *d_lims, save_bits)
    dret_codes, dret_file = _quantize(dret * defl_scale, *d_lims, save_bits)

    datasets: list[_Dataset] = []
    spec_frames = (
        (0, "Spec forward", n_fw, fw_by_line, zext_file, False),
        (1, "Spec backward", n_bw, bw_by_line, zret_file, True),
    )
    curve_codes = {
        (0, 0): dext_codes,
        (0, 7): zext_codes,
        (1, 0): dret_codes,
        (1, 7): zret_codes,
    }
    counts_of = {0: ext_points, 1: ret_points}
    for group, frame, width, counts, z_file, reverse in spec_frames:
        # Dim0 is the Z-Axis each line ramps over; here the span of its sensor
        line_z = [z_file[r, c, : counts_of[group][r, c]] for r, c in order]
        mins = [float(np.min(z)) for z in line_z]
        ranges = [float(np.max(z) - np.min(z)) for z in line_z]
        dim0_min = min(mins)
        dim0_range = max(m + s for m, s in zip(mins, ranges)) - dim0_min
        for channel, name, unit, (dim_range, dim_min) in (
            (0, "Deflection", defl_unit, d_range),
            (7, "Z-Axis Sensor", z_unit, z_range),
        ):
            keys = {
                "Version": "3",
                "Points": str(width),
                "Lines": str(lines),
                "Frame": frame,
                "CurLine": str(lines - 1),
                "Dim0Name": "Z-Axis",
                "Dim0Unit": z_unit,
                "Dim0Range": _g(dim0_range),
                "Dim0Min": _g(dim0_min),
                "Dim1Name": "SpecPoint",
                "Dim1Range": str(lines - 1),
                "Dim1Min": "1",
                "Dim2Name": name,
                "Dim2Unit": unit,
                "Dim2Range": dim_range,
                "Dim2Min": dim_min,
                **_line_dims(counts, mins, ranges),
                **_save_keys(save_bits),
            }
            codes = _rows(
                curve_codes[group, channel], counts_of[group], order, width, reverse
            )
            datasets.append(_Dataset(group, channel, keys, codes))

    # tip-sample separation and deflection, the inputs of the slope and adhesion
    # analyses, stored like the raw curves
    separations = (
        zext_file - dext_file / defl_scale * 1e-9 / z_scale,
        zret_file - dret_file / defl_scale * 1e-9 / z_scale,
    )
    indentation_frames = (
        ("IndentationZSensorFwd0", 0, "Tip-Sample Separation", z_unit, separations[0]),
        ("IndentationZSensorBwd0", 1, "Tip-Sample Separation", z_unit, separations[1]),
        ("IndentationDeflFwd0", 0, "Deflection", defl_unit, dext_file),
        ("IndentationDeflBwd0", 1, "Deflection", defl_unit, dret_file),
    )
    group = 2
    for frame, half, name, unit, values in indentation_frames:
        if name == "Deflection":
            dim_range, dim_min = d_range
        else:
            span = _g(2 * float(np.nanmax(np.abs(values))))
            dim_range, dim_min = span, _g(-float(span))
        grid_codes, _ = _quantize(values, float(dim_range), float(dim_min), save_bits)
        width = (n_fw, n_bw)[half]
        counts = (fw_by_line, bw_by_line)[half]
        keys = _result_keys(group, frame, width, lines, name, unit, dim_range, dim_min)
        keys.update(_line_dims(counts))
        keys.update(_save_keys(save_bits))
        codes = _rows(grid_codes, counts_of[half], order, width, bool(half))
        datasets.append(_Dataset(group, 0, keys, codes))
        group += 1

    results = _results(
        zext_file, zret_file, dext_file, dret_file, ext_points, ret_points
    )
    images = {}
    image_units = {}
    force_unit = defl_unit
    slope_unit = f"{defl_unit}/{z_unit}"
    for frame, width, what in RESULT_FRAMES:
        column0 = results[what]
        unit = (
            z_unit
            if ("height" in what or "window" in what)
            else slope_unit
            if "slope" in what
            else force_unit
        )
        if width == 2:  # the two ends of the fit window
            values = np.stack([column0[0], column0[1]], axis=-1)
        else:
            column0 = np.asarray(column0)
            line_values = column0[by_line]
            running = (
                np.minimum.accumulate(line_values),
                np.maximum.accumulate(line_values),
            )
            columns = [column0]
            for extra in running[: width - 1]:
                grid = np.empty(shape)
                grid[by_line] = extra
                columns.append(grid)
            values = np.stack(columns, axis=-1)
        dim_range, dim_min = _result_range(values)
        grid_codes, decoded = _quantize(
            values, float(dim_range), float(dim_min), save_bits
        )
        keys = _result_keys(
            group, frame, width, lines, f"NameZ{group - 2}", unit, dim_range, dim_min
        )
        keys.update(_line_dims([width] * lines))
        keys.update(_save_keys(save_bits))
        codes = grid_codes[by_line]
        datasets.append(_Dataset(group, 0, keys, codes))
        if frame in RESULT_IMAGES:
            image_name, sign = RESULT_IMAGES[frame]
            images[image_name] = sign * decoded[..., 0]
            image_units[image_name] = unit
        group += 1

    # the map's height is where each extend half stopped, its highest force
    images["MapHeight"] = np.take_along_axis(
        zext_file, ext_points[..., None] - 1, axis=-1
    )[..., 0]
    image_units["MapHeight"] = z_unit

    scan_lines, scan_points = scan_shape
    for frame in ("Scan forward", "Scan backward"):
        topography = -4.6e-6 + rng.normal(0.0, 20e-9, scan_shape)
        for channel, name, full_scale in (
            (1, "Z-Axis", Z_AXIS_FULL_SCALE),
            (7, "Z-Axis Sensor", Z_SENSOR_FULL_SCALE),
        ):
            dim_range, dim_min = _symmetric_range(full_scale / z_scale)
            codes, _ = _quantize(
                topography / z_scale, float(dim_range), float(dim_min), save_bits
            )
            keys = {
                "Version": "2",
                "Points": str(scan_points),
                "Lines": str(scan_lines),
                "Frame": frame,
                "CurLine": str(scan_lines - 1),
                "Dim0Name": "X*",
                "Dim0Unit": "m",
                "Dim0Range": _g(scan_size[0] * 1e-9),
                "Dim0Min": "0",
                "Dim1Name": "Y*",
                "Dim1Unit": "m",
                "Dim1Range": _g(scan_size[1] * 1e-9),
                "Dim1Min": "0",
                "Dim2Name": name,
                "Dim2Unit": z_unit,
                "Dim2Range": dim_range,
                "Dim2Min": dim_min,
                **_save_keys(save_bits),
            }
            datasets.append(_Dataset(group, channel, keys, codes))
        group += 1

    # the map extents are the positions of its outermost points
    half_x = (scan_size[0] - scan_size[0] / cols) / 2 * 1e-9
    half_y = (scan_size[1] - scan_size[1] / rows) / 2 * 1e-9
    xmin, xmax, ymin, ymax = _g(-half_x), _g(half_x), _g(-half_y), _g(half_y)
    scansize = (
        (float(xmax) - float(xmin)) * cols / (cols - 1) * 1e9,
        (float(ymax) - float(ymin)) * rows / (rows - 1) * 1e9,
    )
    mod_time_str = _g(mod_time)

    header = _header(
        datasets,
        shape=shape,
        points=points,
        map_row=f"{xmin};{xmax};{ymin};{ymax};{cols};{rows};0;1",
        k_str=k_str,
        k_unit=k_unit,
        sens_full_scale=sens_full_scale,
        defl_unit=defl_unit,
        defl_full_scale=_g(defl_full_scale),
        mod_time=mod_time_str,
        scan_size=scan_size,
        scan_shape=scan_shape,
    )
    dtype = f"<i{save_bits // 8}"
    path = Path(directory) / filename
    with path.open("wb") as f:
        f.write(header.encode("utf-8"))
        f.write(b"#!")
        for dataset in datasets:
            f.write(dataset.codes.astype(dtype).tobytes())

    # the reader's curves: z in nm, deflection in nm of cantilever bend
    def nm(z_file, d_file):
        return z_file * z_scale * 1e9, d_file / defl_scale

    zext_nm, dext_nm = nm(zext_file, dext_file)
    zret_nm, dret_nm = nm(zret_file, dret_file)
    volume = SyntheticVolume(
        "SpecMap",
        zext_nm,
        zret_nm,
        dext_nm,
        dret_nm,
        ext_points,
        ret_points,
        order,
    )
    return SyntheticFile(
        path=path,
        volumes=[volume],
        images=images,
        k=k_si,
        defl_sens=sens * 1e9,
        t_step=float(mod_time_str) / n_fw,
        scansize=scansize,
        image_units=image_units,
        units={"Z-Axis Sensor": z_unit, "Deflection": defl_unit, "k": k_unit},
    )


def _result_keys(group, frame, width, lines, name, unit, dim_range, dim_min):
    i = group - 2
    return {
        "Version": "3",
        "Points": str(width),
        "Lines": str(lines),
        "Frame": frame,
        "CurLine": str(lines - 1),
        "Dim0Name": f"NameX{i}",
        "Dim0Unit": f"Unit{i}",
        "Dim0Range": "0",
        "Dim0Min": "0",
        "Dim1Range": "1",
        "Dim1Min": "0",
        "Dim2Name": name,
        "Dim2Unit": unit,
        "Dim2Range": dim_range,
        "Dim2Min": dim_min,
    }


def _results(zext, zret, dext, dret, ext_points, ret_points):
    """Per-curve analysis results from the decoded curves, in the file's units."""

    def at(a, i):
        return np.take_along_axis(a, i[..., None], axis=-1)[..., 0]

    mid, end = at(zext, ext_points // 2), at(zext, ext_points - 1)
    d_mid, d_end = at(dext, ext_points // 2), at(dext, ext_points - 1)
    ret_start, ret_mid = zret[..., 0], at(zret, ret_points // 2)
    dret_start, dret_mid = dret[..., 0], at(dret, ret_points // 2)

    def at_min(z, d):
        i = np.nanargmin(d, axis=-1)[..., None]
        return (
            np.take_along_axis(d, i, axis=-1)[..., 0],
            np.take_along_axis(z, i, axis=-1)[..., 0],
        )

    ret_force, ret_height = at_min(zret, dret)
    ext_force, ext_height = at_min(zext, dext)
    with np.errstate(divide="ignore", invalid="ignore"):
        ext_slope = np.abs(np.nan_to_num((d_end - d_mid) / (end - mid)))
        ret_slope = np.abs(
            np.nan_to_num((dret_start - dret_mid) / (ret_start - ret_mid))
        )
    return {
        "ext window": (mid, end),
        "ext slope": ext_slope,
        "ext force": np.abs(d_end),
        "ret adhesion force": ret_force,
        "ret adhesion height": ret_height,
        "ret window": (ret_mid, ret_start),
        "ret slope": ret_slope,
        "ret force": np.abs(dret_start),
        "ext adhesion force": ext_force,
        "ext adhesion height": ext_height,
    }


def _header(
    datasets,
    *,
    shape,
    points,
    map_row,
    k_str,
    k_unit,
    sens_full_scale,
    defl_unit,
    defl_full_scale,
    mod_time,
    scan_size,
    scan_shape,
) -> str:
    rows, cols = shape
    n_fw, n_bw = points
    groups = [
        ("Spec forward", "1", "8"),
        ("Spec backward", "1", "8"),
        *(
            (frame, str(254 + i), "1")
            for i, frame in enumerate(
                (
                    "IndentationZSensorFwd0",
                    "IndentationZSensorBwd0",
                    "IndentationDeflFwd0",
                    "IndentationDeflBwd0",
                    *(frame for frame, _, _ in RESULT_FRAMES),
                ),
                start=2,
            )
        ),
        ("Scan forward", "0", "8"),
        ("Scan backward", "0", "8"),
    ]
    dataset_keys = {"Version": "2", "GroupCount": str(len(groups))}
    for g, (name, ident, count) in enumerate(groups):
        dataset_keys[f"Gr{g}-Name"] = name
        dataset_keys[f"Gr{g}-ID"] = ident
        dataset_keys[f"Gr{g}-Count"] = count
    for d in datasets:
        dataset_keys[f"Gr{d.group}-Ch{d.channel}"] = f"DataSet-{d.group}:{d.channel}"
    dataset_keys.update(
        {
            "SetInfos": "DataSet-Info",
            "InfoSetCount": "4",
            "InfoSet0": "DataSetInfos",
            "InfoSet1": "Calibration",
            "InfoSet2": "Parameters",
            "InfoSet3": "SpecInfos",
        }
    )
    spec_info = {
        "Mod. output": "Z-Axis",
        "Modulation time": f"{mod_time} s",
        "Relative": "true",
        "Data points": str(n_fw),
    }
    scan_info = {
        "Image size": f"{_g(scan_size[0] / 1000)}µm",
        "Points": f"{scan_shape[1]} ",
        "Lines": f"{scan_shape[0]} ",
    }
    global_info = {
        "Measurement environment": "Air",
        "Op. mode": "Static Force",
        "Head type": "FlexAFM",
    }
    sections = [
        ("DataSet", dataset_keys),
        (
            "DataSet-Info",
            {
                "-- Spec --": "--------",
                **spec_info,
                "-- Scan --": "--------",
                **scan_info,
                "-- Global --": "--------",
                **global_info,
            },
        ),
        (
            "DataSet\\DataSetInfos",
            {
                "SubSectionCount": "3",
                "SubSection0": "Spec",
                "SubSection1": "Scan",
                "SubSection2": "Global",
            },
        ),
        ("DataSet\\DataSetInfos\\Spec", {"SubSectionCount": "0", **spec_info}),
        ("DataSet\\DataSetInfos\\Scan", {"SubSectionCount": "0", **scan_info}),
        ("DataSet\\DataSetInfos\\Global", {"SubSectionCount": "0", **global_info}),
        (
            "DataSet\\Calibration",
            {
                "SubSectionCount": "2",
                "SubSection0": "Scanhead",
                "SubSection1": "Cantilever",
            },
        ),
        (
            "DataSet\\Calibration\\Scanhead",
            {
                "SubSectionCount": "0",
                "Version": "9",
                "HeadTyp": "FlexAFM",
                "InCount": "6",
                "In0": f"0,Channel0,Deflection,{defl_unit},{defl_full_scale},0",
                "In1": f"1,Channel1,Z-Axis,m,{_g(Z_AXIS_FULL_SCALE)},0",
                "In2": "2,Channel2,Amplitude,V,10,0",
                "In3": "3,Channel3,Friction force,V,10,0",
                "In4": "4,Channel4,Tip current,A,0.0001,0",
                "In5": f"5,TipSignalDC,Deflection,m,{sens_full_scale},0",
            },
        ),
        (
            "DataSet\\Calibration\\Cantilever",
            {
                "SubSectionCount": "0",
                "Version": "2",
                "Name": "Synthetic",
                "PropCount": "2",
                "Prop0": f"D[{k_str}]*[{k_unit}]",
                "Prop1": "D[190000]*[Hz]",
            },
        ),
        (
            "DataSet\\Parameters",
            {
                "SubSectionCount": "2",
                "SubSection0": "Imaging",
                "SubSection1": "Spectroscopy",
            },
        ),
        (
            "DataSet\\Parameters\\Imaging",
            {
                "SubSectionCount": "0",
                "ScanRange": (
                    f"V[{_g(scan_size[0] * 1e-9)},{_g(scan_size[1] * 1e-9)},0]*[m,m,m]"
                ),
                "ScanOffset": "V[0,0,0]*[m,m,m]",
                "Datapoints": f"V[{scan_shape[1]},{scan_shape[0]},0]*[,,]",
            },
        ),
        (
            "DataSet\\Parameters\\Spectroscopy",
            {
                "SubSectionCount": "0",
                "ModuleLevel": "L[1]*[]",
                "Repetition": "L[1]*[]",
                "FwModTime": f"D[{mod_time}]*[s]",
                "BwModTime": f"D[{mod_time}]*[s]",
                "ModRelValue": "B[1]*[]",
                "FwModDatapoints": f"L[{n_fw}]*[]",
                "BwModDatapoints": f"L[{n_bw}]*[]",
                "FwModRange": "D[5e-07]*[m]",
                "BwModRange": "D[-5e-07]*[m]",
                "FwModMode": "L[1]*[]",
                "BwModMode": "L[0]*[]",
                "FwModStopMode": "L[1]*[]",
                "BwModStopMode": "L[1]*[]",
            },
        ),
        (
            "DataSet\\SpecInfos",
            {
                "SubSectionCount": "2",
                "SubSection0": "SpecHeader",
                "SubSection1": "SpecMapTable",
            },
        ),
        ("DataSet\\SpecInfos\\SpecHeader", {"SubSectionCount": "0", "SpecMode": "Map"}),
        (
            "DataSet\\SpecInfos\\SpecMapTable",
            {"SubSectionCount": "0", "Count": "1", "Map0": map_row},
        ),
        *((f"DataSet-{d.group}:{d.channel}", d.keys) for d in datasets),
    ]
    return "".join(_section(name, keys) for name, keys in sections) + "\r\n"
