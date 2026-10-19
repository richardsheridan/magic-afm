"""Asylum Research ARH5 files, the HDF5 conversions of ARDF force maps.

The file keeps the instrument's settings as one text attribute ``Note`` on the root
group: ``Key: value`` lines (some without the space), a variable-length string with
ASCII character set that holds UTF-8 bytes such as "°C". Lengths are in m, forces in N,
the spring constant in N/m and ``InvOLS``, the deflection sensitivity, in m/V. Maps sit
in the ``Image`` group as float32 (rows, cols) datasets named by quantity and scan
direction, e.g. ``HeightRetrace``.

A fast force map (``FFM``) holds one group per captured direction, numbered from ``0``
in capture order, so a retrace-only capture keeps its retrace in ``FFM/0``. Each group
carries ``Channels`` and ``Segments`` attributes (arrays of the same kind of string) and
one float32 (rows, cols, points) dataset per channel, each curve the extend half then
the retract half in time order. The layout, attribute types, chunking and note keys
follow a real conversion of a 32 x 32 map of 2000 point curves captured on retrace only.
Being square, it cannot say whether the first axis of the maps and curve datasets is the
slow one, the lines, or the fast one; the builder puts the lines first, as the reader
does. Which value of ``FastMapCapture`` means trace only is a guess (1; the real files
say 2 for retrace only and 3 for both).

A slow force map (``ForceMap/0``) has curves of different lengths, so each is its own
(channels, points) dataset named ``"r:c"``, beside an int32 ``Segments`` dataset of
(rows, cols, 3) indices: where the extend and retract segments end, then the index of
the curve's last sample, which closes the away segment. The channels are ``Raw``,
``Defl`` and ``ZSnsr``, and z is ``Raw``. The sample rate is ``NumPtsPerSec``; the note
also keeps ``FastMapZRate`` from the fast-map settings, which a slow map does not use.
The Segments convention and the channels follow a real conversion of a 5 x 6 slow map,
and so do the image names: ``MapHeight`` and ``MapAdhesion``, then per-curve values
``TriggerTime`` and ``Seconds`` in s, ``XLVDT`` and ``YLVDT`` in m, and
``HeaterTemperature`` and ``HeaterHumidity``, NaN without a heater. The meaning of
``"r:c"`` is not checked against one.
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

from pathlib import Path

import h5py
import numpy as np

from . import Index, SyntheticFile, SyntheticVolume, force_curves, point_counts

ASCII = h5py.string_dtype("ascii")
# FastMapCapture: which scan directions a fast force map kept
CAPTURES = {"trace": ("Trace",), "retrace": ("Retrace",), "both": ("Trace", "Retrace")}
CAPTURE_CODES = {"trace": 1, "retrace": 2, "both": 3}
# the drive signal runs this far above the Z sensor in the real file, in m
DRIVE_OFFSET = 1.27e-6


def _g(x: float) -> str:
    """A number as the note writes it: "0.462", "5.5471e-08", "2000000"."""
    return f"{x:.10g}"


def _strings(values) -> np.ndarray:
    return np.array([v.encode("ascii") for v in values], dtype=object)


def _note(lines: dict[str, str]) -> bytes:
    return "\n".join(f"{key}:{value}" for key, value in lines.items()).encode()


def _common_note(shape, k, invols, scan_size):
    """Note lines every map has, and the values they decode to."""
    rows, cols = shape
    k_str, invols_str = _g(k), _g(invols * 1e-9)
    fast_str, slow_str = _g(scan_size[0] * 1e-9), _g(scan_size[1] * 1e-9)
    lines = {
        "Version": "18.28.45",
        "ForceNote": "",
        "ScanSize": f" {_g(max(scan_size) * 1e-9)}",
        "FastScanSize": f" {fast_str}",
        "SlowScanSize": f" {slow_str}",
        "InvOLS": f" {invols_str}",
        "SpringConstant": f" {k_str}",
        "FastRatio": " 1",
        "SlowRatio": f" {_g(scan_size[1] / scan_size[0])}",
        "ScanPoints": f" {cols}",
        "ScanLines": f" {rows}",
        "Start Inside Temp": " 25.4 °C",
    }
    expected = {
        "k": float(k_str),
        "defl_sens": float(invols_str) * 1e9,
        "scansize": (float(fast_str) * 1e9, float(slow_str) * 1e9),
    }
    return lines, expected


def _maps(zext, dext, dret, k, rng) -> dict[str, np.ndarray]:
    """Per-pixel maps the Asylum software computes, in m, N and Pa."""
    z_top = np.nanmax(zext, axis=-1) * 1e-9
    young = rng.uniform(1e6, 1e7, z_top.shape)
    young[rng.random(z_top.shape) < 0.2] = np.nan  # failed fits
    maps = {
        "Adhesion": -np.nanmin(dret, axis=-1) * 1e-9 * k,
        "Force": np.nanmax(dext, axis=-1) * 1e-9 * k,
        "Height": -(z_top + DRIVE_OFFSET),
        "YoungsJKR": young,
        "ZSensor": -z_top,
    }
    return {name: m.astype(np.float32) for name, m in maps.items()}


# what each map holds; the file states no units
MAP_UNITS = {
    "Adhesion": "N",
    "Force": "N",
    "Height": "m",
    "YoungsJKR": "Pa",
    "ZSensor": "m",
}


# the images of a slow force map; the heater images have no unit the reader knows
FORCE_MAP_UNITS = {
    "MapAdhesion": "N",
    "MapHeight": "m",
    "TriggerTime": "s",
    "Seconds": "s",
    "XLVDT": "m",
    "YLVDT": "m",
    "HeaterTemperature": "V",
    "HeaterHumidity": "V",
}


def _image_units(images: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: MAP_UNITS[name.removesuffix("Retrace").removesuffix("Trace")]
        for name in images
    }


def build_arh5_ffm(
    directory: Path,
    shape: Index = (3, 4),
    points: int = 32,
    *,
    capture: str = "both",
    seed: int = 0,
    k: float = 0.462,
    invols: float = 55.471,
    scan_size: tuple[float, float] = (1000.0, 1000.0),
    rate: float = 1000.0,
    filename: str = "synthetic.h5",
) -> SyntheticFile:
    """Write a fast force map ARH5 file and say what it holds.

    ``points`` per half curve, so each curve holds twice as many; ``capture`` is
    "trace", "retrace" or "both". ``k`` in N/m, ``invols`` in nm/V,
    ``scan_size`` in nm along the fast then the slow axis, ``rate`` the curves
    per second. The volumes are named after the direction they hold.
    """
    cols = shape[1]
    rng = np.random.default_rng(seed)
    note, expected = _common_note(shape, k, invols, scan_size)
    n_total = 2 * points
    samples_str = _g(rate * n_total)
    note.update(
        {
            "NumPtsPerSec": samples_str,
            "ImagingMode": "Fast Force Map",
            "ScanDown": " 0",
            "FastMapZRate": f" {_g(rate)}",
            "FastMapCapture": f" {CAPTURE_CODES[capture]}",
        }
    )
    counts = point_counts(shape, points)
    volumes = []
    path = Path(directory) / filename
    with h5py.File(path, "w") as f:
        f.attrs.create("Note", _note(note), dtype=ASCII)
        image_group = f.create_group("Image")
        for i, direction in enumerate(CAPTURES[capture]):
            zext, zret, dext, dret = force_curves(
                rng, counts, counts, z_far=700.0, ramp=480.0
            )
            raw = (np.concatenate([zext, zret], axis=-1) * 1e-9).astype(np.float32)
            defl = (np.concatenate([dext, dret], axis=-1) * 1e-9).astype(np.float32)
            drive = (raw + DRIVE_OFFSET).astype(np.float32)
            group = f.create_group(f"FFM/{i}")
            group.attrs.create(
                "Channels", _strings(["Raw", "Defl", "Drive"]), dtype=ASCII
            )
            group.attrs.create("Segments", _strings(["Ext", "Ret"]), dtype=ASCII)
            chunks = (1, min(cols, 16), n_total)
            for name, data in (("Defl", defl), ("Drive", drive), ("Raw", raw)):
                group.create_dataset(name, data=data, chunks=chunks)
            for name, image in _maps(zext, dext, dret, k, rng).items():
                image_group.create_dataset(
                    name + direction, data=image, chunks=image.shape
                )
            # what the stored float32 metres are in nm
            z_nm = raw.astype(np.float64) * 1e9
            d_nm = defl.astype(np.float64) * 1e9
            volumes.append(
                SyntheticVolume(
                    direction,
                    z_nm[..., :points],
                    z_nm[..., points:],
                    d_nm[..., :points],
                    d_nm[..., points:],
                    counts,
                    counts,
                    tuple(np.ndindex(shape)),
                )
            )
        images = {name: image[()] for name, image in image_group.items()}
    return SyntheticFile(
        path=path,
        volumes=volumes,
        images=images,
        image_units=_image_units(images),
        k=expected["k"],
        defl_sens=expected["defl_sens"],
        t_step=1 / float(samples_str),
        scansize=expected["scansize"],
    )


def build_arh5_force_map(
    directory: Path,
    shape: Index = (3, 4),
    points: tuple[int, int] = (32, 32),
    *,
    ext_points=None,
    ret_points=None,
    away_points: int = 8,
    channels: tuple[str, ...] = ("Raw", "Defl", "ZSnsr"),
    samples_per_second: float = 2000.0,
    fast_map_rate: float = 6.1996,
    seed: int = 0,
    k: float = 0.462,
    invols: float = 55.471,
    scan_size: tuple[float, float] = (1000.0, 1000.0),
    filename: str = "synthetic.h5",
) -> SyntheticFile:
    """Write a slow force map ARH5 file and say what it holds.

    ``points`` are the extend and retract points of a full curve; ``ext_points`` and
    ``ret_points`` (an int or a (rows, cols) array) give each curve's own lengths, by
    default ``points``. Each curve ends with ``away_points`` more as the tip keeps
    withdrawing. ``channels`` are the stored channels in order; z is ``Raw``, or
    ``ZSnsr`` when ``Raw`` is left out. ``samples_per_second`` is the note's
    ``NumPtsPerSec``, and ``fast_map_rate`` its ``FastMapZRate``, a fast-map setting
    that has no bearing on this map. Other arguments are as for build_arh5_ffm.
    """
    rng = np.random.default_rng(seed)
    ext_points = point_counts(shape, points[0] if ext_points is None else ext_points)
    ret_points = point_counts(shape, points[1] if ret_points is None else ret_points)
    note, expected = _common_note(shape, k, invols, scan_size)
    samples_str = _g(samples_per_second)
    note.update(
        {
            "NumPtsPerSec": samples_str,
            "FMapScanDown": " 0",
            "FastMapZRate": f" {_g(fast_map_rate)}",
        }
    )
    zext, zret, dext, dret = force_curves(
        rng, ext_points, ret_points + away_points, z_far=700.0, ramp=480.0
    )
    path = Path(directory) / filename
    with h5py.File(path, "w") as f:
        f.attrs.create("Note", _note(note), dtype=ASCII)
        group = f.create_group("ForceMap/0")
        group.attrs.create("Channels", _strings(channels), dtype=ASCII)
        zname = "Raw" if "Raw" in channels else "ZSnsr"
        segments = np.stack(
            [
                ext_points,
                ext_points + ret_points,
                ext_points + ret_points + away_points - 1,
            ],
            axis=-1,
        ).astype(np.int32)
        group.create_dataset("Segments", data=segments)
        zsnsr_ext = zext.copy()
        for r, c in np.ndindex(shape):
            ne, nr = ext_points[r, c], ret_points[r, c] + away_points
            z = np.concatenate([zext[r, c, :ne], zret[r, c, :nr]]) * 1e-9
            d = np.concatenate([dext[r, c, :ne], dret[r, c, :nr]]) * 1e-9
            signals = {"Raw": z + DRIVE_OFFSET, "Defl": d, "ZSnsr": z}
            stored = {
                name: signals[name].astype(np.float32)
                for name in ("Raw", "Defl", "ZSnsr")
            }
            curve = np.stack([stored[name] for name in channels])
            group.create_dataset(f"{r}:{c}", data=curve)
            # what the stored float32 metres are in nm
            z_nm = stored[zname].astype(np.float64) * 1e9
            d_nm = stored["Defl"].astype(np.float64) * 1e9
            zsnsr_ext[r, c, :ne] = stored["ZSnsr"][:ne].astype(np.float64) * 1e9
            zext[r, c, :ne], zret[r, c, :nr] = z_nm[:ne], z_nm[ne:]
            dext[r, c, :ne], dret[r, c, :nr] = d_nm[:ne], d_nm[ne:]
        image_group = f.create_group("Image")
        # the maps the software computes come from the Z sensor
        maps = _maps(zsnsr_ext, dext, dret, k, rng)
        rows, cols = np.indices(shape)
        values = {
            "MapAdhesion": maps["Adhesion"],
            "MapHeight": maps["Height"],
            "TriggerTime": ext_points / float(samples_str),
            "Seconds": 3661431000.0 + rows * shape[1] + cols,
            "XLVDT": cols * scan_size[0] * 1e-9 / max(shape[1] - 1, 1),
            "YLVDT": rows * scan_size[1] * 1e-9 / max(shape[0] - 1, 1),
            "HeaterTemperature": np.full(shape, np.nan),
            "HeaterHumidity": np.full(shape, np.nan),
        }
        for name, image in values.items():
            image = np.asarray(image, np.float32)
            image_group.create_dataset(name, data=image, chunks=image.shape)
        images = {name: image[()] for name, image in image_group.items()}
    # the away segment is not part of the curve
    n_ret = ret_points.max()
    zret, dret = zret[..., :n_ret].copy(), dret[..., :n_ret].copy()
    past_end = np.arange(n_ret) >= ret_points[..., None]
    zret[past_end] = dret[past_end] = np.nan
    volume = SyntheticVolume(
        "FMAP", zext, zret, dext, dret, ext_points, ret_points, None
    )
    return SyntheticFile(
        path=path,
        volumes=[volume],
        images=images,
        image_units=dict(FORCE_MAP_UNITS),
        k=expected["k"],
        defl_sens=expected["defl_sens"],
        t_step=1 / float(samples_str),
        scansize=expected["scansize"],
    )
