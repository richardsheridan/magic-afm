"""Bruker Nanoscope force files: PeakForce capture (.pfc) and force volume (.spm).

A Nanoscope file opens with a text header in windows-1252 with CRLF line endings:
``\\*Section`` lines each followed by ``\\Key: value`` lines, ending with
``\\*File list end``, a SUB byte (0x1A) and zeros up to the ``Data length`` of the
``Force file list`` section, where the data begin. A section repeats once per channel:
one ``Ciao image list`` per image and one ``Ciao force image list`` per force channel,
each giving the ``Data offset`` and ``Data length`` of its little-endian signed
integers. Values decode through a ``Z scale`` line such as
``V [Sens. DeflSens] (0.0003750000 V/LSB) 2.500000 V``: the integer times a volts per
LSB times the sensitivity named in brackets, a ``@Sens. DeflSens: V 35.00000 nm/V`` line
of the ``Ciao scan list`` (``Sens. Zsens`` is in the ``Scanner list``). For a force
channel the volts per LSB is the number in parentheses; for an image it is the last
number, the full scale, over 2**(8 * Bytes/pixel). The builder prints the one as the
other, so the two readings agree; in the real captures they differ in the fifth
significant digit. ``Z offset`` lines are written as 0 V: no real file says whether or
how a nonzero offset applies (the real captures have 0 V on the image and -0.776 V on
the force channel).

A PeakForce capture stores no z, only the deflection: per pixel, one record of
``Samps/line`` points per half, lines one after another. A record is one period of the
oscillating drive and holds its first half in reverse time order, then its second half
in time order. The deflection lags the drive by ``Sync Distance`` samples of
``Sample Points`` per period, and each tap peaks that far into its record: at 68, 70 and
75 of 256 points in the real captures at 0.5, 1 and 2 kHz, whose Sample Points times PFT
Freq is always 512 kHz. So a record starts where the drive is nearest the surface, and
the curve of a pixel, from the farthest point of the drive to the farthest, runs from
the end of the record before its own into its own. The first pixel's curve starts before
the first record; the expected curves are NaN where the file does not hold them. The
first sample in time of every real record, the last stored point of its first half, is
INT16_MIN: a marker in place of a sample, so the expected curves are NaN there too. The
z of a curve is a model, not data: the drive is a cosine of ``Peak Force Amplitude``
about the pixel's ``Height`` image value, nearest the surface in the middle of the curve
once shifted by the lag's fraction of a point.

The layout, section order, keys and channel set follow a real capture (software
0x09000103, a 32 x 32 map of 128 + 128 point curves at 1 kHz). No real file settles:
whether ``Sample Points`` stays at 512 kHz over PFT Freq when a curve has other than 128
+ 128 points; on a grid that is not square, that the lines come first, that
``Scan Size`` is the fast axis and ``Aspect Ratio`` fast to slow (the builder's pixels
are square); in what order the records were taken, beyond running on across neighbouring
records, so the builder takes the stored order as time order and copies
``Capture direction`` and ``Frame direction`` as found; what point of the tap the Height
image gives; and the ``Sync Distance QNM`` key, which no real capture has but a newer
force volume carries beside ``Sync Distance``, so that the builder writes it the same
way and takes it as the lag. The skipped-point capture, whose ``force/line`` (16 0) is
half the scan list's ``Samps/line`` (32) and whose force ``Data length`` is a quarter of
a full capture's, is not built.

A force volume stores z and deflection as force channels of their own, in the set and
order of the real file: ``Deflection Error``, ``Peak Force Error``, ``Deflection Error``
again and ``Height Sensor``, then a ``Height Sensor`` image. Each force channel holds
per pixel, lines one after another, the extend half from the surface outward, the
reverse of time order, then the retract half from the surface outward, in time order.
Every channel is int32 in the file, but ``Bytes/pixel`` says 2 on the deflection and
peak force error channels and 4 on the heights: in both real force volumes it gives the
step the values sit on (a volts per LSB of the input's full scale over 2**16 or 2**32,
so the 2 byte channels hold values that fit in int16), not their width. The two
Deflection Error channels hold the same values and differ only in the full scale on
their Z scale line (0.5 V and 2.5 V); Peak Force Error holds small noise. The image is
minus the height at the nearest point of each curve, the first stored point of its
extend half. Every key that gives the grid agrees: ``Samps/line`` and ``Lines`` of the
``Ciao scan list``, ``force/line``, ``Samples / line`` and ``FV Lines`` of the
``Ciao force list``, and the image's ``Samps/line`` and ``Number of lines``.

The layout, section order, keys and channel set follow a real force volume (software
0x09400202, a 32 x 32 map of 512 + 512 point curves). That file ends with 12288 bytes no
header line describes; the builder leaves them out. No real file settles grids that are
not square (as for a capture), the meaning of the image's -6.021 V Z offset (written 0 V
here), how long a ramp takes, or the order the curves were taken. The builder's time
step takes a ramp, extend and retract, as 1 / ``Scan rate`` of the Ciao force list (the
real file's ``Forward vel.`` is twice the ramp size times that rate, so it says no
more), while the ramp script lines of the real file, ``Segment Samples`` 1041, 2083 and
1041 at a ``Sample Period`` of 12 us, make a ramp about 50 ms, not 1 s.
``FV Line Direction`` says ``Bidirection``, but neighbouring rows of the real image
match better as stored than with every other row reversed, so the builder stores every
line the same way.

Two arguments write what the real files hold and today's reader gets wrong, for
tests of a fix: ``scan_list_grid`` writes a Ciao scan list grid unlike the rest,
as a second real force volume has (16 x 8 where every other key says 8 x 8), and
``invalid_pixels`` writes INT32_MIN over the extend half of Peak Force Error,
the second Deflection Error and Height Sensor and into the image, as the real
file does at one pixel while its header says ``Invalid Data Flag: None``. The
expected extend halves and image are NaN at those pixels.
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

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import Index, SyntheticFile, SyntheticVolume, force_curves, point_counts

INT16_MIN = -(2**15)
INT32_MIN = -(2**31)
# the deflection and height sensor inputs span 24.576 V
INPUT_VOLTS = 24.576
# full scale of the Height image's Z scale line in the real captures, in V
HEIGHT_FULL_SCALE = 354.4026
# Sample Points times PFT Freq in every real capture
PFC_SAMPLE_RATE_KHZ = 512.0
PFC_HEADER_BYTES = 40960
# the Sens. Phase soft scale of every real scan list: a factor of 1 with no unit
PHASE_SENS = "1.000000"
SPM_HEADER_BYTES = 80960


def _g(x: float) -> str:
    """A number as the header writes most settings: "0.46", "139.325", "75"."""
    return f"{x:g}"


def _fixed(x: float, digits: int = 7) -> str:
    """A number in fixed point to 7 significant digits: "35.00000", "0.0003750000"."""
    if x == 0:
        return "0"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:.{decimals}f}"


def _z_scale(sens: str, v_per_lsb: str, value: str) -> str:
    return f"V [{sens}] ({v_per_lsb} V/LSB) {value} V"


def _v_per_lsb(bytes_per_pixel: int) -> str:
    """Volts per LSB of a 24.576 V input with values of the given size."""
    return _fixed(INPUT_VOLTS / 2 ** (8 * bytes_per_pixel))


def _aspect(shape: Index) -> str:
    """Aspect Ratio of a scan with square pixels: fast to slow, in lowest terms."""
    rows, cols = shape
    d = math.gcd(rows, cols)
    return f"{cols // d}:{rows // d}"


def _image_scan_size(fast: float, slow: float) -> str:
    """The two-valued Scan Size of an image, in um (written ~m) from 1 um up."""
    if max(fast, slow) >= 1000:
        return f"{_g(fast / 1000)} {_g(slow / 1000)} ~m"
    return f"{_g(fast)} {_g(slow)} nm"


def _header(sections, size: int) -> bytes:
    lines = []
    for name, keys in sections:
        lines.append(f"\\*{name}")
        lines.extend(f"\\{key}: {value}" for key, value in keys.items())
    lines.append("\\*File list end")
    text = "".join(line + "\r\n" for line in lines).encode("windows-1252")
    if len(text) >= size:
        raise ValueError("header longer than its Data length", len(text), size)
    return text + b"\x1a" + bytes(size - len(text) - 1)


def edit_nanoscope_header(path: Path, edit, size: int = PFC_HEADER_BYTES) -> None:
    """Apply edit(lines) to the header lines of a capture, keeping its size."""
    data = path.read_bytes()
    end = data.index(b"\x1a")
    lines = data[:end].decode("windows-1252").split("\r\n")
    text = "\r\n".join(edit(lines)).encode("windows-1252")
    assert len(text) < size
    header = text + b"\x1a" + bytes(size - len(text) - 1)
    path.write_bytes(header + data[size:])


def _quantize(values: np.ndarray, lsb: float, dtype) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes for values and what they decode to."""
    codes = np.rint(values / lsb)
    info = np.iinfo(dtype)
    if codes.min() <= info.min or codes.max() > info.max:
        raise ValueError("values outside the channel's range", lsb, dtype)
    codes = codes.astype(dtype)
    return codes, codes * lsb


def _phase_image(rng, shape: Index, bytes_per_pixel: int, dtype, entry: dict):
    """A Phase image on the unitless soft scale, with its Ciao image list entry.

    Gives the codes, their values in V and the entry, a copy of the given image's with
    the Phase Image Data and Z scale lines.
    """
    lsb = _v_per_lsb(bytes_per_pixel)
    full = _fixed(INPUT_VOLTS)
    codes, volts = _quantize(
        rng.uniform(-5.0, 5.0, shape), float(full) / 2 ** (8 * bytes_per_pixel), dtype
    )
    entry = entry | {
        "Data length": str(codes.nbytes),
        "@2:Image Data": 'S [Phase] "Phase"',
        "@2:Z scale": _z_scale("Sens. Phase", lsb, full),
        "@2:Z offset": _z_scale("Sens. Phase", lsb, "0"),
    }
    return codes, volts * float(PHASE_SENS), entry


def build_pfc(
    directory: Path,
    shape: Index = (4, 4),
    points: int = 128,
    *,
    seed: int = 0,
    k: float = 0.46,
    defl_sens: float = 35.0,
    z_sens: float = 7.01,
    scan_size: float = 2000.0,
    rate: float = 1.0,
    sync_distance: float = 139.325,
    sync_distance_qnm: float | None = None,
    sample_points: int | None = None,
    amplitude: float = 75.0,
    extend_marker: bool = True,
    phase_image: bool = False,
    filename: str = "synthetic.000.pfc",
) -> SyntheticFile:
    """Write a PeakForce capture of shape (rows, cols) and say what it holds.

    ``points`` per half curve, so each curve holds twice as many. ``k`` in N/m,
    ``defl_sens`` and ``z_sens`` (of the Height image) in nm/V, ``scan_size``
    the fast axis in nm, ``rate`` the PFT Freq in kHz, ``amplitude`` the Peak
    Force Amplitude in nm. ``sync_distance`` is the lag of the deflection behind
    the drive in samples of ``sample_points`` per period (by default 512 kHz
    over ``rate``); it must stay under half a period. ``sync_distance_qnm``
    adds that newer key, whose value is then the lag. ``extend_marker`` writes
    the INT16_MIN marker of the real captures. ``phase_image`` adds a Phase image
    after the Height image, on the scan list's unitless ``Sens. Phase`` soft scale
    (``V 1.000000 `` in the real files), so its values are in V.
    """
    rows, cols = shape
    n_curves = rows * cols
    if n_curves < 2:
        raise ValueError("a capture needs two pixels, one to precede the other", shape)
    npts = 2 * points
    rate_str = _g(rate)
    if sample_points is None:
        sample_points = round(PFC_SAMPLE_RATE_KHZ / float(rate_str))
    sync_str = _g(sync_distance)
    qnm_str = None if sync_distance_qnm is None else _g(sync_distance_qnm)
    lag = float(sync_str if qnm_str is None else qnm_str) * npts / sample_points
    lag_points = math.floor(lag)
    if not 0 <= lag_points < points:
        raise ValueError("the lag must be under half a period", lag, points)
    amplitude_str = _g(amplitude)
    k_str = _g(k)
    sens_str = _fixed(defl_sens)
    z_sens_str = _fixed(z_sens)
    v_per_lsb = _fixed(INPUT_VOLTS / 2**16)
    height_full = _fixed(HEIGHT_FULL_SCALE)
    height_v_per_lsb = _fixed(float(height_full) / 2**16)
    fast = float(_g(scan_size))
    slow = fast * rows / cols

    rng = np.random.default_rng(seed)
    counts = point_counts(shape, points)
    _, _, dext, dret = force_curves(
        rng, counts, counts, z_far=-amplitude, ramp=2 * amplitude
    )
    curves = np.concatenate([dext, dret], axis=-1).reshape(n_curves, npts)
    # Lay the curves end to end, each starting half a period before its tap peaks lag
    # points into its own record. The tip taps on at the last pixel after the map, which
    # fills the end of the last record.
    first = points - lag_points  # where the first record starts in curve 0
    stream = np.concatenate([curves, curves[-1:]]).ravel()
    stored = stream[first : first + n_curves * npts]
    codes, decoded = _quantize(stored, float(v_per_lsb) * float(sens_str), np.int16)
    if extend_marker:
        codes[::npts] = INT16_MIN
        decoded[::npts] = np.nan
    known = np.full_like(stream, np.nan)
    known[first : first + n_curves * npts] = decoded
    d = known[: n_curves * npts].reshape(rows, cols, npts)
    records = codes.reshape(n_curves, npts)
    force_data = np.concatenate(
        [records[:, :points][:, ::-1], records[:, points:]], axis=1
    ).astype("<i2")

    height_lsb = float(height_full) / 2**16 * float(z_sens_str)
    height_codes, height = _quantize(
        rng.uniform(100.0, 300.0, shape), height_lsb, np.int16
    )
    phase = 2 * np.pi * (np.arange(npts) - (lag - lag_points)) / npts
    z = height[..., None] - float(amplitude_str) * np.cos(phase)

    image_offset = PFC_HEADER_BYTES
    image_length = height_codes.nbytes
    force_offset = image_offset + image_length
    height_entry = {
        "Data offset": str(image_offset),
        "Data length": str(image_length),
        "Bytes/pixel": "2",
        "Start context": "OL1BIG",
        "Data type": "AFM",
        "Frame direction": "Down",
        "Samps/line": str(cols),
        "Number of lines": str(rows),
        "Aspect Ratio": _aspect(shape),
        "Scan Size": _image_scan_size(fast, slow),
        "Line Direction": "Retrace",
        "@2:Image Data": 'S [Height] "Height"',
        "@2:Z scale": _z_scale("Sens. Zsens", height_v_per_lsb, height_full),
        "@2:Z offset": _z_scale("Sens. Zsens", height_v_per_lsb, "0"),
    }
    image_sections = [("Ciao image list", height_entry)]
    image_codes = [height_codes.astype("<i2")]
    images = {"Height": height}
    image_units = {"Height": "nm"}
    if phase_image:
        phase_codes, phase, phase_entry = _phase_image(
            rng, shape, 2, np.int16, height_entry | {"Data offset": str(force_offset)}
        )
        force_offset += phase_codes.nbytes
        image_sections.append(("Ciao image list", phase_entry))
        image_codes.append(phase_codes.astype("<i2"))
        images["Phase"] = phase
        image_units["Phase"] = "V"
    scan_list = {
        "Parameter select": "Main",
        "Operating mode": "Image",
        "Scan Size": f"{_g(fast)} nm",
        "Samps/line": str(cols),
        "Lines": str(rows),
        "Aspect Ratio": _aspect(shape),
        "Capture direction": "Down",
        "Sample Points": str(sample_points),
        "Sync Distance": sync_str,
        "Peak Force Amplitude": amplitude_str,
        "PFT Freq": f"{rate_str} KHz",
        "@MicroscopeList": 'S [SoftHarmoniX] "PeakForce QNM"',
        "@Sens. DeflSens": f"V {sens_str} nm/V",
    }
    if qnm_str is not None:
        scan_list["Sync Distance QNM"] = qnm_str
    if phase_image:
        scan_list["@Sens. Phase"] = f"V {PHASE_SENS} "
    sections = [
        (
            "Force file list",
            {
                "Version": "0x09000103",
                "Date": "01:26:23 PM Thu Dec 28 2023",
                "Start context": "FVOL",
                "Data length": str(PFC_HEADER_BYTES),
            },
        ),
        ("Equipment list", {"Description": "MultiMode 8", "Microscope": "MultiMode 8"}),
        (
            "Scanner list",
            {"Scanner type": "AFM", "@Sens. Zsens": f"V {z_sens_str} nm/V"},
        ),
        ("Ciao scan list", scan_list),
        (
            "Ciao force list",
            {
                "Samps/line": f"{points} {points}",
                "force/line": f"{cols} 0",
                "Samples / line": f"{cols} 0",
                "Z direction": "Retract",
                "@4:Image Data": 'S [DeflectionError] "Deflection Error"',
            },
        ),
        *image_sections,
        (
            "Ciao force image list",
            {
                "Data offset": str(force_offset),
                "Data length": str(force_data.nbytes),
                "Bytes/pixel": "2",
                "Start context": "FOL",
                "Data type": "FORCE",
                "Frame direction": "Up",
                "Samps/line": f"{points} {points}",
                "Z direction": "Retract",
                "Spring Constant": k_str,
                "@4:Image Data": 'S [DeflectionError] "Deflection Error"',
                "@4:Z scale": _z_scale("Sens. DeflSens", v_per_lsb, "2.500000"),
                "@4:Z offset": _z_scale("Sens. DeflSens", v_per_lsb, "0"),
            },
        ),
    ]
    path = Path(directory) / filename
    with path.open("wb") as f:
        f.write(_header(sections, PFC_HEADER_BYTES))
        for codes in image_codes:
            f.write(codes.tobytes())
        f.write(force_data.tobytes())

    volume = SyntheticVolume(
        "Deflection Error",
        z[..., :points],
        z[..., points:],
        d[..., :points],
        d[..., points:],
        counts,
        counts,
        tuple(np.ndindex(shape)),
    )
    return SyntheticFile(
        path=path,
        volumes=[volume],
        images=images,
        k=float(k_str),
        defl_sens=float(sens_str),
        t_step=1 / (float(rate_str) * 1e3 * npts),
        scansize=(fast, slow),
        image_units=image_units,
        sync_dist=lag,
    )


def build_spm(
    directory: Path,
    shape: Index = (4, 4),
    points: int = 512,
    *,
    seed: int = 0,
    k: float = 1.9,
    defl_sens: float = 115.1046,
    z_sens: float = 840.1727,
    scan_size: float = 50.0,
    ramp_rate: float = 1.00469,
    scan_list_grid: Index | None = None,
    invalid_pixels: Iterable[Index] = (),
    phase_image: bool = False,
    filename: str = "synthetic.0_00000.spm",
) -> SyntheticFile:
    """Write a force volume of shape (rows, cols) and say what it holds.

    ``points`` per half curve, so each curve holds twice as many. ``k`` in N/m,
    ``defl_sens`` and ``z_sens`` (of the Height Sensor) in nm/V, ``scan_size``
    the fast axis in nm, ``ramp_rate`` the ramps per second. ``scan_list_grid``
    (rows, cols) replaces the grid of the Ciao scan list alone; ``invalid_pixels``
    lists the (row, col) of pixels marked invalid. Both write what the reader
    gets wrong today; see the module docstring. ``phase_image`` adds a Phase image
    after the Height Sensor image, on the scan list's unitless ``Sens. Phase`` soft
    scale (``V 1.000000 `` in the real files), so its values are in V.
    """
    rows, cols = shape
    scan_rows, scan_cols = shape if scan_list_grid is None else scan_list_grid
    invalid = list(invalid_pixels)
    k_str = _g(k)
    sens_str = _fixed(defl_sens)
    z_sens_str = _fixed(z_sens)
    force_sens_str = _fixed(float(k_str) * float(sens_str))
    rate_str = _g(ramp_rate)
    full_scale = _fixed(INPUT_VOLTS)
    fast = float(_g(scan_size))
    slow = fast * rows / cols

    rng = np.random.default_rng(seed)
    counts = point_counts(shape, points)
    zext, zret, dext, dret = force_curves(rng, counts, counts, z_far=-275.0, ramp=65.0)
    # values on the step their Bytes/pixel says, as in the real force volumes
    z_lsb = float(_v_per_lsb(4)) * float(z_sens_str)
    d_lsb = float(_v_per_lsb(2)) * float(sens_str)
    zext_codes, zext = _quantize(zext, z_lsb, np.int32)
    zret_codes, zret = _quantize(zret, z_lsb, np.int32)
    dext_codes, dext = _quantize(dext, d_lsb, np.int16)
    dret_codes, dret = _quantize(dret, d_lsb, np.int16)

    def stored(ext, ret):
        """(rows, cols, 2, points) int32 as stored: both halves from the surface."""
        return np.stack([ext[..., ::-1], ret], axis=2).astype("<i4")

    height = stored(zext_codes, zret_codes)
    deflection = stored(dext_codes, dret_codes)
    error = rng.integers(-5, 6, deflection.shape).astype("<i4")
    image_codes = -height[:, :, 0, 0]
    image_lsb = float(full_scale) / 2**32 * float(z_sens_str)
    image = image_codes * image_lsb
    deflection_2v5 = deflection.copy()
    for r, c in invalid:
        for channel in (error, deflection_2v5, height):
            channel[r, c, 0] = INT32_MIN
        image_codes[r, c] = INT32_MIN
        zext[r, c] = dext[r, c] = image[r, c] = np.nan

    # Image Data, sensitivity, Bytes/pixel, full scale of the Z scale and FV scale
    # lines, values
    defl = 'S [DeflectionError] "Deflection Error"'
    full = full_scale
    channels = [
        (defl, "Sens. DeflSens", 2, "0.5000000", "2.500000", deflection),
        ('S [] "Peak Force Error"', "Sens. ForceDeflSens", 2, full, full, error),
        (defl, "Sens. DeflSens", 2, "2.500000", "2.500000", deflection_2v5),
        ('S [ZSensor] "Height Sensor"', "Sens. ZsensSens", 4, full, full, height),
    ]
    offset = SPM_HEADER_BYTES
    force_sections = []
    for image_data, sens, bytes_per_pixel, value, fv_value, data in channels:
        lsb = _v_per_lsb(bytes_per_pixel)
        force_sections.append(
            (
                "Ciao force image list",
                {
                    "Data offset": str(offset),
                    "Data length": str(data.nbytes),
                    "Bytes/pixel": str(bytes_per_pixel),
                    "Start context": "FOL",
                    "Data type": "AFM",
                    "Frame direction": "Up",
                    "Invalid Data Flag": "None",
                    "Invalid Data Fill": "None",
                    "Samps/line": f"{points} {points}",
                    "Z direction": "Retract",
                    "Spring Constant": k_str,
                    "@4:Image Data": image_data,
                    "@4:Z scale": _z_scale(sens, lsb, value),
                    "@4:Z offset": _z_scale(sens, lsb, "0"),
                    "@4:FV scale": _z_scale(sens, lsb, fv_value),
                },
            )
        )
        offset += data.nbytes
    image_codes = image_codes.astype("<i4")
    height_entry = {
        "Data offset": str(offset),
        "Data length": str(image_codes.nbytes),
        "Bytes/pixel": "4",
        "Start context": "OL1BIG",
        "Data type": "AFM",
        "Frame direction": "Up",
        "Invalid Data Flag": "None",
        "Invalid Data Fill": "None",
        "Samps/line": str(cols),
        "Number of lines": str(rows),
        "Aspect Ratio": _aspect(shape),
        "Scan Size": _image_scan_size(fast, slow),
        "Line Direction": "Retrace",
        "@2:Image Data": 'S [ZSensor] "Height Sensor"',
        "@2:Z scale": _z_scale("Sens. ZsensSens", _v_per_lsb(4), full_scale),
        "@2:Z offset": _z_scale("Sens. ZsensSens", _v_per_lsb(4), "0"),
    }
    image_sections = [("Ciao image list", height_entry)]
    stored_images = [image_codes]
    images = {"Height Sensor": image}
    image_units = {"Height Sensor": "nm"}
    scan_list_phase = {}
    if phase_image:
        phase_offset = offset + image_codes.nbytes
        phase_codes, phase, phase_entry = _phase_image(
            rng, shape, 4, np.int32, height_entry | {"Data offset": str(phase_offset)}
        )
        image_sections.append(("Ciao image list", phase_entry))
        stored_images.append(phase_codes.astype("<i4"))
        images["Phase"] = phase
        image_units["Phase"] = "V"
        scan_list_phase["@Sens. Phase"] = f"V {PHASE_SENS} "
    sections = [
        (
            "Force file list",
            {
                "Version": "0x09400202",
                "Date": "05:50:40 PM Fri Apr 30 2021",
                "Start context": "FVOL",
                "Data length": str(SPM_HEADER_BYTES),
            },
        ),
        ("Equipment list", {"Description": "PF-TUNA Dimension Icon"}),
        ("Scanner list", {"Scanner type": "Dim 4000"}),
        (
            "Ciao scan list",
            {
                "Parameter select": "Main",
                "Operating mode": "Force Volume",
                "Scan Size": f"{_g(fast)} nm",
                "Samps/line": str(scan_cols),
                "Lines": str(scan_rows),
                "Aspect Ratio": _aspect(shape),
                "Capture direction": "Up",
                "PFT Freq": "1 kHz",
                "@MicroscopeList": 'S [SoftHarmoniX] "PeakForce QNM"',
                "@Sens. DeflSens": f"V {sens_str} nm/V",
                "@Sens. ZsensSens": f"V {z_sens_str} nm/V",
                "@Sens. ForceDeflSens": f"V {force_sens_str} nN/V",
                **scan_list_phase,
            },
        ),
        (
            "Ciao force list",
            {
                "Scan rate": rate_str,
                "Samps/line": f"{points} {points}",
                "force/line": str(cols),
                "Samples / line": str(cols),
                "Z direction": "Retract",
                "X Type": "Height Sensor",
                "FV Line Direction": "Bidirection",
                "FV Lines": str(rows),
                "FV Aspect Ratio": _aspect(shape),
                "@4:Image Data": 'S [DeflectionError] "Deflection Error"',
            },
        ),
        ("Probe List", {"Save/Load Tip Model": "Spherical"}),
        *force_sections,
        *image_sections,
    ]
    path = Path(directory) / filename
    with path.open("wb") as f:
        f.write(_header(sections, SPM_HEADER_BYTES))
        for *_, data in channels:
            f.write(data.tobytes())
        for codes in stored_images:
            f.write(codes.tobytes())

    volume = SyntheticVolume(
        "Deflection Error",
        zext,
        zret,
        dext,
        dret,
        counts,
        counts,
        tuple(np.ndindex(shape)),
    )
    return SyntheticFile(
        path=path,
        volumes=[volume],
        images=images,
        k=float(k_str),
        defl_sens=float(sens_str),
        t_step=1 / (float(rate_str) * 2 * points),
        scansize=(fast, slow),
        image_units=image_units,
    )
