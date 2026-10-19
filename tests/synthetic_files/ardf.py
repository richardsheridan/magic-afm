"""Asylum Research ARDF files, the fast and slow force maps the Asylum software saves.

An ARDF file is a run of little-endian sections, each opening with a 16 byte header: a
CRC32, the section's size in bytes, a four letter name and flags. The CRC (zlib's)
covers the rest of the section from the size field on. A table (``FTOC``, ``TTOC``,
``IMAG``, ``VOLM``, ``VTOC`` and the image's ``IBOX``) has a 32 byte header whose size
field counts only that header and whose CRC covers only it; after the name come the
table's full size, its number of slots and their stride, and then the slots, each an
entry section with a CRC of its own or zeros when unused. Tables keep spare slots: the
file table 32, the file text table 16, an image or volume table 4 and a text table 4.
Table headers carry flags 0x10000, those of ``IMAG`` and ``VOLM`` 0x30000, every other
section 0.

The file holds, in this order: the ``ARDF`` header; the file table, with one entry per
image and per volume pointing at its ``IMAG`` or ``VOLM``; a text table and the note it
points at, ``Key: value`` lines (some without the space) ending in CR, in windows-1252.
A ``TEXT`` section holds its index in its text table, the length of the text and the
text, then at least one zero byte up to a multiple of 8 bytes. Then each image: its
table, whose ``NEXT`` entry points at the next image or the first volume and whose
``THMB`` entry points at its thumbnail; a text table; an ``IDEF`` with points per line,
lines, the steps between pixels in m along the fast and the slow axis, their units, the
image's name and its units; the ``IBOX`` table with one ``IDAT`` per line of float32
values, line 0 first; and a ``GAMI``. The first image alone has three texts, its own
note just after its ``GAMI`` and two more at the end of the file; every other image has
one, at the end, whose index is 1 although it fills the table's first slot. Every image
says its units are m, whatever it holds: the maps are in m, N and Pa.

Then each volume, one per captured direction: its table, with a ``NEXT`` entry pointing
at the next volume if there is one and an ``NSET`` entry giving how many curves it
holds; an empty text table; a ``VDEF`` with points per line, lines, the steps along each
axis and in time, their units and the segment names "Ext;Ret;"; one ``VCHN`` per channel
(``Raw``, ``Defl``, ``Drive``, in m unless the builder is told otherwise); an ``XDEF``
naming the experiment's extra values; a ``VTOC`` with one slot per line, filled in the
order the lines were taken, each ``VOFF`` entry giving the line's first curve; and an
``MLOV``. The curves follow, in the order they were taken. Each is a ``VSET`` (its index
within its volume, line, point, a type, the offset of the previous curve of the same
volume and the offset just past this curve's data), a ``VNAM`` "LineNNNNPointNNNN" and
one ``VDAT`` per channel of float32 in its unit, the extend half then the retract half
in time order, after the indices of where each segment starts and of the last point. The
trace runs along a line from point 0, the retrace back from the last point; with both
kept, each line holds its trace then its retrace, of types 10 and 11, while a lone
direction has type 5. ``ScanDown: 1`` in the note puts line 0 first, ``ScanDown: 0`` the
last line. After the curves come the thumbnails, 8-bit renderings of the images 128
pixels along the longer side of the scan, and the texts at the end.

A scan stopped part way holds only the lines it reached: the volume's ``NSET`` counts
those curves, the ``VTOC`` slots past them are zeros, and each ``IBOX`` has zeros where
the missing lines' ``IDAT`` sections would be. The expected images are NaN on those
lines, as the file holds no value there.

The layout, field values, flags, padding and note keys follow twelve real files
from one instrument: 32 x 32 maps of 1000 + 1000 point curves with the trace,
the retrace or both kept, taken from line 0 or from the last line, complete or
stopped after 16 to 19 lines; maps of 64 lines of 32 points and of 32 lines of
64 points; scans twice as long along one axis as along the other; and 2500 +
2500 point curves. Built with the same settings, each comes out with the same
sections, sizes (texts aside), fields and links as the real file. The ``Drive``
channel of the real files sits about 1.1 um above ``Raw``, varying by about 10
nm from curve to curve; the builder adds a constant. The Height and ZSensor
images are minus the largest Drive and Raw value of each curve, as in the real
files; Force and Adhesion are close to the Asylum ones but not baseline
corrected; the thumbnails are scaled nearest-neighbour copies, where the real
ones are interpolated and wrap around. No reader looks at the thumbnails, the
image notes or the meaning of the flags, ``GAMI``, ``MLOV`` or ``XDEF``, which
are copied as found. No real file settles: whether a ``TEXT`` whose text ends
just on a multiple of 8 bytes gets 8 zero bytes after it, as the builder
writes, or none; how a ``VNAM`` grows for names longer than 17 characters;
whether an ``IDAT`` is padded when a line has an odd number of points; how a
thumbnail's shorter side is rounded for aspect ratios other than 1 and 2;
whether a scan can stop part way along a line (the builder stops after whole
lines); or a stopped scan that kept both directions, which the builder writes
as a complete one cut short. The main note's ``FMapScanPoints`` and
``FMapScanLines`` are the settings of a slow force map, not this grid, so the
builder leaves them out.

A slow force map, written with ``slow_map``, follows one real file of 5 lines of 6
points. Its single volume has a ``VDEF`` naming the segments "Ext;Ret;Away;", channels
``Raw``, ``Defl`` and ``ZSnsr``, and curves of type 2, each as long as its own extend,
retract and Away segments: a ramp stops early once the force triggers, and the Away
segment carries the tip further from the surface after the retract. Each ``VDAT`` gives
where the extend, retract and Away segments start and the index of the last point. The
curves snake: the first line taken runs from its last point, the next from point 0,
and so on, and each curve's index is its place on the grid, line by line, whatever the
order it was taken in. An ``XDAT`` with the experiment's values for the curve follows
each curve's ``VDAT`` sections. The note gives the ramp rate, ``ForceScanRate``, and
the sample rate, ``NumPtsPerSec``, so an untriggered curve has ``NumPtsPerSec /
ForceScanRate / 2`` points per half; its ``FastMapZRate`` is a fast force map setting
left over in the note, which says nothing of this map. The images' note sits after the
last image and no texts end the file. The images are named ``MapHeight`` and
``MapAdhesion``, as in the real file. No real file
settles which way the first line of a slow map runs when its line count is even, nor
how a slow map stopped part way or one with a dwell segment is laid out.
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

import os
import struct
import zlib
from pathlib import Path

import numpy as np

from . import Index, SyntheticFile, SyntheticVolume, force_curves, point_counts

HEADER = struct.Struct("<LL4sL")
TABLE = struct.Struct("<QLL")
TABLE_FLAGS, ITEM_FLAGS = 0x10000, 0x30000
POINTER = struct.Struct("<Q")
TOFF = struct.Struct("<QQ")
IDEF = struct.Struct("<LLQQdd32s32s32s32s")
VDEF = struct.Struct("<LL24sddd32s32s32s32sQ")
VCHN = struct.Struct("<32s32s")
VOFF = struct.Struct("<LLQQ")
VSET = struct.Struct("<LLLLQQ")
VNAM = struct.Struct("<LLLL32s")
VDAT = struct.Struct("<10L")
THUMBNAIL = struct.Struct("<LLLL")
THUMBNAIL_SIZE = 128
XDEF_SIZE = 96
XDEF_TEXT = b"XLVDT;YLVDT;Seconds;TriggerTime;HeaterTemperature;HeaterHumidity;"
CHANNELS = (("Raw", "m"), ("Defl", "m"), ("Drive", "m"))
# the lengths a channel can be written in, in m
LENGTH_UNITS = {"m": 1.0, "nm": 1e-9}
MAP_NAMES = ("Height", "ZSensor", "Adhesion", "Force", "YoungsJKR")
# what each map holds, whatever unit the file's IDEF gives it
MAP_UNITS = {
    "Height": "m",
    "ZSensor": "m",
    "Adhesion": "N",
    "Force": "N",
    "YoungsJKR": "Pa",
}
# FastMapCapture: which scan directions a fast force map kept, one volume each
CAPTURES = {"trace": ("Trace",), "retrace": ("Retrace",), "both": ("Trace", "Retrace")}
CAPTURE_CODES = {"trace": 1, "retrace": 2, "both": 3}
VSET_TYPES = {"trace": (5,), "retrace": (5,), "both": (10, 11)}
# the drive signal runs about this far above the Z sensor in the real files, in m
DRIVE_OFFSET = 1.1e-6
SLOW_CHANNELS = (("Raw", "m"), ("Defl", "m"), ("ZSnsr", "m"))
SLOW_MAP_UNITS = {"MapHeight": "m", "MapAdhesion": "N"}
SLOW_SEGMENTS = b"Ext;Ret;Away;"
SLOW_VSET_TYPE = 2
# force index, line, point, how many values, and the values XDEF names
XDAT = struct.Struct("<LLLL6d")
# the Z sensor reads within a few tenths of a nm of Raw in the real slow map, in m
ZSNSR_OFFSET = -0.1e-9
END_TEXT = (
    "Date:  2000-01-01\rTime:  12:00:00 PM\rSeconds:  3029572800\r"
    "End Inside Temp:  25.000 °C\rEnd Outside Temp:  22.000 °C\r"
    "End BaseLine: 0.15\rLog File:Start\rLog File:Stop\r"
)


def _g(x: float) -> str:
    """A number as the note writes it: "0.462", "5.5471e-08", "2000000"."""
    return f"{x:.10g}"


def _cstring(text: str) -> bytes:
    return text.encode("windows-1252")


def _note(lines: dict[str, str]) -> str:
    return "".join(f"{key}:{value}\r" for key, value in lines.items())


class _Sections:
    """A file built section by section; CRCs are filled in once pointers are.

    Offsets are positions in the file. One run of bytes, the hole, can be left
    unwritten: it is not held in memory, and the file holds zeros there, sparse where
    the filesystem allows.
    """

    def __init__(self):
        self.buf = bytearray()
        self.crcs: list[tuple[int, int]] = []
        self.hole_at = 0  # where the hole starts in buf
        self.hole = 0

    def tell(self) -> int:
        """The file offset of the next section."""
        return len(self.buf) + self.hole

    def skip(self, nbytes: int) -> None:
        """Leave nbytes unwritten before the next section."""
        if self.hole:
            raise ValueError("a file has one hole at most")
        self.hole_at, self.hole = len(self.buf), nbytes

    def _at(self, offset: int) -> int:
        """Where a file offset lies in buf."""
        if offset < self.hole_at:
            return offset
        if offset < self.hole_at + self.hole:
            raise ValueError("offset lies in the hole", offset)
        return offset - self.hole

    def add(self, name: bytes, payload: bytes = b"", flags: int = 0) -> int:
        offset = self.tell()
        size = HEADER.size + len(payload)
        self.buf += HEADER.pack(0, size, name, flags) + payload
        self.crcs.append((offset, size))
        return offset

    def table(self, name: bytes, slots: int, stride: int, flags: int) -> int:
        size = HEADER.size + TABLE.size + slots * stride
        offset = self.add(name, TABLE.pack(size, slots, stride), flags)
        self.buf += bytes(slots * stride)
        return offset

    def entry(self, table: int, slot: int, name: bytes, payload: bytes) -> int:
        _, slots, stride = TABLE.unpack_from(self.buf, self._at(table) + HEADER.size)
        size = HEADER.size + len(payload)
        if not (slot < slots and size == stride):
            raise ValueError("entry does not fit the table", name, slot, size)
        offset = table + HEADER.size + TABLE.size + slot * stride
        at = self._at(offset)
        self.buf[at : at + size] = HEADER.pack(0, size, name, 0) + payload
        self.crcs.append((offset, size))
        return offset

    def text(self, index: int, text: str) -> int:
        encoded = _cstring(text)
        fields = struct.pack("<LL", index, len(encoded))
        unpadded = HEADER.size + len(fields) + len(encoded)
        padding = bytes(8 - unpadded % 8)
        return self.add(b"TEXT", fields + encoded + padding)

    def patch(self, offset: int, fmt: struct.Struct, *values) -> None:
        fmt.pack_into(self.buf, self._at(offset) + HEADER.size, *values)

    def write(self, path: Path) -> None:
        """Fill in the CRCs and write the file, seeking over the hole."""
        for offset, size in self.crcs:
            at = self._at(offset)
            crc = zlib.crc32(self.buf[at + 4 : at + size])
            struct.pack_into("<L", self.buf, at, crc)
        with open(path, "wb") as file:
            file.write(self.buf[: self.hole_at])
            file.seek(self.hole, os.SEEK_CUR)
            file.write(self.buf[self.hole_at :])


def _maps(raw, defl, drive, points, k, rng) -> dict[str, np.ndarray]:
    """Per-pixel maps the Asylum software computes, in m, N and Pa."""
    raw, defl, drive = (a.astype(np.float64) for a in (raw, defl, drive))
    young = rng.uniform(1e7, 3e7, raw.shape[:2])
    young[rng.random(young.shape) < 0.2] = np.nan  # failed fits
    maps = {
        "Height": -drive.max(axis=-1),
        "ZSensor": -raw.max(axis=-1),
        "Adhesion": -defl[..., points:].min(axis=-1) * k,
        "Force": defl[..., :points].max(axis=-1) * k,
        "YoungsJKR": young,
    }
    return {name: m.astype(np.float32) for name, m in maps.items()}


def _in_unit(metres: np.ndarray, unit: str) -> np.ndarray:
    """float32 values in m as float32 values in unit, the same array for m."""
    if unit == "m":
        return metres
    return (metres.astype(np.float64) / LENGTH_UNITS[unit]).astype(np.float32)


def _thumbnail(image: np.ndarray, scan_size: tuple[float, float]) -> bytes:
    """The image as 8-bit grey levels, nearest neighbour, in the scan's aspect.

    The longer side of the scan gets 128 pixels. The fields are the rows (along
    the slow axis), the pixels per row and the bits per pixel.
    """
    lines, points = image.shape
    longer = max(scan_size)
    n_rows = round(THUMBNAIL_SIZE * scan_size[1] / longer)
    n_cols = round(THUMBNAIL_SIZE * scan_size[0] / longer)
    rows = np.arange(n_rows) * lines // n_rows
    cols = np.arange(n_cols) * points // n_cols
    grid = image[rows[:, None], cols].astype(np.float64)
    finite = np.isfinite(grid)
    low = grid[finite].min() if finite.any() else 0.0
    span = np.ptp(grid[finite]) if finite.any() else 0.0
    grey = np.where(finite, (grid - low) / (span or 1.0) * 255, 0.0)
    fields = THUMBNAIL.pack(n_rows, n_cols, 8, 0)
    return fields + grey.round().astype(np.uint8).tobytes()


def build_ardf(
    directory: Path,
    shape: Index = (4, 4),
    points: int = 32,
    *,
    capture: str = "trace",
    scan_down: bool = False,
    lines_taken: int | None = None,
    seed: int = 0,
    k: float = 0.462,
    invols: float = 55.471,
    scan_size: tuple[float, float] = (1000.0, 1000.0),
    rate: float = 1000.0,
    channel_units: dict[str, str] | None = None,
    gap: int = 0,
    filename: str = "synthetic.ARDF",
    slow_map: bool = False,
    ext_points=None,
    ret_points=None,
    away_points=8,
) -> SyntheticFile:
    """Write a fast or slow force map ARDF file and say what it holds.

    ``shape`` is (lines, points per line): rows are the slow axis, columns the fast one.
    ``points`` per half curve, so each curve holds twice as many; ``capture`` is
    "trace", "retrace" or "both". ``scan_down`` is the note's ScanDown: True takes line
    0 first, False the last line first. With ``lines_taken`` the scan stops after that
    many lines, a stopped scan. ``k`` in N/m, ``invols`` in nm/V, ``scan_size`` in nm
    along the fast then the slow axis, ``rate`` the curves per second. The volumes are
    named after the direction they hold. ``channel_units`` maps a channel name to the
    unit its ``VCHN`` declares and its curves are stored in, "m" or "nm", by default m.
    ``gap`` bytes of zeros, left unwritten so that the file is sparse where the
    filesystem allows, follow the file's text table, so that everything the file's
    pointers address (the note, images, volumes, curves, thumbnails and texts) sits that
    much further on.

    With ``slow_map`` the file is a slow force map of one volume, named "FMAP" as the
    reader names it. ``points`` is then the points per half of an untriggered curve,
    ``rate`` the untriggered curves per second, and ``ext_points``, ``ret_points`` and
    ``away_points`` the points of each curve's extend, retract and Away segments, an int
    or a (lines, points) array, the first two ``points`` unless given.
    """
    if slow_map:
        if capture != "trace":
            raise ValueError("a slow force map holds one volume", capture)
        return _build_slow_map(
            directory,
            shape,
            points,
            ext_points=points if ext_points is None else ext_points,
            ret_points=points if ret_points is None else ret_points,
            away_points=away_points,
            scan_down=scan_down,
            lines_taken=lines_taken,
            seed=seed,
            k=k,
            invols=invols,
            scan_size=scan_size,
            rate=rate,
            channel_units=channel_units,
            gap=gap,
            filename=filename,
        )
    if ext_points is not None or ret_points is not None:
        raise ValueError("a fast force map has halves of equal length", points)
    lines, cols = shape
    if lines < 2 or cols < 2:
        raise ValueError("a map needs at least two lines of two points", shape)
    lines_taken = lines if lines_taken is None else lines_taken
    if not 0 < lines_taken <= lines:
        raise ValueError("lines_taken must be within the grid", lines_taken, lines)
    units = dict(CHANNELS) | (channel_units or {})
    if set(units) != {name for name, _ in CHANNELS}:
        raise ValueError("channel_units names an unknown channel", channel_units)
    rng = np.random.default_rng(seed)
    directions = CAPTURES[capture]
    vset_types = dict(zip(directions, VSET_TYPES[capture], strict=True))
    n_total = 2 * points

    k_str, invols_str = _g(k), _g(invols * 1e-9)
    fast_str, slow_str = _g(scan_size[0] * 1e-9), _g(scan_size[1] * 1e-9)
    samples_str = _g(rate * n_total)
    smaller = min(scan_size)
    note = {
        "Version": "18.28.45",
        "FastMapCapture": str(CAPTURE_CODES[capture]),
        "NumPtsPerSec": samples_str,
        "ScanSize": f" {_g(max(scan_size) * 1e-9)}",
        "FastScanSize": f" {fast_str}",
        "SlowScanSize": f" {slow_str}",
        "ImagingMode": "Fast Force Map",
        "InvOLS": f" {invols_str}",
        "SpringConstant": f" {k_str}",
        "ScanDown": f" {int(scan_down)}",
        "FastRatio": f" {_g(scan_size[0] / smaller)}",
        "SlowRatio": f" {_g(scan_size[1] / smaller)}",
        "FastMapZRate": f" {_g(rate)}",
        "UserCalcUnit": " °",
    }
    image_note = {
        "ScanSize": note["ScanSize"],
        "FastScanSize": note["FastScanSize"],
        "SlowScanSize": note["SlowScanSize"],
        "ScanPoints": f" {cols}",
        "ScanLines": f" {lines}",
        "ImagingMode": note["ImagingMode"],
        "InvOLS": note["InvOLS"],
        "SpringConstant": note["SpringConstant"],
        "FastRatio": note["FastRatio"],
        "SlowRatio": note["SlowRatio"],
        "FastMapZRate": note["FastMapZRate"],
    }
    x_step = float(fast_str) / (cols - 1)
    y_step = float(slow_str) / (lines - 1)
    t_step = 1 / float(samples_str)

    # the lines in the order they were taken, and the points of each direction
    line_order = list(range(lines)) if scan_down else list(range(lines))[::-1]
    line_order = line_order[:lines_taken]
    point_orders = {"Trace": list(range(cols)), "Retrace": list(range(cols))[::-1]}
    acquired = np.zeros(shape, dtype=bool)
    acquired[line_order] = True

    counts = point_counts(shape, points)
    stored = {}
    written = {}
    volumes = []
    for direction in directions:
        zext, zret, dext, dret = force_curves(
            rng, counts, counts, z_far=700.0, ramp=480.0
        )
        raw = (np.concatenate([zext, zret], axis=-1) * 1e-9).astype(np.float32)
        defl = (np.concatenate([dext, dret], axis=-1) * 1e-9).astype(np.float32)
        drive = (raw + DRIVE_OFFSET).astype(np.float32)
        stored[direction] = (raw, defl, drive)
        # the channels in their units, and what those float32 values are in nm, NaN
        # where never taken
        written[direction] = tuple(
            _in_unit(a, units[name])
            for a, (name, _) in zip((raw, defl, drive), CHANNELS, strict=True)
        )
        z_file, d_file, _ = written[direction]
        z_nm = np.where(
            acquired[..., None],
            z_file.astype(np.float64) * LENGTH_UNITS[units["Raw"]] * 1e9,
            np.nan,
        )
        d_nm = np.where(
            acquired[..., None],
            d_file.astype(np.float64) * LENGTH_UNITS[units["Defl"]] * 1e9,
            np.nan,
        )
        order = tuple((r, c) for r in line_order for c in point_orders[direction])
        volumes.append(
            SyntheticVolume(
                direction,
                z_nm[..., :points],
                z_nm[..., points:],
                d_nm[..., :points],
                d_nm[..., points:],
                counts,
                counts,
                order,
                None if lines_taken == lines else acquired,
            )
        )
    maps = {d: _maps(*stored[d], points, k, rng) for d in directions}
    images = {name + d: maps[d][name] for name in MAP_NAMES for d in directions}

    f = _Sections()
    f.add(b"ARDF")
    ftoc = f.table(b"FTOC", 32, 24, TABLE_FLAGS)
    ttoc = f.table(b"TTOC", 16, 32, TABLE_FLAGS)
    f.skip(gap)
    f.entry(ttoc, 0, b"TOFF", TOFF.pack(0, f.text(0, _note(note))))

    image_tables = []
    next_entry = None
    for i, (name, image) in enumerate(images.items()):
        imag = f.table(b"IMAG", 4, 24, ITEM_FLAGS)
        f.entry(ftoc, i, b"IMAG", POINTER.pack(imag))
        if next_entry is not None:
            f.patch(next_entry, POINTER, imag)
        next_entry = f.entry(imag, 0, b"NEXT", POINTER.pack(0))
        image_ttoc = f.table(b"TTOC", 4, 32, TABLE_FLAGS)
        image_tables.append((imag, image_ttoc))
        idef = (cols, lines, 0, 0, x_step, y_step, b"m", b"m", _cstring(name), b"m")
        f.add(b"IDEF", IDEF.pack(*idef))
        ibox = f.table(b"IBOX", lines, HEADER.size + 4 * cols, TABLE_FLAGS)
        for line in sorted(line_order):
            f.entry(ibox, line, b"IDAT", image[line].astype("<f4").tobytes())
        f.add(b"GAMI")
        if i == 0:
            text = f.text(0, _note(image_note))
            f.entry(image_ttoc, 0, b"TOFF", TOFF.pack(0, text))

    vtocs = {}
    for i, direction in enumerate(directions):
        volm = f.table(b"VOLM", 4, 24, ITEM_FLAGS)
        f.entry(ftoc, len(images) + i, b"VOLM", POINTER.pack(volm))
        f.patch(next_entry, POINTER, volm)
        slot = 0
        if i + 1 < len(directions):
            next_entry = f.entry(volm, slot, b"NEXT", POINTER.pack(0))
            slot += 1
        f.entry(volm, slot, b"NSET", POINTER.pack(lines_taken * cols))
        f.table(b"TTOC", 4, 32, TABLE_FLAGS)
        vdef = (cols, lines, b"", x_step, y_step, t_step, b"m", b"m", b"s")
        f.add(b"VDEF", VDEF.pack(*vdef, b"Ext;Ret;", 2))
        for channel, _ in CHANNELS:
            f.add(b"VCHN", VCHN.pack(_cstring(channel), _cstring(units[channel])))
        xdef = struct.pack("<LL", 0, len(XDEF_TEXT)) + XDEF_TEXT
        f.add(b"XDEF", xdef + bytes(XDEF_SIZE - HEADER.size - len(xdef)))
        vtocs[direction] = f.table(b"VTOC", lines, 40, TABLE_FLAGS)
        f.add(b"MLOV")

    force_index = dict.fromkeys(directions, 0)
    previous = dict.fromkeys(directions, 0)
    segments = (0, points, n_total - 1, 0, 0)
    for slot, line in enumerate(line_order):
        for direction in directions:
            for p in point_orders[direction]:
                index = force_index[direction]
                fields = (index, line, p, vset_types[direction], previous[direction])
                vset = f.add(b"VSET", VSET.pack(*fields, 0))
                if p == point_orders[direction][0]:
                    voff = VOFF.pack(index, line, p, vset)
                    f.entry(vtocs[direction], slot, b"VOFF", voff)
                name = f"Line{line:04d}Point{p:04d}".encode()
                f.add(b"VNAM", VNAM.pack(index, line, p, len(name), name))
                for channel, data in enumerate(written[direction]):
                    vdat = VDAT.pack(index, line, p, n_total, channel, *segments)
                    f.add(b"VDAT", vdat + data[line, p].astype("<f4").tobytes())
                # the curve points on to whatever section follows it
                f.patch(vset, VSET, *fields, f.tell())
                previous[direction] = vset
                force_index[direction] += 1

    stored_images = {}
    for (imag, _), (name, image) in zip(image_tables, images.items(), strict=True):
        # the thumbnail shows the zeros the file holds on lines never taken
        held = np.where(acquired, image, np.float32(0))
        thumbnail = f.add(b"THMB", _thumbnail(held, scan_size))
        f.entry(imag, 1, b"THMB", POINTER.pack(thumbnail))
        stored_images[name] = np.where(acquired, image, np.float32(np.nan))
    for i, (_, image_ttoc) in enumerate(image_tables):
        if i == 0:
            ranges = "".join(
                f"Display Offset {j}: 0\rDisplay Range {j}: "
                f"{np.nanmax(image) - np.nanmin(image):.1g}\rColorMap {j}: Grays256\r"
                for j, image in enumerate(images.values())
            )
            # this copy of ScanDown is the opposite of the note's in every real file
            text = f"ScanDown:{int(not scan_down)}\r{ranges}SavedPlanefit:None\r"
            f.entry(image_ttoc, 1, b"TOFF", TOFF.pack(1, f.text(1, text)))
        else:
            text = f.text(1, "SavedPlanefit:None\r")
            f.entry(image_ttoc, 0, b"TOFF", TOFF.pack(1, text))
    first_ttoc = image_tables[0][1]
    f.entry(first_ttoc, 2, b"TOFF", TOFF.pack(2, f.text(2, END_TEXT)))

    path = Path(directory) / filename
    f.write(path)
    return SyntheticFile(
        path=path,
        volumes=volumes,
        images=stored_images,
        k=float(k_str),
        defl_sens=float(invols_str) * 1e9,
        t_step=t_step,
        scansize=(float(fast_str) * 1e9, float(slow_str) * 1e9),
        image_units={
            name + d: MAP_UNITS[name] for name in MAP_NAMES for d in directions
        },
        units=units,
    )


def _build_slow_map(
    directory: Path,
    shape: Index,
    points: int,
    *,
    ext_points,
    ret_points,
    away_points,
    scan_down: bool,
    lines_taken: int | None,
    seed: int,
    k: float,
    invols: float,
    scan_size: tuple[float, float],
    rate: float,
    channel_units: dict[str, str] | None,
    gap: int,
    filename: str,
) -> SyntheticFile:
    """Write a slow force map ARDF file; build_ardf with slow_map says how."""
    lines, cols = shape
    if lines < 2 or cols < 2:
        raise ValueError("a map needs at least two lines of two points", shape)
    lines_taken = lines if lines_taken is None else lines_taken
    if not 0 < lines_taken <= lines:
        raise ValueError("lines_taken must be within the grid", lines_taken, lines)
    units = dict(SLOW_CHANNELS) | (channel_units or {})
    if set(units) != {name for name, _ in SLOW_CHANNELS}:
        raise ValueError("channel_units names an unknown channel", channel_units)
    rng = np.random.default_rng(seed)
    counts = tuple(
        point_counts(shape, n) for n in (ext_points, ret_points, away_points)
    )

    k_str, invols_str = _g(k), _g(invols * 1e-9)
    fast_str, slow_str = _g(scan_size[0] * 1e-9), _g(scan_size[1] * 1e-9)
    samples_str = _g(rate * 2 * points)
    smaller = min(scan_size)
    note = {
        "Version": "16.08.202",
        "ForceScanRate": f" {_g(rate)}",
        "NumPtsPerSec": f" {samples_str}",
        "FMapScanPoints": f" {cols}",
        "FMapScanLines": f" {lines}",
        "ScanSize": f" {_g(max(scan_size) * 1e-9)}",
        "FastScanSize": f" {fast_str}",
        "SlowScanSize": f" {slow_str}",
        "ImagingMode": "Contact",
        "InvOLS": f" {invols_str}",
        "SpringConstant": f" {k_str}",
        "ScanDown": f" {int(scan_down)}",
        "FastRatio": f" {_g(scan_size[0] / smaller)}",
        "SlowRatio": f" {_g(scan_size[1] / smaller)}",
        # left over from a fast force map, and not a whole fraction of this one's rate
        "FastMapZRate": f" {_g(3 * rate)}",
        "UserCalcUnit": " °",
    }
    image_note = {
        key: note[key]
        for key in (
            "ScanSize",
            "FastScanSize",
            "SlowScanSize",
            "ImagingMode",
            "InvOLS",
            "SpringConstant",
            "FastRatio",
            "SlowRatio",
            "FMapScanPoints",
            "FMapScanLines",
            "FastMapZRate",
        )
    }
    x_step = float(fast_str) / (cols - 1)
    y_step = float(slow_str) / (lines - 1)
    t_step = 1 / float(samples_str)

    # the lines in the order they were taken, each the other way from the one before
    line_order = list(range(lines)) if scan_down else list(range(lines))[::-1]
    line_order = line_order[:lines_taken]
    order = []
    for i, line in enumerate(line_order):
        line_points = range(cols)[::-1] if i % 2 == 0 else range(cols)
        order.extend((line, p) for p in line_points)
    acquired = np.zeros(shape, dtype=bool)
    acquired[line_order] = True

    ext, ret, away = counts
    zext, zret, dext, dret = force_curves(rng, ext, ret, z_far=700.0, ramp=480.0)
    # the Away segment goes on from where the retract stopped, off the surface
    n_ret, n_away = int(ret.max()), int(away.max())
    last = np.take_along_axis(zret, ret[..., None] - 1, axis=-1)
    zaway = last - 480.0 / max(n_ret - 1, 1) * np.arange(1, n_away + 1)
    daway = rng.normal(0.0, 0.05, zaway.shape)
    past_end = np.arange(n_away) >= away[..., None]
    zaway[past_end] = np.nan
    daway[past_end] = np.nan

    # each channel's segments in m, then in its unit, as float32
    raw = [(z * 1e-9).astype(np.float32) for z in (zext, zret, zaway)]
    defl = [(d * 1e-9).astype(np.float32) for d in (dext, dret, daway)]
    zsnsr = [
        (z + ZSNSR_OFFSET + rng.normal(0.0, 0.03e-9, z.shape)).astype(np.float32)
        for z in raw
    ]
    written = [
        [_in_unit(segment, units[name]) for segment in channel]
        for channel, (name, _) in zip((raw, defl, zsnsr), SLOW_CHANNELS, strict=True)
    ]

    def in_nm(segment, name):
        """A segment as written, in nm, NaN where never taken."""
        nm = segment.astype(np.float64) * LENGTH_UNITS[units[name]] * 1e9
        return np.where(acquired[..., None], nm, np.nan)

    volume = SyntheticVolume(
        "FMAP",
        in_nm(written[0][0], "Raw"),
        in_nm(written[0][1], "Raw"),
        in_nm(written[1][0], "Defl"),
        in_nm(written[1][1], "Defl"),
        ext,
        ret,
        tuple(order),
        None if lines_taken == lines else acquired,
    )
    raw_m = np.concatenate(raw, axis=-1).astype(np.float64)
    images = {
        "MapHeight": (-np.nanmax(raw_m, axis=-1)).astype(np.float32),
        "MapAdhesion": (-np.nanmin(defl[1].astype(np.float64), axis=-1) * k).astype(
            np.float32
        ),
    }

    f = _Sections()
    f.add(b"ARDF")
    ftoc = f.table(b"FTOC", 32, 24, TABLE_FLAGS)
    ttoc = f.table(b"TTOC", 16, 32, TABLE_FLAGS)
    f.skip(gap)
    f.entry(ttoc, 0, b"TOFF", TOFF.pack(0, f.text(0, _note(note))))

    image_tables = []
    next_entry = None
    for i, (name, image) in enumerate(images.items()):
        imag = f.table(b"IMAG", 4, 24, ITEM_FLAGS)
        f.entry(ftoc, i, b"IMAG", POINTER.pack(imag))
        if next_entry is not None:
            f.patch(next_entry, POINTER, imag)
        next_entry = f.entry(imag, 0, b"NEXT", POINTER.pack(0))
        image_tables.append((imag, f.table(b"TTOC", 4, 32, TABLE_FLAGS)))
        idef = (cols, lines, 0, 0, x_step, y_step, b"m", b"m", _cstring(name), b"m")
        f.add(b"IDEF", IDEF.pack(*idef))
        ibox = f.table(b"IBOX", lines, HEADER.size + 4 * cols, TABLE_FLAGS)
        for line in sorted(line_order):
            f.entry(ibox, line, b"IDAT", image[line].astype("<f4").tobytes())
        f.add(b"GAMI")
    # the first image's note follows the last image
    text = f.text(0, _note(image_note))
    f.entry(image_tables[0][1], 0, b"TOFF", TOFF.pack(0, text))

    volm = f.table(b"VOLM", 4, 24, ITEM_FLAGS)
    f.entry(ftoc, len(images), b"VOLM", POINTER.pack(volm))
    f.patch(next_entry, POINTER, volm)
    f.entry(volm, 0, b"NSET", POINTER.pack(lines_taken * cols))
    f.table(b"TTOC", 4, 32, TABLE_FLAGS)
    vdef = (cols, lines, b"", x_step, y_step, t_step, b"m", b"m", b"s")
    f.add(b"VDEF", VDEF.pack(*vdef, SLOW_SEGMENTS, 3))
    for channel, _ in SLOW_CHANNELS:
        f.add(b"VCHN", VCHN.pack(_cstring(channel), _cstring(units[channel])))
    xdef = struct.pack("<LL", 0, len(XDEF_TEXT)) + XDEF_TEXT
    f.add(b"XDEF", xdef + bytes(XDEF_SIZE - HEADER.size - len(xdef)))
    vtoc = f.table(b"VTOC", lines, 40, TABLE_FLAGS)
    f.add(b"MLOV")

    previous = 0
    for i, (line, p) in enumerate(order):
        index = line * cols + p
        lengths = [int(n[line, p]) for n in counts]
        total = sum(lengths)
        # where each segment starts, then the last point
        segments = (0, lengths[0], lengths[0] + lengths[1], total - 1, 0)
        fields = (index, line, p, SLOW_VSET_TYPE, previous)
        vset = f.add(b"VSET", VSET.pack(*fields, 0))
        if i % cols == 0:
            f.entry(vtoc, i // cols, b"VOFF", VOFF.pack(index, line, p, vset))
        name = f"Line{line:04d}Point{p:04d}".encode()
        f.add(b"VNAM", VNAM.pack(index, line, p, len(name), name))
        for channel, segments_written in enumerate(written):
            values = np.concatenate(
                [s[line, p, :n] for s, n in zip(segments_written, lengths, strict=True)]
            )
            vdat = VDAT.pack(index, line, p, total, channel, *segments)
            f.add(b"VDAT", vdat + values.astype("<f4").tobytes())
        lvdt = (p * x_step, line * y_step)
        seconds, trigger = 3029572800.0 + i, lengths[0] / float(samples_str)
        xdat = XDAT.pack(index, line, p, 6, *lvdt, seconds, trigger, np.nan, np.nan)
        f.add(b"XDAT", xdat)
        # the curve points on to whatever section follows it
        f.patch(vset, VSET, *fields, f.tell())
        previous = vset

    stored_images = {}
    for (imag, _), (name, image) in zip(image_tables, images.items(), strict=True):
        held = np.where(acquired, image, np.float32(0))
        thumbnail = f.add(b"THMB", _thumbnail(held, scan_size))
        f.entry(imag, 1, b"THMB", POINTER.pack(thumbnail))
        stored_images[name] = np.where(acquired, image, np.float32(np.nan))

    path = Path(directory) / filename
    f.write(path)
    return SyntheticFile(
        path=path,
        volumes=[volume],
        images=stored_images,
        k=float(k_str),
        defl_sens=float(invols_str) * 1e9,
        t_step=t_step,
        scansize=(float(fast_str) * 1e9, float(slow_str) * 1e9),
        image_units=dict(SLOW_MAP_UNITS),
        units=units,
    )
