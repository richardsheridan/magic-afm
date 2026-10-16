"""Asylum Research ARDF files, the fast force maps the Asylum software saves.

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
(``Raw``, ``Defl``, ``Drive``, in m); an ``XDEF`` naming the experiment's extra values;
a ``VTOC`` with one slot per line, filled in the order the lines were taken, each
``VOFF`` entry giving the line's first curve; and an ``MLOV``. The curves follow, in the
order they were taken. Each is a ``VSET`` (its index within its volume, line, point, a
type, the offset of the previous curve of the same volume and the offset just past this
curve's data), a ``VNAM`` "LineNNNNPointNNNN" and one ``VDAT`` per channel of float32 m,
the extend half then the retract half in time order, after the indices of where each
segment starts and of the last point. The trace runs along a line from point 0, the
retrace back from the last point; with both kept, each line holds its trace then its
retrace, of types 10 and 11, while a lone direction has type 5. ``ScanDown: 1`` in the
note puts line 0 first, ``ScanDown: 0`` the last line. After the curves come the
thumbnails, 8-bit renderings of the images 128 pixels along the longer side of the scan,
and the texts at the end.

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
builder leaves them out. Slow force maps, whose curves have extend, retract and
dwell segments, are not built.
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
MAP_NAMES = ("Height", "ZSensor", "Adhesion", "Force", "YoungsJKR")
# FastMapCapture: which scan directions a fast force map kept, one volume each
CAPTURES = {"trace": ("Trace",), "retrace": ("Retrace",), "both": ("Trace", "Retrace")}
CAPTURE_CODES = {"trace": 1, "retrace": 2, "both": 3}
VSET_TYPES = {"trace": (5,), "retrace": (5,), "both": (10, 11)}
# the drive signal runs about this far above the Z sensor in the real files, in m
DRIVE_OFFSET = 1.1e-6
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
    """A file built section by section; CRCs are filled in once pointers are."""

    def __init__(self):
        self.buf = bytearray()
        self.crcs: list[tuple[int, int]] = []

    def add(self, name: bytes, payload: bytes = b"", flags: int = 0) -> int:
        offset = len(self.buf)
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
        _, slots, stride = TABLE.unpack_from(self.buf, table + HEADER.size)
        size = HEADER.size + len(payload)
        if not (slot < slots and size == stride):
            raise ValueError("entry does not fit the table", name, slot, size)
        offset = table + HEADER.size + TABLE.size + slot * stride
        self.buf[offset : offset + size] = HEADER.pack(0, size, name, 0) + payload
        self.crcs.append((offset, size))
        return offset

    def text(self, index: int, text: str) -> int:
        encoded = _cstring(text)
        fields = struct.pack("<LL", index, len(encoded))
        unpadded = HEADER.size + len(fields) + len(encoded)
        padding = bytes(8 - unpadded % 8)
        return self.add(b"TEXT", fields + encoded + padding)

    def patch(self, offset: int, fmt: struct.Struct, *values) -> None:
        fmt.pack_into(self.buf, offset + HEADER.size, *values)

    def finish(self) -> bytes:
        for offset, size in self.crcs:
            crc = zlib.crc32(self.buf[offset + 4 : offset + size])
            struct.pack_into("<L", self.buf, offset, crc)
        return bytes(self.buf)


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
    filename: str = "synthetic.ARDF",
) -> SyntheticFile:
    """Write a fast force map ARDF file and say what it holds.

    ``shape`` is (lines, points per line): rows are the slow axis, columns the
    fast one. ``points`` per half curve, so each curve holds twice as many;
    ``capture`` is "trace", "retrace" or "both". ``scan_down`` is the note's
    ScanDown: True takes line 0 first, False the last line first. With
    ``lines_taken`` the scan stops after that many lines, a stopped scan. ``k``
    in N/m, ``invols`` in nm/V, ``scan_size`` in nm along the fast then the slow
    axis, ``rate`` the curves per second. The volumes are named after the
    direction they hold.
    """
    lines, cols = shape
    if lines < 2 or cols < 2:
        raise ValueError("a map needs at least two lines of two points", shape)
    lines_taken = lines if lines_taken is None else lines_taken
    if not 0 < lines_taken <= lines:
        raise ValueError("lines_taken must be within the grid", lines_taken, lines)
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
    volumes = []
    for direction in directions:
        zext, zret, dext, dret = force_curves(
            rng, counts, counts, z_far=700.0, ramp=480.0
        )
        raw = (np.concatenate([zext, zret], axis=-1) * 1e-9).astype(np.float32)
        defl = (np.concatenate([dext, dret], axis=-1) * 1e-9).astype(np.float32)
        drive = (raw + DRIVE_OFFSET).astype(np.float32)
        stored[direction] = (raw, defl, drive)
        # what the stored float32 metres are in nm, NaN where never taken
        z_nm = np.where(acquired[..., None], raw.astype(np.float64) * 1e9, np.nan)
        d_nm = np.where(acquired[..., None], defl.astype(np.float64) * 1e9, np.nan)
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
        for channel, unit in CHANNELS:
            f.add(b"VCHN", VCHN.pack(_cstring(channel), _cstring(unit)))
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
                for channel, data in enumerate(stored[direction]):
                    vdat = VDAT.pack(index, line, p, n_total, channel, *segments)
                    f.add(b"VDAT", vdat + data[line, p].astype("<f4").tobytes())
                # the curve points on to whatever section follows it
                f.patch(vset, VSET, *fields, len(f.buf))
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
    path.write_bytes(f.finish())
    return SyntheticFile(
        path=path,
        volumes=volumes,
        images=stored_images,
        k=float(k_str),
        defl_sens=float(invols_str) * 1e9,
        t_step=t_step,
        scansize=(float(fast_str) * 1e9, float(slow_str) * 1e9),
        image_units=dict.fromkeys(images, "m"),
        units=dict(CHANNELS),
    )
