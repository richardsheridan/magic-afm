"""MagicAFM Data Readers

This module has readers for file types that this package supports.
Generally, they have the structure of a FVFile that has a dict of Images
and a list of Volumes. Only minimal metadata is read from the storage until
one of the get_* methods is called. FVFiles are constructed via a `parse`
classmethod that takes an opened mmap or h5py.File. It's your responsibility
to ensure that it stays open as long as you intend to use the FVFile.

This module should not depend on other parts of the package and weird dependencies
should be loaded lazily. (I'm looking at you h5py..)
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

__all__ = [
    "Image",
    "Volume",
    "FVFile",
    "FileFormatError",
    "open_h5",
    "mmap_path_read_only",
    "convert_ardf",
    "ARH5ForceMapVolume",
    "ARH5Image",
    "ARH5FFMVolume",
    "ARH5File",
    "ARDFImage",
    "ARDFVolume",
    "ARDFFile",
    "NanoscopeImage",
    "NanoscopeVolume",
    "NanoscopeFile",
    "NIDVolume",
    "NIDMapHeightImage",
    "NIDMapImage",
    "NIDFile",
    "SUFFIX_FVFILE_MAP",
    "Index",
    "Curve",
]

import mmap
import re
import struct
from bisect import bisect_left
from collections.abc import Iterable
from functools import partial
from subprocess import PIPE
from typing import Any, Protocol, TypeAlias, Union

try:
    from collections.abc import Buffer
except ImportError:
    Buffer: TypeAlias = Union[mmap.mmap, memoryview, bytes, bytearray]

try:
    from subprocess import STARTF_USESHOWWINDOW, STARTUPINFO
except ImportError:

    def STARTUPINFO(*a, **kw):
        return None

    STARTF_USESHOWWINDOW = None

import numpy as np
from attrs import evolve, field, frozen

_NANOMETER_UNIT_CONVERSION = (
    1e9  # maybe we can intelligently read this from the file someday
)
_TOC_STRUCT = struct.Struct("<QLL")

# decimal exponents of the SI prefixes seen in unit strings (micro as micro sign or mu)
_SI_PREFIX_EXPONENTS = {
    "": 0,
    "m": -3,
    "\u00b5": -6,
    "\u03bc": -6,
    "u": -6,
    "n": -9,
    "p": -12,
}


def _si_prefix_exponent(unit: str, base: str) -> int | None:
    """Decimal exponent of unit as a prefixed base, such as -6 for "µm" and "m"."""
    if not unit.endswith(base):
        return None
    return _SI_PREFIX_EXPONENTS.get(unit[: len(unit) - len(base)])


def _split_si_prefix(unit: str) -> tuple[int, str]:
    """Decimal exponent and base of a prefixed unit, such as (-9, "m") for "nm".

    A unit that is one character, or starts with no known prefix, is unprefixed.
    """
    exponent = _SI_PREFIX_EXPONENTS.get(unit[:1])
    if exponent is None or len(unit) == 1:
        return 0, unit
    return exponent, unit[1:]


def _nano_factor(unit: str, base: str) -> float | None:
    """Factor converting unit to nano-base, such as 1e3 for "µm" and "m", or None."""
    exponent = _si_prefix_exponent(unit, base)
    if exponent is None:
        return None
    return 10.0 ** (9 + exponent)


def _nm_per_unit(unit: str) -> float:
    """Factor converting a length in unit to nm, raising FileFormatError if unknown."""
    factor = _nano_factor(unit, "m")
    if factor is None:
        raise FileFormatError("Expected a length unit.", unit)
    return factor


def _nan_curve(npts: int) -> np.ndarray:
    """A read-only all-NaN curve with npts points in each half."""
    x = np.full(shape=(2, 2, npts), fill_value=np.nan, dtype=np.float32)
    x.setflags(write=False)
    return x


def _halves(z, d) -> "Curve":
    """((z_ext, z_ret), (d_ext, d_ret)) from z and d each holding two halves."""
    return (z[0], z[1]), (d[0], d[1])


###############################################
############### Typing stuff ##################
###############################################


Index: TypeAlias = tuple[int, ...]
Curve: TypeAlias = tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
_ChanMap: TypeAlias = dict[str, tuple[int, "_ARDFVchan"]]
_StepInfo: TypeAlias = tuple[tuple[float, str], ...]


class Image(Protocol):
    name: str
    shape: Index
    units: str

    def get_image(self) -> np.ndarray:
        """Get the image from disk."""
        ...


class Volume(Protocol):
    name: str
    shape: Index

    def get_curve(self, r: int, c: int) -> Curve:
        """Efficiently get a specific curve from disk, each half as long as stored."""
        ...

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        ...

    def iter_curves(self) -> Iterable[tuple[Index, Curve]]:
        """Iterate over curves lazily in on-disk order."""
        ...

    def get_all_curves(self) -> np.ndarray:
        """Eagerly load (2, rows, cols, 2, n) curves, cut to the shortest half."""
        ...


class FVFile(Protocol):
    headers: dict[str, Any]
    images: dict[str, Image]
    volumes: list[Volume]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]
    sync_dist: float | None
    trace: int | None

    @staticmethod
    def check_type(data: Buffer): ...

    @classmethod
    def parse(cls, data: Buffer) -> "FVFile": ...


###############################################
################## Helpers ####################
###############################################


class FileFormatError(ValueError):
    """A file is malformed or uses a variant this module cannot read."""


_DIRECTION_NAMES = {0: "Retrace", 1: "Trace"}


def _select_volume(fvfile: FVFile, trace: int | None) -> Volume:
    """The volume named for trace, else the first if trace or directions are absent."""
    names = [v.name for v in fvfile.volumes]
    directions = [n for n in names if n in _DIRECTION_NAMES.values()]
    if trace is None or not directions:
        return fvfile.volumes[0]
    wanted = _DIRECTION_NAMES[int(trace)]
    if wanted not in names:
        raise ValueError(f"Requested {wanted} but the file holds only {directions}")
    return fvfile.volumes[names.index(wanted)]


def _check_force_bpp(length, ncurves, npts, grid) -> int:
    """Bytes per point of a force section that holds ncurves curves of npts points.

    grid names the header keys the caller took ncurves from, with their values.
    """
    bpp, rem = divmod(length, ncurves * npts)
    if rem or bpp not in (2, 4):
        raise FileFormatError(
            f"force data length {length} does not fit {ncurves} curves of {npts}"
            f" points; the capture may skip points ({grid})"
        )
    return bpp


def _nonzero(value, key, section):
    """Value of a header count or rate the reader divides by, if it is not zero."""
    if not value:
        raise FileFormatError(f"{key} is zero in {section}")
    return value


def open_h5(path):
    import h5py

    return h5py.File(path, "r")


def mmap_path_read_only(path):
    with open(path, mode="rb", buffering=0) as file:
        return mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)


def _decode_cstring(cstring: bytes):
    return cstring.rstrip(b"\0").decode("windows-1252")


_SEARCH_CHUNK_BYTES = 1 << 20


def _find_in_buffer(data: Buffer, sub: bytes, start: int = 0, end: int | None = None):
    """Lowest index of sub in any buffer object, with bytes.find semantics."""
    find = getattr(data, "find", None)
    if find is not None:
        return find(sub, start) if end is None else find(sub, start, end)
    with memoryview(data).cast("B") as view:
        start, end, _ = slice(start, end).indices(len(view))
        # overlapping windows so that a straddling match is not missed
        step = _SEARCH_CHUNK_BYTES - len(sub) + 1
        for pos in range(start, end, step):
            index = bytes(view[pos : min(pos + _SEARCH_CHUNK_BYTES, end)]).find(sub)
            if index >= 0:
                return pos + index
    return -1


###############################################
################## FVFiles ####################
###############################################


###############################################
################### Asylum ####################
###############################################


def _parse_ar_note(note: Iterable[str]):
    # The notes have a very regular key-value structure
    # convert to dict for later access
    return dict(
        line.split(":", 1) for line in note if ":" in line and "@Line:" not in line
    )


async def convert_ardf(
    ardf_path, *, h5file_path=None, conv_path="ARDFtoHDF5.exe", pbar=None
):
    """Turn an ARDF into a corresponding ARH5, returning the path.

    Requires converter executable available from Asylum Research"""
    import trio

    ardf_path = trio.Path(ardf_path)
    if h5file_path is None:
        h5file_path = ardf_path.with_suffix(".h5")
    else:
        h5file_path = trio.Path(h5file_path)
    conv_path = trio.Path(conv_path)

    if pbar is None:
        # Just display the raw subprocess output
        pipe = None
    else:
        # set up pbar and pipes for custom display
        pbar.set_description_str("Converting " + ardf_path.name)
        pipe = PIPE

    async def reading_stdout():
        """Store up stdout in our own buffer to check for Failed at the end."""
        stdout = bytearray()
        async for bytes_ in proc.stdout:
            stdout.extend(bytes_)
        stdout = stdout.decode()
        if "Failed" in stdout:
            raise RuntimeError(stdout)
        else:
            print(stdout)

    async def reading_stderr():
        """Parse the percent complete display to send to our own progressbar"""
        async for bytes_ in proc.stderr:
            i = bytes_.rfind(b"\x08") + 1  # first thing on right not a backspace
            most_recent_numeric_output = bytes_[i:-1]  # crop % sign
            if most_recent_numeric_output:
                try:
                    n = round(float(most_recent_numeric_output.decode()), 1)
                except ValueError:
                    # I don't know what causes this, but I'd
                    # rather carry on than have a fatal error
                    pass
                else:
                    pbar.update(n - pbar.n)

    if not await conv_path.is_file():
        raise FileNotFoundError(f"Could not locate converter at conv_path={conv_path}")
    try:
        async with trio.open_nursery() as nursery:
            proc = await nursery.start(
                partial(
                    trio.run_process,
                    [conv_path, ardf_path, h5file_path],
                    stderr=pipe,
                    stdout=pipe,
                    # suppress a console on windows
                    startupinfo=STARTUPINFO(dwFlags=STARTF_USESHOWWINDOW),
                )
            )
            if pbar is not None:
                nursery.start_soon(reading_stdout)
                nursery.start_soon(reading_stderr)
    except BaseException:
        with trio.CancelScope(shield=True):
            await h5file_path.unlink(missing_ok=True)
        raise

    return h5file_path


class ARH5ForceMapVolume:
    name: str
    shape: Index

    def __init__(self, h5data):
        self.name = "FMAP"
        self.force_curves = h5data["ForceMap"]["0"]
        # ForceMap Segments can contain 3 or 4 endpoint indices for each indent array
        self.segments = self.force_curves["Segments"][:, :, :]  # XXX Read h5data
        im_r, im_c, num_segments = self.segments.shape
        self.shape = im_r, im_c

        # Generally segments are [Ext, Dwell, Ret, Away] or [Ext, Ret, Away] for magic,
        # we don't dwell. new converter ensures this
        if num_segments != 3:
            raise FileFormatError(
                f"force map has {num_segments} segments per curve, expected 3"
            )

        # this is all necessary because the arrays are not of uniform length
        # get_all_curves cuts every half down to the length of the shortest
        self.extlens = self.segments[:, :, 0]
        self.minext = self.split = np.min(self.extlens)
        self.extretlens = self.segments[:, :, 1]
        self.minret = np.min(self.extretlens - self.extlens)
        self.npts = self.minext + self.minret

        # We only care about 2 channels, Defl and ZSnsr
        # Convert channels array to a map that can be used to index into ForceMap data by name
        # chanmap should always be {'Defl':1,'ZSnsr':2} but it's cheap to calculate
        chanmap = {
            key: index for index, key in enumerate(self.force_curves.attrs["Channels"])
        }
        # We could slice with "1:" if chanmap were constant but I'm not sure if it is
        self.defl_zsnsr_row_slice = [chanmap["Defl"], chanmap["ZSnsr"]]

    def get_curve(self, r, c):
        # Because of the nonuniform arrays, each indent gets its own dataset
        # indexed by 'row:column' e.g. '1:1'.
        curve = self.force_curves[f"{r}:{c}"]  # XXX Read h5data
        split = self.extlens[r, c]
        end = self.extretlens[r, c]
        # Index into the data and grab the Defl and Zsnsr ext and ret arrays as one 2D array
        defl_zsnsr = curve[self.defl_zsnsr_row_slice, :end]  # XXX Read h5data
        # reverse axis zero so data is ordered zsnsr,defl like we did for FFM
        z, d = (defl_zsnsr[::-1] * _NANOMETER_UNIT_CONVERSION).astype(np.float32)
        return (z[:split], z[split:]), (d[:split], d[split:])

    def iter_indices(self) -> Iterable[Index]:
        for index in self.force_curves:
            if index == "Segments":
                continue
            r, c = list(map(int, index.split(":")))
            yield r, c

    def iter_curves(self):
        for r, c in self.iter_indices():
            yield (r, c), self.get_curve(r, c)

    def get_all_curves(self):
        # Cut both halves to one length, keeping the points nearest the surface
        n = min(self.minext, self.minret)
        x = np.empty((2,) + self.shape + (2, n), dtype=np.float32)
        for (r, c), ((zext, zret), (dext, dret)) in self.iter_curves():
            x[0, r, c] = zext[len(zext) - n :], zret[:n]
            x[1, r, c] = dext[len(dext) - n :], dret[:n]
        return x


_ASYLUM_BASE_UNITS = {
    "Height": "m",
    "ZSensor": "m",
    "Adhesion": "N",
    "Force": "N",
    "YoungsJKR": "Pa",
}


def _asylum_units(name: str) -> str:
    """Unit of an Asylum image's values by its base name, or "V" when unknown."""
    base = name.removesuffix("Retrace").removesuffix("Trace")
    return _ASYLUM_BASE_UNITS.get(base, "V")


@frozen
class ARH5Image:
    data: dict
    name: str
    shape: Index
    units: str

    def get_image(self) -> np.ndarray:
        """Get the image from disk."""
        return self.data[:]


@frozen
class ARH5FFMVolume:
    name: str
    shape: Index
    # step_info: _StepInfo
    _zreader: np.ndarray
    _dreader: np.ndarray

    def get_curve(self, r, c):
        z = self._zreader[r, c].reshape((2, -1)) * _NANOMETER_UNIT_CONVERSION
        d = self._dreader[r, c].reshape((2, -1)) * _NANOMETER_UNIT_CONVERSION
        return _halves(z.astype(np.float32), d.astype(np.float32))

    def iter_indices(self) -> Iterable[Index]:
        return np.ndindex(self.shape)

    def iter_curves(self) -> Iterable[tuple[Index, Curve]]:
        for index in self.iter_indices():
            yield index, self.get_curve(*index)

    def get_all_curves(self):
        z = self._zreader[:] * _NANOMETER_UNIT_CONVERSION
        d = self._dreader[:] * _NANOMETER_UNIT_CONVERSION
        z = z.reshape(z.shape[:-1] + (2, -1))
        d = d.reshape(d.shape[:-1] + (2, -1))
        return np.stack((z, d))


@frozen
class ARH5File:
    headers: dict[str, Any] = field(repr=lambda x: f"<dict with {len(x)} entries>")
    images: dict[str, Image]
    volumes: list[Volume]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]
    sync_dist: float | None = None
    trace: int | None = None

    @classmethod
    def parse(cls, h5data):
        notes = _parse_ar_note(h5data.attrs["Note"].splitlines())
        images = {
            name: ARH5Image(img, name, img.shape, _asylum_units(name))
            for name, img in h5data["Image"].items()
        }

        k = float(notes["SpringConstant"])
        scansize = (
            float(notes["FastScanSize"]) * _NANOMETER_UNIT_CONVERSION,
            float(notes["SlowScanSize"]) * _NANOMETER_UNIT_CONVERSION,
        )
        # NOTE: aspect is redundant to scansize
        # self.aspect = float(self.notes["SlowRatio"]) / float(self.notes["FastRatio"])
        defl_sens = float(notes["InvOLS"]) * _NANOMETER_UNIT_CONVERSION
        rate = float(notes["FastMapZRate"])
        volumes = []
        if "FFM" in h5data:
            # self.scandown = bool(self.notes["ScanDown"])
            # FastMapCapture: 1 trace (a guess), 2 retrace, 3 both. No note or group
            # attribute says which group is which, so assume FFM/0 and FFM/1 follow
            # capture order, trace first.
            capture = int(notes.get("FastMapCapture", 3))
            names = {1: ["Trace"], 2: ["Retrace"]}.get(capture, ["Trace", "Retrace"])
            ffm = h5data["FFM"]
            groups = [(ffm[g], name) for g, name in zip("01", names) if g in ffm]
            if "Raw" in ffm:
                groups.append((ffm, names[0]))
            for group, name in groups:
                volumes.append(
                    ARH5FFMVolume(
                        name, group["Raw"].shape[:2], group["Raw"], group["Defl"]
                    )
                )
            npts = volumes[0]._zreader.shape[-1]
        else:
            # self.scandown = bool(self.notes["FMapScanDown"])
            volumes.append(ARH5ForceMapVolume(h5data))
            npts = volumes[0].npts
        t_step = 1 / rate / npts
        trace = 1 if len(volumes) > 1 else None
        return cls(notes, images, volumes, k, defl_sens, t_step, scansize, trace=trace)


@frozen
class _ARDFHeader:
    data: Buffer
    offset: int
    crc: int = field(repr=hex)
    size: int
    name: bytes
    flags: int = field(repr=hex)
    _struct = struct.Struct("<LL4sL")

    @classmethod
    def unpack(cls, data: Buffer, offset: int):
        return cls(data, offset, *cls._struct.unpack_from(data, offset))

    def validate(self):
        # ImHex poly 0x4c11db7 init 0xffffffff xor out 0xffffffff reflect in and out
        import zlib

        crc = zlib.crc32(
            memoryview(self.data)[self.offset + 4 : self.offset + self.size]
        )
        if self.crc != crc:
            raise FileFormatError(
                f"Invalid section. Expected {self.crc:X}, got {crc:X}.", self
            )
        return True


@frozen
class _ARDFTableOfContents:
    data: Buffer
    offset: int
    size: int
    entries: list[tuple[_ARDFHeader, int]]
    _entry_struct = struct.Struct("<LL4sLQ")

    @classmethod
    def unpack(cls, header: _ARDFHeader):
        if header.size != 32:
            raise FileFormatError("Malformed table of contents", header)
        header.validate()
        data = header.data
        offset = header.offset
        size, nentries, stride = _TOC_STRUCT.unpack_from(data, offset + 16)
        if stride != 24:
            raise FileFormatError(f"table of contents stride {stride}, expected 24")
        if size - 32 != nentries * stride:
            raise FileFormatError(
                f"table of contents size {size} does not fit {nentries} entries"
            )
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, pointer = cls._entry_struct.unpack_from(data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = _ARDFHeader(data, toc_offset, *header)
            if entry_header.name not in {b"IMAG", b"VOLM", b"NEXT", b"THMB", b"NSET"}:
                raise FileFormatError("Malformed table of contents.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)


@frozen
class _ARDFTextTableOfContents:
    data: Buffer
    offset: int
    size: int
    entries: list[tuple[_ARDFHeader, int]]
    _entry_struct = struct.Struct("<LL4sLQQ")

    @classmethod
    def unpack(cls, header: _ARDFHeader):
        if header.size != 32 or header.name != b"TTOC":
            raise FileFormatError("Malformed text table of contents.", header)
        header.validate()
        offset = header.offset
        data = header.data
        size, nentries, stride = _TOC_STRUCT.unpack_from(data, offset + 16)
        if stride != 32:
            raise FileFormatError(f"text table stride {stride}, expected 32")
        if size - 32 != nentries * stride:
            raise FileFormatError(
                f"text table size {size} does not fit {nentries} entries"
            )
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, _, pointer = cls._entry_struct.unpack_from(data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = _ARDFHeader(data, toc_offset, *header)
            if entry_header.name != b"TOFF":
                raise FileFormatError("Malformed text table entry.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)

    def decode_entry(self, index: int):
        # maybe could be another TEXT class but we'll read it straight in
        entry_header, pointer = self.entries[index]
        text_header = _ARDFHeader.unpack(self.data, pointer)
        if text_header.name != b"TEXT":
            raise FileFormatError("Malformed text section.", text_header)
        text_header.validate()
        i, text_len = struct.unpack_from("<LL", self.data, pointer + 16)
        if i != index:
            raise FileFormatError(f"text entry {i} found at index {index}")
        offset = text_header.offset + 24
        if not text_len < text_header.size - 24:
            raise FileFormatError(f"text length {text_len} overruns its section")
        text = memoryview(self.data)[offset : offset + text_len]
        return bytes(text).replace(b"\r", b"\n").decode("windows-1252")


@frozen
class _ARDFVolumeTableOfContents:
    offset: int
    size: int
    lines: list[int]
    pointers: list[int]
    _voff_struct = struct.Struct("<LLQQ")

    @classmethod
    def unpack(cls, header: _ARDFHeader):
        # cant reuse exact _ARDFTableOfContents for VTOC, but structure is similar
        data = header.data
        offset = header.offset
        if header.size != 32 or header.name != b"VTOC":
            raise FileFormatError("Malformed volume table of contents.", header)
        size, nentries, stride = _TOC_STRUCT.unpack_from(data, offset + 16)
        if stride != 40:
            raise FileFormatError(f"volume table stride {stride}, expected 40")
        if size - 32 != nentries * stride:
            raise FileFormatError(
                f"volume table size {size} does not fit {nentries} entries"
            )
        lines, pointers = [], []
        for i, toc_offset in enumerate(
            range(header.offset + 32, header.offset + size, stride)
        ):
            entry_header = _ARDFHeader.unpack(data, toc_offset)
            if not entry_header.crc:
                # scan was interrupted, and vtoc is zero-filled
                break
            if entry_header.name != b"VOFF":
                raise FileFormatError("Malformed volume table entry.", entry_header)
            entry_header.validate()
            force_index, line, point, pointer = cls._voff_struct.unpack_from(
                data, toc_offset + 16
            )
            lines.append(line)
            pointers.append(pointer)
        return cls(offset, size, lines, pointers)


@frozen
class _ARDFVchan:
    name: str
    unit: str
    _struct = struct.Struct("<32s32s")

    @classmethod
    def unpack(cls, header: _ARDFHeader):
        if header.size != 80 or header.name != b"VCHN":
            raise FileFormatError("Malformed channel definition.", header)
        header.validate()
        name, unit = cls._struct.unpack_from(header.data, header.offset + 16)
        return cls(_decode_cstring(name), _decode_cstring(unit))


@frozen
class _ARDFXdef:
    offset: int
    size: int
    xdef: list[str]
    _struct = struct.Struct("<LL")

    @classmethod
    def unpack(cls, header: _ARDFHeader):
        if header.size != 96 or header.name != b"XDEF":
            raise FileFormatError("Malformed experiment definition.", header)
        header.validate()
        # Experiment definition is just a string
        _, nchars = cls._struct.unpack_from(header.data, header.offset + 16)
        if _ != 0:
            raise FileFormatError(f"experiment definition reserved field is {_}")
        if nchars > header.size:
            raise FileFormatError("Experiment definition too long.", header, nchars)
        xdef = memoryview(header.data)[header.offset + 24 : header.offset + 24 + nchars]
        xdef = bytes(xdef).decode("windows-1252").split(";")[:-1]
        return cls(header.offset, header.size, xdef)


@frozen
class _ARDFVset:
    data: Buffer
    offset: int
    size: int
    force_index: int
    line: int
    point: int
    # vtype seems to differ between different FV modes.
    # FMAP with ext;ret;dwell shows 0b10 = 2 everywhere.
    # FFM with just trace or just retrace shows 0b101 = 5 everywhere.
    # FFM storing both shows 0b1010 = 10 for trace
    # and 0b1011 = 11 for retrace.
    vtype: int = field(repr=bin)
    prev_vset_offset: int
    next_vset_offset: int
    _struct = struct.Struct("<LLLLQQ")

    @classmethod
    def unpack(cls, vset_header: _ARDFHeader):
        if vset_header.name != b"VSET":
            raise FileFormatError("malformed VSET header", vset_header)
        vset_header.validate()
        data = vset_header.data
        offset = vset_header.offset
        size = cls._struct.size + 16
        return cls(data, offset, size, *cls._struct.unpack_from(data, offset + 16))


@frozen
class _ARDFVdata:
    data: Buffer
    offset: int
    force_index: int
    line: int
    point: int
    nfloats: int
    channel: int
    seg_offsets: tuple[int, ...]
    _struct = struct.Struct("<10L")

    @classmethod
    def unpack(cls, header: _ARDFHeader):
        data = header.data
        offset = header.offset
        (
            force_index,
            line,
            point,
            nfloats,
            channel,
            *seg_offsets,
        ) = cls._struct.unpack_from(data, offset + 16)
        return cls(
            data, offset, force_index, line, point, nfloats, channel, seg_offsets
        )

    @property
    def array_offset(self):
        return self.offset + self._struct.size + 16

    @property
    def next_offset(self):
        return self.array_offset + self.nfloats * 4

    def get_ndarray(self, factor: float | None = None):
        with memoryview(self.data) as v:  # assert data is open, and hold it open
            x = np.ndarray(
                shape=self.nfloats,
                dtype="<f4",
                buffer=v,
                offset=self.array_offset,
            ).astype("f4", copy=True)
        if factor is not None:
            x *= factor
        return x


@frozen
class ARDFImage:
    data: Buffer
    ibox_offset: int
    name: str
    shape: Index
    units: str
    step_info: _StepInfo
    acquired: np.ndarray | None = field(default=None, repr=False)
    _struct = struct.Struct("<LLQQdd32s32s32s32s")

    @classmethod
    def parse_imag(cls, imag_header: _ARDFHeader):
        if imag_header.name != b"IMAG":
            raise FileFormatError("Malformed image header.", imag_header)
        imag_toc = _ARDFTableOfContents.unpack(imag_header)

        # don't use NEXT or THMB data, step over
        ttoc_header = _ARDFHeader.unpack(imag_toc.data, imag_toc.offset + imag_toc.size)
        ttoc = _ARDFTextTableOfContents.unpack(ttoc_header)

        # don't use TTOC or TOFF, step over
        idef_header = _ARDFHeader.unpack(ttoc.data, ttoc.offset + ttoc.size)
        if idef_header.name != b"IDEF" or idef_header.size != cls._struct.size + 16:
            raise FileFormatError("Malformed image definition.", idef_header)
        idef_header.validate()
        points, lines, _, __, x_step, y_step, *cstrings = cls._struct.unpack_from(
            idef_header.data,
            idef_header.offset + 16,
        )
        if _ or __:
            raise FileFormatError(f"image definition reserved fields are {_}, {__}")
        x_unit, y_unit, name, units = list(map(_decode_cstring, cstrings))
        # IDEF says m for every image, so trust it only for length maps
        if (known := _asylum_units(name)) != "m":
            units = known

        return cls(
            imag_header.data,
            idef_header.offset + idef_header.size,
            name,
            (lines, points),
            units,
            ((x_step, x_unit), (y_step, y_unit)),
        )

    def get_image(self):
        ibox_header = _ARDFHeader.unpack(self.data, self.ibox_offset)
        if ibox_header.size != 32 or ibox_header.name != b"IBOX":
            raise FileFormatError("Malformed image layout.", ibox_header)
        ibox_header.validate()
        data_offset = ibox_header.offset + ibox_header.size + 16  # past IDAT header
        ibox_size, lines, stride = _TOC_STRUCT.unpack_from(
            ibox_header.data, ibox_header.offset + 16
        )
        points = (stride - 16) // 4  # less IDAT header
        if (lines, points) != self.shape:
            raise FileFormatError(
                f"image data is {points}x{lines}, definition says {self.shape}"
            )
        # elide image data validation and map into an array directly
        with memoryview(self.data) as v:  # assert data is open, and hold it open
            arr = np.ndarray(
                shape=(lines, points),
                dtype="<f4",
                buffer=v,
                offset=data_offset,
                strides=(stride, 4),
            ).astype("f4", copy=True)
        gami_header = _ARDFHeader.unpack(
            ibox_header.data, ibox_header.offset + ibox_size
        )
        if gami_header.size != 16 or gami_header.name != b"GAMI":
            raise FileFormatError("Malformed image layout.", gami_header)
        gami_header.validate()
        if self.acquired is not None:
            arr[~self.acquired] = np.nan
        return arr


@frozen
class _ARDFFFMReader:
    data: Buffer
    array_view: np.ndarray = field(repr=False)
    array_offset: int  # hard to recover from views
    channels: list[int]  # [z, d]
    factors: np.ndarray = field(repr=False)  # float32 to nm, shaped to broadcast
    # seg_offsets is weird. you'd think it would contain the starting index
    # for each segment. However, it always has a trailing value of 1-nfloats,
    # and nonexistent segments get a zero. For regular/FFM data, we'll just
    # assume that the second offset maps to our "split" concept.
    seg_offsets: tuple
    up: bool
    trace: bool

    @classmethod
    def parse(cls, first_vset_header: _ARDFHeader, points: int, lines: int, channels):
        data = first_vset_header.data
        # just walk past these first headers to find our data_offset
        first_vset = _ARDFVset.unpack(first_vset_header)
        vset_stride = first_vset.next_vset_offset - first_vset_header.offset
        if first_vset.vtype == 5:
            line_stride = vset_stride * points
        else:
            # better be "both" setting see _ARDFVset
            if first_vset.vtype not in {10, 11}:
                raise FileFormatError(f"unknown volume set type {first_vset.vtype}")
            line_stride = vset_stride * points * 2
        up = first_vset.line == 0
        trace = first_vset.point == 0

        first_vnam_header = _ARDFHeader.unpack(
            data, first_vset_header.offset + first_vset_header.size
        )
        if first_vnam_header.name != b"VNAM":
            raise FileFormatError("Malformed volume name", first_vnam_header)
        first_vnam_header.validate()
        first_vdat_header = _ARDFHeader.unpack(
            data, first_vnam_header.offset + first_vnam_header.size
        )
        if first_vdat_header.name != b"VDAT":
            raise FileFormatError("Malformed volume data")
        first_vdat = _ARDFVdata.unpack(first_vdat_header)
        return cls(
            data=data,
            array_view=np.ndarray(
                shape=(lines, points, len(channels), first_vdat.nfloats),
                dtype="<f4",
                buffer=data,
                offset=first_vdat.array_offset,
                strides=(
                    line_stride,
                    vset_stride,
                    first_vdat_header.size,
                    4,
                ),
                # match up array and image coordinates
            )[:: 1 if up else -1, :: 1 if trace else -1],
            array_offset=first_vdat.array_offset,
            channels=[channels["Raw"][0], channels["Defl"][0]],
            factors=np.array(
                [_nm_per_unit(channels[name][1].unit) for name in ("Raw", "Defl")],
                dtype="f4",
            )[:, None],
            seg_offsets=first_vdat.seg_offsets,
            up=up,
            trace=trace,
        )

    def get_curve(self, r, c):
        """Efficiently get a specific curve from disk."""
        with memoryview(self.data):  # assert data is open, and hold it open
            x = self.array_view[r, c, self.channels]  # advanced indexing copies
        x = x.astype("f4", copy=False)
        x *= self.factors
        return _halves(*x.reshape((len(self.channels), 2, -1)))

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        # undo the array reversals in the constructor method
        lines, points = self.array_view.shape[:2]
        lines_iter = range(lines)
        if not self.up:
            lines_iter = reversed(lines_iter)
        for line in lines_iter:
            points_iter = range(points)
            if not self.trace:
                points_iter = reversed(points_iter)
            for point in points_iter:
                yield line, point

    def iter_curves(self) -> Iterable[tuple[Index, Curve]]:
        """Iterate over curves lazily in on-disk order."""
        # TODO: cleverly use np.nditer?
        for index in self.iter_indices():
            yield index, self.get_curve(*index)

    def get_all_curves(self) -> np.ndarray:
        """Eagerly load all curves into memory."""
        with memoryview(self.data):  # assert data is open, and hold it open
            # advanced indexing triggers a copy
            loaded_data = self.array_view[:, :, self.channels, :]
        # avoid a second copy with inplace op
        loaded_data *= self.factors
        # reshape assuming equal points on extend and retract
        loaded_data = loaded_data.reshape(loaded_data.shape[:-1] + (2, -1))
        # make it look like z, d
        return np.moveaxis(loaded_data, 2, 0)


@frozen
class _ARDFForceMapReader:
    data: Buffer
    vtoc: _ARDFVolumeTableOfContents
    lines: int
    points: int
    vtype: int
    channels: _ChanMap
    nancurve: np.ndarray = field(repr=False)
    _seen_vsets: dict[Index, _ARDFVset] = field(init=False, factory=dict)

    @property
    def zname(self):
        return "Raw" if "Raw" in self.channels else "ZSnsr"

    @property
    def zfactor(self) -> float:
        return _nm_per_unit(self.channels[self.zname][1].unit)

    @property
    def dfactor(self) -> float:
        return _nm_per_unit(self.channels["Defl"][1].unit)

    def traverse_vsets(self, pointer: int):
        while True:
            header = _ARDFHeader.unpack(self.data, pointer)
            if header.name != b"VSET":
                break
            vset = _ARDFVset.unpack(header)
            index = (vset.line, vset.point, vset.vtype)
            if index not in self._seen_vsets:
                self._seen_vsets[index] = vset
            yield vset  # .line, vset.point, vset.
            pointer = vset.next_vset_offset

    def traverse_vdats(self, pointer):
        vnam_header = _ARDFHeader.unpack(self.data, pointer)
        if vnam_header.name != b"VNAM":
            raise FileFormatError("Malformed volume name", vnam_header)
        vnam_header.validate()
        pointer = vnam_header.offset + vnam_header.size
        while True:
            vdat_header = _ARDFHeader.unpack(self.data, pointer)
            if vdat_header.name != b"VDAT":
                break
            vdat_header.validate()  # opportunity to read data with gil released
            yield _ARDFVdata.unpack(vdat_header)
            pointer = vdat_header.offset + vdat_header.size

    def get_curve(self, r: int, c: int) -> Curve:
        """Efficiently get a specific curve from disk."""
        if not (0 <= r < self.lines and 0 <= c < self.points):
            raise ValueError("Invalid index:", (self.lines, self.points), (r, c))

        index = r, c, self.vtype
        if index not in self._seen_vsets:
            # bisect row pointer
            if self.vtoc.lines[0] > self.vtoc.lines[-1]:
                # probably reversed
                sl = np.s_[::-1]
            else:
                sl = np.s_[:]
            i = bisect_left(self.vtoc.lines[sl], r)
            if i >= len(self.vtoc.lines) or r != int(self.vtoc.lines[sl][i]):
                return _halves(*self.nancurve)
            # read entire line of the vtoc
            for vset in self.traverse_vsets(int(self.vtoc.pointers[sl][i])):
                if vset.line != r:
                    break
                # traverse_vsets implicitly fills in seen_vsets

        try:
            vset = self._seen_vsets[index]
        except KeyError:
            # index was missing from the line, probably due to "Stop"
            return _halves(*self.nancurve)

        zxr, dxr = self.nancurve
        for vdat in self.traverse_vdats(vset.offset + vset.size):
            _, ext, *_ = vdat.seg_offsets
            ret = 2 * ext
            if vdat.channel == self.channels[self.zname][0]:
                z = vdat.get_ndarray(self.zfactor)
                zxr = z[:ret].reshape(2, ext)
            elif vdat.channel == self.channels["Defl"][0]:
                d = vdat.get_ndarray(self.dfactor)
                dxr = d[:ret].reshape(2, ext)
        return _halves(zxr, dxr)

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        for vset in self.traverse_vsets(self.vtoc.pointers[0]):
            if vset.vtype != self.vtype:
                continue
            yield vset.line, vset.point

    def iter_curves(self) -> Iterable[tuple[Index, Curve]]:
        """Iterate over curves lazily in on-disk order."""
        zname = self.zname
        for vset in self.traverse_vsets(self.vtoc.pointers[0]):
            if vset.vtype != self.vtype:
                continue
            zxr, dxr = self.nancurve
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                _, ext, *_ = vdat.seg_offsets
                ret = 2 * ext
                if vdat.channel == self.channels[zname][0]:
                    z = vdat.get_ndarray(self.zfactor)
                    zxr = z[:ret].reshape(2, ext)
                elif vdat.channel == self.channels["Defl"][0]:
                    d = vdat.get_ndarray(self.dfactor)
                    dxr = d[:ret].reshape(2, ext)
            yield (vset.line, vset.point), _halves(zxr, dxr)

    def get_all_curves(self) -> np.ndarray:
        """Eagerly load all curves into memory."""
        minext = 0xFFFFFFFF
        vdats = {}
        for vset in self.traverse_vsets(self.vtoc.pointers[0]):
            if vset.vtype != self.vtype:
                continue
            x = vdats[vset.line, vset.point] = [None, None]
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                if vdat.channel == self.channels[self.zname][0]:
                    x[0] = vdat  # zvdat
                elif vdat.channel == self.channels["Defl"][0]:
                    x[1] = vdat  # dvdat
            if None in x:
                raise FileFormatError(f"missing vdat channel: {x}, {self.channels}")
            minext = min(minext, vdat.seg_offsets[1])
        del vset, vdat, x
        minfloats = 2 * minext
        x = np.full((self.lines, self.points, 2, minfloats), np.nan, dtype=np.float32)
        for (r, c), (zvdat, dvdat) in vdats.items():
            # center the slice, as the stacked array splits each curve halfway
            floats = zvdat.seg_offsets[1] * 2
            halfextra = (floats - minfloats) // 2  # even - even -> even
            sl = np.s_[halfextra : minfloats + halfextra]
            # TODO: verify turnaround point against iter_curves
            x[r, c, :, :] = zvdat.get_ndarray()[sl], dvdat.get_ndarray()[sl]
        x[:, :, 0] *= self.zfactor
        x[:, :, 1] *= self.dfactor
        return np.moveaxis(x.reshape(x.shape[:-1] + (2, -1)), 2, 0)


@frozen
class ARDFVolume:
    volm_offset: int
    name: str
    shape: Index
    step_info: _StepInfo
    xdef: _ARDFXdef
    _reader: _ARDFForceMapReader | _ARDFFFMReader
    _struct = struct.Struct("<LL24sddd32s32s32s32sQ")

    @classmethod
    def parse_volm(cls, volm_header: _ARDFHeader, npts: int | None = None):
        data = volm_header.data
        if volm_header.size != 32 or volm_header.name != b"VOLM":
            raise FileFormatError("Malformed volume header.", volm_header)
        # the next headers look a lot like VSET is a table of contents, but I've only
        # seen NEXT and NSET headers. NEXT shows up if "both" trace and retrace data
        # are inside. I'll assume the last entry is always NSET, the number of VSETs.
        volm_toc = _ARDFTableOfContents.unpack(volm_header)
        nset_header, nsets = volm_toc.entries[-1]
        nset_header.validate()
        ttoc_header = _ARDFHeader.unpack(data, volm_toc.offset + volm_toc.size)
        ttoc = _ARDFTextTableOfContents.unpack(ttoc_header)

        # don't use TTOC or TOFF, step over

        # cls essentially represents VDEF plus its linkage down to VSET
        # so this unpacking is intentionally inlined here.
        vdef_header = _ARDFHeader.unpack(data, ttoc.offset + ttoc.size)
        if vdef_header.name != b"VDEF" or vdef_header.size != cls._struct.size + 16:
            raise FileFormatError("Malformed volume definition.", vdef_header)
        vdef_header.validate()
        unpack_from = cls._struct.unpack_from  # line wrapping
        points, lines, _, x_step, y_step, t_step, *cstrings, nseg = unpack_from(
            vdef_header.data, vdef_header.offset + 16
        )
        if sum(_) != 0:
            raise FileFormatError(f"volume definition reserved fields are {_}")
        complete = points * lines == nsets
        x_unit, y_unit, t_unit, seg_names = list(map(_decode_cstring, cstrings))
        seg_names = seg_names.split(";")[:-1]
        if nseg != len(seg_names):
            raise FileFormatError(
                f"volume definition has {nseg} segments but names {seg_names}"
            )

        # Implicit table of channels here smh
        offset = vdef_header.offset + vdef_header.size
        channels: _ChanMap = {}
        for i in range(5):
            header = _ARDFHeader.unpack(data, offset)
            if header.name != b"VCHN":
                break
            vchn = _ARDFVchan.unpack(header)
            offset = header.offset + header.size
            if _si_prefix_exponent(vchn.unit, "m") is None:
                raise FileFormatError(
                    f"channel {vchn.name} has unit {vchn.unit!r}, expected a length"
                )
            channels[vchn.name] = (i, vchn)
        else:
            raise FileFormatError("Got too many channels.", channels)

        xdef = _ARDFXdef.unpack(header)
        vtoc_header = _ARDFHeader.unpack(data, xdef.offset + xdef.size)
        vtoc = _ARDFVolumeTableOfContents.unpack(vtoc_header)

        mlov_header = _ARDFHeader.unpack(data, vtoc.offset + vtoc.size)
        if mlov_header.size != 16 or mlov_header.name != b"MLOV":
            raise FileFormatError("Malformed volume table of contents.", mlov_header)
        mlov_header.validate()

        # Check if each offset is regularly spaced
        # optimize for LARGE regular case (FMaps are SMALL)
        first_vset_header = _ARDFHeader.unpack(data, vtoc.pointers[0])
        if complete and not np.any(np.diff(np.diff(vtoc.pointers))):
            reader = _ARDFFFMReader.parse(first_vset_header, points, lines, channels)
            name = "Trace" if reader.trace else "Retrace"
        else:
            first_vset = _ARDFVset.unpack(first_vset_header)
            args = (data, vtoc, lines, points, first_vset.vtype, channels)
            if npts is None:
                # no usable note, so fall back on the longest acquired curve
                probe = _ARDFForceMapReader(*args, _nan_curve(0))
                npts = max(
                    vdat.seg_offsets[1]
                    for vset in probe.traverse_vsets(vtoc.pointers[0])
                    for vdat in probe.traverse_vdats(vset.offset + vset.size)
                )
            reader = _ARDFForceMapReader(*args, _nan_curve(npts))
            name = "FMAP"

        return cls(
            volm_header.offset,
            name,
            (lines, points),
            ((x_step, x_unit), (y_step, y_unit), (t_step, t_unit)),
            xdef,
            reader,
        )

    def acquired_mask(self) -> np.ndarray | None:
        """Mask of the curves the file holds, or None for a complete regular scan."""
        if not isinstance(self._reader, _ARDFForceMapReader):
            return None
        mask = np.zeros(self.shape, dtype=bool)
        for rc in self.iter_indices():
            mask[rc] = True
        return mask

    def get_curve(self, r: int, c: int) -> Curve:
        """Efficiently get a specific curve from disk."""
        return self._reader.get_curve(r, c)

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        return self._reader.iter_indices()

    def iter_curves(self) -> Iterable[tuple[Index, Curve]]:
        """Iterate over curves lazily in on-disk order."""
        return self._reader.iter_curves()

    def get_all_curves(self) -> np.ndarray:
        """Eagerly load all curves into memory."""
        return self._reader.get_all_curves()


@frozen
class ARDFFile:
    headers: dict[str, str] = field(repr=lambda x: f"<dict with {len(x)} entries>")
    images: dict[str, Image]
    volumes: list[Volume]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]
    trace: int | None
    sync_dist: float | None = None

    @staticmethod
    def check_type(data: Buffer):
        file_header = _ARDFHeader.unpack(data, 0)
        if file_header.size != 16 or file_header.name != b"ARDF":
            raise FileFormatError("Not an ARDF file.", file_header)
        file_header.validate()
        return file_header

    @classmethod
    def parse(cls, data: Buffer):
        file_header = cls.check_type(data)
        ftoc_header = _ARDFHeader.unpack(data, offset=file_header.size)
        if ftoc_header.name != b"FTOC":
            raise FileFormatError("Malformed ARDF file table of contents.", ftoc_header)
        ftoc = _ARDFTableOfContents.unpack(ftoc_header)
        ttoc_header = _ARDFHeader.unpack(data, offset=ftoc.offset + ftoc.size)
        ttoc = _ARDFTextTableOfContents.unpack(ttoc_header)
        if len(ttoc.entries) != 1:
            raise FileFormatError(
                f"text table has {len(ttoc.entries)} entries, expected 1"
            )
        notes = _parse_ar_note(ttoc.decode_entry(0).splitlines())
        try:
            npts = round(
                float(notes["NumPtsPerSec"]) / float(notes["FastMapZRate"]) / 2
            )
        except (KeyError, ValueError, ZeroDivisionError):
            npts = None
        images = {}
        volumes = []
        for item, pointer in ftoc.entries:
            item.validate()
            item = _ARDFHeader.unpack(data, pointer)
            if item.name == b"IMAG":
                item = ARDFImage.parse_imag(item)
                images[item.name] = item
            elif item.name == b"VOLM":
                item = ARDFVolume.parse_volm(item, npts)
                # volumes[item.name] = item
                volumes.append(item)
            else:
                raise FileFormatError(f"Unknown TOC entry {item.name}.", item)

        # a stopped scan leaves its images unwritten past the curves it acquired
        acquired = None
        for volume in volumes:
            mask = volume.acquired_mask()
            if mask is not None:
                acquired = mask if acquired is None else acquired | mask
        if acquired is not None and not acquired.all():
            acquired.setflags(write=False)
            images = {
                name: evolve(image, acquired=acquired)
                if image.shape == acquired.shape
                else image
                for name, image in images.items()
            }

        k = float(notes["SpringConstant"])
        # slight numerical differences here
        # lines, points = volumes[0].shape
        # xsize = lines * volumes[0].step_info[0][0]
        # ysize = points * volumes[0].step_info[1][0]
        scansize = (
            float(notes["FastScanSize"]) * _NANOMETER_UNIT_CONVERSION,
            float(notes["SlowScanSize"]) * _NANOMETER_UNIT_CONVERSION,
        )
        # NOTE: aspect is redundant to scansize
        # self.aspect = float(self.notes["SlowRatio"]) / float(self.notes["FastRatio"])
        defl_sens = float(notes["InvOLS"]) * _NANOMETER_UNIT_CONVERSION
        t_step = volumes[0].step_info[-1][0]
        trace = 1 if len(volumes) > 1 else None
        return cls(notes, images, volumes, k, defl_sens, t_step, scansize, trace)


###############################################
################### Bruker ####################
###############################################


# noinspection PyUnboundLocalVariable
def _parse_nanoscope_header(header_lines: Iterable[str]):
    """Convert header from a Nanoscope file to a convenient nested dict

    header_lines can be an opened file object or a list of strings or anything
    that iterates the header line-by-line."""

    header = {}
    for line in header_lines:
        if not line.startswith("\\"):
            raise FileFormatError(f"header line without backslash: {line!r}")
        line = line[1:].strip()  # strip leading slash and newline

        if line.startswith("*"):
            # we're starting a new section
            section_name = line[1:]
            current_section = {}
            if section_name == "File list end":
                break  # THIS IS THE **NORMAL** WAY TO END THE FOR LOOP
            if section_name in header:
                header[section_name].append(current_section)
            else:
                header[section_name] = [current_section]
        else:
            # add key, value pairs for this section
            key, value = line.split(":", maxsplit=1)
            # Colon special case for "groups"
            if key.startswith("@") and key[1].isdigit() and len(key) == 2:
                key2, value = value.split(":", maxsplit=1)
                key = key + ":" + key2

            current_section[key] = value.strip()
    else:
        raise FileFormatError("File ended too soon")
    if (not header) or ("" in header):
        raise FileFormatError("File is empty or not a Bruker data file")

    return header


@frozen
class NanoscopeImage:
    data: Buffer
    name: str
    offset: int
    length: int
    bpp: int
    unit_scale: float
    unit_offset: float
    shape: Index
    units: str

    @classmethod
    def parse_image(cls, data, header, h, name):
        length = int(h["Data length"])
        offset = int(h["Data offset"])
        r = _nonzero(int(h["Number of lines"]), "Number of lines", name)
        c = _nonzero(int(h["Samps/line"]), "Samps/line", name)
        bpp, rem = divmod(length, r * c)
        if rem or bpp not in (2, 4):
            raise FileFormatError(
                f"Data length {length} of image {name!r} is not {r} x {c} samples"
                " of 2 or 4 bytes"
            )
        value = h["@2:Z scale"]
        hard_scale = float(value.split()[-2]) / (2 ** (bpp * 8))
        hard_offset = float(h["@2:Z offset"].split()[-2]) / (2 ** (bpp * 8))
        soft_scale_name = "@" + value[1 + value.find("[") : value.find("]")]
        try:
            soft_scale_string = header["Ciao scan list"][0][soft_scale_name]
        except KeyError:
            soft_scale_string = header["Scanner list"][0][soft_scale_name]
        soft_scale_fields = soft_scale_string.split()
        # a unitless soft scale leaves the image in the hard scale's unit
        soft_unit = (
            soft_scale_fields[2] if len(soft_scale_fields) > 2 else value.split()[0]
        )
        exponent, units = _split_si_prefix(soft_unit.split("/")[0])
        soft_scale = float(soft_scale_fields[1]) / 10.0**-exponent
        unit_scale = hard_scale * soft_scale
        unit_offset = hard_offset * soft_scale
        return cls(
            data=data,
            name=name,
            offset=offset,
            length=length,
            bpp=bpp,
            unit_scale=unit_scale,
            unit_offset=unit_offset,
            shape=(r, c),
            units=units,
        )

    def get_image(self) -> np.ndarray:
        with memoryview(self.data) as v:  # assert data is open, and hold it open
            z_floats = np.ndarray(
                shape=self.shape,
                dtype=f"<i{self.bpp}",
                buffer=v,
                offset=self.offset,
            ).astype("f4", copy=True)
        z_floats *= self.unit_scale
        z_floats += self.unit_offset
        return z_floats


def _nanoscope_curve_scale(header, subheader, sens_key: str) -> float:
    """nm per LSB of a force curve channel, checking the units the header declares."""
    sens_string = header["Ciao scan list"][0][sens_key]
    _, sens, sens_unit = sens_string.split()
    if sens_unit != "nm/V":
        raise FileFormatError(f"{sens_key} unit is {sens_unit!r}, expected nm/V")
    value = subheader["@4:Z scale"]
    lsb, lsb_unit = value[1 + value.find("(") : value.find(")")].split()
    if lsb_unit != "V/LSB":
        raise FileFormatError(f"Z scale unit is {lsb_unit!r}, expected V/LSB")
    return float(sens) * float(lsb)


@frozen
class _FFVReader:
    data: Buffer
    name: str
    ints: np.ndarray = field(repr=False)
    scale: float
    _soft_scale_map = {
        "Height Sensor": "@Sens. ZsensSens",
        "Height": "@Sens. ZSens",
        "Deflection Error": "@Sens. DeflSens",
    }

    @classmethod
    def parse(cls, header, data, subheader, shape):
        r, c = shape
        name = subheader["@4:Image Data"].split('"')[1]
        offset = int(subheader["Data offset"])
        length = int(subheader["Data length"])
        n_ext, _ = tuple(map(int, subheader["Samps/line"].split()))
        cfl = header["Ciao force list"][0]
        grid = f"FV Lines: {cfl['FV Lines']}, force/line: {cfl['force/line']}"
        bpp = _check_force_bpp(length, r * c, 2 * n_ext, grid)
        ints = np.ndarray(
            shape=(r, c, 2, n_ext), dtype=f"<i{bpp}", buffer=data, offset=offset
        )
        scale = _nanoscope_curve_scale(header, subheader, cls._soft_scale_map[name])
        return cls(data, name, ints, scale)

    def get_curve(self, r, c):
        with memoryview(self.data):  # assert data is open, and hold it open
            f = self.ints[r, c].astype("f4")
        f *= self.scale
        return f[0, ::-1], f[1]


@frozen
class _QNMDReader:
    data: Buffer
    name: str
    ints: np.ndarray = field(repr=False)
    shape: Index
    scale: float

    @classmethod
    def parse(cls, header, data, subheader, shape):
        # QNM doesn't have its own height data, so it must be synthesized
        # I've chosen to do this by combining a height image with a sine wave

        r, c = shape
        length = int(subheader["Data length"])
        offset = int(subheader["Data offset"])
        n_ext, _ = tuple(map(int, subheader["Samps/line"].split()))
        npts = 2 * n_ext
        force_line = header["Ciao force list"][0]["force/line"]
        samps_line = header["Ciao scan list"][0]["Samps/line"]
        grid = f"force/line: {force_line} against Samps/line: {samps_line}"
        bpp = _check_force_bpp(length, r * c, npts, grid)
        d_ints = np.ndarray(
            shape=(r * c, 2, n_ext), dtype=f"<i{bpp}", buffer=data, offset=offset
        )

        d_scale = _nanoscope_curve_scale(header, subheader, "@Sens. DeflSens")
        name = subheader["@4:Image Data"].split('"')[1]
        return cls(data, name, d_ints, shape, d_scale)

    def get_curve(self, r, c, sync_dist_int):
        i = np.ravel_multi_index((r, c), self.shape)
        with memoryview(self.data):  # assert data is open, and hold it open
            d = self.ints[i]
            extend_marker = d[0, -1] == -32768
            d = d.astype("f4")
        d *= self.scale
        # flip extend segment
        d[0, :] = d[0, ::-1]
        # remove extend segment marker
        if extend_marker:
            d[0, 0] = d[0, 1]
        d = d.reshape(-1)
        # rotate the record so the tap lands at the split, wrapping its own end
        npts = len(d)
        n_ext = npts // 2  # halves are equal, see shape in parse
        sync_dist = (sync_dist_int - n_ext) % npts  # according to Bruker via Bede
        d = np.roll(d, -sync_dist)
        return d.reshape((2, -1))


@frozen
class _QNMZReader:
    name: str
    amp: float
    npts: int
    height_for_z: np.ndarray = field(repr=False)

    @classmethod
    def parse(cls, header, d_reader: _QNMDReader, height_for_z):
        # QNM doesn't have its own height data, so it must be synthesized
        # I've chosen to do this by combining a height image with a sine wave

        amp = np.float32(header["Ciao scan list"][0]["Peak Force Amplitude"])
        npts = d_reader.ints[0].size
        return cls(d_reader.name + " z_basis", amp, npts, height_for_z)

    def get_curve(self, r, c, sync_dist_frac):
        phase = -2 * np.pi * sync_dist_frac / self.npts
        # need to infer z from amp/height
        z_basis = -self.amp * (
            np.cos(
                np.linspace(
                    phase,
                    phase + 2 * np.pi,
                    self.npts,
                    endpoint=False,
                    dtype=np.float32,
                )
            )
        )
        z_basis += self.height_for_z[r, c]
        return z_basis.reshape((2, -1))


@frozen
class NanoscopeVolume:
    name: str
    shape: Index
    step_info: _StepInfo
    sync_dist: float | None
    _zreader: _FFVReader | _QNMZReader
    _dreader: _FFVReader | _QNMDReader

    @classmethod
    def parse(cls, shape, scansize, t_step, sync_dist, z_reader, d_reader):
        slowpx, fastpx = shape
        x_step = scansize[0] / _NANOMETER_UNIT_CONVERSION / fastpx
        y_step = scansize[1] / _NANOMETER_UNIT_CONVERSION / slowpx

        # self.scandown = {"Down": True, "Up": False}[
        #     header["FV"]["Deflection Error"]["Frame direction"]
        # ]
        return cls(
            d_reader.name,
            shape,
            ((x_step, "m"), (y_step, "m"), (t_step, "s")),
            sync_dist,
            z_reader,
            d_reader,
        )

    def get_curve(self, r: int, c: int) -> Curve:
        """Efficiently get a specific curve from disk."""
        if self.sync_dist is None:
            z, d = self._zreader.get_curve(r, c), self._dreader.get_curve(r, c)
            return _halves(z, d)
        else:
            sync_dist_int = int(self.sync_dist)
            sync_dist_frac = self.sync_dist - sync_dist_int
            return _halves(
                self._zreader.get_curve(r, c, sync_dist_frac),
                self._dreader.get_curve(r, c, sync_dist_int),
            )

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        # TODO: ensure on-disk order
        return np.ndindex(self.shape)

    def iter_curves(self) -> Iterable[tuple[Index, Curve]]:
        """Iterate over curves lazily in on-disk order."""
        for index in self.iter_indices():
            yield index, self.get_curve(*index)

    def get_all_curves(self) -> np.ndarray:
        """Eagerly load all curves into memory."""
        x = np.empty(
            self.shape + (2, 2, self._dreader.ints.shape[-1]), dtype=np.float32
        )
        for index, curve in self.iter_curves():
            x[*index, ...] = curve
        return np.moveaxis(x, 2, 0)


@frozen
class NanoscopeFile:
    headers: dict[str, Any] = field(repr=lambda x: f"<dict with {len(x)} entries>")
    images: dict[str, Image]
    volumes: list[Volume]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]
    sync_dist: float | None = None
    trace: int | None = None

    @staticmethod
    def check_type(data: Buffer):
        # Check for magic string indicating nanoscope
        magic = b"\\*Force file list\r\n"
        start = bytes(memoryview(data)[: len(magic)])
        if not start == magic:
            raise FileFormatError("Not a nanoscope file.", start)

    @classmethod
    def parse(cls, data: Buffer):
        cls.check_type(data)
        # End of header is demarcated by a SUB byte (26 = 0x1A)
        header_end_pos = _find_in_buffer(data, b"\x1a")
        if header_end_pos < 0:
            raise FileFormatError(
                "No stop byte found, are you sure this is a Nanoscope file?"
            )
        header = _parse_nanoscope_header(
            bytes(memoryview(data)[:header_end_pos]).decode("windows-1252").splitlines()
        )

        # Header items that I should be reading someday
        # \Frame direction: Down
        # \Line Direction: Retrace
        # \Z direction: Retract

        # and in the image lists
        # \Aspect Ratio: 1:1
        # \Scan Size: 800 800 nm
        csl = header["Ciao scan list"][0]
        cfl = header["Ciao force list"][0]
        # deflection and height data are separate "force images", first of each name
        force_sections = {}
        for subheader in header["Ciao force image list"]:
            name = subheader["@4:Image Data"].split('"')[1]
            force_sections.setdefault(name, subheader)
        # any section with Height in its name marks a force volume, else QNM
        is_ffv = any("Height" in name for name in force_sections)
        if is_ffv:
            fv_lines = int(cfl["FV Lines"].split()[0])
            slowpx = _nonzero(fv_lines, "FV Lines", "Ciao force list")
            fv_cols = int(cfl["force/line"].split()[0])
            fastpx = _nonzero(fv_cols, "force/line", "Ciao force list")
        else:
            fastpx = _nonzero(int(csl["Samps/line"]), "Samps/line", "Ciao scan list")
            slowpx = _nonzero(int(csl["Lines"]), "Lines", "Ciao scan list")
        shape = slowpx, fastpx
        ratestr, unit = csl["PFT Freq"].split()
        if unit.lower() != "khz":
            raise FileFormatError(f"PFT Freq unit is {unit!r}, expected kHz")
        rate = _nonzero(float(ratestr), "PFT Freq", "Ciao scan list")
        force_line = cfl["force/line"].split()[0]
        npts = _nonzero(int(force_line), "force/line", "Ciao force list")
        t_step = 1 / (rate * 1000) / npts
        for subheader in header["Ciao force image list"]:
            n_ext = subheader["Samps/line"].split()[0]
            _nonzero(int(n_ext), "Samps/line", "Ciao force image list")

        scansize, units = csl["Scan Size"].split()
        if units == "nm":
            factor = 1.0
        elif units == "pm":
            factor = 0.001
        elif units == "~m":  # microns?!
            factor = 1000.0
        else:
            raise FileFormatError("unknown units:", units)

        # TODO: tuple(map(float,header[""Ciao scan list""]["Aspect Ratio"].split(":")))
        ratio = float(scansize) * factor / max(fastpx, slowpx)
        scansize = (fastpx * ratio, slowpx * ratio)

        images = {}
        for entry in header["Ciao image list"]:
            name = entry["@2:Image Data"].split('"')[1]
            images[name] = NanoscopeImage.parse_image(data, header, entry, name)
        if not is_ffv:
            deflections = header["Ciao force image list"]
            # QNM data doesn't contain height data which we synthesize from an image
            try:
                height_for_z = images["Height Sensor"].get_image()
            except KeyError:
                height_for_z = images["Height"].get_image()
            height_for_z *= _NANOMETER_UNIT_CONVERSION
            try:
                sync_dist_str = csl["Sync Distance QNM"]
            except KeyError:
                sync_dist_str = csl["Sync Distance"]
            points_per_curve = sum(map(int, deflections[0]["Samps/line"].split()))
            try:
                samples_per_period = float(csl["Sample Points"])
            except KeyError:
                # Sample Points times PFT Freq is 512 kHz in every real capture
                samples_per_period = 512 / rate
            else:
                if not samples_per_period > 0:
                    raise FileFormatError(
                        f"Sample Points is {samples_per_period:g}, expected positive"
                    )
            sync_dist = float(sync_dist_str) * points_per_curve / samples_per_period
            volumes = [
                NanoscopeVolume.parse(
                    shape,
                    scansize,
                    t_step,
                    sync_dist,
                    _QNMZReader.parse(header, d_reader, height_for_z),
                    d_reader,
                )
                for d_reader in (
                    _QNMDReader.parse(header, data, d_subh, shape)
                    for d_subh in deflections
                )
            ]
        else:
            sync_dist = None
            # z from Height Sensor (the measured z) or else Height; d from Deflection
            # Error; sections of other names go unread
            z_subh = force_sections.get("Height Sensor", force_sections.get("Height"))
            if z_subh is None:
                raise FileFormatError("no Height Sensor or Height force section")
            try:
                d_subh = force_sections["Deflection Error"]
            except KeyError:
                raise FileFormatError("no Deflection Error force section") from None
            volumes = [
                NanoscopeVolume.parse(
                    shape,
                    scansize,
                    t_step,
                    sync_dist,
                    _FFVReader.parse(header, data, z_subh, shape),
                    _FFVReader.parse(header, data, d_subh, shape),
                )
            ]
        # volumes = {v.name:v for v in volumes}
        k = float(header["Ciao force image list"][0]["Spring Constant"])
        defl_sens = float(csl["@Sens. DeflSens"].split()[1])
        return cls(header, images, volumes, k, defl_sens, t_step, scansize, sync_dist)


###############################################
################## Nanosurf ###################
###############################################


_NANOSURF_VALUE_REGEX = re.compile(r"(\S)\[(.*)\]\*\[(.*)\]")
_NANOSURF_LINE_POINTS_REGEX = re.compile(r"LineDim\d*Points")
_NANOSURF_PARAM_TYPES = {
    "Frame": str,
    "Points": int,
    "Lines": int,
    "SaveBits": int,
    "Dim0Min": float,
    "Dim0Range": float,
    "Dim0Unit": str,
    "Dim1Min": float,
    "Dim1Range": float,
    "Dim1Unit": str,
    "Dim2Min": float,
    "Dim2Range": float,
    "Dim2Unit": str,
    "Dim2Name": str,
}
# frames of the raw curves and images, everything else is an analysis result
_NANOSURF_RAW_FRAMES = (
    "Spec forward",
    "Spec backward",
    "Scan forward",
    "Scan backward",
)
# result frame -> image name and sign; adhesion is stored as a negative force
_NANOSURF_RESULT_IMAGES = {
    "MaxAdhesionOut00": ("MaxRetAdhesionForce", -1.0),
    "SnapInOut00": ("MaxExtAdhesionForce", -1.0),
    "MaxAdhesionOut10": ("MaxRetAdhesionHeight", 1.0),
    "SnapInOut10": ("MaxExtAdhesionHeight", 1.0),
}


def _parse_nanosurf_value(string: str):
    """Decode a typed Nanosurf value such as "D[25.03]*[N/m]" to (value, unit)"""
    ty, val, unit = _NANOSURF_VALUE_REGEX.search(string).groups()
    if ty == "D":
        val = float(val)
    elif ty == "B":
        val = bool(int(val))
    elif ty == "L":
        val = int(val)
    elif ty == "V":
        val = [float(v) for v in val.split(",")]
        unit = unit.split(",")
    return val, unit


def _parse_nanosurf_line_points(section: dict[str, str]):
    """List the point count of each line of a dataset section with ragged lines"""
    return [
        int(value)
        for key, value in section.items()
        if _NANOSURF_LINE_POINTS_REGEX.match(key)
    ]


def _parse_nanosurf_header(header_lines: Iterable[str]):
    """Convert header from a Nanosurf file to a convenient dict of sections

    The header is INI-like, a "[name]" line starting a section of "key=value"
    lines. header_lines can be an opened file object or a list of strings or
    anything that iterates the header line-by-line. Derived items are added
    under the keys "datasets", "param", "cantilever", "spec_map",
    "image_center" and "thermal_tune".

    Copyright (C) Nanosurf AG - All Rights Reserved (2021)

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    “Software”), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

    header = {}
    current_section = {}
    for line in header_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            # we're starting a new section
            current_section = header[line[1:-1]] = {}
        else:
            # add key, value pairs for this section
            key, _, value = line.partition("=")
            current_section[key.strip()] = value.strip()

    datasets = []
    ds = header["DataSet"]
    for gr in range(int(ds["GroupCount"])):
        for g in range(int(ds[f"Gr{gr}-Count"])):
            key = f"Gr{gr}-Ch{g}"
            if key in ds:
                datasets.append(ds[key])

    param = {}
    for term, ty in _NANOSURF_PARAM_TYPES.items():
        param[term] = [ty(header[d][term]) for d in datasets if term in header[d]]

    # for spectroscopy with maps, lines are ragged
    line_points = []
    for d in datasets:
        h = header[d]
        points = _parse_nanosurf_line_points(h) if "LineDim0Min" in h else []
        line_points.append(np.array(points, dtype=int))
    param["LinePoints"] = line_points

    cantilever = header.get("DataSet\\Calibration\\Cantilever", {})
    key = "DataSet\\Calibration\\Scanhead"
    if key in header:
        deflection_input = header[key]["In5"]
        sensitivity = float(deflection_input.split(",")[4]) / 10.0
        cantilever["Sensitivity"] = str(sensitivity)

    spec_map = None
    key = "DataSet\\SpecInfos"
    if key in header:
        n = int(header[key]["SubSectionCount"])
        sec_names = [header[key][f"SubSection{i}"] for i in range(n)]
        spec_mode = header[key + "\\" + sec_names[0]]["SpecMode"]
        table = header[key + "\\" + sec_names[1]]
        count = int(table["Count"])
        spec_map = np.array(
            [table[spec_mode[:3] + str(i)].split(";") for i in range(count)]
        ).astype(float)

    image_center = []
    key = "DataSet\\Parameters\\Imaging"
    if key in header:
        image_center, _ = _parse_nanosurf_value(header[key]["ScanOffset"])

    thermal_tune = None
    info = header.get("DataSet-Info", {})
    if "-- Thermal Tuning --" in info:
        thermal_tune = {"q_factor": float(info["Q Factor:"])}
        for name, key, pattern in (
            ("peak_freq", "Frequency:", r"(\d+\.\d+)(\S*)"),
            ("spring_const", "Spring Constant:", r"(\d+\.\d+)\s(\S*)"),
            ("peak_amp", "Peak Value:", r"(\d+\.\d+)(\S*)"),
        ):
            value, unit = re.search(pattern, info[key]).groups()
            thermal_tune[name] = value
            thermal_tune[name + "_unit"] = unit

    header["datasets"] = datasets
    header["param"] = param
    header["cantilever"] = cantilever
    header["spec_map"] = spec_map
    header["image_center"] = image_center
    header["thermal_tune"] = thermal_tune
    return header


@frozen
class _NIDDataset:
    data: Buffer
    name: str
    frame: str
    channel: str
    unit: str
    offset: int
    points: int
    lines: int
    bpp: int
    cur_line: int
    unit_scale: float
    unit_offset: float
    line_points: np.ndarray = field(repr=False)

    @classmethod
    def parse(cls, data: Buffer, h, name, offset, line_points=None):
        bits = int(h["SaveBits"])
        unit_range = float(h["Dim2Range"])
        if line_points is None:
            line_points = _parse_nanosurf_line_points(h)
        line_points = np.asarray(line_points, dtype=int)
        return cls(
            data=data,
            name=name,
            frame=h["Frame"],
            channel=h["Dim2Name"],
            unit=h["Dim2Unit"],
            offset=offset,
            points=int(h["Points"]),
            lines=int(h["Lines"]),
            bpp=bits // 8,
            cur_line=int(h["CurLine"]),
            unit_scale=unit_range / 2**bits,
            unit_offset=unit_range / 2 + float(h["Dim2Min"]),
            line_points=line_points,
        )

    @property
    def nbytes(self) -> int:
        return self.points * self.lines * self.bpp

    def _get_floats(self, shape: Index | int, offset: int, index=...) -> np.ndarray:
        with memoryview(self.data) as v:  # assert data is open, and hold it open
            floats = np.ndarray(
                shape=shape,
                dtype=f"<i{self.bpp}",
                buffer=v,
                offset=offset,
            )[index].astype("f4", copy=True)
        floats *= self.unit_scale
        floats += self.unit_offset
        return floats

    def _get_block_floats(self, index=...) -> np.ndarray:
        # the dataset is stored as one (lines, points) block
        return self._get_floats((self.lines, self.points), self.offset, index)

    def get_ndarray(self) -> np.ndarray:
        return self._get_block_floats()

    def get_line(self, line: int) -> np.ndarray:
        offset = self.offset + line * self.points * self.bpp
        return self._get_floats(self.points, offset)

    def get_column(self, column: int, n_lines: int) -> np.ndarray:
        return self._get_block_floats(np.s_[:n_lines, column])

    def get_line_ends(self, n_lines: int) -> np.ndarray:
        # each line ends at its own LineDim{n}Points
        lines = np.arange(n_lines)
        return self._get_block_floats((lines, self.line_points[:n_lines] - 1))


def _nid_map_line_to_rc(line: int, nx: int) -> Index:
    # the map is scanned as a serpentine starting from the bottom row
    r, c = divmod(line, nx)
    return r, (c if r % 2 == 0 else nx - 1 - c)


def _nid_map_rc_to_line(r: int, c: int, nx: int) -> int:
    return r * nx + (c if r % 2 == 0 else nx - 1 - c)


def _nid_map_from_lines(values: np.ndarray, shape: Index) -> np.ndarray:
    # lines that were never scanned have no value, so start from nan
    image = np.full(shape, np.nan, dtype=np.float32)
    for line, value in enumerate(values):
        r, c = _nid_map_line_to_rc(line, shape[1])
        image[r, c] = value
    return image


@frozen
class NIDVolume:
    name: str
    shape: Index
    n_lines: int
    k: float
    _zfactors: np.ndarray = field(repr=False)  # float32 fwd and bwd to nm
    _dfactors: np.ndarray = field(repr=False)
    _zfwd: _NIDDataset
    _dfwd: _NIDDataset
    _zbwd: _NIDDataset
    _dbwd: _NIDDataset
    nancurve: np.ndarray = field(repr=False)

    @classmethod
    def parse(cls, spec, spec_map_row, k, defl_sens):
        """Volume of the spectroscopy map; k in N/m and defl_sens in nm/V."""
        zfwd = spec["Spec forward", "Z-Axis Sensor"]
        dfwd = spec["Spec forward", "Deflection"]
        zbwd = spec["Spec backward", "Z-Axis Sensor"]
        dbwd = spec["Spec backward", "Deflection"]

        def zfactor(d):
            factor = _nano_factor(d.unit, "m")
            if factor is None:
                raise FileFormatError(
                    "Expected curve data in a length.", d.frame, d.channel, d.unit
                )
            return factor

        def dfactor(d):
            if d.unit == "V":
                return defl_sens
            if (factor := _nano_factor(d.unit, "N")) is not None:
                return factor / k
            if (factor := _nano_factor(d.unit, "m")) is not None:
                return factor
            raise FileFormatError(
                "Expected curve data in a force, a length or V.",
                d.frame,
                d.channel,
                d.unit,
            )

        zfactors = np.array([[zfactor(zfwd)], [zfactor(zbwd)]], dtype="f4")
        dfactors = np.array([[dfactor(dfwd)], [dfactor(dbwd)]], dtype="f4")
        nx, ny = int(spec_map_row[4]), int(spec_map_row[5])
        # a stopped scan leaves the lines past CurLine unwritten
        n_lines = min(
            min(d.cur_line + 1, d.lines, len(d.line_points))
            for d in (zfwd, dfwd, zbwd, dbwd)
        )
        nancurve = _nan_curve(min(zfwd.points, zbwd.points))
        return cls(
            "SpecMap",
            (ny, nx),
            n_lines,
            k,
            zfactors,
            dfactors,
            zfwd,
            dfwd,
            zbwd,
            dbwd,
            nancurve,
        )

    def _line_npts(self, line: int) -> tuple[int, int]:
        # rows are zero filled past LineDim{n}Points
        return int(self._zfwd.line_points[line]), int(self._zbwd.line_points[line])

    def _get_line_curve(self, line: int, n: int | None = None) -> Curve:
        nf, nb = self._line_npts(line)
        ne, nr = (nf, nb) if n is None else (n, n)
        zf, zb = self._zfactors[:, 0]
        df, db = self._dfactors[:, 0]
        zext = self._zfwd.get_line(line)[nf - ne : nf] * zf
        dext = self._dfwd.get_line(line)[nf - ne : nf] * df
        # backward rows are stored far-to-near
        zret = self._zbwd.get_line(line)[nb - nr : nb][::-1] * zb
        dret = self._dbwd.get_line(line)[nb - nr : nb][::-1] * db
        return (zext, zret), (dext, dret)

    def get_curve(self, r: int, c: int) -> Curve:
        """Efficiently get a specific curve from disk."""
        ny, nx = self.shape
        if not (0 <= r < ny and 0 <= c < nx):
            raise ValueError("Invalid index:", self.shape, (r, c))
        line = _nid_map_rc_to_line(r, c, nx)
        if line >= self.n_lines:
            return _halves(*self.nancurve)
        return self._get_line_curve(line)

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        nx = self.shape[1]
        for line in range(self.n_lines):
            yield _nid_map_line_to_rc(line, nx)

    def iter_curves(self) -> Iterable[tuple[Index, Curve]]:
        """Iterate over curves lazily in on-disk order."""
        nx = self.shape[1]
        for line in range(self.n_lines):
            yield _nid_map_line_to_rc(line, nx), self._get_line_curve(line)

    def get_all_curves(self) -> np.ndarray:
        """Eagerly load all curves into memory."""
        nx = self.shape[1]
        n_min = min(n for line in range(self.n_lines) for n in self._line_npts(line))
        x = np.full(self.shape + (2, 2, n_min), np.nan, dtype=np.float32)
        for line in range(self.n_lines):
            r, c = _nid_map_line_to_rc(line, nx)
            x[r, c] = self._get_line_curve(line, n_min)
        return np.moveaxis(x, 2, 0)


@frozen
class NIDMapHeightImage:
    name: str
    shape: Index
    n_lines: int
    _zfwd: _NIDDataset

    @property
    def units(self) -> str:
        return self._zfwd.unit

    def get_image(self) -> np.ndarray:
        """Get the image from disk."""
        # the extend curve ends at max force, and rows are zero filled past it
        heights = self._zfwd.get_line_ends(self.n_lines)
        return _nid_map_from_lines(heights, self.shape)


@frozen
class NIDMapImage:
    name: str
    shape: Index
    n_lines: int
    sign: float
    _dataset: _NIDDataset

    @property
    def units(self) -> str:
        return self._dataset.unit

    def get_image(self) -> np.ndarray:
        """Get the image from disk."""
        values = self._dataset.get_column(0, self.n_lines)
        values *= self.sign
        return _nid_map_from_lines(values, self.shape)


@frozen
class NIDFile:
    headers: dict[str, Any] = field(repr=lambda x: f"<dict with {len(x)} entries>")
    images: dict[str, Image]
    volumes: list[Volume]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]
    sync_dist: float | None = None
    trace: int | None = None

    @staticmethod
    def check_type(data: Buffer):
        # Check for magic string indicating Nanosurf
        magic = b"[DataSet]"
        start = bytes(memoryview(data)[: len(magic)])
        if not start == magic:
            raise FileFormatError("Not a Nanosurf NID file.", start)

    @classmethod
    def parse(cls, data: Buffer):
        cls.check_type(data)
        # End of header is demarcated by the bytes b"#!"
        header_end_pos = _find_in_buffer(data, b"#!")
        if header_end_pos < 0:
            raise FileFormatError(
                "No stop bytes found, are you sure this is a NID file?"
            )
        header = _parse_nanosurf_header(
            bytes(memoryview(data)[:header_end_pos]).decode("utf-8").splitlines()
        )

        # datasets are stored contiguously in header order right after the marker
        offset = header_end_pos + 2
        datasets = {}
        for name, points in zip(header["datasets"], header["param"]["LinePoints"]):
            dataset = _NIDDataset.parse(data, header[name], name, offset, points)
            datasets[name] = dataset
            offset += dataset.nbytes
        nbytes = memoryview(data).nbytes
        if offset > nbytes:
            raise FileFormatError(
                "Datasets overrun the end of the file.", offset, nbytes
            )

        k, k_unit = _parse_nanosurf_value(header["cantilever"]["Prop0"])
        k_exponent = _si_prefix_exponent(k_unit, "N/m")
        if k_exponent is None:
            raise FileFormatError(
                "Expected spring constant in a force per length.", k_unit
            )
        k *= 10.0**k_exponent
        defl_sens = (
            float(header["cantilever"]["Sensitivity"]) * _NANOMETER_UNIT_CONVERSION
        )
        spectroscopy = header["DataSet\\Parameters\\Spectroscopy"]
        mod_time, _ = _parse_nanosurf_value(spectroscopy["FwModTime"])
        mod_points, _ = _parse_nanosurf_value(spectroscopy["FwModDatapoints"])
        t_step = mod_time / mod_points

        spec_mode = header["DataSet\\SpecInfos\\SpecHeader"]["SpecMode"]
        if spec_mode != "Map" or len(header["spec_map"]) != 1:
            raise FileFormatError(
                "Only single spectroscopy maps are supported.",
                spec_mode,
                len(header["spec_map"]),
            )
        xmin, xmax, ymin, ymax, nx, ny = header["spec_map"][0][:6]
        if nx < 2 or ny < 2:
            raise FileFormatError(
                f"spectroscopy map grid is {nx:g} columns by {ny:g} rows, expected at "
                "least two of each"
            )
        # spectroscopy datasets are identified by frame and channel, not name
        spec = {(d.frame, d.channel): d for d in datasets.values()}
        volume = NIDVolume.parse(spec, header["spec_map"][0], k, defl_sens)
        image = NIDMapHeightImage(
            "MapHeight",
            volume.shape,
            volume.n_lines,
            spec["Spec forward", "Z-Axis Sensor"],
        )
        images = {image.name: image}

        # datasets on the map grid whose channel is a NameZ<n> placeholder hold
        # per-curve analysis results in column 0, later columns hold the live chart's
        # running min/max except Slope frames, which hold a fit window, slope and
        # intercept
        n_map = volume.shape[0] * volume.shape[1]
        for d in datasets.values():
            if (
                d.frame not in _NANOSURF_RAW_FRAMES
                and not d.frame.startswith("Slope")
                and d.lines == n_map
                and re.fullmatch(r"NameZ\d+", d.channel)
            ):
                name, sign = _NANOSURF_RESULT_IMAGES.get(d.frame, (d.frame, 1.0))
                images[name] = NIDMapImage(name, volume.shape, volume.n_lines, sign, d)

        # map extents are inclusive of the outermost points, so pitch is span/(n-1)
        scansize = (
            float((xmax - xmin) * nx / (nx - 1) * _NANOMETER_UNIT_CONVERSION),
            float((ymax - ymin) * ny / (ny - 1) * _NANOMETER_UNIT_CONVERSION),
        )
        return cls(header, images, [volume], k, defl_sens, t_step, scansize)


SUFFIX_FVFILE_MAP = {
    ".ardf": (ARDFFile, mmap_path_read_only),
    ".h5": (ARH5File, open_h5),
    ".spm": (NanoscopeFile, mmap_path_read_only),
    ".pfc": (NanoscopeFile, mmap_path_read_only),
    ".nid": (NIDFile, mmap_path_read_only),
}
