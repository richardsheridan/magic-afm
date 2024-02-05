"""Magic AFM Data Readers

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
import struct
from collections.abc import Collection, Iterable
from functools import partial
from mmap import mmap
from subprocess import PIPE
from typing import Protocol, TypeAlias, Any

try:
    from subprocess import STARTF_USESHOWWINDOW, STARTUPINFO
except ImportError:
    STARTUPINFO = lambda *a, **kw: None
    STARTF_USESHOWWINDOW = None

import numpy as np

from attrs import frozen, field

NANOMETER_UNIT_CONVERSION = (
    1e9  # maybe we can intelligently read this from the file someday
)
NANCURVE = np.full(shape=(2, 2, 100), fill_value=np.nan, dtype=np.float32)
NANCURVE.setflags(write=False)
TOC_STRUCT = struct.Struct("<QLL")

###############################################
############### Typing stuff ##################
###############################################


Index: TypeAlias = tuple[int, ...]
ZDArrays: TypeAlias = Collection[np.ndarray]
ChanMap: TypeAlias = dict[str, tuple[int, "ARDFVchan"]]
StepInfo: TypeAlias = tuple[tuple[float, str], ...]


class Image(Protocol):
    name: str
    shape: Index

    def get_image(self) -> np.ndarray:
        """Get the image from disk."""
        ...


class Volume(Protocol):
    name: str
    shape: Index

    def get_curve(self, r: int, c: int) -> ZDArrays:
        """Efficiently get a specific curve from disk."""
        ...

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        ...

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        ...

    def get_all_curves(self) -> ZDArrays:
        """Eagerly load all curves into memory."""
        ...


class FVFile(Protocol):
    headers: dict[str, Any]
    images: dict[str, Image]
    volumes: list[Volume]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]

    @classmethod
    def parse(cls, data) -> "FVFile":
        # TODO: data should be data: Buffer after 3.12
        ...


###############################################
################## Helpers ####################
###############################################


def open_h5(path):
    import h5py

    return h5py.File(path, "r")


def mmap_path_read_only(path):
    import mmap

    with open(path, mode="rb", buffering=0) as file:
        return mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)


def decode_cstring(cstring: bytes):
    return cstring.rstrip(b"\0").decode("windows-1252")


# noinspection PyUnboundLocalVariable
def parse_nanoscope_header(header_lines: Iterable[str]):
    """Convert header from a Nanoscope file to a convenient nested dict

    header_lines can be an opened file object or a list of strings or anything
    that iterates the header line-by-line."""

    header = {}
    for line in header_lines:
        assert line.startswith("\\")
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
        raise ValueError("File ended too soon")
    if (not header) or ("" in header):
        raise ValueError("File is empty or not a Bruker data file")

    return header


def parse_ar_note(note: Iterable[str]):
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


###############################################
################## FVFiles ####################
###############################################


###############################################
################### Asylum ####################
###############################################


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

        # Generally segments are [Ext, Dwell, Ret, Away] or [Ext, Ret, Away]
        # for magic, we don't dwell. new converter ensures this assertion
        assert num_segments == 3

        # this is all necessary because the arrays are not of uniform length
        # We will cut all arrays down to the length of the smallest
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

    def _shared_get_part(self, curve, s):
        # Index into the data and grab the Defl and Zsnsr ext and ret arrays as one 2D array
        defl_zsnsr = curve[self.defl_zsnsr_row_slice, :]  # XXX Read h5data

        # we are happy to throw away data far from the surface to square up the data
        # Also reverse axis zero so data is ordered zsnsr,defl like we did for FFM
        return (
            defl_zsnsr[::-1, (s - self.minext) : (s + self.minret)]
            * NANOMETER_UNIT_CONVERSION
        )

    def get_curve(self, r, c):
        # Because of the nonuniform arrays, each indent gets its own dataset
        # indexed by 'row:column' e.g. '1:1'.
        curve = self.force_curves[f"{r}:{c}"]  # XXX Read h5data
        split = self.extlens[r, c]

        z, d = self._shared_get_part(curve, split)
        split = self.minext
        return (z[:split], z[split:]), (d[:split], d[split:])

    def iter_indices(self) -> Iterable[Index]:
        for index in self.force_curves:
            if index == "Segments":
                continue
            r, c = list(map(int, index.split(":")))
            yield r, c

    def iter_curves(self):
        for index, curve in self.force_curves.items():
            # Unfortunately they threw in segments here too, so we skip over it
            if index == "Segments":
                continue
            # Because of the nonuniform arrays, each indent gets its own dataset
            # indexed by 'row:column' e.g. '1:1'. We could start with the shape and index
            # manually, but the string munging is easier for me to think about
            r, c = list(map(int, index.split(":")))
            split = self.extlens[r, c]
            yield (r, c), self._shared_get_part(curve, split)

    def get_all_curves(self):
        im_r, im_c, num_segments = self.segments.shape
        x = np.empty((im_r, im_c, 2, self.minext + self.minret), dtype=np.float32)
        for index, curve in self.iter_curves():
            x[*index, :, :] = curve
        return x.reshape(x.shape[:-1] + (2, -1))


@frozen
class ARH5Image:
    data: dict
    name: str
    shape: Index

    def get_image(self) -> np.ndarray:
        """Get the image from disk."""
        return self.data[:]


@frozen
class ARH5FFMVolume:
    name: str
    shape: Index
    # step_info: StepInfo
    _zreader: np.ndarray
    _dreader: np.ndarray

    def get_curve(self, r, c):
        z = self._zreader[r, c].reshape((2, -1))
        d = self._dreader[r, c].reshape((2, -1))
        return z * NANOMETER_UNIT_CONVERSION, d * NANOMETER_UNIT_CONVERSION

    def iter_indices(self) -> Iterable[Index]:
        return np.ndindex(self.shape)

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        for index in self.iter_indices():
            yield index, self.get_curve(*index)

    def get_all_curves(self):
        z = self._zreader[:] * NANOMETER_UNIT_CONVERSION
        d = self._dreader[:] * NANOMETER_UNIT_CONVERSION
        return z, d


@frozen
class ARH5File:
    headers: dict[str, Any] = field(repr=lambda x: f"<dict with {len(x)} entries>")
    images: dict[str, Image]
    volumes: list[Volume]
    k: float
    defl_sens: float
    t_step: float
    scansize: tuple[float, float]

    @classmethod
    def parse(cls, h5data):
        notes = parse_ar_note(h5data.attrs["Note"].splitlines())
        images = {
            name: ARH5Image(img, name, img.shape)
            for name, img in h5data["Image"].items()
        }

        k = float(notes["SpringConstant"])
        scansize = (
            float(notes["FastScanSize"]) * NANOMETER_UNIT_CONVERSION,
            float(notes["SlowScanSize"]) * NANOMETER_UNIT_CONVERSION,
        )
        # NOTE: aspect is redundant to scansize
        # self.aspect = float(self.notes["SlowRatio"]) / float(self.notes["FastRatio"])
        defl_sens = float(notes["InvOLS"]) * NANOMETER_UNIT_CONVERSION
        rate = float(notes["FastMapZRate"])
        volumes = []
        if "FFM" in h5data:
            # self.scandown = bool(self.notes["ScanDown"])
            if "0" in h5data["FFM"]:
                volumes.append(
                    ARH5FFMVolume(
                        "Trace",
                        h5data["FFM"]["0"]["Raw"].shape[:2],
                        h5data["FFM"]["0"]["Raw"],
                        h5data["FFM"]["0"]["Defl"],
                    )
                )
            if "1" in h5data["FFM"]:
                volumes.append(
                    ARH5FFMVolume(
                        "Retrace",
                        h5data["FFM"]["1"]["Raw"].shape[:2],
                        h5data["FFM"]["1"]["Raw"],
                        h5data["FFM"]["1"]["Defl"],
                    )
                )
            if "Raw" in h5data["FFM"]:
                volumes.append(
                    ARH5FFMVolume(
                        "Trace",
                        h5data["FFM"]["Raw"].shape[:2],
                        h5data["FFM"]["Raw"],
                        h5data["FFM"]["Defl"],
                    )
                )
            npts = volumes[0]._zreader.shape[-1]
        else:
            # self.scandown = bool(self.notes["FMapScanDown"])
            volumes.append(ARH5ForceMapVolume(h5data))
            npts = volumes[0].npts
        t_step = 1 / rate / npts
        return cls(notes, images, volumes, k, defl_sens, t_step, scansize)


@frozen
class ARDFHeader:
    data: mmap
    offset: int
    crc: int = field(repr=hex)
    size: int
    name: bytes
    flags: int = field(repr=hex)
    _struct = struct.Struct("<LL4sL")

    @classmethod
    def unpack(cls, data: mmap, offset: int):
        return cls(data, offset, *cls._struct.unpack_from(data, offset))

    def validate(self):
        # ImHex poly 0x4c11db7 init 0xffffffff xor out 0xffffffff reflect in and out
        import zlib

        crc = zlib.crc32(
            memoryview(self.data)[self.offset + 4 : self.offset + self.size]
        )
        if self.crc != crc:
            raise ValueError(
                f"Invalid section. Expected {self.crc:X}, got {crc:X}.", self
            )
        return True


@frozen
class ARDFTableOfContents:
    data: mmap
    offset: int
    size: int
    entries: list[tuple[ARDFHeader, int]]
    _entry_struct = struct.Struct("<LL4sLQ")

    @classmethod
    def unpack(cls, header: ARDFHeader):
        if header.size != 32:
            raise ValueError("Malformed table of contents", header)
        header.validate()
        data = header.data
        offset = header.offset
        size, nentries, stride = TOC_STRUCT.unpack_from(data, offset + 16)
        assert stride == 24
        assert size - 32 == nentries * stride, (size, nentries, stride)
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, pointer = cls._entry_struct.unpack_from(data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = ARDFHeader(data, toc_offset, *header)
            if entry_header.name not in {b"IMAG", b"VOLM", b"NEXT", b"THMB", b"NSET"}:
                raise ValueError("Malformed table of contents.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)


@frozen
class ARDFTextTableOfContents:
    data: mmap
    offset: int
    size: int
    entries: list[tuple[ARDFHeader, int]]
    _entry_struct = struct.Struct("<LL4sLQQ")

    @classmethod
    def unpack(cls, header: ARDFHeader):
        if header.size != 32 or header.name != b"TTOC":
            raise ValueError("Malformed text table of contents.", header)
        header.validate()
        offset = header.offset
        data = header.data
        size, nentries, stride = TOC_STRUCT.unpack_from(data, offset + 16)
        assert stride == 32, stride
        assert size - 32 == nentries * stride, (size, nentries, stride)
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, _, pointer = cls._entry_struct.unpack_from(data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = ARDFHeader(data, toc_offset, *header)
            if entry_header.name != b"TOFF":
                raise ValueError("Malformed text table entry.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)

    def decode_entry(self, index: int):
        # maybe could be another TEXT class but we'll read it straight in
        entry_header, pointer = self.entries[index]
        text_header = ARDFHeader.unpack(self.data, pointer)
        if text_header.name != b"TEXT":
            raise ValueError("Malformed text section.", text_header)
        text_header.validate()
        i, text_len = struct.unpack_from("<LL", self.data, pointer + 16)
        assert i == index, (i, index)
        offset = text_header.offset + 24
        assert text_len < text_header.size - 24, (text_len, text_header)
        self.data.seek(offset)
        text = self.data.read(text_len)
        # self.data.seek(0)  # seems unneeded currently
        return text.replace(b"\r", b"\n").decode("windows-1252")


@frozen
class ARDFVolumeTableOfContents:
    offset: int
    size: int
    lines: np.ndarray
    points: np.ndarray
    pointers: np.ndarray
    _voff_struct = struct.Struct("<LLQQ")

    @classmethod
    def unpack(cls, header: ARDFHeader):
        # cant reuse exact ARDFTableOfContents for VTOC, but structure is similar
        data = header.data
        offset = header.offset
        if header.size != 32 or header.name != b"VTOC":
            raise ValueError("Malformed volume table of contents.", header)
        size, nentries, stride = TOC_STRUCT.unpack_from(data, offset + 16)
        assert stride == 40, stride
        assert size - 32 == nentries * stride, (size, nentries, stride)
        vtoc_arr = np.zeros(
            nentries,
            dtype=[  # match _voff_struct.format
                ("force_index", "L"),
                ("line", "L"),
                ("point", "Q"),
                ("pointer", "Q"),
            ],
        )
        for i, toc_offset in enumerate(
            range(header.offset + 32, header.offset + size, stride)
        ):
            entry_header = ARDFHeader.unpack(data, toc_offset)
            if not entry_header.crc:
                # scan was interrupted, and vtoc is zero-filled
                vtoc_arr = vtoc_arr[:i].copy()
                break
            if entry_header.name != b"VOFF":
                raise ValueError("Malformed volume table entry.", entry_header)
            entry_header.validate()
            vtoc_arr[i] = cls._voff_struct.unpack_from(data, toc_offset + 16)
        return cls(
            offset, size, vtoc_arr["line"], vtoc_arr["point"], vtoc_arr["pointer"]
        )


@frozen
class ARDFVchan:
    name: str
    unit: str
    _struct = struct.Struct("<32s32s")

    @classmethod
    def unpack(cls, header: ARDFHeader):
        if header.size != 80 or header.name != b"VCHN":
            raise ValueError("Malformed channel definition.", header)
        header.validate()
        name, unit = cls._struct.unpack_from(header.data, header.offset + 16)
        return cls(decode_cstring(name), decode_cstring(unit))


@frozen
class ARDFXdef:
    offset: int
    size: int
    xdef: list[str]
    _struct = struct.Struct("<LL")

    @classmethod
    def unpack(cls, header: ARDFHeader):
        if header.size != 96 or header.name != b"XDEF":
            raise ValueError("Malformed experiment definition.", header)
        header.validate()
        # Experiment definition is just a string
        _, nchars = cls._struct.unpack_from(header.data, header.offset + 16)
        assert _ == 0, _
        if nchars > header.size:
            raise ValueError("Experiment definition too long.", header, nchars)
        header.data.seek(header.offset + 24)
        xdef = header.data.read(nchars).decode("windows-1252").split(";")[:-1]
        # data.seek(0)  # seems unneeded currently
        return cls(header.offset, header.size, xdef)


@frozen
class ARDFVset:
    data: mmap
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
    def unpack(cls, vset_header: ARDFHeader):
        if vset_header.name != b"VSET":
            raise ValueError("malformed VSET header", vset_header)
        vset_header.validate()
        data = vset_header.data
        offset = vset_header.offset
        size = cls._struct.size + 16
        return cls(data, offset, size, *cls._struct.unpack_from(data, offset + 16))


@frozen
class ARDFVdata:
    data: mmap
    offset: int
    force_index: int
    line: int
    point: int
    nfloats: int
    channel: int
    seg_offsets: tuple[int, ...]
    _struct = struct.Struct("<10L")

    @classmethod
    def unpack(cls, header: ARDFHeader):
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

    def get_ndarray(self):
        with memoryview(self.data):  # assert data is open, and hold it open
            return (
                np.ndarray(
                    shape=self.nfloats,
                    dtype="<f4",
                    buffer=self.data,
                    offset=self.array_offset,
                )
                * NANOMETER_UNIT_CONVERSION
            )


@frozen
class ARDFImage:
    data: mmap
    ibox_offset: int
    name: str
    shape: Index
    units: str
    step_info: StepInfo
    _struct = struct.Struct("<LLQQdd32s32s32s32s")

    @classmethod
    def parse_imag(cls, imag_header: ARDFHeader):
        if imag_header.name != b"IMAG":
            raise ValueError("Malformed image header.", imag_header)
        imag_toc = ARDFTableOfContents.unpack(imag_header)

        # don't use NEXT or THMB data, step over
        ttoc_header = ARDFHeader.unpack(imag_toc.data, imag_toc.offset + imag_toc.size)
        ttoc = ARDFTextTableOfContents.unpack(ttoc_header)

        # don't use TTOC or TOFF, step over
        idef_header = ARDFHeader.unpack(ttoc.data, ttoc.offset + ttoc.size)
        if idef_header.name != b"IDEF" or idef_header.size != cls._struct.size + 16:
            raise ValueError("Malformed image definition.", idef_header)
        idef_header.validate()
        points, lines, _, __, x_step, y_step, *cstrings = cls._struct.unpack_from(
            idef_header.data,
            idef_header.offset + 16,
        )
        assert not (_ or __), (_, __)
        x_unit, y_unit, name, units = list(map(decode_cstring, cstrings))

        return cls(
            imag_header.data,
            idef_header.offset + idef_header.size,
            name,
            (points, lines),
            units,
            ((x_step, x_unit), (y_step, y_unit)),
        )

    def get_image(self):
        ibox_header = ARDFHeader.unpack(self.data, self.ibox_offset)
        if ibox_header.size != 32 or ibox_header.name != b"IBOX":
            raise ValueError("Malformed image layout.", ibox_header)
        ibox_header.validate()
        data_offset = ibox_header.offset + ibox_header.size + 16  # past IDAT header
        ibox_size, lines, stride = TOC_STRUCT.unpack_from(
            ibox_header.data, ibox_header.offset + 16
        )
        points = (stride - 16) // 4  # less IDAT header
        assert (points, lines) == self.shape
        # elide image data validation and map into an array directly
        with memoryview(self.data):  # assert data is open, and hold it open
            arr = np.ndarray(
                shape=(lines, points),
                dtype="<f4",
                buffer=self.data,
                offset=data_offset,
                strides=(stride, 4),
            ).astype("f4")
        gami_header = ARDFHeader.unpack(
            ibox_header.data, ibox_header.offset + ibox_size
        )
        if gami_header.size != 16 or gami_header.name != b"GAMI":
            raise ValueError("Malformed image layout.", gami_header)
        gami_header.validate()
        return arr


@frozen
class ARDFFFMReader:
    data: mmap  # keep checking our mmap is open so array_view cannot segfault
    array_view: np.ndarray = field(repr=False)
    array_offset: int  # hard to recover from views
    channels: list[int]  # [z, d]
    # seg_offsets is weird. you'd think it would contain the starting index
    # for each segment. However, it always has a trailing value of 1-nfloats,
    # and nonexistent segments get a zero. For regular/FFM data, we'll just
    # assume that the second offset maps to our "split" concept.
    seg_offsets: tuple
    up: bool
    trace: bool

    @classmethod
    def parse(cls, first_vset_header: ARDFHeader, points: int, lines: int, channels):
        data = first_vset_header.data
        # just walk past these first headers to find our data_offset
        first_vset = ARDFVset.unpack(first_vset_header)
        vset_stride = first_vset.next_vset_offset - first_vset_header.offset
        if first_vset.vtype == 5:
            line_stride = vset_stride * points
        else:
            # better be "both" setting see ARDFVset
            assert first_vset.vtype in {10, 11}, first_vset
            line_stride = vset_stride * points * 2
        up = first_vset.line == 0
        trace = first_vset.point == 0

        first_vnam_header = ARDFHeader.unpack(
            data, first_vset_header.offset + first_vset_header.size
        )
        if first_vnam_header.name != b"VNAM":
            raise ValueError("Malformed volume name", first_vnam_header)
        first_vnam_header.validate()
        first_vdat_header = ARDFHeader.unpack(
            data, first_vnam_header.offset + first_vnam_header.size
        )
        if first_vdat_header.name != b"VDAT":
            raise ValueError("Malformed volume data")
        first_vdat = ARDFVdata.unpack(first_vdat_header)
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
            seg_offsets=first_vdat.seg_offsets,
            up=up,
            trace=trace,
        )

    def get_curve(self, r, c):
        """Efficiently get a specific curve from disk."""
        with memoryview(self.data):  # assert data is open, and hold it open
            x = self.array_view[r, c, self.channels]  # advanced indexing copies
        return x.reshape((len(self.channels), 2, -1)) * NANOMETER_UNIT_CONVERSION

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

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        # TODO: cleverly use np.nditer?
        for index in self.iter_indices():
            yield index, self.get_curve(*index)

    def get_all_curves(self) -> ZDArrays:
        """Eagerly load all curves into memory."""
        with memoryview(self.data):  # assert data is open, and hold it open
            # advanced indexing triggers a copy
            loaded_data = self.array_view[:, :, self.channels, :]
        # avoid a second copy with inplace op
        loaded_data *= NANOMETER_UNIT_CONVERSION
        # reshape assuming equal points on extend and retract
        return loaded_data.reshape(self.array_view.shape[:-1] + (2, -1))


@frozen
class ARDFForceMapReader:
    data: mmap
    vtoc: ARDFVolumeTableOfContents
    lines: int
    points: int
    vtype: int
    channels: ChanMap
    _seen_vsets: dict[Index, ARDFVset] = field(init=False, factory=dict)

    @property
    def zname(self):
        return "Raw" if "Raw" in self.channels else "ZSnsr"

    def traverse_vsets(self, pointer: int):
        while True:
            header = ARDFHeader.unpack(self.data, pointer)
            if header.name != b"VSET":
                break
            vset = ARDFVset.unpack(header)
            index = (vset.line, vset.point, vset.vtype)
            if index not in self._seen_vsets:
                self._seen_vsets[index] = vset
            yield vset  # .line, vset.point, vset.
            pointer = vset.next_vset_offset

    def traverse_vdats(self, pointer):
        vnam_header = ARDFHeader.unpack(self.data, pointer)
        if vnam_header.name != b"VNAM":
            raise ValueError("Malformed volume name", vnam_header)
        vnam_header.validate()
        pointer = vnam_header.offset + vnam_header.size
        while True:
            vdat_header = ARDFHeader.unpack(self.data, pointer)
            if vdat_header.name != b"VDAT":
                break
            vdat_header.validate()  # opportunity to read data with gil released
            yield ARDFVdata.unpack(vdat_header)
            pointer = vdat_header.offset + vdat_header.size

    def get_curve(self, r: int, c: int) -> ZDArrays:
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
            i = self.vtoc.lines[sl].searchsorted(r)
            if i >= len(self.vtoc.lines) or r != int(self.vtoc.lines[sl][i]):
                return NANCURVE
            # read entire line of the vtoc
            for vset in self.traverse_vsets(int(self.vtoc.pointers[sl][i])):
                if vset.line != r:
                    break
                # traverse_vsets implicitly fills in seen_vsets

        vset = self._seen_vsets[index]

        zxr, dxr = NANCURVE
        for vdat in self.traverse_vdats(vset.offset + vset.size):
            s = vdat.seg_offsets
            if vdat.channel == self.channels[self.zname][0]:
                z = vdat.get_ndarray()
                zxr = z[: s[1]], z[s[1] : s[2]]
            elif vdat.channel == self.channels["Defl"][0]:
                d = vdat.get_ndarray()
                dxr = d[: s[1]], d[s[1] : s[2]]
        return zxr, dxr

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        for vset in self.traverse_vsets(int(self.vtoc.pointers[0])):
            if vset.vtype != self.vtype:
                continue
            yield vset.line, vset.point

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        zname = self.zname
        for vset in self.traverse_vsets(int(self.vtoc.pointers[0])):
            if vset.vtype != self.vtype:
                continue
            zxr, dxr = NANCURVE
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                s = vdat.seg_offsets
                if vdat.channel == self.channels[zname][0]:
                    z = vdat.get_ndarray()
                    zxr = z[: s[1]], z[s[1] : s[2]]
                elif vdat.channel == self.channels["Defl"][0]:
                    d = vdat.get_ndarray()
                    dxr = d[: s[1]], d[s[1] : s[2]]
            yield (vset.line, vset.point), (zxr, dxr)

    def get_all_curves(self) -> ZDArrays:
        """Eagerly load all curves into memory."""
        minext = 0xFFFFFFFF
        vdats = {}
        for vset in self.traverse_vsets(int(self.vtoc.pointers[0])):
            if vset.vtype != self.vtype:
                continue
            x = vdats[vset.line, vset.point] = [None, None]
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                if vdat.channel == self.channels["ZSnsr"][0]:
                    x[0] = vdat  # zvdat
                elif vdat.channel == self.channels["Defl"][0]:
                    x[1] = vdat  # dvdat
                minext = min(minext, vdat.seg_offsets[1])
            assert None not in x, f"missing vdat channel: {x}, {self.channels}"
        del vset, vdat, x
        minfloats = 2 * minext
        x = np.empty((self.lines, self.points, 2, minfloats), dtype=np.float32)
        for (r, c), (zvdat, dvdat) in vdats.items():
            # code elsewhere assumes split is halfway through
            floats = zvdat.seg_offsets[1] * 2
            halfextra = (floats - minfloats) // 2  # even - even -> even
            sl = np.s_[halfextra : minfloats + halfextra]
            # TODO: verify turnaround point against iter_curves
            x[r, c, :, :] = zvdat.get_ndarray()[sl], dvdat.get_ndarray()[sl]
        return x.reshape(x.shape[:-1] + (2, -1))


@frozen
class ARDFVolume:
    volm_offset: int
    name: str
    shape: Index
    step_info: StepInfo
    xdef: ARDFXdef
    _reader: ARDFForceMapReader | ARDFFFMReader
    _struct = struct.Struct("<LL24sddd32s32s32s32sQ")

    @classmethod
    def parse_volm(cls, volm_header: ARDFHeader):
        data = volm_header.data
        if volm_header.size != 32 or volm_header.name != b"VOLM":
            raise ValueError("Malformed volume header.", volm_header)
        # the next headers look a lot like VSET is a table of contents, but I've only
        # seen NEXT and NSET headers. NEXT shows up if "both" trace and retrace data
        # are inside. I'll assume the last entry is always NSET, the number of VSETs.
        volm_toc = ARDFTableOfContents.unpack(volm_header)
        nset_header, nsets = volm_toc.entries[-1]
        nset_header.validate()
        ttoc_header = ARDFHeader.unpack(data, volm_toc.offset + volm_toc.size)
        ttoc = ARDFTextTableOfContents.unpack(ttoc_header)

        # don't use TTOC or TOFF, step over

        # cls essentially represents VDEF plus its linkage down to VSET
        # so this unpacking is intentionally inlined here.
        vdef_header = ARDFHeader.unpack(data, ttoc.offset + ttoc.size)
        if vdef_header.name != b"VDEF" or vdef_header.size != cls._struct.size + 16:
            raise ValueError("Malformed volume definition.", vdef_header)
        vdef_header.validate()
        unpack_from = cls._struct.unpack_from  # line wrapping
        points, lines, _, x_step, y_step, t_step, *cstrings, nseg = unpack_from(
            vdef_header.data, vdef_header.offset + 16
        )
        assert sum(_) == 0, _
        complete = points * lines == nsets
        x_unit, y_unit, t_unit, seg_names = list(map(decode_cstring, cstrings))
        seg_names = seg_names.split(";")[:-1]
        assert nseg == len(seg_names)

        # Implicit table of channels here smh
        offset = vdef_header.offset + vdef_header.size
        channels: ChanMap = {}
        for i in range(5):
            header = ARDFHeader.unpack(data, offset)
            if header.name != b"VCHN":
                break
            vchn = ARDFVchan.unpack(header)
            offset = header.offset + header.size
            # TODO: apply NANOMETER_UNIT_CONVERSION depending on this
            assert vchn.unit == "m"
            channels[vchn.name] = (i, vchn)
        else:
            raise RuntimeError("Got too many channels.", channels)

        xdef = ARDFXdef.unpack(header)
        vtoc_header = ARDFHeader.unpack(data, xdef.offset + xdef.size)
        vtoc = ARDFVolumeTableOfContents.unpack(vtoc_header)

        mlov_header = ARDFHeader.unpack(data, vtoc.offset + vtoc.size)
        if mlov_header.size != 16 or mlov_header.name != b"MLOV":
            raise ValueError("Malformed volume table of contents.", mlov_header)
        mlov_header.validate()

        # Check if each offset is regularly spaced
        # optimize for LARGE regular case (FMaps are SMALL)
        first_vset_header = ARDFHeader.unpack(data, int(vtoc.pointers[0]))
        if complete and not np.any(np.diff(np.diff(vtoc.pointers.astype(np.uint64)))):
            reader = ARDFFFMReader.parse(first_vset_header, points, lines, channels)
            name = "Trace" if reader.trace else "Retrace"
        else:
            first_vset = ARDFVset.unpack(first_vset_header)
            reader = ARDFForceMapReader(
                data,
                vtoc,
                lines,
                points,
                first_vset.vtype,
                channels,
            )
            name = "FMAP"

        return cls(
            volm_header.offset,
            name,
            (points, lines),
            ((x_step, x_unit), (y_step, y_unit), (t_step, t_unit)),
            xdef,
            reader,
        )

    def get_curve(self, r: int, c: int) -> ZDArrays:
        """Efficiently get a specific curve from disk."""
        return self._reader.get_curve(r, c)

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        return self._reader.iter_indices()

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        return self._reader.iter_curves()

    def get_all_curves(self) -> ZDArrays:
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

    @classmethod
    def parse(cls, data: mmap):
        file_header = ARDFHeader.unpack(data, 0)
        if file_header.size != 16 or file_header.name != b"ARDF":
            raise ValueError("Not an ARDF file.", file_header)
        file_header.validate()
        ftoc_header = ARDFHeader.unpack(data, offset=file_header.size)
        if ftoc_header.name != b"FTOC":
            raise ValueError("Malformed ARDF file table of contents.", ftoc_header)
        ftoc = ARDFTableOfContents.unpack(ftoc_header)
        ttoc_header = ARDFHeader.unpack(data, offset=ftoc.offset + ftoc.size)
        ttoc = ARDFTextTableOfContents.unpack(ttoc_header)
        assert len(ttoc.entries) == 1
        notes = parse_ar_note(ttoc.decode_entry(0).splitlines())
        images = {}
        volumes = []
        for item, pointer in ftoc.entries:
            item.validate()
            item = ARDFHeader.unpack(data, pointer)
            if item.name == b"IMAG":
                item = ARDFImage.parse_imag(item)
                images[item.name] = item
            elif item.name == b"VOLM":
                item = ARDFVolume.parse_volm(item)
                # volumes[item.name] = item
                volumes.append(item)
            else:
                raise RuntimeError(f"Unknown TOC entry {item.name}.", item)

        k = float(notes["SpringConstant"])
        # slight numerical differences here
        # lines, points = volumes[0].shape
        # xsize = lines * volumes[0].step_info[0][0]
        # ysize = points * volumes[0].step_info[1][0]
        scansize = (
            float(notes["FastScanSize"]) * NANOMETER_UNIT_CONVERSION,
            float(notes["SlowScanSize"]) * NANOMETER_UNIT_CONVERSION,
        )
        # NOTE: aspect is redundant to scansize
        # self.aspect = float(self.notes["SlowRatio"]) / float(self.notes["FastRatio"])
        defl_sens = float(notes["InvOLS"]) * NANOMETER_UNIT_CONVERSION
        t_step = volumes[0].step_info[-1][0]
        trace = 1 if len(volumes) > 1 else None
        return cls(notes, images, volumes, k, defl_sens, t_step, scansize, trace)


###############################################
################### Bruker ####################
###############################################


@frozen
class NanoscopeImage:
    data: mmap
    name: str
    offset: int
    length: int
    bpp: int
    unit_scale: float
    unit_offset: float
    shape: Index

    @classmethod
    def parse_image(cls, data, header, h, name):
        length = int(h["Data length"])
        offset = int(h["Data offset"])
        r = int(h["Number of lines"])
        c = int(h["Samps/line"])
        bpp = length // (r * c)
        value = h["@2:Z scale"]
        hard_scale = float(value.split()[-2]) / (2 ** (bpp * 8))
        hard_offset = float(h["@2:Z offset"].split()[-2]) / (2 ** (bpp * 8))
        soft_scale_name = "@" + value[1 + value.find("[") : value.find("]")]
        try:
            soft_scale_string = header["Ciao scan list"][0][soft_scale_name]
        except KeyError:
            soft_scale_string = header["Scanner list"][0][soft_scale_name]
        soft_scale = float(soft_scale_string.split()[1]) / NANOMETER_UNIT_CONVERSION
        unit_scale = np.float32(hard_scale * soft_scale)
        unit_offset = np.float32(hard_offset * soft_scale)
        return cls(
            data=data,
            name=name,
            offset=offset,
            length=length,
            bpp=bpp,
            unit_scale=unit_scale,
            unit_offset=unit_offset,
            shape=(r, c),
        )

    def get_image(self) -> np.ndarray:
        z_floats = np.zeros(self.shape, dtype=np.float32)
        with memoryview(self.data):  # assert data is open, and hold it open
            z_ints = np.ndarray(
                shape=self.shape,
                dtype=f"<i{self.bpp}",
                buffer=self.data,
                offset=self.offset,
            )
            z_floats += z_ints
        z_floats *= self.unit_scale
        z_floats += self.unit_offset
        return z_floats


@frozen
class FFVReader:
    data: mmap
    name: str
    ints: np.ndarray = field(repr=False)
    scale: np.float32
    _soft_scale_map = {
        "Height Sensor": "@Sens. ZsensSens",
        "Height": "@Sens. ZSens",
        "Deflection Error": "@Sens. DeflSens",
    }

    @classmethod
    def parse(cls, header, data, subheader):
        c = int(header["Ciao scan list"][0]["Samps/line"])
        r = int(header["Ciao scan list"][0]["Lines"])
        name = subheader["@4:Image Data"].split('"')[1]
        offset = int(subheader["Data offset"])
        length = int(subheader["Data length"])
        # npts = int(subheader["Sample Points"])  # only QNM?
        n_ext, n_ret = tuple(map(int, subheader["Samps/line"].split()))
        bpp = length // (r * c * (n_ret + n_ext))
        ints = np.ndarray(
            shape=(r, c, 2, n_ext), dtype=f"<i{bpp}", buffer=data, offset=offset
        )
        value = subheader["@4:Z scale"]
        soft_scale = float(
            header["Ciao scan list"][0][cls._soft_scale_map[name]].split()[1]
        )
        hard_scale = float(value[1 + value.find("(") : value.find(")")].split()[0])
        scale = np.float32(soft_scale * hard_scale)
        return cls(data, name, ints, scale)

    def get_curve(self, r, c):
        with memoryview(self.data):  # assert data is open, and hold it open
            f = self.ints[r, c] * self.scale
        return f[0, ::-1], f[1]


@frozen
class QNMDReader:
    data: mmap
    name: str
    ints: np.ndarray = field(repr=False)
    shape: Index
    scale: np.float32
    sync_dist_slice: slice
    phasors: np.ndarray
    sync_frac: float

    @classmethod
    def parse(cls, header, data, subheader, shape, sync_dist):
        # QNM doesn't have its own height data, so it must be synthesized
        # I've chosen to do this by combining a height image with a sine wave

        r, c = shape
        length = int(subheader["Data length"])
        offset = int(subheader["Data offset"])
        # npts = int(header["Ciao scan list"][0]["Sample Points"]) only at 2kHz?
        n_ext, n_ret = tuple(map(int, subheader["Samps/line"].split()))
        assert n_ext == n_ret
        npts = n_ext + n_ret
        bpp = length // (r * c * npts)
        d_ints = np.ndarray(
            shape=(r * c, 2, n_ext), dtype=f"<i{bpp}", buffer=data, offset=offset
        )

        value = subheader["@4:Z scale"]
        soft_scale = float(header["Ciao scan list"][0]["@Sens. DeflSens"].split()[1])
        hard_scale = float(value[1 + value.find("(") : value.find(")")].split()[0])
        d_scale = np.float32(soft_scale * hard_scale)
        name = subheader["@4:Image Data"].split('"')[1]
        sync_int = int(sync_dist)
        sync_frac = sync_dist - sync_int
        sync_dist = (sync_int - n_ext) % npts  # according to Bruker via Bede
        sync_dist_slice = np.s_[sync_dist : sync_dist + npts]
        # e^(-i*2*pi*omega[n]*phase_shift_in_seconds),
        # time cancels out if sync dist has units of sampling rate
        phasors = np.exp(-2j * np.pi * np.fft.rfftfreq(npts) * sync_frac)
        return cls(
            data, name, d_ints, shape, d_scale, sync_dist_slice, phasors, sync_frac
        )

    def get_curve(self, r, c):
        i = np.ravel_multi_index((r, c), self.shape)
        if i:
            i -= 1  # due to applying sync_dist, the first pixel is OOB
        with memoryview(self.data):  # assert data is open, and hold it open
            d = self.ints[i : i + 2] * self.scale
        # flip extend segments
        d[:, 0, :] = d[:, 0, ::-1]
        # remove extend segment marker
        if d[0, 0, 0] == -32768 * self.scale:
            d[:, 0, 0] = d[:, 0, 1]
        d = d.reshape(-1)
        # "roll" across 2 indents (usually mostly consisting of the 2nd indent)
        d = d[self.sync_dist_slice]
        if self.sync_frac:
            d_fft = np.fft.rfft(d)
            d_fft *= self.phasors
            d = np.fft.irfft(d_fft)
        return d.reshape((2, -1))


@frozen
class QNMZReader:
    name: str
    z_basis: np.ndarray = field(repr=False)
    height_for_z: np.ndarray = field(repr=False)

    @classmethod
    def parse(cls, header, d_reader: QNMDReader, height_for_z):
        # QNM doesn't have its own height data, so it must be synthesized
        # I've chosen to do this by combining a height image with a sine wave

        amp = np.float32(header["Ciao scan list"][0]["Peak Force Amplitude"])
        npts = d_reader.ints[0].size
        # phase = d_reader.sync_dist / npts * 2 * np.pi
        phase = np.pi
        # TODO: apply sync dist here rather than d?
        z_basis = amp * np.cos(
            np.linspace(
                phase, phase + 2 * np.pi, npts, endpoint=False, dtype=np.float32
            )
        )
        return cls(d_reader.name + " z_basis", z_basis, height_for_z)

    def get_curve(self, r, c):
        # need to infer z from amp/height
        z = self.z_basis + self.height_for_z[r, c]
        return z.reshape((2, -1))


@frozen
class NanoscopeVolume:
    name: str
    shape: Index
    step_info: StepInfo
    _zreader: FFVReader | QNMZReader
    _dreader: FFVReader | QNMDReader

    @classmethod
    def parse(cls, header, z_reader, d_reader):
        npts = int(header["Ciao force list"][0]["force/line"].split()[0])
        rate, unit = header["Ciao scan list"][0]["PFT Freq"].split()
        assert unit.lower() == "khz"
        rate = float(rate) * 1000
        t_step = 1 / rate / npts

        scansize, units = header["Ciao scan list"][0]["Scan Size"].split()
        if units == "nm":
            factor = 1.0
        elif units == "pm":
            factor = 0.001
        elif units == "~m":  # microns?!
            factor = 1000.0
        else:
            raise ValueError("unknown units:", units)
        scansize = float(scansize)

        # TODO: tuple(map(float,header[""Ciao scan list""]["Aspect Ratio"].split(":")))
        fastpx = int(header["Ciao scan list"][0]["Samps/line"])
        slowpx = int(header["Ciao scan list"][0]["Lines"])
        shape = (slowpx, fastpx)
        # TODO: nonsquare images
        x_step = scansize * factor / NANOMETER_UNIT_CONVERSION / (fastpx - 1)
        y_step = scansize * factor / NANOMETER_UNIT_CONVERSION / (slowpx - 1)

        # self.scandown = {"Down": True, "Up": False}[
        #     header["FV"]["Deflection Error"]["Frame direction"]
        # ]
        return cls(
            d_reader.name,
            shape,
            ((x_step, "m"), (y_step, "m"), (t_step, "s")),
            z_reader,
            d_reader,
        )

    def get_curve(self, r: int, c: int) -> ZDArrays:
        """Efficiently get a specific curve from disk."""
        return self._zreader.get_curve(r, c), self._dreader.get_curve(r, c)

    def iter_indices(self) -> Iterable[Index]:
        """Iterate over force curve indices in on-disk order"""
        # TODO: ensure on-disk order
        for index in np.ndindex(self.shape):
            yield index

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        for index in self.iter_indices():
            yield index, self.get_curve(*index)

    def get_all_curves(self) -> ZDArrays:
        """Eagerly load all curves into memory."""
        x = np.empty(
            self.shape + (2, 2, self._dreader.ints.shape[-1] // 2), dtype=np.float32
        )
        for index, curve in self.iter_curves():
            x[*index, ...] = curve
        return x


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

    @classmethod
    def parse(cls, data: mmap):
        # Check for magic string indicating nanoscope
        magic = b"\\*Force file list\r\n"
        start = data[: len(magic)]
        if not start == magic:
            raise ValueError("Not a nanoscope file.", start)
        # End of header is demarcated by a SUB byte (26 = 0x1A)
        # Longest header so far was 80 kB,
        # stop there to avoid searching gigabytes before fail
        header_end_pos = data.find(b"\x1A", 0, 80960)
        if header_end_pos < 0:
            raise ValueError(
                "No stop byte found, are you sure this is a Nanoscope file?"
            )
        header = parse_nanoscope_header(
            data[:header_end_pos].decode("windows-1252").splitlines()
        )

        # Header items that I should be reading someday
        # \Frame direction: Down
        # \Line Direction: Retrace
        # \Z direction: Retract

        # and in the image lists
        # \Aspect Ratio: 1:1
        # \Scan Size: 800 800 nm
        csl = header["Ciao scan list"][0]
        fastpx = int(csl["Samps/line"])
        slowpx = int(csl["Lines"])
        shape = slowpx, fastpx

        images = {}
        for entry in header["Ciao image list"]:
            name = entry["@2:Image Data"].split('"')[1]
            images[name] = NanoscopeImage.parse_image(data, header, entry, name)
        # deflection and height data are separate "force images"
        # we've got to pair them up
        heights = []
        deflections = []
        for subheader in header["Ciao force image list"]:
            name = subheader["@4:Image Data"].split('"')[1]
            if "Height" in name:
                heights.append(subheader)
            else:
                deflections.append(subheader)
        if not heights:
            # QNM data doesn't contain height data which we synthesize from an image
            try:
                height_for_z = images["Height Sensor"].get_image()
            except KeyError:
                height_for_z = images["Height"].get_image()

            sync_dist = round(
                float(
                    csl["Sync Distance QNM"]
                    if "Sync Distance QNM" in csl
                    else csl["Sync Distance"]
                )
                # actually sum(Samps/line) / Sample Points, but it works
                * float(csl["PFT Freq"].split()[0]) / 2,
                2,
            )
            volumes = [
                NanoscopeVolume.parse(
                    header, QNMZReader.parse(header, d_reader, height_for_z), d_reader
                )
                for d_reader in (
                    QNMDReader.parse(header, data, d_subh, shape, sync_dist)
                    for d_subh in deflections
                )
            ]
        else:
            sync_dist = None
            volumes = [
                NanoscopeVolume.parse(
                    header,
                    FFVReader.parse(header, data, z_subh),
                    FFVReader.parse(header, data, d_subh),
                )
                for z_subh, d_subh in zip(heights, deflections)
            ]
        # volumes = {v.name:v for v in volumes}
        k = float(header["Ciao force image list"][0]["Spring Constant"])
        defl_sens = float(csl["@Sens. DeflSens"].split()[1])
        npts = int(header["Ciao force list"][0]["force/line"].split()[0])
        rate, unit = csl["PFT Freq"].split()
        assert unit.lower() == "khz"
        rate = float(rate) * 1000
        t_step = 1 / rate / npts

        scansize, units = csl["Scan Size"].split()
        if units == "nm":
            factor = 1.0
        elif units == "pm":
            factor = 0.001
        elif units == "~m":  # microns?!
            factor = 1000.0
        else:
            raise ValueError("unknown units:", units)

        # TODO: tuple(map(float,header[""Ciao scan list""]["Aspect Ratio"].split(":")))
        ratio = float(scansize) * factor / max(fastpx, slowpx)
        scansize = (fastpx * ratio, slowpx * ratio)
        return cls(header, images, volumes, k, defl_sens, t_step, scansize, sync_dist)


SUFFIX_FVFILE_MAP = {
    ".ardf": (ARDFFile, mmap_path_read_only),
    ".h5": (ARH5File, open_h5),
    ".spm": (NanoscopeFile, mmap_path_read_only),
    ".pfc": (NanoscopeFile, mmap_path_read_only),
}
