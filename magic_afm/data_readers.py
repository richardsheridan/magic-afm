"""Magic AFM Data Readers

This module abstracts the different file types that this package supports.
Generally, they have the structure of a BaseForceVolumeFile subclass that has-a
worker class. The File class opens file objects and parses metadata, while
the worker class does the actual reads from the disk. Generally, the File class
asyncifies the worker's disk reads with threads, although this is not a rule.
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
import abc
import struct
import threading
import time
from collections.abc import Collection, Iterable
from functools import partial
from subprocess import PIPE
from typing import Protocol, TypeAlias

try:
    from subprocess import STARTF_USESHOWWINDOW, STARTUPINFO
except ImportError:
    STARTUPINFO = lambda *a, **kw: None
    STARTF_USESHOWWINDOW = None

import attrs
import numpy as np
import trio

from . import calculation
from .async_tools import make_cancel_poller, trs

CACHED_OPEN_PATHS = {}


def eventually_evict_path(path):
    path_lock = CACHED_OPEN_PATHS[path][-1]
    while path_lock.acquire(timeout=10.0):
        # someone reset our countdown
        pass
    # time's up, kick it out
    del CACHED_OPEN_PATHS[path]
    return


def mmap_path_read_only(path):
    import mmap

    with open(path, mode="rb", buffering=0) as file:
        return mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)


async def convert_ardf(
    ardf_path, *, h5file_path=None, conv_path="ARDFtoHDF5.exe", pbar=None
):
    """Turn an ARDF into a corresponding ARH5, returning the path.

    Requires converter executable available from Asylum Research"""
    ardf_path = trio.Path(ardf_path)
    if h5file_path is None:
        h5file_path = ardf_path.with_suffix(".h5")
    else:
        h5file_path = trio.Path(h5file_path)

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
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Please acquire ARDFtoHDF5.exe and "
            "place it in the application's root folder."
        ) from None
    except:
        with trio.CancelScope(shield=True):
            await h5file_path.unlink(missing_ok=True)
        raise

    return h5file_path


@attrs.frozen
class ForceVolumeParams:
    image_names: set[str]
    k: float
    defl_sens: float
    npts: int
    split: int
    sync_dist: int


class BaseForceVolumeFile(metaclass=abc.ABCMeta):
    """Consistent interface across filetypes for Magic AFM GUI

    I would not recommend re-using this or its subclasses for an external application.
    Prefer wrapping a worker class."""

    _basic_units_map = {}
    _default_heightmap_names = ()

    def __init__(self, path):
        self.scansize = None
        self.path = path
        self._units_map = self._basic_units_map.copy()
        self._calc_images = {}
        self._file_image_names = set()
        self._trace = None
        self._worker = None
        self.k = None
        self.defl_sens = None
        self.npts = None
        # self.scandown = True
        self.split = None
        self.rate = None
        self.sync_dist = 0

    @property
    def trace(self):
        return self._trace

    @property
    def image_names(self):
        return self._calc_images.keys() | self._file_image_names

    @property
    def initial_image_name(self):
        for name in self._default_heightmap_names:
            if name in self._file_image_names:
                return name
        else:
            return None

    @property
    def parameters(self):
        return ForceVolumeParams(
            k=self.k,
            defl_sens=self.defl_sens,
            image_names=self._file_image_names,
            npts=self.npts,
            split=self.split,
            sync_dist=self.sync_dist,
        )

    @abc.abstractmethod
    async def ainitialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    async def aclose(self):
        raise NotImplementedError

    async def __aenter__(self):
        await self.ainitialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def get_image_units(self, image_name):
        image_name = self.strip_trace(image_name)
        return self._units_map.get(image_name, "V")

    def add_image(self, image_name, units, image):
        self._calc_images[image_name] = image
        image_name = self.strip_trace(image_name)
        self._units_map[image_name] = units

    @staticmethod
    def strip_trace(image_name):
        for suffix in ("trace", "Trace", "retrace", "Retrace"):
            image_name = image_name.removesuffix(suffix)
        return image_name

    async def get_force_curve(self, r, c):
        return await trs(self.get_force_curve_sync, r, c)

    @abc.abstractmethod
    def get_force_curve_sync(self, r, c):
        raise NotImplementedError

    @abc.abstractmethod
    async def get_image(self, image_name):
        raise NotImplementedError


class DemoForceVolumeFile(BaseForceVolumeFile):
    def __init__(self, path):
        path = trio.Path(path)
        super().__init__(path)
        self._file_image_names.add("Demo")
        self._default_heightmap_names = ("Demo",)
        self.scansize = 100, 100
        self.k = 10
        self.defl_sens = 5
        self.rate = 100

    async def ainitialize(self):
        await trio.sleep(0)
        self.npts = 1024
        self.split = self.npts // 2
        self.delta = -15 * (
            np.cos(np.linspace(0, np.pi * 2, self.npts, endpoint=False)) + 0.5
        )

    async def aclose(self):
        pass

    def get_force_curve_sync(self, r, c):
        gen = np.random.default_rng(seed=(r, c))
        parms = (1, 10, 0.1, -2, 1, 0, 0, 1)
        fext = calculation.force_curve(
            calculation.red_extend, self.delta[: self.split], *parms
        )
        fret = calculation.force_curve(
            calculation.red_retract, self.delta[self.split :], *parms
        )
        d = np.concatenate((fext, fret)) / self.k
        z = self.delta + d
        d += gen.normal(scale=0.1, size=d.size)
        z += gen.normal(scale=0.01, size=z.size)
        return z, d

    async def get_image(self, image_name):
        await trio.sleep(0)
        try:
            image = self._calc_images[image_name]
        except KeyError:
            image = np.zeros((64, 64), dtype=np.float32)
        return image


NANOMETER_UNIT_CONVERSION = (
    1e9  # maybe we can intelligently read this from the file someday
)


class ForceMapWorker:
    def __init__(self, h5data):
        self.force_curves = h5data["ForceMap"]["0"]
        # ForceMap Segments can contain 3 or 4 endpoint indices for each indent array
        self.segments = self.force_curves["Segments"][:, :, :]  # XXX Read h5data
        im_r, im_c, num_segments = self.segments.shape

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

    def get_force_curve(self, r, c):
        # Because of the nonuniform arrays, each indent gets its own dataset
        # indexed by 'row:column' e.g. '1:1'.
        curve = self.force_curves[f"{r}:{c}"]  # XXX Read h5data
        split = self.extlens[r, c]

        return self._shared_get_part(curve, split)

    def get_all_curves(self, _poll_for_cancel=(lambda: None)):
        im_r, im_c, num_segments = self.segments.shape
        x = np.empty((im_r, im_c, 2, self.minext + self.minret), dtype=np.float32)
        for index, curve in self.force_curves.items():
            # Unfortunately they threw in segments here too, so we skip over it
            if index == "Segments":
                continue
            _poll_for_cancel()
            # Because of the nonuniform arrays, each indent gets its own dataset
            # indexed by 'row:column' e.g. '1:1'. We could start with the shape and index
            # manually, but the string munging is easier for me to think about
            r, c = index.split(":")
            r, c = int(r), int(c)
            split = self.extlens[r, c]

            x[r, c, :, :] = self._shared_get_part(curve, split)
        z = x[:, :, 0, :]
        d = x[:, :, 1, :]
        return z, d


class FFMSingleWorker:
    def __init__(self, raw, defl):
        self.raw = raw
        self.defl = defl
        self.npts = raw.shape[-1]
        self.split = self.npts // 2

    def get_force_curve(self, r, c):
        z = self.raw[r, c]
        d = self.defl[r, c]
        return z * NANOMETER_UNIT_CONVERSION, d * NANOMETER_UNIT_CONVERSION

    def get_all_curves(self, _poll_for_cancel=(lambda: None)):
        _poll_for_cancel()
        z = self.raw[:] * NANOMETER_UNIT_CONVERSION
        _poll_for_cancel()
        d = self.defl[:] * NANOMETER_UNIT_CONVERSION
        _poll_for_cancel()
        return z, d


class FFMTraceRetraceWorker:
    def __init__(self, raw_trace, defl_trace, raw_retrace, defl_retrace):
        self.raw_trace = raw_trace
        self.defl_trace = defl_trace
        self.raw_retrace = raw_retrace
        self.defl_retrace = defl_retrace
        self.trace = True
        self.npts = raw_trace.shape[-1]
        self.split = self.npts // 2

    def get_force_curve(self, r, c):
        if self.trace:
            z = self.raw_trace[r, c]
            d = self.defl_trace[r, c]
        else:
            z = self.raw_retrace[r, c]
            d = self.defl_retrace[r, c]
        return z * NANOMETER_UNIT_CONVERSION, d * NANOMETER_UNIT_CONVERSION

    def get_all_curves(self, _poll_for_cancel=(lambda: None)):
        _poll_for_cancel()
        if self.trace:
            z = self.raw_trace[:] * NANOMETER_UNIT_CONVERSION
            _poll_for_cancel()
            d = self.defl_trace[:] * NANOMETER_UNIT_CONVERSION
        else:
            z = self.raw_retrace[:] * NANOMETER_UNIT_CONVERSION
            _poll_for_cancel()
            d = self.defl_retrace[:] * NANOMETER_UNIT_CONVERSION
        _poll_for_cancel()
        return z, d


def open_h5(path):
    import h5py

    return h5py.File(path, "r")


class ARH5File(BaseForceVolumeFile):
    _basic_units_map = {
        "Adhesion": "N",
        "Height": "m",
        "IndentationHertz": "m",
        "YoungsHertz": "Pa",
        "YoungsJKR": "Pa",
        "YoungsDMT": "Pa",
        "ZSensor": "m",
        "MapAdhesion": "N",
        "MapHeight": "m",
        "Force": "N",
    }
    _default_heightmap_names = ("MapHeight", "ZSensorTrace", "ZSensorRetrace")

    @BaseForceVolumeFile.trace.setter
    def trace(self, trace):
        self._worker.trace = trace
        self._trace = trace

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_h5data"]
        del state["_worker"]
        del state["_images"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            h5data, images, worker, path_lock = CACHED_OPEN_PATHS[self.path]
        except KeyError:
            h5data = open_h5(self.path)
            images = h5data["Image"]
            worker = self._choose_worker(h5data)
            path_lock = threading.Lock()
            path_lock.acquire()
            CACHED_OPEN_PATHS[self.path] = h5data, images, worker, path_lock
            threading.Thread(
                target=eventually_evict_path, args=(self.path,), daemon=True
            ).start()
        else:
            # reset thread countdown
            try:
                path_lock.release()
            except RuntimeError:
                pass  # no problem if unreleased
        self._h5data = h5data
        self._images = images
        self._worker = worker

    async def ainitialize(self):
        self._h5data = h5data = await trio.to_thread.run_sync(open_h5, self.path)

        # The notes have a very regular key-value structure
        # convert to dict for later access
        def note_parser():
            return dict(
                line.split(":", 1)
                for line in h5data.attrs["Note"].split("\n")
                if ":" in line and "@Line:" not in line
            )

        self.notes = await trs(note_parser)
        worker = await trs(self._choose_worker, h5data)
        images, image_names = await trs(
            lambda: (h5data["Image"], set(h5data["Image"].keys()))
        )
        self._worker = worker
        self._images = images
        self._file_image_names.update(image_names)

        self.k = float(self.notes["SpringConstant"])
        self.scansize = (
            float(self.notes["FastScanSize"]) * NANOMETER_UNIT_CONVERSION,
            float(self.notes["SlowScanSize"]) * NANOMETER_UNIT_CONVERSION,
        )
        # NOTE: aspect is redundant to scansize
        # self.aspect = float(self.notes["SlowRatio"]) / float(self.notes["FastRatio"])
        self.defl_sens = self._defl_sens_orig = (
            float(self.notes["InvOLS"]) * NANOMETER_UNIT_CONVERSION
        )
        self.rate = float(self.notes["FastMapZRate"])
        self.npts, self.split = worker.npts, worker.split

    async def aclose(self):
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(self._h5data.close)

    def _choose_worker(self, h5data):
        if "FFM" in h5data:
            # self.scandown = bool(self.notes["ScanDown"])
            if "1" in h5data["FFM"]:
                worker = FFMTraceRetraceWorker(
                    h5data["FFM"]["0"]["Raw"],
                    h5data["FFM"]["0"]["Defl"],
                    h5data["FFM"]["1"]["Raw"],
                    h5data["FFM"]["1"]["Defl"],
                )
                self._trace = True
            elif "0" in h5data["FFM"]:
                worker = FFMSingleWorker(
                    h5data["FFM"]["0"]["Raw"], h5data["FFM"]["0"]["Defl"]
                )
            else:
                worker = FFMSingleWorker(h5data["FFM"]["Raw"], h5data["FFM"]["Defl"])
        else:
            # self.scandown = bool(self.notes["FMapScanDown"])
            worker = ForceMapWorker(h5data)
        return worker

    def get_force_curve_sync(self, r, c):
        z, d = self._worker.get_force_curve(r, c)
        if self.defl_sens != self._defl_sens_orig:
            d *= self.defl_sens / self._defl_sens_orig
        return z, d

    async def get_all_curves(self):
        z, d = await trs(self._worker.get_all_curves, make_cancel_poller())
        if self.defl_sens != self._defl_sens_orig:
            d *= self.defl_sens / self._defl_sens_orig
        return z, d

    async def get_image(self, image_name):
        if image_name in self._calc_images:
            await trio.sleep(0)
            image = self._calc_images[image_name]
        else:
            image = await trs(self._sync_get_image, image_name)
        return image

    def _sync_get_image(self, image_name):
        return self._images[image_name][:]


class BrukerWorkerBase(metaclass=abc.ABCMeta):
    def __init__(self, header, mm, s):
        self.header = header  # get_image
        self.mm = mm  # get_image
        self.split = s  # get_force_curve
        self.version = header["Force file list"]["Version"].strip()  # get_image

    @abc.abstractmethod
    def get_force_curve(self, r, c, defl_sens, sync_dist):
        raise NotImplementedError

    def get_image(self, image_name):
        h = self.header["Image"][image_name]

        value = h["@2:Z scale"]
        bpp = bruker_bpp_fix(h["Bytes/pixel"], self.version)
        hard_scale = float(value.split()[-2]) / (2 ** (bpp * 8))
        hard_offset = float(h["@2:Z offset"].split()[-2]) / (2 ** (bpp * 8))
        soft_scale_name = "@" + value[1 + value.find("[") : value.find("]")]
        try:
            soft_scale_string = self.header["Ciao scan list"][soft_scale_name]
        except KeyError:
            soft_scale_string = self.header["Scanner list"][soft_scale_name]
        soft_scale = float(soft_scale_string.split()[1]) / NANOMETER_UNIT_CONVERSION
        scale = np.float32(hard_scale * soft_scale)
        data_length = int(h["Data length"])
        offset = int(h["Data offset"])
        r = int(h["Number of lines"])
        c = int(h["Samps/line"])
        assert data_length == r * c * bpp
        # scandown = h["Frame direction"] == "Down"
        z_ints = np.ndarray(
            shape=(r, c), dtype=f"i{bpp}", buffer=self.mm, offset=offset
        )
        z_floats = z_ints * scale + np.float32(hard_offset * soft_scale)
        return z_floats


class FFVWorker(BrukerWorkerBase):
    def __init__(self, header, mm, s):
        super().__init__(header, mm, s)
        arbitrary_image = next(iter(header["Image"].values()))
        r = int(arbitrary_image["Number of lines"])
        c = int(arbitrary_image["Samps/line"])
        data_name = header["Ciao force list"]["@4:Image Data"].split('"')[1]
        for name in [data_name, "Height Sensor"]:
            subheader = header["FV"][name]
            offset = int(subheader["Data offset"])
            bpp = bruker_bpp_fix(subheader["Bytes/pixel"], self.version)
            length = int(subheader["Data length"])
            npts = length // (r * c * bpp)
            data = np.ndarray(
                shape=(r, c, npts), dtype=f"i{bpp}", buffer=mm, offset=offset
            )
            if name == "Height Sensor":
                self.z_ints = data
                value = subheader["@4:Z scale"]
                soft_scale = float(
                    header["Ciao scan list"]["@Sens. ZsensSens"].split()[1]
                )
                hard_scale = float(
                    value[1 + value.find("(") : value.find(")")].split()[0]
                )
                self.z_scale = np.float32(soft_scale * hard_scale)
            else:
                self.d_ints = data
                value = subheader["@4:Z scale"]
                self.defl_hard_scale = float(
                    value[1 + value.find("(") : value.find(")")].split()[0]
                )

    def get_force_curve(self, r, c, defl_sens, sync_dist):
        s = self.split
        defl_scale = np.float32(defl_sens * self.defl_hard_scale)

        d = self.d_ints[r, c] * defl_scale
        d[:s] = d[s - 1 :: -1]

        z = self.z_ints[r, c] * self.z_scale
        z[:s] = z[s - 1 :: -1]

        return z, np.roll(d, -sync_dist) if sync_dist else d


class QNMWorker(BrukerWorkerBase):
    def __init__(self, header, mm, s):
        super().__init__(header, mm, s)
        arbitrary_image = next(iter(header["Image"].values()))
        r = int(arbitrary_image["Number of lines"])
        c = int(arbitrary_image["Samps/line"])
        data_name = header["Ciao force list"]["@4:Image Data"].split('"')[1]
        subheader = header["FV"][data_name]
        bpp = bruker_bpp_fix(subheader["Bytes/pixel"], self.version)
        length = int(subheader["Data length"])
        offset = int(subheader["Data offset"])
        npts = length // (r * c * bpp)

        self.d_ints = np.ndarray(
            shape=(r, c, npts), dtype=f"i{bpp}", buffer=mm, offset=offset
        )
        value = subheader["@4:Z scale"]
        self.defl_hard_scale = float(
            value[1 + value.find("(") : value.find(")")].split()[0]
        )

        try:
            image = self.get_image("Height Sensor")
        except KeyError:
            image = self.get_image("Height")
        image *= NANOMETER_UNIT_CONVERSION
        self.height_for_z = image
        amp = np.float32(header["Ciao scan list"]["Peak Force Amplitude"])
        phase = s / npts * 2 * np.pi
        self.z_basis = amp * np.cos(
            np.linspace(
                phase, phase + 2 * np.pi, npts, endpoint=False, dtype=np.float32
            )
        )

    def get_force_curve(self, r, c, defl_sens, sync_dist):
        s = self.split
        defl_scale = np.float32(defl_sens * self.defl_hard_scale)

        d = self.d_ints[r, c] * defl_scale
        d[:s] = d[s - 1 :: -1]
        # remove blip
        if d[0] == -32768 * defl_scale:
            d[0] = d[1]
        d = np.roll(d, s - sync_dist)  # TODO roll across two adjacent indents

        # need to infer z from amp/height
        z = self.z_basis + self.height_for_z[r, c]

        return z, d


class NanoscopeFile(BaseForceVolumeFile):
    _basic_units_map = {
        "Height Sensor": "m",
        "Height": "m",
    }
    _default_heightmap_names = ("Height Sensor", "Height")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_mm"]
        del state["_worker"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            mm, worker, path_lock = CACHED_OPEN_PATHS[self.path]
        except KeyError:
            self._mm = mm = mmap_path_read_only(self.path)
            worker, _ = self._choose_worker(self.header)
            path_lock = threading.Lock()
            path_lock.acquire()
            CACHED_OPEN_PATHS[self.path] = mm, worker, path_lock
            threading.Thread(
                target=eventually_evict_path, args=(self.path,), daemon=True
            ).start()
        else:
            # reset thread countdown
            try:
                path_lock.release()
            except RuntimeError:
                pass  # no problem if unreleased
        self._mm = mm
        self._worker = worker

    async def ainitialize(self):
        self._mm = await trio.to_thread.run_sync(mmap_path_read_only, self.path)

        # End of header is demarcated by a SUB byte (26 = 0x1A)
        # Longest header so far was 80 kB, stop there to avoid searching gigabytes before fail
        header_end_pos = await trs(self._mm.find, b"\x1A", 0, 80960)
        if header_end_pos < 0:
            raise ValueError(
                "No stop byte found, are you sure this is a Nanoscope file?"
            )
        self.header = header = parse_nanoscope_header(
            self._mm[:header_end_pos]  # will be cached from find call
            .decode("windows-1252")
            .splitlines()
        )

        # Header items that I should be reading someday
        # \Frame direction: Down
        # \Line Direction: Retrace
        # \Z direction: Retract

        # and in the image lists
        # \Aspect Ratio: 1:1
        # \Scan Size: 800 800 nm

        self._file_image_names.update(header["Image"].keys())

        data_name = header["Ciao force list"]["@4:Image Data"].split('"')[1]

        data_header = header["FV"][data_name]
        self.split, *_ = map(int, data_header["Samps/line"].split())
        self.npts = int(header["Ciao force list"]["force/line"].split()[0])
        rate, unit = header["Ciao scan list"]["PFT Freq"].split()
        assert unit.lower() == "khz"
        self.rate = float(rate) * 1000

        scansize, units = header["Ciao scan list"]["Scan Size"].split()
        if units == "nm":
            factor = 1.0
        elif units == "pm":
            factor = 0.001
        elif units == "~m":  # microns?!
            factor = 1000.0
        else:
            raise ValueError("unknown units:", units)

        # TODO: tuple(map(float,header[""Ciao scan list""]["Aspect Ratio"].split(":")))
        fastpx = int(header["Ciao scan list"]["Samps/line"])
        slowpx = int(header["Ciao scan list"]["Lines"])
        ratio = float(scansize) * factor / max(fastpx, slowpx)
        self.scansize = (fastpx * ratio, slowpx * ratio)

        # self.scandown = {"Down": True, "Up": False}[
        #     header["FV"]["Deflection Error"]["Frame direction"]
        # ]

        self.k = float(data_header["Spring Constant"])
        self.defl_sens = float(header["Ciao scan list"]["@Sens. DeflSens"].split()[1])
        value = data_header["@4:Z scale"]
        self.defl_hard_scale = float(
            value[1 + value.find("(") : value.find(")")].split()[0]
        )

        self._worker, self.sync_dist = await trs(self._choose_worker, header)

    def _choose_worker(self, header):
        if "Height Sensor" in header["FV"]:
            return FFVWorker(header, self._mm, self.split), 0
        else:
            return (
                QNMWorker(header, self._mm, self.split),
                int(
                    round(
                        float(
                            header["Ciao scan list"]["Sync Distance QNM"]
                            if "Sync Distance QNM" in header["Ciao scan list"]
                            else header["Ciao scan list"]["Sync Distance"]
                        )
                    )
                ),
            )

    async def aclose(self):
        self._worker = None
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(self._mm.close)

    def get_force_curve_sync(self, r, c):
        return self._worker.get_force_curve(r, c, self.defl_sens, self.sync_dist)

    async def get_image(self, image_name):
        if image_name in self._calc_images:
            await trio.sleep(0)
            image = self._calc_images[image_name]
        else:
            image = await trs(self._worker.get_image, image_name)
        return image


# noinspection PyUnboundLocalVariable
def parse_nanoscope_header(header_lines):
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
            if section_name == "File list end":
                break  # THIS IS THE **NORMAL** WAY TO END THE FOR LOOP
            if section_name in header:
                # repeat section name, which we interpret as a list of sections
                if header[section_name] is current_section:
                    header[section_name] = [current_section]
                current_section = {}
                header[section_name].append(current_section)
            else:
                current_section = {}
                header[section_name] = current_section
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

    # link headers from [section][index][key] to [Image/FV][Image Data name][key]
    header["Image"] = {}
    for entry in header["Ciao image list"]:
        if type(entry) is str:
            # a single image in this file, rather than a list of images
            name = header["Ciao image list"]["@2:Image Data"].split('"')[1]
            header["Image"][name] = header["Ciao image list"]
            break
        name = entry["@2:Image Data"].split('"')[1]
        # assert name not in header["Image"]  # assume data is redundant for now
        header["Image"][name] = entry
    header["FV"] = {}
    for entry in header["Ciao force image list"]:
        if type(entry) is str:
            # a single force image in this file, rather than a list of images
            name = header["Ciao force image list"]["@4:Image Data"].split('"')[1]
            header["FV"][name] = header["Ciao force image list"]
            break
        name = entry["@4:Image Data"].split('"')[1]
        # assert name not in header["FV"]  # assume data is redundant for now
        header["FV"][name] = entry

    return header


def bruker_bpp_fix(bpp, version):
    if (
        version > "0x09200000"
    ):  # Counting on lexical ordering here, hope zeros don't change...
        return 4
    else:
        return int(bpp)


def decode_cstring(cstring: bytes):
    return cstring.rstrip(b"\0").decode("windows-1252")


def parse_AR_note(note: str):
    # The notes have a very regular key-value structure
    # convert to dict for later access
    return dict(
        line.split(":", 1)
        for line in note.split("\n")
        if ":" in line and "@Line:" not in line
    )


@attrs.frozen
class ARDFHeader:
    data: memoryview
    offset: int
    crc: int = attrs.field(repr=hex)
    size: int
    name: bytes
    flags: int = attrs.field(repr=hex)

    @classmethod
    def unpack(cls, data: memoryview, offset: int):
        return cls(data, offset, *struct.unpack_from("<LL4sL", data, offset))

    def validate(self):
        # ImHex poly 0x4c11db7 init 0xffffffff xor out 0xffffffff reflect in and out
        import zlib

        crc = zlib.crc32(self.data[self.offset + 4 : self.offset + self.size])
        if self.crc != crc:
            raise ValueError(
                f"Invalid section. Expected {self.crc:X}, got {crc:X}.", self
            )
        return True


@attrs.frozen
class ARDFTableOfContents:
    data: memoryview
    offset: int
    size: int
    entries: list[tuple[ARDFHeader, int]]

    @classmethod
    def unpack(cls, data: memoryview, offset: int):
        size, nentries, stride = struct.unpack_from("<QLL", data, offset + 16)
        assert stride == 24
        assert size - 32 == nentries * stride, (size, nentries, stride)
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, pointer = struct.unpack_from("<LL4sLQ", data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = ARDFHeader(data, toc_offset, *header)
            if entry_header.name not in {b"IMAG", b"VOLM", b"NEXT", b"THMB", b"NSET"}:
                raise ValueError("Malformed table of contents.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)


@attrs.frozen
class ARDFTextTableOfContents:
    data: memoryview = attrs.field()
    offset: int
    size: int
    entries: list[tuple[ARDFHeader, int]]

    @classmethod
    def unpack(cls, data: memoryview, offset: int):
        size, nentries, stride = struct.unpack_from("<QLL", data, offset + 16)
        assert stride == 32, stride
        assert size - 32 == nentries * stride, (size, nentries, stride)
        entries = []
        for toc_offset in range(offset + 32, offset + size, stride):
            *header, _, pointer = struct.unpack_from("<LL4sLQQ", data, toc_offset)
            if not pointer:
                break  # rest is null padding
            entry_header = ARDFHeader(data, toc_offset, *header)
            if entry_header.name != b"TOFF":
                raise ValueError("Malformed text table entry.", entry_header)
            entry_header.validate()
            entries.append((entry_header, pointer))
        return cls(data, offset, size, entries)

    def decode_entry(self, index: int):
        entry_header, pointer = self.entries[index]
        *header, i, text_len = struct.unpack_from("<LL4sLLL", self.data, pointer)
        text_header = ARDFHeader(self.data, pointer, *header)
        if text_header.name != b"TEXT":
            raise ValueError("Malformed text section.", text_header)
        text_header.validate()
        assert i == index, (i, index)
        offset = text_header.offset + 24
        assert text_len < text_header.size - 24, (text_len, text_header)
        return (
            self.data[offset : offset + text_len]
            .tobytes()
            .replace(b"\r", b"\n")
            .decode("windows-1252")
        )


@attrs.frozen
class ARDFVset:
    data: memoryview
    offset: int
    size: int
    force_index: int
    line: int
    point: int
    vtype: int
    prev_vset_offset: int
    next_vset_offset: int

    @classmethod
    def unpack(cls, data: memoryview, offset: int):
        vset_format = "<LLLLQQ"
        size = 48  # 16 + struct.calcsize(vset_format)  # constant
        return cls(
            data, offset, size, *struct.unpack_from(vset_format, data, offset + 16)
        )


@attrs.frozen
class ARDFVdata:
    data: memoryview
    offset: int
    size: int
    force_index: int
    line: int
    point: int
    nfloats: int
    channel: int
    seg_offsets: tuple[int, ...]

    @classmethod
    def unpack(cls, header: ARDFHeader):
        data = header.data
        offset = header.offset
        vdat_format = "<10L"
        size = 56  # struct.calcsize(vdat_format) + 16  # constant
        (force_index, line, point, nfloats, channel, *seg_offsets) = struct.unpack_from(
            vdat_format, data, offset + 16
        )
        return cls(
            data, offset, size, force_index, line, point, nfloats, channel, seg_offsets
        )

    @property
    def array_offset(self):
        return self.offset + self.size

    @property
    def next_offset(self):
        return self.array_offset + self.nfloats * 4

    def get_ndarray(self):
        return (
            np.ndarray(
                shape=self.nfloats,
                dtype=np.float32,
                buffer=self.data,
                offset=self.array_offset,
            )
            * NANOMETER_UNIT_CONVERSION
        )


@attrs.frozen
class ARDFImage:
    data: memoryview
    ibox_offset: int
    name: str
    units: str
    x_step: float
    y_step: float
    x_units: str
    y_units: str

    @classmethod
    def parse_imag(cls, imag_header: ARDFHeader):
        if imag_header.size != 32 or imag_header.name != b"IMAG":
            raise ValueError("Malformed image header.", imag_header)
        imag_header.validate()
        imag_toc = ARDFTableOfContents.unpack(imag_header.data, imag_header.offset)

        # don't use NEXT or THMB data, step over
        ttoc_header = ARDFHeader.unpack(imag_toc.data, imag_toc.offset + imag_toc.size)
        if ttoc_header.size != 32 or ttoc_header.name != b"TTOC":
            raise ValueError("Malformed image text table of contents.", ttoc_header)
        ttoc_header.validate()
        ttoc = ARDFTextTableOfContents.unpack(ttoc_header.data, ttoc_header.offset)

        # don't use TTOC or TOFF, step over
        idef_header = ARDFHeader.unpack(ttoc.data, ttoc.offset + ttoc.size)
        idef_format = "<LLQQdd32s32s32s32s"
        if (
            idef_header.name != b"IDEF"
            or idef_header.size != struct.calcsize(idef_format) + 16
        ):
            raise ValueError("Malformed image definition.", idef_header)
        idef_header.validate()
        points, lines, _, __, x_step, y_step, *cstrings = struct.unpack_from(
            idef_format,
            idef_header.data,
            idef_header.offset + 16,
        )
        assert not (_ or __), (_, __)
        x_units, y_units, name, units = list(map(decode_cstring, cstrings))

        return cls(
            imag_header.data,
            idef_header.offset + idef_header.size,
            name,
            units,
            x_step,
            y_step,
            x_units,
            y_units,
        )

    def get_ndarray(self):
        ibox_header = ARDFHeader.unpack(self.data, self.ibox_offset)
        if ibox_header.size != 32 or ibox_header.name != b"IBOX":
            raise ValueError("Malformed image layout.", ibox_header)
        ibox_header.validate()
        data_offset = ibox_header.offset + ibox_header.size + 16  # past IDAT header
        ibox_size, lines, stride = struct.unpack_from(
            "<QLL", ibox_header.data, ibox_header.offset + 16
        )  # TODO: invert lines? it's just a negative sign on stride
        points = (stride - 16) // 4  # less IDAT header
        # elide image data validation and map into an array directly
        arr = np.ndarray(
            shape=(lines, points),
            dtype=np.float32,
            buffer=self.data,
            offset=data_offset,
            strides=(stride, 4),
        ).copy()
        gami_header = ARDFHeader.unpack(
            ibox_header.data, ibox_header.offset + ibox_size
        )
        if gami_header.size != 16 or gami_header.name != b"GAMI":
            raise ValueError("Malformed image layout.", gami_header)
        gami_header.validate()
        return arr


Index: TypeAlias = tuple[int, ...]
ZDArrays: TypeAlias = Collection[np.ndarray]
ChanMap: TypeAlias = dict[str, tuple[int, str]]


class FVReader(Protocol):
    def get_curve(self, r: int, c: int) -> ZDArrays:
        """Efficiently get a specific curve from disk."""
        ...

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        ...

    def get_all_curves(self) -> ZDArrays:
        """Eagerly load all curves into memory."""
        ...


@attrs.frozen
class ARDFFFMReader:
    array_view: np.ndarray
    array_offset: int  # hard to recover from views
    channels: list[int]  # [z, d]
    # seg_offsets is weird. you'd think it would contain the starting index
    # for each segment. However, it always has a trailing value of 1-nfloats,
    # and nonexistent segments get a zero. For regular/FFM data, we'll just
    # assume that the second offset maps to our "split" concept.
    seg_offsets: tuple
    scandown: bool
    trace: bool

    @classmethod
    def parse(cls, first_vset_header: ARDFHeader, points: int, lines: int, channels):
        data = first_vset_header.data
        # just walk past these first headers to find our data_offset
        if first_vset_header.name != b"VSET":
            raise ValueError("Malformed volume set", first_vset_header)
        first_vset_header.validate()
        first_vset = ARDFVset.unpack(first_vset_header.data, first_vset_header.offset)
        vset_stride = first_vset.next_vset_offset - first_vset_header.offset

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
            array_view=np.ndarray(
                shape=(lines, points, len(channels), first_vdat.nfloats),
                dtype=np.float32,
                buffer=data,
                offset=first_vdat.array_offset,
                strides=(
                    vset_stride * points,
                    vset_stride,
                    first_vdat_header.size,
                    4,
                ),
            ),
            array_offset=first_vdat.array_offset,
            channels=[channels["Raw"][0], channels["Defl"][0]],
            seg_offsets=first_vdat.seg_offsets,
            scandown=first_vdat.line != 0,
            trace=first_vdat.point == 0,
        )

    def get_curve(self, r, c):
        """Efficiently get a specific curve from disk."""
        return (
            self.array_view[
                r,
                c,
                self.channels,
            ]
            * NANOMETER_UNIT_CONVERSION
        )

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        # For now assume our strides don't bounce around
        # TODO: cleverly use np.nditer?
        for point in np.ndindex(self.array_view.shape[:2]):
            yield point, self.get_curve(*point)

    def get_all_curves(self) -> ZDArrays:
        """Eagerly load all curves into memory."""
        # advanced indexing triggers a copy
        loaded_data = self.array_view[:, :, self.channels, :]
        # avoid a second copy with inplace op
        loaded_data *= NANOMETER_UNIT_CONVERSION
        # a view with shuffled indices is possible, but
        # most other classes return tuples anyway
        return loaded_data[:, :, 0, :], loaded_data[:, :, 1, :]


@attrs.frozen(eq=False)
class ARDFForceMapReader:
    data: memoryview
    vtoc_lines: np.ndarray
    vtoc_pointers: np.ndarray
    lines: int
    points: int
    channels: ChanMap
    seen_vsets: dict[Index, ARDFVset] = attrs.field(init=False, factory=dict)

    def traverse_vsets(self, pointer: int):
        while True:
            header = ARDFHeader.unpack(self.data, pointer)
            if header.name != b"VSET":
                break
            header.validate()
            vset = ARDFVset.unpack(self.data, pointer)
            index = (vset.line, vset.point)
            if index not in self.seen_vsets:
                self.seen_vsets[index] = vset
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
            # vdat_header.validate()  # TODO: check if meaningfully slow on bigger data
            yield ARDFVdata.unpack(vdat_header)
            pointer = vdat_header.offset + vdat_header.size

    def get_curve(self, r: int, c: int) -> ZDArrays:
        """Efficiently get a specific curve from disk."""
        if not (0 <= r < self.lines and 0 <= c < self.points):
            raise ValueError("Invalid index:", (self.lines, self.points), (r, c))

        if (r, c) not in self.seen_vsets:
            # bisect row pointer
            if self.vtoc_lines[0]:
                # probably reversed
                sl = np.s_[::-1]
            else:
                sl = np.s_[:]
            i = self.vtoc_lines[sl].searchsorted(r)
            # read entire line of the vtoc
            for vset in self.traverse_vsets(int(self.vtoc_pointers[sl][i])):
                if vset.line != r:
                    break
                # transverse_vsets implicitly fills in seen_vsets

        vset = self.seen_vsets[r, c]

        for vdat in self.traverse_vdats(vset.offset + vset.size):
            # zname = "Raw" if "Raw" in self.channels else "ZSnsr"
            # code elsewhere assumes split is halfway through
            if vdat.channel == self.channels["ZSnsr"][0]:
                z = vdat.get_ndarray()[: vdat.seg_offsets[1] * 2]
            elif vdat.channel == self.channels["Defl"][0]:
                d = vdat.get_ndarray()[: vdat.seg_offsets[1] * 2]
        return z, d

    def iter_curves(self) -> Iterable[tuple[Index, ZDArrays]]:
        """Iterate over curves lazily in on-disk order."""
        # zname = "Raw" if "Raw" in self.channels else "ZSnsr"
        for vset in self.traverse_vsets(int(self.vtoc_pointers[0])):
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                # code elsewhere assumes split is halfway through
                if vdat.channel == self.channels["ZSnsr"][0]:
                    z = vdat.get_ndarray()[: vdat.seg_offsets[1] * 2]
                elif vdat.channel == self.channels["Defl"][0]:
                    d = vdat.get_ndarray()[: vdat.seg_offsets[1] * 2]
            yield (vset.line, vset.point), (z, d)

    def get_all_curves(self) -> ZDArrays:
        """Eagerly load all curves into memory."""
        minext = 0xFFFFFFFF
        vdats = {}
        for vset in self.traverse_vsets(int(self.vtoc_pointers[0])):
            for vdat in self.traverse_vdats(vset.offset + vset.size):
                if vdat.channel == self.channels["ZSnsr"][0]:
                    zvdat = vdat
                elif vdat.channel == self.channels["Defl"][0]:
                    dvdat = vdat
            vdats[vset.line, vset.point] = zvdat, dvdat
            minext = min(minext, vdat.seg_offsets[1])
        del vset, vdat
        minfloats = 2 * minext
        x = np.empty((self.lines, self.points, 2, minfloats), dtype=np.float32)
        for (r, c), (zvdat, dvdat) in vdats.items():
            # code elsewhere assumes split is halfway through
            floats = zvdat.seg_offsets[1] * 2
            halfextra = (floats - minfloats) // 2  # even - even -> even
            sl = np.s_[halfextra : minfloats + halfextra]
            # TODO: verify turnaround point against iter_curves
            x[r, c, :, :] = zvdat.get_ndarray()[sl], dvdat.get_ndarray()[sl]
        z = x[:, :, 0, :]
        d = x[:, :, 1, :]
        return z, d


@attrs.frozen
class ARDFVolume:
    volm_offset: int
    reader: FVReader
    shape: tuple[int, int]
    x_step: float
    y_step: float
    t_step: float
    x_units: str
    y_units: str
    t_units: str
    xdef: list[str]

    @classmethod
    def parse_volm(cls, volm_header: ARDFHeader):
        data = volm_header.data
        # TODO: wrap every struct.unpack_from with a new attrs class with .unpack()
        if volm_header.size != 32 or volm_header.name != b"VOLM":
            raise ValueError("Malformed volume header.", volm_header)
        volm_header.validate()
        # NSET looks a lot like a table of contents, but the pointer is just nset
        volm_toc = ARDFTableOfContents.unpack(data, volm_header.offset)
        assert len(volm_toc.entries) == 1
        nset_header, nset = volm_toc.entries[0]
        nset_header.validate()
        ttoc_header = ARDFHeader.unpack(data, volm_toc.offset + volm_toc.size)
        if ttoc_header.size != 32 or ttoc_header.name != b"TTOC":
            raise ValueError("Malformed volume text table of contents.", ttoc_header)
        ttoc_header.validate()
        ttoc = ARDFTextTableOfContents.unpack(data, ttoc_header.offset)

        # don't use TTOC or TOFF, step over
        vdef_header = ARDFHeader.unpack(data, ttoc.offset + ttoc.size)
        vdef_format = "<LL24sddd32s32s32s32sQ"
        if (
            vdef_header.name != b"VDEF"
            or vdef_header.size != struct.calcsize(vdef_format) + 16
        ):
            raise ValueError(
                "Malformed volume definition.",
                vdef_header,
                struct.calcsize(vdef_format),
            )
        vdef_header.validate()
        points, lines, _, x_step, y_step, t_step, *cstrings, nseg = struct.unpack_from(
            vdef_format,
            vdef_header.data,
            vdef_header.offset + 16,
        )
        assert sum(_) == 0, _
        assert points * lines == nset, (points, lines, nset)
        x_units, y_units, t_units, seg_names = list(map(decode_cstring, cstrings))
        seg_names = seg_names.split(";")[:-1]
        assert nseg == len(seg_names)

        # Implicit table of channels here smh
        offset = vdef_header.offset + vdef_header.size
        channels: ChanMap = {}
        for i in range(5):
            header = ARDFHeader.unpack(data, offset)
            if header.name != b"VCHN":
                break
            if header.size != 80:
                raise ValueError("Unexpected channel definition size.", header)
            header.validate()
            name, unit = list(
                map(
                    decode_cstring,
                    struct.unpack_from("<32s32s", data, header.offset + 16),
                )
            )
            offset = header.offset + header.size
            # TODO: apply NANOMETER_UNIT_CONVERSION depending on this
            assert unit == "m"
            channels[name] = (i, "nm")
        else:
            raise RuntimeError("Got too many channels.", channels)

        if header.size != 96 or header.name != b"XDEF":
            raise ValueError("Malformed experiment definition.", header)
        header.validate()
        # Experiment definition is just a string
        _, nchars = struct.unpack_from("<LL", data, header.offset + 16)
        assert _ == 0, _
        if nchars > header.size:
            raise ValueError("Experiment definition too long.", header, nchars)
        xdef = (
            data[header.offset + 24 : header.offset + 24 + nchars]
            .tobytes()
            .decode("windows-1252")
            .split(";")[:-1]
        )

        vtoc_header = ARDFHeader.unpack(data, header.offset + header.size)
        if vtoc_header.size != 32 or vtoc_header.name != b"VTOC":
            raise ValueError("Malformed volume table of contents.", vtoc_header)
        # cant reuse exact ARDFTableOfContents, but structure is similar
        size, nentries, stride = struct.unpack_from(
            "<QLL", data, vtoc_header.offset + 16
        )
        assert stride == 40, stride
        assert size - 32 == nentries * stride, (size, nentries, stride)
        vtoc_lines = []
        vtoc_pointers = []
        for toc_offset in range(
            vtoc_header.offset + 32, vtoc_header.offset + size, stride
        ):
            *header, force_index, line, point, pointer = struct.unpack_from(
                "<LL4sLLLQQ", data, toc_offset
            )
            entry_header = ARDFHeader(data, toc_offset, *header)
            if entry_header.name != b"VOFF":
                raise ValueError("Malformed volume table entry.", entry_header)
            entry_header.validate()
            vtoc_lines.append(line)
            vtoc_pointers.append(pointer)

        mlov_header = ARDFHeader.unpack(data, vtoc_header.offset + size)
        if mlov_header.size != 16 or mlov_header.name != b"MLOV":
            raise ValueError("Malformed volume table of contents.", mlov_header)
        mlov_header.validate()

        # Check if each offset is regularly spaced
        # optimize for LARGE regular case (FMaps are SMALL)
        pointer_arr = np.array(vtoc_pointers, np.int64)
        voff_strides = np.diff(pointer_arr)
        if np.all(voff_strides == voff_strides[0]):
            first_vset_header = ARDFHeader.unpack(data, vtoc_pointers[0])
            reader = ARDFFFMReader.parse(first_vset_header, points, lines, channels)
        else:
            reader = ARDFForceMapReader(
                data,
                np.array(vtoc_lines, np.uint32),
                pointer_arr,
                lines,
                points,
                channels,
            )

        return cls(
            volm_header.offset,
            reader,
            (points, lines),
            x_step,
            y_step,
            t_step,
            x_units,
            y_units,
            t_units,
            xdef,
        )


@attrs.define
class ARDFWorker:
    notes: str = attrs.field(repr=False)
    images: dict[str, ARDFImage]
    volumes: list[ARDFVolume]
    _view: memoryview

    @classmethod
    def parse_ardf(cls, ardf_view):
        ardf_view = memoryview(ardf_view)
        file_header = ARDFHeader.unpack(ardf_view, 0)
        if file_header.size != 16 or file_header.name != b"ARDF":
            raise ValueError("Not an ARDF file.", file_header)
        file_header.validate()
        ftoc_header = ARDFHeader.unpack(ardf_view, offset=file_header.size)
        if ftoc_header.size != 32 or ftoc_header.name != b"FTOC":
            raise ValueError("Malformed ARDF file table of contents.", ftoc_header)
        ftoc_header.validate()
        ftoc = ARDFTableOfContents.unpack(ardf_view, offset=file_header.size)
        ttoc_header = ARDFHeader.unpack(ardf_view, offset=ftoc.offset + ftoc.size)
        if ttoc_header.size != 32 or ttoc_header.name != b"TTOC":
            raise ValueError("Malformed ARDF text table of contents.", ftoc_header)
        ttoc_header.validate()
        ttoc = ARDFTextTableOfContents.unpack(ardf_view, offset=ftoc.offset + ftoc.size)
        assert len(ttoc.entries) == 1
        notes = parse_AR_note(ttoc.decode_entry(0))
        images = {}
        volumes = []
        for item, pointer in ftoc.entries:
            item.validate()
            item = ARDFHeader.unpack(ardf_view, pointer)
            if item.name == b"IMAG":
                item = ARDFImage.parse_imag(item)
                images[item.name] = item
                # assert isinstance(item.get_ndarray(), np.ndarray)
            elif item.name == b"VOLM":
                volumes.append(ARDFVolume.parse_volm(item))
            else:
                raise RuntimeError(f"Unknown TOC entry {item.name}.", item)
        return cls(notes, images, volumes, ardf_view)

    def close(self):
        self._view.release()


class ARDFFile(BaseForceVolumeFile):
    _basic_units_map = {
        "Adhesion": "N",
        "Height": "m",
        "IndentationHertz": "m",
        "YoungsHertz": "Pa",
        "YoungsJKR": "Pa",
        "YoungsDMT": "Pa",
        "ZSensor": "m",
        "MapAdhesion": "N",
        "MapHeight": "m",
        "Force": "N",
    }
    _default_heightmap_names = ("MapHeight", "ZSensorTrace", "ZSensorRetrace")
    _worker: ARDFWorker | None = None
    _mm = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_mm"]
        del state["_worker"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            mm, worker, path_lock = CACHED_OPEN_PATHS[self.path]
        except KeyError:
            self._mm = mm = mmap_path_read_only(self.path)
            worker = ARDFWorker.parse_ardf(mm)
            path_lock = threading.Lock()
            path_lock.acquire()
            CACHED_OPEN_PATHS[self.path] = mm, worker, path_lock
            threading.Thread(
                target=eventually_evict_path, args=(self.path,), daemon=True
            ).start()
        else:
            # reset thread countdown
            try:
                path_lock.release()
            except RuntimeError:
                pass  # no problem if unreleased
        self._mm = mm
        self._worker = worker

    async def ainitialize(self):
        self._mm = await trio.to_thread.run_sync(mmap_path_read_only, self.path)
        self._worker = await trio.to_thread.run_sync(ARDFWorker.parse_ardf, self._mm)
        self._file_image_names.update(self._worker.images.keys())

        self.k = float(self._worker.notes["SpringConstant"])
        lines, points = self._worker.volumes[0].shape
        # slight numerical differences here
        # xsize = lines * self._worker.volumes[0].x_step
        # ysize = points * self._worker.volumes[0].y_step
        self.scansize = (
            float(self._worker.notes["FastScanSize"]) * NANOMETER_UNIT_CONVERSION,
            float(self._worker.notes["SlowScanSize"]) * NANOMETER_UNIT_CONVERSION,
        )
        # NOTE: aspect is redundant to scansize
        # self.aspect = float(self.notes["SlowRatio"]) / float(self.notes["FastRatio"])
        self.defl_sens = self._defl_sens_orig = (
            float(self._worker.notes["InvOLS"]) * NANOMETER_UNIT_CONVERSION
        )
        self.rate = float(self._worker.notes["FastMapZRate"])
        # TODO: don't assume FFM data
        self.npts = self._worker.volumes[0].reader.array_view.shape[-1]
        self.split = self.npts // 2

    async def aclose(self):
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if self._mm is not None:
            with trio.CancelScope(shield=True):
                await trio.to_thread.run_sync(self._mm.close)

    def get_force_curve_sync(self, r, c):
        return self._worker.volumes[0].reader.get_curve(r, c)

    async def get_image(self, image_name):
        if image_name in self._calc_images:
            await trio.sleep(0)
            image = self._calc_images[image_name]
        else:
            image = await trs(self._worker.images[image_name].get_ndarray)
        return image


SUFFIX_FVFILE_MAP = {
    ".ardf": ARDFFile,
    ".h5": ARH5File,
    ".spm": NanoscopeFile,
    ".pfc": NanoscopeFile,
}
