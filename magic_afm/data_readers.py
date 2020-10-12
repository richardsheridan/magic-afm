import abc
from subprocess import PIPE, STARTF_USESHOWWINDOW, STARTUPINFO

import numpy as np
import trio

from . import calculation
from .async_tools import trs, make_cancel_poller


def mmap_file_read_only(pathlib_path):
    import mmap

    with pathlib_path.open("rb", buffering=0) as file:
        return mmap.mmap(file.fileno(), 0, None, mmap.ACCESS_READ)


async def convert_ardf(ardf_path, conv_path="ARDFtoHDF5.exe", pbar=None):
    """Turn an ARDF into a corresponding ARH5, returning the path.

    Requires converter executable available from Asylum Research"""
    ardf_path = trio.Path(ardf_path)
    h5file_path = ardf_path.with_suffix(".h5")

    if pbar is None:
        pipe = None
    else:
        pbar.set_description_str("Converting " + ardf_path.name)
        pipe = PIPE

    async def reading_stdout():
        stdout = bytearray()
        async for bytes_ in proc.stdout:
            stdout.extend(bytes_)
        stdout = stdout.decode()
        if "Failed" in stdout:
            raise RuntimeError(stdout)
        else:
            print(stdout)

    async def reading_stderr():
        async for bytes_ in proc.stderr:
            i = bytes_.rfind(b"\x08") + 1  # first thing on right not a backspace
            most_recent_numeric_output = bytes_[i:-1]  # crop % sign
            if most_recent_numeric_output:
                pbar.update(float(most_recent_numeric_output.decode()) - pbar.n)

    try:
        async with await trio.open_process(
            [str(conv_path), str(ardf_path), str(h5file_path),],
            stderr=pipe,
            stdout=pipe,
            startupinfo=STARTUPINFO(dwFlags=STARTF_USESHOWWINDOW),
        ) as proc:
            if pbar is not None:
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(reading_stdout)
                    nursery.start_soon(reading_stderr)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Please acquire ARDFtoHDF5.exe and place it in the application's root folder."
        )
    except:
        with trio.CancelScope(shield=True):
            await h5file_path.unlink(missing_ok=True)
        raise
    finally:
        if pbar is not None:
            pbar.close()

    return h5file_path


class BaseForceVolumeFile(metaclass=abc.ABCMeta):

    _basic_units_map = {}

    def __init__(self, path):
        self.path = path
        self._units_map = self._basic_units_map.copy()
        self._calc_images = {}
        self._file_image_names = set()
        self.params = {}
        self._trace = None
        self.sync_dist = 0
        self._worker = None

    @property
    def trace(self):
        return self._trace

    @property
    def image_names(self):
        return self._calc_images.keys() | self._file_image_names

    @abc.abstractmethod
    def ainitialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def aclose(self):
        raise NotImplementedError

    async def __aenter__(self):
        await self.ainitialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def get_image_units(self, image_name):
        image_name = self.strip_trace(image_name)
        return self._units_map.get(image_name, "V")

    def add_image(self, image_name, units, scandown, image):
        self._calc_images[image_name] = image, scandown
        image_name = self.strip_trace(image_name)
        self._units_map[image_name] = units

    @staticmethod
    def strip_trace(image_name):
        image_name_l = image_name.lower()
        # python 3.9+
        # image_name = image_name.removesuffix("retrace").removesuffix("trace")
        if image_name_l.endswith("retrace"):
            image_name = image_name[:-6]
        elif image_name_l.endswith("trace"):
            image_name = image_name[:-5]
        return image_name

    @abc.abstractmethod
    async def get_force_curve(self, r, c):
        raise NotImplementedError

    @abc.abstractmethod
    async def get_image(self, image_name):
        raise NotImplementedError


class DemoForceVolumeFile(BaseForceVolumeFile):
    def __init__(self, path):
        path = trio.Path(path)
        super().__init__(path)
        self._file_image_names.add("Demo")
        self.scansize = 100
        self.k = 1
        self.invols = 1
        self.scandown = True

    async def ainitialize(self):
        await trio.sleep(0)
        self.shape = (64, 64)
        self.npts = 1024
        self.split = self.npts // 2
        self.delta = (np.cos(np.linspace(0, np.pi * 2, self.npts, endpoint=False)) - 0.90) * 25

    async def aclose(self):
        pass

    async def get_force_curve(self, r, c):
        await trio.sleep(0)
        gen = np.random.default_rng(seed=(r, c))
        fext = calculation.force_curve(
            calculation.red_extend, self.delta[: self.split], 1, 10, 1, -10, 1, 0, 0, 10
        )
        fret = calculation.force_curve(
            calculation.red_retract, self.delta[self.split:], 1, 10, 1, -10, 1, 0, 0, 10
        )
        d = np.concatenate((fext, fret))
        z = self.delta + d
        d += gen.normal(scale=0.1, size=d.size)
        z += gen.normal(scale=0.01, size=z.size)
        return z, d

    async def get_image(self, image_name):
        await trio.sleep(0)
        return np.zeros(self.shape, dtype=np.float32), True


NANOMETER_UNIT_CONVERSION = 1e9  # maybe we can intelligently read this from the file someday


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
        self.chanmap = {
            key.decode("utf8"): index
            for index, key in enumerate(self.force_curves.attrs["Channels"])
        }

    def _shared_get_part(self, curve, s):
        # Index into the data and grab the Defl and Zsnsr ext and ret arrays as one 2D array
        # We could slice with "1:" if chanmap were constant but I'm not sure if it is
        defl_zsnsr_rows = [self.chanmap["Defl"], self.chanmap["ZSnsr"]]
        defl_zsnsr = curve[defl_zsnsr_rows, :]  # XXX Read h5data

        # we are happy to throw away data far from the surface to square up the data
        # Also reverse axis zero so data is ordered zsnsr,defl like we did for FFM
        return defl_zsnsr[::-1, (s - self.minext) : (s + self.minret)] * NANOMETER_UNIT_CONVERSION

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

    @BaseForceVolumeFile.trace.setter
    def trace(self, trace):
        self._worker.trace = trace
        self._trace = trace

    async def ainitialize(self):
        import h5py

        self._h5data = h5data = await trio.to_thread.run_sync(h5py.File, self.path, "r")
        # The notes have a very regular key-value structure, so we convert to dict for later access
        self.notes = await trs(
            dict,
            (
                line.split(":", 1)
                for line in h5data.attrs["Note"].decode("utf8").split("\n")
                if ":" in line and "@Line:" not in line
            ),
        )
        worker = await trs(self._choose_worker, h5data)
        images, image_names = await trs(lambda: (h5data["Image"], set(h5data["Image"].keys())))
        self._worker = worker
        self._images = images
        self._file_image_names = image_names
        self.shape = await trs(lambda name: images[name].shape, next(iter(image_names)))

        self.k = float(self.notes["SpringConstant"])
        self.scansize = float(self.notes["ScanSize"]) * NANOMETER_UNIT_CONVERSION
        self.invols = self._invols_orig = float(self.notes["InvOLS"]) * NANOMETER_UNIT_CONVERSION
        self.npts, self.split = worker.npts, worker.split

    async def aclose(self):
        with trio.CancelScope(shield=True):
            await trs(self._h5data.close)

    def _choose_worker(self, h5data):
        if "FFM" in h5data:
            self._scandown = bool(self.notes["ScanDown"])
            if "1" in h5data["FFM"]:
                worker = FFMTraceRetraceWorker(
                    h5data["FFM"]["0"]["Raw"],
                    h5data["FFM"]["0"]["Defl"],
                    h5data["FFM"]["1"]["Raw"],
                    h5data["FFM"]["1"]["Defl"],
                )
                self._trace = True
            elif "0" in h5data["FFM"]:
                worker = FFMSingleWorker(h5data["FFM"]["0"]["Raw"], h5data["FFM"]["0"]["Defl"],)
            else:
                worker = FFMSingleWorker(h5data["FFM"]["Raw"], h5data["FFM"]["Defl"],)
        else:
            self._scandown = bool(self.notes["FMapScanDown"])
            worker = ForceMapWorker(h5data)
        return worker

    async def get_force_curve(self, r, c):
        z, d = await trs(self._worker.get_force_curve, r, c)
        if self.invols != self._invols_orig:
            d *= self.invols / self._invols_orig
        return z, d

    async def get_all_curves(self):
        z, d = await trs(self._worker.get_all_curves, make_cancel_poller())
        if self.invols != self._invols_orig:
            d *= self.invols / self._invols_orig
        return z, d

    async def get_image(self, image_name):
        if image_name in self._calc_images:
            await trio.sleep(0)
            image, _ = self._calc_images[image_name]
        else:
            image = await trs(self._sync_get_image, image_name)
        return image, self._scandown

    def _sync_get_image(self, image_name):
        return self._images[image_name][:]


class BrukerWorkerBase(metaclass=abc.ABCMeta):
    def __init__(self, header, mm, shape, s):
        self.header = header  # get_image
        self.mm = mm  # get_image
        self.shape = shape  # get_image
        self.split = s  # get_force_curve
        self.version = header[header["first"]]["Version"].strip()  # get_image

    @abc.abstractmethod
    def get_force_curve(self, r, c, invols, sync_dist):
        raise NotImplementedError

    def get_image(self, image_name):
        soft_scale = self.header["Ciao scan list"]["@Sens. ZsensSens"].split()[1]
        soft_scale = float(soft_scale)
        h = self.header["Image"][image_name]

        value = h["@2:Z scale"]
        hard_scale = float(value[1 + value.find("(") : value.find(")")].split()[0])
        scale = np.float32(hard_scale * soft_scale / NANOMETER_UNIT_CONVERSION)
        data_length = int(h["Data length"])
        offset = int(h["Data offset"])
        bpp = bruker_bpp_fix(h["Bytes/pixel"], self.version)
        assert data_length == self.shape[0] * self.shape[1] * bpp
        scandown = h["Frame direction"] == "Down"
        return (
            np.ndarray(shape=self.shape, dtype=f"i{bpp}", buffer=self.mm, offset=offset,) * scale,
            scandown,
        )


class FFVWorker(BrukerWorkerBase):
    def __init__(self, header, mm, shape, s):
        super().__init__(header, mm, shape, s)
        r, c = shape
        data_name = header["Ciao force list"]["@4:Image Data"].split('"')[1]
        for name in [data_name, "Height Sensor"]:
            subheader = header["FV"][name]
            offset = int(subheader["Data offset"])
            bpp = bruker_bpp_fix(subheader["Bytes/pixel"], self.version)
            length = int(subheader["Data length"])
            npts = length // (r * c * bpp)
            data = np.ndarray(shape=(r, c, npts), dtype=f"i{bpp}", buffer=mm, offset=offset,)
            if name == "Height Sensor":
                self.z_ints = data
                value = subheader["@4:Z scale"]
                soft_scale = float(header["Ciao scan list"]["@Sens. ZsensSens"].split()[1])
                hard_scale = float(value[1 + value.find("(") : value.find(")")].split()[0])
                self.z_scale = np.float32(soft_scale * hard_scale)
            else:
                self.d_ints = data
                value = subheader["@4:Z scale"]
                self.defl_hard_scale = float(
                    value[1 + value.find("(") : value.find(")")].split()[0]
                )

    def get_force_curve(self, r, c, invols, sync_dist):
        s = self.s
        defl_scale = np.float32(invols * self.defl_hard_scale)

        d = self.d_ints[r, c] * defl_scale
        d[:s] = d[s - 1 :: -1]

        z = self.z_ints[r, c] * self.z_scale
        z[:s] = z[s - 1 :: -1]

        return z, np.roll(d, -sync_dist) if sync_dist else d


class QNMWorker(BrukerWorkerBase):
    def __init__(self, header, mm, shape, s):
        super().__init__(header, mm, shape, s)
        r, c = shape
        data_name = header["Ciao force list"]["@4:Image Data"].split('"')[1]
        subheader = header["FV"][data_name]
        bpp = bruker_bpp_fix(subheader["Bytes/pixel"], self.version)
        length = int(subheader["Data length"])
        offset = int(subheader["Data offset"])
        npts = length // (r * c * bpp)

        self.d_ints = np.ndarray(shape=(r, c, npts), dtype=f"i{bpp}", buffer=mm, offset=offset,)
        value = subheader["@4:Z scale"]
        self.defl_hard_scale = float(value[1 + value.find("(") : value.find(")")].split()[0])

        image, scandown = self.get_image("Height Sensor")
        image *= NANOMETER_UNIT_CONVERSION
        self.height_for_z = image, scandown
        amp = np.float32(header["Ciao scan list"]["Peak Force Amplitude"])
        self.z_basis = -amp * np.cos(
            np.linspace(0, 2 * np.pi, npts, endpoint=False, dtype=np.float32)
        )

    def get_force_curve(self, r, c, invols, sync_dist):
        s = self.s
        defl_scale = np.float32(invols * self.defl_hard_scale)

        d = self.d_ints[r, c] * defl_scale
        d[:s] = d[s - 1 :: -1]
        # remove blip
        if d[0] == -32768 * defl_scale:
            d[0] = d[1]
        d = np.roll(d, s - sync_dist)  # TODO roll across two adjacent indents

        # need to infer z from amp/height
        image, scandown = self.height_for_z
        z = self.z_basis - image[r, c]

        return z, d


class NanoscopeFile(BaseForceVolumeFile):
    _basic_units_map = {
        "Height Sensor": "m",
    }

    async def ainitialize(self):
        self._mm = await trio.to_thread.run_sync(mmap_file_read_only, self.path._wrapped)
        self.header = header = await trio.to_thread.run_sync(
            read_nanoscope_header, self.path._wrapped
        )
        # \Frame direction: Down
        # \Line Direction: Retrace
        # \Z direction: Retract

        self._file_image_names = header["Image"].keys()

        data_name = header["Ciao force list"]["@4:Image Data"].split('"')[1]

        data_header = header["FV"][data_name]
        length = int(data_header["Data length"])
        version = header[header["first"]]["Version"].strip()
        bpp = bruker_bpp_fix(data_header["Bytes/pixel"], version)
        self.shape = r, c = (
            int(header["Ciao scan list"]["Samps/line"].split()[0]),
            int(header["Ciao scan list"]["Lines"]),
        )

        self.npts = npts = length // (bpp * r * c)
        self.split = npts // 2
        rate, unit = header["Ciao scan list"]["PFT Freq"].split()
        assert unit.lower() == "khz"
        self.rate = float(rate)

        scansize, units = header["Ciao scan list"]["Scan Size"].split()
        if units == "nm":
            factor = 1.0
        elif units == "pm":
            factor = 0.001
        else:  # probably microns but not sure how it's spelled atm
            factor = 1000.0

        self.scansize = float(scansize) * factor

        self.k = float(data_header["Spring Constant"])
        self.invols = float(header["Ciao scan list"]["@Sens. DeflSens"].split()[1])
        value = data_header["@4:Z scale"]
        self.defl_hard_scale = float(value[1 + value.find("(") : value.find(")")].split()[0])

        self._worker, self.sync_dist = await trs(self._choose_worker, header)

    def _choose_worker(self, header):
        if "Height Sensor" in header["FV"]:
            return FFVWorker(header, self._mm, self.shape, self.s), 0
        else:
            return (
                QNMWorker(header, self._mm, self.shape, self.s),
                int(round(float(header["Ciao scan list"]["Sync Distance"]))),
            )

    async def aclose(self):
        self._worker = None
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(self._mm.close)

    async def get_force_curve(self, r, c):
        return await trs(self._worker.get_force_curve, r, c, self.invols, self.sync_dist)

    async def get_image(self, image_name):
        if image_name in self._calc_images:
            await trio.sleep(0)
            image, scandown = self._calc_images[image_name]
        else:
            image, scandown = await trs(self._worker.get_image, image_name)
        return image, scandown


def read_nanoscope_header(path):
    header = {}
    section = ""
    with path.open("r") as f:
        for line in f:
            assert line.startswith("\\")
            line = line[1:].strip()  # strip leading slash and newline

            if line.startswith("*"):
                if line.startswith("*File list end"):
                    break  # THIS IS THE **NORMAL** WAY TO END THE FOR LOOP
                if section == line[1:]:
                    if header[section] is current:
                        header[section] = [current]
                    current = {}
                    header[section].append(current)
                else:
                    section = line[1:]
                    if not header:
                        header["first"] = section
                    current = {}
                    header[section] = current
                continue

            key, value = line.split(":", maxsplit=1)
            # Colon special case for "groups"
            if key.startswith("@") and key[1].isdigit() and len(key) == 2:
                key2, value = value.split(":", maxsplit=1)
                key = key + ":" + key2

            current[key] = value.strip()
        else:
            raise ValueError("File ended too soon")
    if (not header) or ("" in header):
        raise ValueError("File is empty or not a Bruker data file")

    # link headers from [section][index][key] to [Image/FV][Image Data name][key]
    header["Image"] = {}
    for entry in header["Ciao image list"]:
        if type(entry) is str:
            name = header["Ciao image list"]["@2:Image Data"].split('"')[1]
            header["Image"][name] = header["Ciao image list"]
            break
        name = entry["@2:Image Data"].split('"')[1]
        # assert name not in header["Image"]  # assume data is redundant for now
        header["Image"][name] = entry
    header["FV"] = {}
    for entry in header["Ciao force image list"]:
        if type(entry) is str:
            name = header["Ciao force image list"]["@4:Image Data"].split('"')[1]
            header["FV"][name] = header["Ciao force image list"]
            break
        name = entry["@4:Image Data"].split('"')[1]
        # assert name not in header["FV"]  # assume data is redundant for now
        header["FV"][name] = entry

    return header


def bruker_bpp_fix(bpp, version):
    if version > "0x09200000":  # Counting on lexical ordering here, hope zeros don't change...
        return 4
    else:
        return int(bpp)


SUFFIX_FVFILE_MAP = {".h5": ARH5File, ".spm": NanoscopeFile, ".pfc": NanoscopeFile}
