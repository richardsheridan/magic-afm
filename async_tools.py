"""
A Docstring
"""

# Copyright (C) 2020  Richard J. Sheridan
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
import subprocess
from contextlib import asynccontextmanager
from functools import partial

import h5py
import numpy as np
import trio
from threadpoolctl import threadpool_limits

LONGEST_IMPERCEPTIBLE_DELAY = 0.016  # seconds

internal_threadpool_limiters = threadpool_limits(1)
cpu_bound_limiter = trio.CapacityLimiter(os.cpu_count())
ctrs = partial(trio.to_thread.run_sync, cancellable=True, limiter=cpu_bound_limiter)
trs = partial(trio.to_thread.run_sync, cancellable=True)


def make_cancel_poller():
    """Uses internal undocumented bits so probably super fragile"""
    _cancel_status = trio.lowlevel.current_task()._cancel_status

    def poll_for_cancel(*args, **kwargs):
        if _cancel_status.effectively_cancelled:
            raise trio.Cancelled._create()

    return poll_for_cancel


async def convert_ardf(
    ardf_path, conv_path=r"X:\Data\AFM\Cypher\ARDFtoHDF5.exe", force=False, pbar=None
):
    """Turn an ARDF path into a corresponding HDF5 path, converting the file if it doesn't exist.

    Can force the conversion with the force flag if necessary (e.g. overwriting with new data).
    Requires converter executable available from Asylum Research"""
    ardf_path = trio.Path(ardf_path)
    #     conv_path = trio.Path(conv_path)
    h5file_path = ardf_path.with_suffix(".h5")

    if (not force) and (await h5file_path.is_file()):
        return h5file_path

    startupinfo = None
    creationflags = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = subprocess.CREATE_NO_WINDOW

    if pbar is None:
        import tqdm

        pbar = tqdm.tqdm(total=100, unit="%",)

    pbar.set_description_str("Converting " + str(ardf_path))

    async def reading_stdout():
        stdout = bytearray()
        async for bytes_ in proc.stdout:
            stdout.extend(bytes_)
        stdout = stdout.decode()
        print(stdout)
        if "Failed" in stdout:
            raise RuntimeError()

    async def reading_stderr():
        async for stuff in proc.stderr:
            i = stuff.rfind(b"\x08") + 1  # first thing on right not a backspace
            most_recent_numeric_output = stuff[i:-1]  # crop % sign
            if most_recent_numeric_output:
                pbar.update(float(most_recent_numeric_output.decode()) - pbar.n)

    try:
        async with await trio.open_process(
            [str(conv_path), str(ardf_path), str(h5file_path),],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            startupinfo=startupinfo,
            creationflags=creationflags,
        ) as proc:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(reading_stdout)
                nursery.start_soon(reading_stderr)
    except FileNotFoundError as e:
        print(
            "Please ensure the full path to the Asylum converter tool "
            "ARDFtoHDF5.exe is in the conv_path argument",
            flush=True,
            sep="\n",
        )
        raise e
    finally:
        pbar.close()

    return h5file_path


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
        self.minext = np.min(self.extlens)
        self.extretlens = self.segments[:, :, 1]
        self.minret = np.min(self.extretlens - self.extlens)

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
        s = self.extlens[r, c]

        return (*self._shared_get_part(curve, s), self.minext)

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
            s = self.extlens[r, c]

            x[r, c, :, :] = self._shared_get_part(curve, s)
        z = x[:, :, 0, :]
        d = x[:, :, 1, :]
        s = self.minext
        return z, d, s


class FFMSingleWorker:
    def __init__(self, drive, defl):
        self.drive = drive
        self.defl = defl

    def get_force_curve(self, r, c):
        z = self.drive[r, c]
        d = self.defl[r, c]
        return z * NANOMETER_UNIT_CONVERSION, d * NANOMETER_UNIT_CONVERSION, len(z) // 2

    def get_all_curves(self, _poll_for_cancel=(lambda: None)):
        _poll_for_cancel()
        z = self.drive[:] * NANOMETER_UNIT_CONVERSION
        _poll_for_cancel()
        d = self.defl[:] * NANOMETER_UNIT_CONVERSION
        _poll_for_cancel()
        s = self.drive.shape[-1] // 2
        return z, d, s


class FFMTraceRetraceWorker:
    def __init__(self, drive_trace, defl_trace, drive_retrace, defl_retrace):
        self.drive_trace = drive_trace
        self.defl_trace = defl_trace
        self.drive_retrace = drive_retrace
        self.defl_retrace = defl_retrace
        self.trace = True

    def get_force_curve(self, r, c):
        if self.trace:
            z = self.drive_trace[r, c]
            d = self.defl_trace[r, c]
        else:
            z = self.drive_retrace[r, c]
            d = self.defl_retrace[r, c]
        return z * NANOMETER_UNIT_CONVERSION, d * NANOMETER_UNIT_CONVERSION, len(z) // 2

    def get_all_curves(self, _poll_for_cancel=(lambda: None)):
        _poll_for_cancel()
        if self.trace:
            z = self.drive_trace[:] * NANOMETER_UNIT_CONVERSION
            _poll_for_cancel()
            d = self.defl_trace[:] * NANOMETER_UNIT_CONVERSION
        else:
            z = self.drive_retrace[:] * NANOMETER_UNIT_CONVERSION
            _poll_for_cancel()
            d = self.defl_retrace[:] * NANOMETER_UNIT_CONVERSION
        _poll_for_cancel()
        s = self.drive_trace.shape[-1] // 2
        return z, d, s


class AsyncARH5File:

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
    }

    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        self._units_map = self._basic_units_map.copy()
        self._calc_images = {}
        self._h5_image_names = set()
        self.params = {}
        self._trace = None

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, trace):
        self._worker.trace = trace
        self._trace = trace

    @property
    def image_names(self):
        return self._calc_images.keys() | self._h5_image_names

    async def ainitialize(self):
        h5data = await trs(h5py.File, self.h5file_path, "r")
        # The notes have a very regular key-value structure, so we convert to dict for later access
        self.notes = await trs(
            dict,
            (
                line.split(":", 1)
                for line in h5data.attrs["Note"].decode("utf8").split("\n")
                if ":" in line
            ),
        )
        worker = await trs(self._choose_worker, h5data)
        images, image_names = await trs(lambda: (h5data["Image"], set(h5data["Image"].keys())))
        self._h5data = h5data
        self._worker = worker
        self._images = images
        self._h5_image_names = image_names
        self.shape = await trs(lambda name: images[name].shape, next(iter(image_names)))
        self.npts = len((await self.get_force_curve(0, 0))[0])

        self.params = dict(
            k=float(self.notes["SpringConstant"]),
            scansize=float(self.notes["ScanSize"]) * NANOMETER_UNIT_CONVERSION,
        )

    async def aclose(self):
        with trio.CancelScope(shield=True):
            await trs(self._h5data.close)

    async def __aenter__(self):
        await self.ainitialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def _choose_worker(self, h5data):
        if "FFM" in h5data:
            self.scandown = bool(self.notes["ScanDown"])
            if "1" in h5data["FFM"]:
                worker = FFMTraceRetraceWorker(
                    h5data["FFM"]["0"]["Drive"],
                    h5data["FFM"]["0"]["Defl"],
                    h5data["FFM"]["1"]["Drive"],
                    h5data["FFM"]["1"]["Defl"],
                )
                self._trace = True
            elif "0" in h5data["FFM"]:
                worker = FFMSingleWorker(h5data["FFM"]["0"]["Drive"], h5data["FFM"]["0"]["Defl"],)
            else:
                worker = FFMSingleWorker(h5data["FFM"]["Drive"], h5data["FFM"]["Defl"],)
        else:
            self.scandown = bool(self.notes["FMapScanDown"])
            worker = ForceMapWorker(h5data)
        return worker

    async def get_force_curve(self, r, c):
        return await trs(self._worker.get_force_curve, r, c)

    async def get_all_curves(self):
        return await trs(self._worker.get_all_curves, make_cancel_poller())

    async def get_image(self, image_name):
        if image_name in self._calc_images:
            return self._calc_images[image_name]
        return await trs(self._images.__getitem__, image_name)

    def get_image_units(self, image_name):
        # python 3.9+
        # image_name = image_name.removesuffix("Trace").removesuffix("Retrace")
        if image_name.endswith("Trace"):
            image_name = image_name[:-5]
        if image_name.endswith("Retrace"):
            image_name = image_name[:-6]
        return self._units_map.get(image_name, "V")

    def add_image(self, image_name, units, image):
        self._units_map[image_name] = units
        self._calc_images[image_name] = image


async def thread_map(sync_fn, job_items, *args, cancellable=True, limiter=cpu_bound_limiter):
    job_items = list(job_items)

    async def thread_worker(item, i, send_chan):
        async with send_chan:
            result = await trio.to_thread.run_sync(
                sync_fn, item, *args, cancellable=cancellable, limiter=limiter
            )
            await send_chan.send((i, result))

    send_chan, recv_chan = trio.open_memory_channel(0)
    async with trio.open_nursery() as nursery:
        for i, item in enumerate(job_items):
            nursery.start_soon(thread_worker, item, i, send_chan.clone())
        await send_chan.aclose()
        async for i, result in recv_chan:
            job_items[i] = result

    return job_items


async def spinner_task(set_spinner, set_normal, task_status):
    spinner_starter_sendchan, spinner_starter_recvchan = trio.open_memory_channel(0)
    spinner_stopper_sendchan, spinner_stopper_recvchan = trio.open_memory_channel(0)

    @asynccontextmanager
    async def spinner_scope():
        with trio.CancelScope(shield=True):
            with trio.fail_after(15):  # assert this should never really block
                await spinner_starter_sendchan.send(None)
                stopper = await spinner_stopper_recvchan.receive()
        with trio.CancelScope() as cancel_scope:
            try:
                yield cancel_scope
            finally:
                stopper()

    task_status.started(spinner_scope)

    async def delayed_spinner(deadline, task_status):
        # absolute deadline to start spinner means requests chain properly
        with trio.CancelScope() as cancel_scope:
            task_status.started(cancel_scope.cancel)
            await trio.sleep_until(deadline)
            set_spinner()

    def get_deadline():
        return trio.current_time() + LONGEST_IMPERCEPTIBLE_DELAY * 5

    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        # wait for first ever spinner scope entry
        await spinner_starter_recvchan.receive()
        # absolute deadline to start spinner means requests chain properly
        deadline = get_deadline()
        while True:
            # get that spinner going
            cancel_delayed_spinner = await nursery.start(delayed_spinner, deadline)
            with trio.CancelScope() as cancel_scope:
                await spinner_stopper_sendchan.send(cancel_scope.cancel)
                # wait for possibly a new scope to enter
                await spinner_starter_recvchan.receive()
                # deadline = deadline

            cancel_delayed_spinner()

            if cancel_scope.cancelled_caught:
                # The final outstanding scope exited
                set_normal()
                # wait for a new first scope entry
                await spinner_starter_recvchan.receive()
                # new first scope, new deadline
                deadline = get_deadline()
