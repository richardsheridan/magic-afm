"""MagicAFM GUI

This is a trio guest-mode async tkinter graphical interface for AFM users to
calculate indentation ratios and modulus sensitivities for their force curve
data in an intuitive and responsive package. By facilitating these sorts of
calculations, we hope to improve the overall systematic error of reported
modulus maps in the greater AFM nanomechanics community.
"""
__author__ = "Richard J. Sheridan"
__app_name__ = __doc__.split("\n", 1)[0]

# noinspection PyUnreachableCode
if __debug__:
    from multiprocessing import parent_process

    assert parent_process() is None, "importing gui code in a worker"

import sys

FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
if FROZEN:
    from ._version import __version__
else:
    from . import make_version

    __version__ = make_version.get()

__short_license__ = f"""{__app_name__} {__version__}
Copyright (C) {__author__}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import ctypes
import enum
import itertools
import math
import os
import pathlib
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import warnings
import webbrowser
from contextlib import nullcontext
from functools import partial, wraps
from math import inf
from tkinter import ttk
from typing import Callable, ClassVar, Optional, Literal

import attrs
import imageio
import matplotlib
import numpy as np
import outcome
import trio
import trio_parallel

from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize, LogNorm
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.table import Table
from matplotlib.ticker import EngFormatter
from matplotlib.transforms import Bbox, BboxTransform
from tqdm.std import TqdmExperimentalWarning
from tqdm.tk import tqdm_tk

from . import async_tools, calculation, data_readers
from .async_tools import LONGEST_IMPERCEPTIBLE_DELAY, TOOLTIP_CANCEL, tooltip_task

warnings.simplefilter("ignore", TqdmExperimentalWarning)
tqdm_tk.monitor_interval = 0

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except AttributeError:
    pass

matplotlib.rcParams["savefig.dpi"] = 300
LAYOUT_ENGINE: Literal["constrained"] = "constrained"

COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "gray",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
    "PuOr",
    "PiYG",
    "BrBG",
]


class TkState(enum.IntFlag):
    """AND/OR these with a tk.Event.state to see which keys were held down"""

    # fmt: off
    SHIFT    = 0b00000000001
    CAPSLOCK = 0b00000000010
    CONTROL  = 0b00000000100
    LALT     = 0b00000001000
    NUMLOCK  = 0b00000010000
    RALT     = 0b00010000000
    MB1      = 0b00100000000
    MB2      = 0b01000000000
    MB3      = 0b10000000000
    ALT=0b100000000000000000
    # fmt: on


@enum.unique
class DispKind(enum.IntEnum):
    zd = enum.auto()
    td = enum.auto()
    # noinspection NonAsciiCharacters
    δf = enum.auto()


@attrs.frozen
class ForceCurveOptions:
    fit_mode: calculation.FitMode
    disp_kind: DispKind
    k: float
    defl_sens: float
    sync_dist: int
    radius: float
    tau: float


@attrs.frozen
class ForceCurveData:
    zxr: tuple[np.ndarray, np.ndarray]
    dxr: tuple[np.ndarray, np.ndarray]
    txr: tuple[np.ndarray, np.ndarray]
    fxr: tuple[np.ndarray, np.ndarray]
    deltaxr: tuple[np.ndarray, np.ndarray]
    # everything else set only if fit
    beta: Optional[np.ndarray] = None
    beta_err: Optional[np.ndarray] = None
    calc_fun: Optional[Callable] = None
    sl: Optional[slice] = None
    f_fit: Optional[np.ndarray] = None
    d_fit: Optional[np.ndarray] = None
    defl: Optional[np.ndarray] = None
    ind: Optional[np.ndarray] = None
    z_tru: Optional[np.ndarray] = None
    mindelta: Optional[np.ndarray] = None
    sens: Optional[np.ndarray] = None
    sse: Optional[np.ndarray] = None


@attrs.frozen
class ImagePoint:
    r: int
    c: int
    x: float
    y: float
    _transforms: ClassVar[dict] = {}

    @classmethod
    def get_transforms(cls, axesimage):
        try:
            return cls._transforms[axesimage]
        except KeyError:
            pass

        xmin, xmax, ymin, ymax = axesimage.get_extent()
        rows, cols = axesimage.get_size()
        if axesimage.origin == "upper":
            ymin, ymax = ymax, ymin
        # noinspection PyTypeChecker
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        # noinspection PyTypeChecker
        array_extent = Bbox([[-0.5, -0.5], [rows - 0.5, cols - 0.5]])
        trans = BboxTransform(boxin=data_extent, boxout=array_extent)
        invtrans = trans.inverted()

        def data_coords_to_array_index(x, y):
            return trans.transform_point([y, x]).round().astype(int)

        def array_index_to_data_coords(r, c):
            return invtrans.transform_point([r, c])[::-1]

        transforms = cls._transforms[axesimage] = (
            data_coords_to_array_index,
            array_index_to_data_coords,
        )
        if len(cls._transforms) > 10:
            cls._transforms = dict(itertools.islice(cls._transforms.items(), 5))
        return transforms

    @classmethod
    def from_index(cls, r, c, axesimage):
        transforms = cls.get_transforms(axesimage)
        return cls(r, c, *transforms[1](r, c))

    @classmethod
    def from_data(cls, x, y, axesimage):
        transforms = cls.get_transforms(axesimage)
        r, c = transforms[0](x, y)
        # center x, y
        x, y = transforms[1](r, c)
        return cls(r, c, x, y)


@attrs.frozen
class ImageStats:
    min: float
    q01: float
    q50: float
    q99: float
    max: float

    @classmethod
    def from_array(cls, array):
        # noinspection PyArgumentList
        return cls(*np.nanquantile(array, [0, 0.01, 0.5, 0.99, 1]).tolist())


class AsyncFigureCanvasTkAgg(FigureCanvasTkAgg):
    def __init__(self, figure, master=None):
        self._resize_pending = None
        self.draw_send, self.draw_recv = trio.open_memory_channel(inf)

        super().__init__(figure, master)

        self._tkcanvas.configure(background="#f0f0f0")
        self._tkcanvas_image_region = "1"

    async def idle_draw_task(self):
        # One of the slowest processes. Stick everything in a thread.
        delay = 0.0
        # Initial draws are extra slow due to figure creation.
        # Make sure they are batched with a special loop.
        async with self.spinner_scope():
            try:
                while True:
                    draw_fn = self.draw_recv.receive_nowait()
                    await trio.to_thread.run_sync(draw_fn)
            except trio.WouldBlock:
                self._maybe_resize()
                # don't set delay based on this, it is exceptionally lengthy
                await trio.to_thread.run_sync(self.draw)
        while True:
            # Sleep until someone sends artist calls
            draw_fn = await self.draw_recv.receive()
            # Set deadline ASAP so delay scope is accurate
            deadline = trio.current_time() + delay
            async with self.spinner_scope():
                # if draw_fn returns a truthy value, no draw needed
                if await trio.to_thread.run_sync(draw_fn):
                    continue
                # Batch rapid artist call requests if a draw is incoming
                # spend roughly equal time building artists and drawing
                while True:
                    with trio.move_on_at(deadline) as delay_scope:
                        draw_fn = await self.draw_recv.receive()
                    if delay_scope.cancelled_caught:
                        break
                    await trio.to_thread.run_sync(draw_fn)
                self._maybe_resize()
                t = trio.current_time()
                await trio.to_thread.run_sync(self.draw)
                # previous delay is not great predictor of next delay
                # for now try exponential moving average
                delay = ((trio.current_time() - t) + delay) / 2.0
                delay = min(delay, 1.0)
                # Funny story, we only want tight layout behavior on resize and
                # a few other special cases, but also we want super().draw()
                # and by extension draw_idle_task to be responsible for running
                # the layout engine.
                # So everywhere desired, send set_layout_engine(LAYOUT_ENGINE)
                # in draw_fn and flag need_draw, and it will be reset here.
                self.figure.set_layout_engine("none")

    # NOTE: self._resize_pending disables draw_idle to avoid a superfluous redraw
    # when super().resize(event) internally calls draw_idle.
    # self.resize() sets self._resize_pending and self._maybe_resize() resets it.

    def draw_idle(self):
        if self._resize_pending is None:
            self.draw_send.send_nowait(bool)

    def resize(self, event):
        if self._resize_pending is None:

            def tight_resize_draw_fn():
                self.figure.set_layout_engine(LAYOUT_ENGINE)

            self.draw_send.send_nowait(tight_resize_draw_fn)
        self._resize_pending = event

    def _maybe_resize(self):
        if self._resize_pending is not None:
            super().resize(self._resize_pending)
            self._resize_pending = None

    def pipe_events_to_trio(
        self, spinner_scope, motion_send_chan, pick_send_chan, tooltip_send_chan
    ):
        self.spinner_scope = spinner_scope
        self.mpl_connect(
            "motion_notify_event",
            lambda mouseevent: motion_send_chan.send_nowait(
                (mouseevent, (mouseevent.guiEvent.x_root, mouseevent.guiEvent.y_root))
            ),
        )
        self.mpl_connect(
            "pick_event",
            lambda pickevent: pick_send_chan.send_nowait(pickevent.mouseevent),
        )
        self.mpl_connect(
            "figure_leave_event",
            impartial(partial(tooltip_send_chan.send_nowait, TOOLTIP_CANCEL)),
        )


class AsyncNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        self.toolitems = tuple(
            x for x in self.toolitems if x[-1] != "configure_subplots"
        )
        self.toolitems += (
            ("Export", "Export calculated maps", "filesave", "export_calculations"),
            (
                "ForceCurves",
                "Export calculated force curves",
                "filesave",
                "export_force_curves",
            ),
        )
        self._prev_filename = ""
        super().__init__(canvas, window)

    def drag_pan(self, event):
        """Callback for dragging in pan/zoom mode."""
        self._drag_pan_event = event

        def drag_pan_draw_fn():
            if self._pan_info is None:
                return
            if self._drag_pan_event is not event:
                return
            for ax in self._pan_info.axes:
                # Using the recorded button at the press is safer than the current
                # button, as multiple buttons can get pressed during motion.
                ax.drag_pan(self._pan_info.button, event.key, event.x, event.y)

        self.canvas.draw_send.send_nowait(drag_pan_draw_fn)

    def save_figure(self, *args):
        # frameon is a optimization of some sort on tkagg backend, but we don't want to enforce
        # tk gray on exported figures, transparency is better
        try:
            self.canvas.figure.set_frameon(False)
            super().save_figure(*args)
        finally:
            self.canvas.figure.set_frameon(True)

    def teach_navbar_to_use_trio(
        self, nursery, get_image_names, get_image_by_name, point_data
    ):
        self._parent_nursery = nursery
        self._get_image_names = get_image_names
        self._get_image_by_name = get_image_by_name
        self._point_data = point_data
        self._wait_cursor_for_draw_cm = nullcontext

    def export_force_curves(self):
        # This method is bound early in init so can't use the usual trick of swapping
        # the callback function during the teaching step
        self._parent_nursery.start_soon(self._aexport_force_curves)

    async def _aexport_force_curves(self):
        if not self._point_data:
            return
        is_fit = next(iter(self._point_data.values())).beta is not None
        if is_fit:
            h = "t (ms); z (nm); d (nm); d_fit (nm)"
        else:
            h = "t (ms); z (nm); d (nm)"
        # fmt: off
        export_filetypes = (
            ("ASCII/TXT/TSV/CSV", "*.asc *.txt *.tsv *.csv"),
            ("NPZ", "*.npz"),
        )
        # must take two positional arguments, fname and array
        exporter_map = {
            ".txt": partial(np.savetxt, fmt='%.8g', header=h, delimiter=" "),
            ".asc": partial(np.savetxt, fmt='%.8g', header=h, delimiter=" "),
            ".tsv": partial(np.savetxt, fmt='%.8g', header=h, delimiter="\t"),
            ".csv": partial(np.savetxt, fmt='%.8g', header=h, delimiter=","),
            ".npz": np.savez_compressed,
        }
        # fmt: on
        defaultextension = ""
        initialdir = os.path.expanduser(matplotlib.rcParams["savefig.directory"])
        initialfile = os.path.basename(self._prev_filename)
        fname = await trio.to_thread.run_sync(
            partial(
                tk.filedialog.asksaveasfilename,
                master=self,
                title="Export force curve",
                filetypes=export_filetypes,
                defaultextension=defaultextension,
                initialdir=initialdir,
                initialfile=initialfile,
            )
        )

        if not fname:
            return  # Cancelled

        # Save dir for next time, unless empty str (i.e., use cwd).
        if initialdir != "":
            matplotlib.rcParams["savefig.directory"] = os.path.dirname(str(fname))
        self._prev_filename = fname
        root, ext = os.path.splitext(fname)
        try:
            exporter = exporter_map[ext]
        except KeyError:
            await trio.to_thread.run_sync(
                partial(
                    tkinter.messagebox.showerror,
                    master=self,
                    title="Export error",
                    message=f"Unknown extension '{ext}'",
                )
            )
            return
        arrays = {}
        for point, data in self._point_data.items():
            if is_fit:
                arrays[f"r{point.r:04d}c{point.c:04d}"] = np.column_stack(
                    [
                        data.t[data.sl],
                        data.z[data.sl],
                        data.d[data.sl],
                        data.d_fit,
                    ]
                )
            else:
                arrays[f"r{point.r:04d}c{point.c:04d}"] = np.column_stack(
                    [
                        data.t,
                        data.z,
                        data.d,
                    ]
                )

        if exporter is np.savez_compressed:
            try:
                await trio.to_thread.run_sync(partial(exporter, fname, **arrays))
            except Exception as e:
                await trio.to_thread.run_sync(
                    partial(
                        tkinter.messagebox.showerror,
                        master=self,
                        title="Export error",
                        message=repr(e),
                    )
                )
        elif isinstance(exporter, partial) and exporter.func is np.savetxt:
            for name, array in arrays.items():
                try:
                    await trio.to_thread.run_sync(
                        exporter, root + "_" + name + ext, array
                    )
                except Exception as e:
                    await trio.to_thread.run_sync(
                        partial(
                            tkinter.messagebox.showerror,
                            master=self,
                            title="Export error",
                            message=repr(e),
                        )
                    )
                    break
        else:
            await trio.to_thread.run_sync(
                partial(
                    tkinter.messagebox.showerror,
                    master=self,
                    title="Export error",
                    message=f"Unknown exporter '{exporter}'",
                )
            )

    def export_calculations(self):
        # This method is bound early in init so can't use the usual trick of swapping
        # the callback function during the teaching step
        self._parent_nursery.start_soon(self._aexport_calculations)

    async def _aexport_calculations(self):
        # NOTE: it's also possible to export image stacks but need a way to indicate that

        # fmt: off
        export_filetypes = (
            ("ASCII/TXT/TSV/CSV", "*.asc *.txt *.tsv *.csv"),
            ("NPY/NPZ", "*.npy *.npz"),
            ################
            ("Many image formats (try an extension)", "*.*")
        )
        # must take two positional arguments, fname and array
        exporter_map = {
            ".txt": partial(np.savetxt, fmt='%.8g'),
            ".asc": partial(np.savetxt, fmt='%.8g'),
            ".tsv": partial(np.savetxt, fmt='%.8g', delimiter="\t"),
            ".csv": partial(np.savetxt, fmt='%.8g', delimiter=","),
            ".npy": np.save,
            ".npz": np.savez_compressed,
        }
        # fmt: on
        defaultextension = ""
        initialdir = os.path.expanduser(matplotlib.rcParams["savefig.directory"])
        initialfile = os.path.basename(self._prev_filename)
        fname = await trio.to_thread.run_sync(
            partial(
                tk.filedialog.asksaveasfilename,
                master=self,
                title="Export calculated images",
                filetypes=export_filetypes,
                defaultextension=defaultextension,
                initialdir=initialdir,
                initialfile=initialfile,
            )
        )

        if not fname:
            return  # Cancelled

        # Save dir for next time, unless empty str (i.e., use cwd).
        if initialdir != "":
            matplotlib.rcParams["savefig.directory"] = os.path.dirname(str(fname))
        self._prev_filename = fname
        root, ext = os.path.splitext(fname)
        exporter = exporter_map.get(ext, imageio.imwrite)
        for image_name in self._get_image_names():
            if image_name.startswith("Calc"):
                # to date, all afm images have been "flipped" on disk, which is why
                # they are displayed with "lower" origin, but on export users
                # have found this confusing, so we manually flip here
                # TODO: solve this at the data reader level?
                image = (await self._get_image_by_name(image_name))[::-1]
                try:
                    await trio.to_thread.run_sync(
                        exporter, root + "_" + image_name[4:] + ext, image
                    )
                except Exception as e:
                    await trio.to_thread.run_sync(
                        partial(
                            tkinter.messagebox.showerror,
                            master=self,
                            title="Export error",
                            message=repr(e),
                        )
                    )
                    break


class TkHost:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._outcome: Optional[outcome.Outcome] = None
        self._guest_tick = bool
        self._tk_func_name = root.register(self._tk_func)

    def _tk_func(self):
        self._guest_tick()

    def run_sync_soon_threadsafe(self, func):
        """Use Tcl "after" command to schedule a function call

        Based on `tkinter source comments <https://github.com/python/cpython/blob/a5d6aba318ead9cc756ba750a70da41f5def3f8f/Modules/_tkinter.c#L1472-L1555>`_
        the issuance of the tcl call to after itself is thread-safe since it is sent
        to the `appropriate thread <https://github.com/python/cpython/blob/a5d6aba318ead9cc756ba750a70da41f5def3f8f/Modules/_tkinter.c#L814-L824>`_ on line 1522.
        Tkapp_ThreadSend effectively uses "after 0" while putting the command in the
        event queue so the `"after idle after 0" <https://wiki.tcl-lang.org/page/after#096aeab6629eae8b244ae2eb2000869fbe377fa988d192db5cf63defd3d8c061>`_ incantation
        is unnecessary here.

        Compare to `tkthread <https://github.com/serwy/tkthread/blob/1f612e1dd46e770bd0d0bb64d7ecb6a0f04875a3/tkthread/__init__.py#L163>`_
        where definitely thread unsafe `eval <https://github.com/python/cpython/blob/a5d6aba318ead9cc756ba750a70da41f5def3f8f/Modules/_tkinter.c#L1567-L1585>`_
        is used to send thread safe signals between tcl interpreters.
        """
        # self.root.after("idle", func) # does a fairly intensive wrapping to each func
        self._guest_tick = func
        self.root.call("after", "idle", self._tk_func_name)

    def run_sync_soon_not_threadsafe(self, func):
        """Use Tcl "after" command to schedule a function call from the main thread

        If .call is called from the Tcl thread, the locking and sending are optimized away
        so it should be fast enough.

        The incantation `"after idle after 0" <https://wiki.tcl-lang.org/page/after#096aeab6629eae8b244ae2eb2000869fbe377fa988d192db5cf63defd3d8c061>`_ avoids blocking the normal event queue with
        an unending stream of idle tasks.
        """
        self._guest_tick = func
        self.root.call("after", "idle", "after", 0, self._tk_func_name)

    def done_callback(self, outcome_):
        """End the Tk app.

        Only really ends if self.root is truly the parent of all other Tk objects!
        thus tk.NoDefaultRoot
        """
        self._outcome = outcome_
        self.root.destroy()

    def run(self, async_fn, *args):
        trio.lowlevel.start_guest_run(
            async_fn,
            *args,
            run_sync_soon_threadsafe=self.run_sync_soon_threadsafe,
            run_sync_soon_not_threadsafe=self.run_sync_soon_not_threadsafe,
            done_callback=self.done_callback,
            restrict_keyboard_interrupt_to_checkpoints=True,
        )
        try:
            self.root.mainloop()
        except BaseException as e:
            if isinstance(self._outcome, outcome.Error):
                raise e from self._outcome.error
            else:
                raise e
        else:
            return self._outcome.unwrap()


def impartial(fn):
    @wraps(fn)
    def impartial_wrapper(*a, **kw):
        return fn()

    return impartial_wrapper


def unbind_mousewheel(widget_with_bind):
    # https://stackoverflow.com/a/44269385
    def empty_scroll_command(event):
        return "break"

    # Windows and OSX
    widget_with_bind.bind("<MouseWheel>", empty_scroll_command)
    # Linux and other *nix systems
    widget_with_bind.bind("<ButtonPress-4>", empty_scroll_command)
    widget_with_bind.bind("<ButtonPress-5>", empty_scroll_command)


class ForceVolumeTkDisplay:
    def __init__(
        self, root, name, initial_values: data_readers.ForceVolumeParams, **kwargs
    ):
        self._traces: list[tuple[tk.Variable, str]] = []
        self._nursery = None
        self._prev_defl_sens = 1.0
        self._prev_k = 1.0
        self._prev_sync_dist = 0
        self._prev_radius = 20.0
        self._prev_tau = 0.0
        self.tkwindow = window = tk.Toplevel(root, **kwargs)
        window.wm_title(name)

        # Build figure
        self.fig = Figure((9, 2.75), facecolor="#f0f0f0", layout=LAYOUT_ENGINE)
        self.canvas = AsyncFigureCanvasTkAgg(self.fig, window)
        self.navbar = AsyncNavigationToolbar2Tk(self.canvas, window)
        self.img_ax = None
        self.plot_ax = None

        def initial_draw_fn():
            self.img_ax, self.plot_ax = self.fig.subplots(1, 2, width_ratios=[1, 1.5])
            self.img_ax.set_anchor("W")
            self.img_ax.set_facecolor((0.8, 0, 0))  # scary red for NaN values of images
            # Need to pre-load something into these labels for change_image_callback
            self.plot_ax.set_xlabel(" ")
            self.plot_ax.set_ylabel(" ")
            self.plot_ax.set_ylim([-1000, 1000])
            self.plot_ax.set_xlim([-1000, 1000])
            # technically needs a draw but later resize will take care of it
            return True

        self.canvas.draw_send.send_nowait(initial_draw_fn)

        # Options and buttons
        self.options_frame = ttk.Frame(root)

        image_opts_frame = ttk.Frame(self.options_frame)

        image_name_labelframe = ttk.Labelframe(image_opts_frame, text="Image")
        self.image_name_strvar = tk.StringVar(
            image_name_labelframe, value="Choose an image..."
        )
        self.image_name_menu = ttk.Combobox(
            image_name_labelframe,
            width=12,
            state="readonly",
            textvariable=self.image_name_strvar,
        )
        unbind_mousewheel(self.image_name_menu)
        self.image_name_menu.pack(fill="x")
        image_name_labelframe.pack(fill="x")

        colormap_labelframe = ttk.Labelframe(image_opts_frame, text="Colormap")
        self.colormap_strvar = tk.StringVar(colormap_labelframe, value="viridis")
        colormap_menu = ttk.Combobox(
            colormap_labelframe,
            state="readonly",
            textvariable=self.colormap_strvar,
            values=COLORMAPS,
            width=max(map(len, COLORMAPS)) - 1,
        )
        unbind_mousewheel(colormap_menu)
        colormap_menu.pack(fill="x")
        colormap_labelframe.pack(fill="x")
        self.log_scale_intvar = tk.IntVar(image_name_labelframe, value=False)
        log_scale_checkbtn = ttk.Checkbutton(
            image_opts_frame, variable=self.log_scale_intvar, text="logarithmic scale"
        )
        log_scale_checkbtn.pack(fill="x")

        manipulate_labelframe = ttk.Labelframe(image_opts_frame, text="Manipulations")
        self.manipulate_strvar = tk.StringVar(
            manipulate_labelframe, value=next(iter(calculation.MANIPULATIONS))
        )
        manipulate_menu = ttk.Combobox(
            manipulate_labelframe,
            state="readonly",
            textvariable=self.manipulate_strvar,
            values=list(calculation.MANIPULATIONS),
        )
        unbind_mousewheel(manipulate_menu)
        manipulate_menu.pack(fill="x")
        manipulate_labelframe.pack(fill="x")

        image_opts_frame.grid(row=1, column=0, rowspan=2)

        disp_labelframe = ttk.Labelframe(self.options_frame, text="Force curve display")
        self.disp_kind_intvar = tk.IntVar(disp_labelframe, value=DispKind.zd.value)
        self._add_trace(self.disp_kind_intvar, self.change_disp_kind_callback)
        disp_zd_button = ttk.Radiobutton(
            disp_labelframe,
            text="d vs. z",
            value=DispKind.zd.value,
            variable=self.disp_kind_intvar,
            padding=4,
        )
        disp_zd_button.pack(side="top")
        disp_td_button = ttk.Radiobutton(
            disp_labelframe,
            text="d vs. t",
            value=DispKind.td.value,
            variable=self.disp_kind_intvar,
            padding=4,
        )
        disp_td_button.pack(side="top")
        disp_deltaf_button = ttk.Radiobutton(
            disp_labelframe,
            text="f vs. δ",
            value=DispKind.δf.value,
            variable=self.disp_kind_intvar,
            padding=4,
        )
        disp_deltaf_button.pack(side="top")
        disp_labelframe.grid(row=0, column=0)

        preproc_labelframe = ttk.Labelframe(self.options_frame, text="Preprocessing")
        self.defl_sens_strvar = tk.StringVar(preproc_labelframe)
        self._add_trace(self.defl_sens_strvar, self.defl_sens_callback)
        self.defl_sens_sbox = ttk.Spinbox(
            preproc_labelframe,
            from_=0,
            to=1e3,
            increment=0.1,
            format="%0.1f",
            width=6,
            textvariable=self.defl_sens_strvar,
        )
        self.defl_sens_sbox.set(initial_values.defl_sens)
        self.defl_sens_sbox.grid(row=0, column=2, sticky="E")
        defl_sens_label = ttk.Label(
            preproc_labelframe, text="Deflection Sens.", justify="left"
        )
        defl_sens_label.grid(row=0, column=0, columnspan=2, sticky="W")
        self.spring_const_strvar = tk.StringVar(preproc_labelframe)
        self._add_trace(self.spring_const_strvar, self.spring_const_callback)
        self.spring_const_sbox = ttk.Spinbox(
            preproc_labelframe,
            from_=0,
            to=1e3,
            increment=0.1,
            format="%0.1f",
            width=6,
            textvariable=self.spring_const_strvar,
        )
        self.spring_const_sbox.set(initial_values.k)
        self.spring_const_sbox.grid(row=1, column=2, sticky="E")
        spring_const_label = ttk.Label(
            preproc_labelframe, text="Spring Constant", justify="left"
        )
        spring_const_label.grid(row=1, column=0, columnspan=2, sticky="W")
        self.sync_dist_strvar = tk.StringVar(preproc_labelframe)
        self._add_trace(self.sync_dist_strvar, self.sync_dist_callback)
        self.sync_dist_sbox = ttk.Spinbox(
            preproc_labelframe,
            from_=-initial_values.sync_dist * 2,
            to=initial_values.sync_dist * 2,
            increment=1,
            format="%0.0f",
            width=6,
            textvariable=self.sync_dist_strvar,
        )
        self.sync_dist_sbox.set(initial_values.sync_dist)
        self.sync_dist_sbox.grid(row=2, column=2, sticky="E")
        sync_dist_label = ttk.Label(
            preproc_labelframe, text="Sync Distance", justify="left"
        )
        sync_dist_label.grid(row=2, column=0, columnspan=2, sticky="W")
        preproc_labelframe.grid(row=0, column=1, sticky="EW")
        preproc_labelframe.grid_columnconfigure(1, weight=1)

        fit_labelframe = ttk.Labelframe(self.options_frame, text="Fit parameters")
        self.fit_intvar = tk.IntVar(
            fit_labelframe, value=calculation.FitMode.SKIP.value
        )
        self._add_trace(self.fit_intvar, self.change_fit_kind_callback)
        fit_skip_button = ttk.Radiobutton(
            fit_labelframe,
            text="Skip",
            value=calculation.FitMode.SKIP.value,
            variable=self.fit_intvar,
        )
        fit_skip_button.grid(row=1, column=0, sticky="W")
        fit_ext_button = ttk.Radiobutton(
            fit_labelframe,
            text="Extend",
            value=calculation.FitMode.EXTEND.value,
            variable=self.fit_intvar,
        )
        fit_ext_button.grid(row=0, column=0, sticky="W")
        fit_ret_button = ttk.Radiobutton(
            fit_labelframe,
            text="Retract",
            value=calculation.FitMode.RETRACT.value,
            variable=self.fit_intvar,
        )
        fit_ret_button.grid(row=0, column=1, sticky="W")
        fit_ret_button = ttk.Radiobutton(
            fit_labelframe,
            text="Both",
            value=calculation.FitMode.BOTH.value,
            variable=self.fit_intvar,
        )
        fit_ret_button.grid(row=1, column=1, sticky="W")

        fit_radius_label = ttk.Label(fit_labelframe, text="Tip radius (nm)")
        fit_radius_label.grid(row=2, column=0, columnspan=2, sticky="W")
        self.radius_strvar = tk.StringVar(fit_labelframe)
        self._add_trace(self.radius_strvar, self.radius_callback)
        self.fit_radius_sbox = ttk.Spinbox(
            fit_labelframe,
            from_=1,
            to=10000,
            increment=0.1,
            format="%0.1f",
            width=6,
            textvariable=self.radius_strvar,
        )
        self.fit_radius_sbox.set(20.0)
        self.fit_radius_sbox.grid(row=2, column=2, sticky="E")
        fit_tau_label = ttk.Label(fit_labelframe, text="DMT-JKR (0-1)", justify="left")
        fit_tau_label.grid(row=3, column=0, columnspan=2, sticky="W")
        self.tau_strvar = tk.StringVar(fit_labelframe)
        self._add_trace(self.tau_strvar, self.tau_callback)
        self.fit_tau_sbox = ttk.Spinbox(
            fit_labelframe,
            from_=0,
            to=1,
            increment=0.05,
            format="%0.2f",
            width=6,
            textvariable=self.tau_strvar,
        )
        self.fit_tau_sbox.set(0.0)
        self.fit_tau_sbox.grid(row=3, column=2, sticky="E")

        self.calc_props_button = ttk.Button(
            fit_labelframe, text="Calculate Property Maps", state="disabled"
        )
        self.calc_props_button.grid(row=4, column=0, columnspan=3)
        fit_labelframe.grid(row=1, column=1, rowspan=3, sticky="EW")

        self.options_frame.grid(row=1, column=0, sticky="NSEW")

        # yes, cheating on trio here
        window.bind("<FocusIn>", impartial(self.options_frame.lift))

        # Window widgets
        size_grip = ttk.Sizegrip(window)

        self.navbar.grid(row=0, sticky="we")
        window.grid_rowconfigure(0, weight=0)

        self.canvas.get_tk_widget().grid(row=1, sticky="wens")
        window.grid_rowconfigure(1, weight=1)
        window.grid_columnconfigure(0, weight=1)

        size_grip.grid(row=1, sticky="es")
        size_grip.lift()

        # tooltips
        from inspect import cleandoc as c

        self.img_ax_tip_text = c(
            """Property map
        
            Left click: plot force curve from point
            Shift: plot multiple curves
            Control: plot continuously"""
        )
        self.plot_ax_tip_text = c(
            """Force Curve
        
            No actions yet!"""
        )
        self.tipwindow = tipwindow = tk.Toplevel(window)
        tipwindow.wm_withdraw()
        # noinspection PyTypeChecker
        tipwindow.wm_overrideredirect(1)
        try:
            # For Mac OS
            # noinspection PyUnresolvedReferences
            tipwindow.tk.call(
                "::tk::unsupported::MacWindowStyle",
                "style",
                tipwindow._w,
                "help",
                "noActivates",
            )
        except tk.TclError:
            pass
        self.tipwindow_strvar = tk.StringVar(tipwindow)
        self.tipwindow_label = tk.Label(
            tipwindow,
            textvariable=self.tipwindow_strvar,
            justify="left",
            relief="solid",
            background="white",
        )
        self.tipwindow_label.pack(ipadx=2)
        self._opts = self.options

    def _add_trace(self, tkvar, callback):
        self._traces.append((tkvar, tkvar.trace_add("write", callback)))

    def destroy(self):
        for tkvar, cbname in self._traces:
            tkvar.trace_remove("write", cbname)
        self.options_frame.destroy()
        self.tkwindow.destroy()

    @property
    def options(self):
        try:
            self._opts = ForceCurveOptions(
                fit_mode=calculation.FitMode(self.fit_intvar.get()),
                disp_kind=DispKind(self.disp_kind_intvar.get()),
                k=float(self.spring_const_strvar.get()),
                defl_sens=float(self.defl_sens_strvar.get()),
                radius=float(self.fit_radius_sbox.get()),
                tau=float(self.fit_tau_sbox.get()),
                sync_dist=int(self.sync_dist_strvar.get()),
            )
        except Exception as e:
            warnings.warn(str(e))
        return self._opts

    def spinner_start(self):
        self.tkwindow.configure(cursor="watch")
        self.options_frame.configure(cursor="watch")

    def spinner_stop(self):
        self.tkwindow.configure(cursor="arrow")
        self.options_frame.configure(cursor="arrow")

    def show_tooltip(self, x, y, text):
        self.tipwindow_strvar.set(text)
        # req -> calculated sync but geometry manager changes unknown
        w = self.tipwindow_label.winfo_reqwidth() + 4  # match with ipadx setting
        h = self.tipwindow_label.winfo_reqheight()
        self.tipwindow.wm_geometry(f"+{x - w}+{y - h}")
        self.tipwindow.wm_deiconify()

    def hide_tooltip(self):
        self.tipwindow.wm_withdraw()

    def reset_image_name_menu(self, names):
        names = list(names)
        longest = max(map(len, names))
        self.image_name_menu.configure(values=names, width=min(longest - 1, 20))

    def replot(self):
        pass

    def replot_tight(self):
        pass

    # noinspection PyTypeChecker
    def teach_display_to_use_trio(
        self,
        nursery,
        redraw_send_chan,
        calc_prop_map_callback,
        change_cmap_callback,
        manipulate_callback,
        change_image_callback,
    ):
        self.tkwindow.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        self.calc_props_button.configure(
            command=lambda: nursery.start_soon(calc_prop_map_callback, self.options)
        )
        self._add_trace(
            self.colormap_strvar,
            impartial(partial(nursery.start_soon, change_cmap_callback)),
        )
        self._add_trace(
            self.manipulate_strvar,
            impartial(partial(nursery.start_soon, manipulate_callback)),
        )
        self._add_trace(
            self.image_name_strvar,
            impartial(partial(nursery.start_soon, change_image_callback)),
        )
        self._add_trace(
            self.log_scale_intvar,
            impartial(partial(nursery.start_soon, change_image_callback)),
        )
        self.replot = partial(redraw_send_chan.send_nowait, False)
        self.replot_tight = partial(redraw_send_chan.send_nowait, True)

        nursery.start_soon(self.canvas.idle_draw_task)

    def defl_sens_callback(self, *args):
        try:
            self._prev_defl_sens = float(self.defl_sens_strvar.get())
        except ValueError:
            self.defl_sens_sbox.configure(foreground="red2")
        else:
            self.defl_sens_sbox.configure(foreground="black")
            self.replot()

    def spring_const_callback(self, *args):
        try:
            self._prev_k = float(self.spring_const_strvar.get())
        except ValueError:
            self.spring_const_sbox.configure(foreground="red2")
        else:
            self.spring_const_sbox.configure(foreground="black")
            self.replot()

    def sync_dist_callback(self, *args):
        try:
            self._prev_sync_dist = int(self.sync_dist_strvar.get())
        except ValueError:
            self.sync_dist_sbox.configure(foreground="red2")
        else:
            self.sync_dist_sbox.configure(foreground="black")
            self.replot()

    def radius_callback(self, *args):
        try:
            self._prev_radius = float(self.radius_strvar.get())
        except ValueError:
            self.fit_radius_sbox.configure(foreground="red2")
        else:
            self.fit_radius_sbox.configure(foreground="black")
            self.replot()

    def tau_callback(self, *args):
        try:
            self._prev_tau = float(self.tau_strvar.get())
        except ValueError:
            self.fit_tau_sbox.configure(foreground="red2")
        else:
            self.fit_tau_sbox.configure(foreground="black")
            self.replot()

    def change_fit_kind_callback(self, *args):
        if self.fit_intvar.get():
            state = "normal"
        else:
            state = "disabled"
        self.calc_props_button.configure(state=state)
        self.replot_tight()

    def change_disp_kind_callback(self, *args):
        self.replot_tight()


async def force_volume_task(
    display: ForceVolumeTkDisplay, opened_fvol: data_readers.AsyncFVFile
):
    # plot_curve_event_response
    plot_curve_cancels_pending = set()
    img_artists = []
    plot_artists = []
    table: Optional[Table] = None
    existing_points = set()
    point_data = {}

    # set in change_image_callback
    colorbar: Optional[Colorbar] = None
    axesimage: Optional[AxesImage] = None
    image_stats: Optional[ImageStats] = None
    unit: Optional[str] = None

    async def calc_prop_map_callback(options: ForceCurveOptions):
        assert options.fit_mode
        img_shape = axesimage.get_size()
        ncurves = img_shape[0] * img_shape[1]
        chunksize = 4

        async with spinner_scope():
            # assign pbar and progress_image ASAP in case of cancel
            with trio.CancelScope() as cancel_scope, tqdm_tk(
                total=ncurves,
                desc=f"Fitting {opened_fvol.path.name} force curves",
                smoothing=0.01,
                # smoothing_time=1,
                mininterval=LONGEST_IMPERCEPTIBLE_DELAY * 2,
                unit=" fits",
                tk_parent=display.tkwindow,
                grab=False,
                leave=False,
                cancel_callback=cancel_scope.cancel,
            ) as pbar:
                progress_image: matplotlib.image.AxesImage
                progress_array = np.zeros(img_shape + (4,), dtype="f4")
                half_red = np.array((1, 0, 0, 0.5), dtype="f4")
                half_green = np.array((0, 1, 0, 0.5), dtype="f4")
                property_map = np.empty(
                    img_shape,
                    dtype=np.dtype(
                        [(name, "f4") for name in calculation.PROPERTY_UNITS_DICT]
                    ),
                )

                def make_progress_image_draw_fn():
                    nonlocal progress_image, progress_array
                    progress_image = display.img_ax.imshow(
                        progress_array,
                        extent=axesimage.get_extent(),
                        origin="lower",
                        animated=True,
                        visible=False,
                        zorder=0.01,
                        interpolation="none",
                    )
                    progress_array = progress_image.get_array()
                    return True

                display.canvas.draw_send.send_nowait(make_progress_image_draw_fn)

                def blit_img_draw_fn():
                    bg = display.canvas.copy_from_bbox(progress_image.clipbox)
                    progress_image.set_visible(True)
                    display.img_ax.draw_artist(progress_image)
                    display.canvas.blit(progress_image.clipbox)
                    progress_image.set_visible(False)
                    display.canvas.restore_region(bg)
                    return True

                async with trio.open_nursery() as pipeline_nursery:
                    force_curve_aiter = await pipeline_nursery.start(
                        async_tools.to_sync_runner_map_unordered,
                        trio_parallel.run_sync,
                        partial(
                            calculation.load_force_curve,
                            opened_fvol,
                            options.fit_mode,
                            options.k,
                        ),
                        opened_fvol,  # np.ndindex(img_shape),
                        chunksize * 8,
                    )
                    d = attrs.asdict(options)
                    del d["disp_kind"]
                    property_aiter = await pipeline_nursery.start(
                        async_tools.to_sync_runner_map_unordered,
                        trio_parallel.run_sync,
                        partial(calculation.calc_properties_imap, **d),
                        force_curve_aiter,
                        chunksize,
                    )
                    async for rc, properties in property_aiter:
                        if properties is None:
                            property_map[rc] = np.nan
                            progress_array[rc] = half_red
                        else:
                            property_map[rc] = properties
                            progress_array[rc] = half_green
                        if pbar.update():
                            display.canvas.draw_send.send_nowait(blit_img_draw_fn)

        def progress_image_cleanup_draw_fn():
            if progress_image is not None:
                progress_image.remove()

        display.canvas.draw_send.send_nowait(progress_image_cleanup_draw_fn)

        if cancel_scope.cancelled_caught:
            return

        combobox_values = list(display.image_name_menu.cget("values"))

        # Actually write out results to external world
        if options.fit_mode == calculation.FitMode.EXTEND:
            extret = "Ext"
        elif options.fit_mode == calculation.FitMode.RETRACT:
            extret = "Ret"
        else:
            extret = "Both"
        for name in calculation.PROPERTY_UNITS_DICT:
            opened_fvol.add_image(
                image_name="Calc" + extret + name,
                units=calculation.PROPERTY_UNITS_DICT[name],
                image=property_map[name].squeeze(),
            )
            if "Calc" + extret + name not in combobox_values:
                combobox_values.append("Calc" + extret + name)

        display.reset_image_name_menu(combobox_values)

    async def change_cmap_callback():
        colormap_name = display.colormap_strvar.get()

        def change_cmap_draw_fn():
            axesimage.set_cmap(colormap_name)

        await display.canvas.draw_send.send(change_cmap_draw_fn)

    async def change_image_callback():
        nonlocal image_stats, unit
        image_name = display.image_name_strvar.get()
        cmap = display.colormap_strvar.get()
        image_array = await opened_fvol.get_image(image_name)
        image_stats = ImageStats.from_array(image_array)
        unit = opened_fvol.get_image_units(image_name)

        positive = image_stats.q01 > 0
        log_scale = display.log_scale_intvar.get()
        norm = Normalize

        if log_scale:
            if positive:
                norm = LogNorm
            else:
                await trio.to_thread.run_sync(
                    partial(
                        tkinter.messagebox.showwarning,
                        master=display.tkwindow,
                        title="Logarithmic scale warning",
                        message=(
                            "Many negative values in image; "
                            "logarithmic scaling ignored.\n"
                            "Consider applying image manipulations."
                        ),
                    )
                )

        fastscansize, slowscansize = opened_fvol.fvfile.scansize

        def change_image_draw_fn():
            nonlocal axesimage, colorbar
            if colorbar is not None:
                colorbar.remove()
            if axesimage is not None:
                axesimage.remove()
            display.img_ax.set_title(image_name.replace("_", "\n"))
            display.img_ax.set_ylabel("Y piezo (nm)")
            display.img_ax.set_xlabel("X piezo (nm)")

            axesimage = display.img_ax.imshow(
                image_array,
                origin="lower",
                extent=(
                    -0.5 * fastscansize / image_array.shape[1],
                    fastscansize + 0.5 * fastscansize / image_array.shape[1],
                    -0.5 * slowscansize / image_array.shape[0],
                    slowscansize + 0.5 * slowscansize / image_array.shape[0],
                ),
                picker=True,
                norm=norm(vmin=image_stats.q01, vmax=image_stats.q99),
                cmap=cmap,
            )
            display.img_ax.autoscale()
            axesimage.get_array().fill_value = np.nan

            colorbar = display.fig.colorbar(
                axesimage, ax=display.img_ax, use_gridspec=True
            )
            display.navbar.update()  # let navbar catch new cax in fig for tooltips
            if unit is not None:
                colorbar.formatter = EngFormatter(unit, places=1)
                colorbar.update_ticks()

            display.fig.set_layout_engine(LAYOUT_ENGINE)

        await display.canvas.draw_send.send(change_image_draw_fn)

    async def redraw_existing_points_task(task_status):
        redraw_send, redraw_recv = trio.open_memory_channel(inf)

        def clear_points_draw_fn():
            for artist in img_artists:
                artist.remove()
            for artist in plot_artists:
                artist.remove()
            display.plot_ax.relim()
            display.plot_ax.set_prop_cycle(None)
            img_artists.clear()
            plot_artists.clear()
            point_data.clear()

        def tight_points_draw_fn():
            display.fig.set_layout_engine(LAYOUT_ENGINE)

        task_status.started(redraw_send)
        while True:
            msg = await redraw_recv.receive()
            # only do work for most recent request
            while True:
                try:
                    msg = redraw_recv.receive_nowait() or msg
                except trio.WouldBlock:
                    break
            await display.canvas.draw_send.send(clear_points_draw_fn)
            async with spinner_scope():
                options = display.options
                for point in existing_points.copy():  # avoid crash on concurrent clear
                    await plot_curve_response(point, options, False)
                if msg:
                    await display.canvas.draw_send.send(tight_points_draw_fn)

    async def manipulate_callback():
        manip_name = display.manipulate_strvar.get()
        current_name = display.image_name_strvar.get()
        current_names = list(display.image_name_menu.cget("values"))
        name = "Calc" + manip_name + "_" + current_name
        if name not in opened_fvol.image_names:
            manip_fn = calculation.MANIPULATIONS[manip_name]
            async with spinner_scope():
                manip_img = await trio.to_thread.run_sync(
                    manip_fn, axesimage.get_array().data
                )
            opened_fvol.add_image(name, unit, manip_img)
            if name not in current_names:
                current_names.append(name)
            display.reset_image_name_menu(current_names)
        display.image_name_menu.set(name)

    async def plot_curve_response(
        point: ImagePoint,
        options: ForceCurveOptions,
        clear_previous: bool,
        *,
        task_status=trio.TASK_STATUS_IGNORED,
    ):
        existing_points.add(point)  # should be before 1st checkpoint

        # XXX: only needed on first plot. Maybe later make optional?
        display.plot_ax.set_autoscale_on(True)

        if clear_previous:
            for cancel_scope in plot_curve_cancels_pending:
                cancel_scope.cancel()
            plot_curve_cancels_pending.clear()
        task_status.started()
        async with spinner_scope():
            with trio.CancelScope() as cancel_scope:
                plot_curve_cancels_pending.add(cancel_scope)

                # Calculation phase
                # Do a few long-running jobs, likely to be canceled
                force_curve = await opened_fvol.get_force_curve(point.r, point.c)
                force_curve_data = await trio.to_thread.run_sync(
                    calculate_force_data,
                    *force_curve,
                    opened_fvol.fvfile.t_step,
                    options,
                    trio.from_thread.check_cancelled,
                )
                del force_curve  # contained in data
            plot_curve_cancels_pending.discard(cancel_scope)

            if cancel_scope.cancelled_caught:
                existing_points.discard(point)
                return

            def plot_point_draw_fn():
                nonlocal table
                # Clearing Phase
                # Clear previous artists and reset plots (faster than .clear()?)
                if table is not None:
                    table.remove()
                    table = None
                if clear_previous:
                    for artist in img_artists:
                        artist.remove()
                    for artist in plot_artists:
                        artist.remove()
                    img_artists.clear()
                    plot_artists.clear()
                    existing_points.clear()
                    point_data.clear()

                    existing_points.add(point)
                    display.plot_ax.relim()
                    display.plot_ax.set_prop_cycle(None)
                point_data[
                    point
                ] = force_curve_data  # unconditional so draw_force_curve gets the latest data

                # Drawing Phase
                # Based options choose plots and collect artists for deletion
                new_artists, color = draw_force_curve(
                    force_curve_data, display.plot_ax, options
                )
                plot_artists.extend(new_artists)
                img_artists.extend(
                    display.img_ax.plot(
                        point.x,
                        point.y,
                        marker="X",
                        markersize=8,
                        linestyle="",
                        markeredgecolor="k",
                        markerfacecolor=color,
                    )
                )
                if options.fit_mode:
                    table = draw_data_table(point_data, display.plot_ax)

            await display.canvas.draw_send.send(plot_point_draw_fn)

    async def mpl_img_pick_event_task(task_status):
        send_chan, recv_chan = trio.open_memory_channel(inf)
        task_status.started(send_chan)
        async for mouseevent in recv_chan:
            if mouseevent.button != MouseButton.LEFT:
                continue
            point = ImagePoint.from_data(mouseevent.xdata, mouseevent.ydata, axesimage)
            shift_held = "shift" in mouseevent.modifiers
            if shift_held and point in existing_points:
                continue
            await nursery.start(
                plot_curve_response, point, display.options, not shift_held
            )

    async def mpl_motion_event_task(task_status):
        send_chan, recv_chan = trio.open_memory_channel(inf)
        task_status.started(send_chan)
        # Peel root_coords out of MouseEvent.guiEvent in MPL callback
        async for mouseevent, root_coords in recv_chan:
            if mouseevent.inaxes is None:
                tooltip_send_chan.send_nowait(TOOLTIP_CANCEL)
                continue
            elif display is None or mouseevent.name != "motion_notify_event":
                continue
            elif mouseevent.inaxes is display.img_ax:
                tooltip_send_chan.send_nowait((*root_coords, display.img_ax_tip_text))
                if "ctrl" in mouseevent.modifiers:
                    pick_send_chan.send_nowait(mouseevent)
            elif mouseevent.inaxes is display.plot_ax:
                tooltip_send_chan.send_nowait((*root_coords, display.plot_ax_tip_text))

    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        spinner_scope = await nursery.start(
            async_tools.spinner_task, display.spinner_start, display.spinner_stop
        )
        tooltip_send_chan = await nursery.start(
            tooltip_task, display.show_tooltip, display.hide_tooltip, 2, 3
        )
        pick_send_chan = await nursery.start(mpl_img_pick_event_task)
        motion_send_chan = await nursery.start(mpl_motion_event_task)
        redraw_send_chan = await nursery.start(redraw_existing_points_task)

        display.navbar.teach_navbar_to_use_trio(
            nursery=nursery,
            get_image_names=partial(display.image_name_menu.cget, "values"),
            get_image_by_name=opened_fvol.get_image,
            point_data=point_data,
        )
        display.canvas.pipe_events_to_trio(
            spinner_scope, motion_send_chan, pick_send_chan, tooltip_send_chan
        )
        display.teach_display_to_use_trio(
            nursery,
            redraw_send_chan,
            calc_prop_map_callback,
            change_cmap_callback,
            manipulate_callback,
            change_image_callback,
        )
        # This causes the initial plotting of figures after next checkpoint
        display.reset_image_name_menu(opened_fvol.image_names)
        display.image_name_strvar.set(opened_fvol.initial_image_name)


def draw_data_table(point_data, ax: Axes):
    assert point_data
    if len(point_data) == 1:
        data = next(iter(point_data.values()))
        exp = np.log10(data.beta[0])
        prefix, fac = {0: ("G", 1), 1: ("M", 1e3), 2: ("k", 1e6)}.get(
            (-exp + 2.7) // 3, ("", 1)
        )
        colLabels = [
            f"$M$ ({prefix}Pa)",
            r"${dM}/{dk} \times {k}/{M}$",
            "$F_{adh}$ (nN)",
            "d (nm)",
            "δ (nm)",
            "d/δ",
            "SSE",
        ]
        table: Table = ax.table(
            [
                [
                    "{:.2f}±{:.2f}".format(data.beta[0] * fac, data.beta_err[0] * fac),
                    "{:.2e}".format(data.sens[0]),
                    "{:.2f}±{:.2f}".format(data.beta[1], data.beta_err[1]),
                    "{:.2f}".format(data.defl),
                    "{:.2f}".format(data.ind),
                    "{:.2f}".format(data.defl / data.ind),
                    "{:.2f}".format(data.sse),
                ],
            ],
            loc="top",
            colLabels=colLabels,
            colLoc="right",
        )
    else:
        m, sens, fadh, defl, ind, rat = np.transpose(
            [
                (
                    data.beta[0],
                    data.sens[0],
                    data.beta[1],
                    data.defl,
                    data.ind,
                    data.defl / data.ind,
                )
                for data in point_data.values()
            ],
        )
        exp = np.log10(np.mean(m))
        prefix, fac = {0: ("G", 1), 1: ("M", 1e3), 2: ("k", 1e6)}.get(
            (-exp + 2.7) // 3, ("", 1)
        )
        colLabels = [
            f"$M$ ({prefix}Pa)",
            r"${dM}/{dk} \times {k}/{M}$",
            "$F_{adh}$ (nN)",
            "d (nm)",
            "δ (nm)",
            "d/δ",
        ]
        table = ax.table(
            [
                [
                    "{:.2f}±{:.2f}".format(np.mean(m) * fac, np.std(m, ddof=1) * fac),
                    "{:.2e}±{:.2e}".format(np.mean(sens), np.std(sens, ddof=1)),
                    "{:.2f}±{:.2f}".format(np.mean(fadh), np.std(fadh, ddof=1)),
                    "{:.2f}±{:.2f}".format(np.mean(defl), np.std(defl, ddof=1)),
                    "{:.2f}±{:.2f}".format(np.mean(ind), np.std(ind, ddof=1)),
                    "{:.2f}±{:.2f}".format(np.mean(rat), np.std(rat, ddof=1)),
                ],
            ],
            loc="top",
            colLabels=colLabels,
            colLoc="right",
        )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(len(colLabels)))
    return table


def draw_force_curve(data: ForceCurveData, plot_ax, options: ForceCurveOptions):
    artists = []
    aex = artists.extend
    aap = artists.append
    if options.disp_kind == DispKind.zd:
        plot_ax.set_xlabel("Z piezo (nm)")
        plot_ax.set_ylabel("Cantilever deflection (nm)")
        aex(plot_ax.plot(data.zxr[0], data.dxr[0], label="Extend"))
        aex(plot_ax.plot(data.zxr[1], data.dxr[1], label="Retract"))
        if options.fit_mode == calculation.FitMode.BOTH:
            aex(plot_ax.plot(np.concatenate(data.zxr), data.d_fit, "--", label="Model"))
        elif options.fit_mode:
            aex(
                plot_ax.plot(
                    data.zxr[options.fit_mode - 1], data.d_fit, "--", label="Model"
                )
            )
        if options.fit_mode:
            aap(
                plot_ax.axvline(
                    data.z_tru, ls=":", c=artists[0].get_color(), label="Surface Z"
                )
            )
    elif options.disp_kind == DispKind.δf:
        plot_ax.set_xlabel("Indentation depth (nm)")
        plot_ax.set_ylabel("Indentation force (nN)")
        if options.fit_mode:
            f_fit = data.f_fit - data.beta[3]
            aex(
                plot_ax.plot(
                    data.deltaxr[0] - data.beta[2],
                    data.fxr[0] - data.beta[3],
                    label="Extend",
                )
            )
            aex(
                plot_ax.plot(
                    data.deltaxr[1] - data.beta[2],
                    data.fxr[1] - data.beta[3],
                    label="Retract",
                )
            )
            if options.fit_mode == calculation.FitMode.BOTH:
                aex(
                    plot_ax.plot(
                        np.concatenate(data.deltaxr) - data.beta[2],
                        f_fit,
                        "--",
                        label="Model",
                    )
                )
            else:
                aex(
                    plot_ax.plot(
                        data.deltaxr[options.fit_mode - 1] - data.beta[2],
                        f_fit,
                        "--",
                        label="Model",
                    )
                )

            mopts = dict(
                marker="X",
                markersize=8,
                linestyle="",
                markeredgecolor="k",
                markerfacecolor="k",
            )
            aex(
                plot_ax.plot(
                    [
                        data.ind + data.mindelta - data.beta[2],
                        data.mindelta - data.beta[2],
                    ],
                    [data.defl * options.k - data.beta[1], -data.beta[1]],
                    label="Max/Crit",
                    **mopts,
                )
            )
        else:
            aex(plot_ax.plot(data.deltaxr[0], data.fxr[0], label="Extend"))
            aex(plot_ax.plot(data.deltaxr[1], data.fxr[1], label="Retract"))
    elif options.disp_kind == DispKind.td:
        plot_ax.set_xlabel("Time (ms)")
        plot_ax.set_ylabel("Deflection (nm)")
        # TODO: mismatching lengths here
        aex(plot_ax.plot(data.txr[0], data.dxr[0], label="Extend"))
        aex(plot_ax.plot(data.txr[1], data.dxr[1], label="Retract"))
        if options.fit_mode == calculation.FitMode.BOTH:
            aex(plot_ax.plot(np.concatenate(data.txr), data.d_fit, label="Model"))
        elif options.fit_mode:
            aex(plot_ax.plot(data.txr[options.fit_mode - 1], data.d_fit, label="Model"))

    else:
        raise ValueError("Unknown DispKind: ", options.disp_kind)
    plot_ax.legend(handles=artists)
    return artists, artists[0].get_color()


def calculate_force_data(zxr, dxr, t_step, options, cancel_poller=bool):
    cancel_poller()
    npts = sum(map(len, zxr))
    t_step *= 1000
    rnpts = calculation.RESAMPLE_NPTS
    if npts > rnpts:
        split = len(zxr[0]) * rnpts // npts
        z = calculation.resample_dset(np.concatenate(zxr), rnpts, True)
        cancel_poller()
        d = calculation.resample_dset(np.concatenate(dxr), rnpts, True)
        cancel_poller()
        zxr = z[:split], z[split:]
        dxr = d[:split], d[split:]
        t_step *= npts / rnpts
        npts = rnpts
    # Transform data to model units
    t = np.linspace(0, npts * t_step, num=npts, endpoint=False, dtype=dxr[0].dtype)
    txr = t[: len(dxr[0])], t[len(dxr[0]) :]
    assert npts == sum(map(len, txr))
    fxr = tuple(map(np.multiply, dxr, (options.k,) * len(dxr)))
    deltaxr = tuple(map(np.subtract, zxr, dxr))

    if not options.fit_mode:
        return ForceCurveData(zxr=zxr, dxr=dxr, txr=txr, fxr=fxr, deltaxr=deltaxr)

    if options.fit_mode == calculation.FitMode.EXTEND:
        delta, f, split = deltaxr[0], fxr[0], None
    elif options.fit_mode == calculation.FitMode.RETRACT:
        delta, f, split = deltaxr[1], fxr[1], None
    elif options.fit_mode == calculation.FitMode.BOTH:
        delta, f, split = np.concatenate(deltaxr), np.concatenate(fxr), len(deltaxr[0])
    else:
        raise ValueError("Unknown fit_mode: ", options.fit_mode)

    cancel_poller()
    optionsdict = attrs.asdict(options)
    beta, beta_err, sse, calc_fun = calculation.fitfun(
        delta, f, **optionsdict, cancel_poller=cancel_poller, split=split
    )
    f_fit = calc_fun(delta, *beta)
    d_fit = f_fit / options.k
    cancel_poller()

    eps = 1e-3
    delta_new, f_new, k_new = calculation.perturb_k(delta, f, eps, options.k)
    optionsdict.pop("k")
    beta_perturb = calculation.fitfun(
        delta_new,
        f_new,
        k_new,
        **optionsdict,
        cancel_poller=cancel_poller,
        split=split,
    )[0]
    sens = (beta_perturb - beta) / beta / eps
    if np.all(np.isfinite(beta)):
        (
            deflection,
            indentation,
            z_true_surface,
            mindelta,
        ) = calculation.calc_def_ind_ztru(f, beta, **attrs.asdict(options), split=split)
    else:
        deflection, indentation, z_true_surface, mindelta = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    return ForceCurveData(
        zxr=zxr,
        dxr=dxr,
        fxr=fxr,
        deltaxr=deltaxr,
        txr=txr,
        beta=beta,
        beta_err=beta_err,
        calc_fun=calc_fun,
        f_fit=f_fit,
        d_fit=d_fit,
        defl=deflection,
        ind=indentation,
        z_tru=z_true_surface,
        mindelta=mindelta,
        sens=sens,
        sse=sse,
    )


async def open_task(root, nursery):
    """Open files using a dialog box, then launch a task for each"""
    # Choose file
    paths = await trio.to_thread.run_sync(
        partial(
            tk.filedialog.askopenfilenames,
            master=root,
            filetypes=[
                ("AFM Data", "*.h5 *.ARDF *.spm *.pfc"),
                ("AR HDF5", "*.h5"),
                ("ARDF", "*.ARDF"),
                ("Nanoscope", "*.spm *.pfc"),
            ],
        )
    )

    for path in paths:
        nursery.start_soon(open_one, root, path)


async def open_one(root, path):
    """Open the supplied path and create a window for data analysis"""
    path = pathlib.Path(path)

    # choose handler based on file suffix
    suffix = path.suffix.lower()
    # if suffix == ".ardf":
    #     with trio.CancelScope() as cscope, tqdm_tk(
    #         tk_parent=root,
    #         cancel_callback=cscope.cancel,
    #         total=100,
    #         unit="%",
    #         leave=False,
    #         smoothing=0.1,
    #         # smoothing_time=0.5,
    #         miniters=1,
    #     ) as pbar:
    #         try:
    #             path = await data_readers.convert_ardf(path, pbar=pbar)
    #         except FileNotFoundError as e:
    #             await trio.to_thread.run_sync(
    #                 partial(
    #                     tkinter.messagebox.showerror,
    #                     master=root,
    #                     title="Converter not found",
    #                     message=str(e),
    #                 )
    #             )
    #             return
    #     if cscope.cancelled_caught:
    #         return
    #     else:
    #         suffix = path.suffix.lower()

    fvfile_cls, opener = data_readers.SUFFIX_FVFILE_MAP[suffix]
    open_thing = await trio.to_thread.run_sync(opener, path)
    try:
        fvfile = await trio.to_thread.run_sync(fvfile_cls.parse, open_thing)
        opened_fv = data_readers.AsyncFVFile.from_fvfile(path, fvfile)
        display = ForceVolumeTkDisplay(root, path.name, opened_fv.parameters)
        await force_volume_task(display, opened_fv)
        display.destroy()
    finally:
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(open_thing.close)


async def demo_task(root):
    opened_fv = data_readers.DemoForceVolumeFile()
    display = ForceVolumeTkDisplay(root, "Demo", opened_fv.parameters)
    await force_volume_task(display, opened_fv)
    display.destroy()


class MyInstrument(trio.abc.Instrument):
    t = 0
    tau = 1.0
    sleep_time = tau / 2
    wake_time = tau / 2

    @property
    def cycle_time(self):
        return self.wake_time + self.sleep_time

    def before_io_wait(self, timeout):
        t = trio.current_time()
        b = math.exp(-self.cycle_time / self.tau)  # b = 1 - alpha
        self.wake_time = (t - self.t) * (1 - b) + b * self.wake_time
        self.t = t

    def after_io_wait(self, timeout):
        t = trio.current_time()
        b = math.exp(-self.cycle_time / self.tau)  # b = 1 - alpha
        self.sleep_time = (t - self.t) * (1 - b) + b * self.sleep_time
        self.t = t


async def about_task(root):
    """Display and control the About menu

    ☒ Make new Toplevel window
    ☒ Show copyright and version info and maybe something else
    ☒ display cute progress bar spinners to diagnose event loops

    """
    top = tk.Toplevel(root)
    top.wm_title(f"About {__app_name__}")
    message = tk.Message(top, text=__short_license__)
    message.pack()
    task = tk.StringVar(top)
    task_label = ttk.Label(top, textvariable=task)
    task_label.pack()
    thread_cpu = tk.StringVar(top)
    thread_cpu_label = ttk.Label(top, textvariable=thread_cpu)
    thread_cpu_label.pack()
    thread_default = tk.StringVar(top)
    thread_default_label = ttk.Label(top, textvariable=thread_default)
    thread_default_label.pack()
    process = tk.StringVar(top)
    process_label = ttk.Label(top, textvariable=process)
    process_label.pack()
    opts = dict(mode="indeterminate", maximum=80, length=300)
    tk_pbar = ttk.Progressbar(top, **opts)
    tk_pbar.pack()
    trio_pbar = ttk.Progressbar(top, **opts)
    trio_pbar.pack()
    timely_trio_pbar = ttk.Progressbar(top, **opts)
    timely_trio_pbar.pack()

    interval = 32

    async def pbar_runner():
        while True:
            trio_pbar.step()
            await trio.sleep(interval / 1000)

    async def pbar_runner_timely():
        t0 = t = trio.current_time()
        while True:
            v = (trio.current_time() - t0) * 1000 / interval
            timely_trio_pbar["value"] = int(round(v))
            t += interval / 1000
            await trio.sleep_until(t)

    async def state_poller_task():
        t = trio.current_time()
        while True:
            task_stats = trio.lowlevel.current_statistics()
            worker_stats = trio_parallel.default_context_statistics()
            task.set(
                f"Tasks living: {task_stats.tasks_living}\n"
                f"Tasks runnable: {task_stats.tasks_runnable}\n"
                f"Unprocessed callbacks: {task_stats.run_sync_soon_queue_size}\n"
                f"Sleep %: {100 * inst.sleep_time / inst.cycle_time :.1f}"
            )
            thread_cpu.set(
                "CPU-bound tasks:"
                + repr(async_tools.cpu_bound_limiter).split(",")[1][:-1]
            )
            thread_default.set(
                "Default threads:"
                + repr(trio.to_thread.current_default_thread_limiter()).split(",")[1][
                    :-1
                ]
            )
            process.set(
                f"Worker processes: {worker_stats.running_workers}/"
                f"{worker_stats.idle_workers+worker_stats.running_workers}"
            )
            t += interval / 1000
            await trio.sleep_until(t)

    inst = MyInstrument()
    trio.lowlevel.add_instrument(inst)
    # run using tcl event loop
    tk_pbar.start(interval)
    # run using trio
    async with trio.open_nursery() as nursery:
        top.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        nursery.start_soon(state_poller_task)
        nursery.start_soon(pbar_runner)
        nursery.start_soon(pbar_runner_timely)
    trio.lowlevel.remove_instrument(inst)
    top.destroy()


async def main_task(root):
    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        # local names of actions
        quit_callback = nursery.cancel_scope.cancel
        open_callback = partial(nursery.start_soon, open_task, root, nursery)
        demo_callback = partial(nursery.start_soon, demo_task, root)
        about_callback = partial(nursery.start_soon, about_task, root)
        help_action = partial(
            webbrowser.open_new,
            "https://github.com/richardsheridan/magic-afm",
        )

        # calls root.destroy by default
        root.protocol("WM_DELETE_WINDOW", quit_callback)

        # Build menus
        menu_frame = tk.Menu(root, relief="groove", tearoff=False)
        root.config(menu=menu_frame)

        file_menu = tk.Menu(menu_frame, tearoff=False)
        file_menu.add_command(
            label="Open...", accelerator="Ctrl+O", underline=0, command=open_callback
        )
        file_menu.bind("<KeyRelease-o>", func=open_callback)
        root.bind_all("<Control-KeyPress-o>", func=impartial(open_callback))
        file_menu.add_command(label="Demo", underline=0, command=demo_callback)
        file_menu.add_command(
            label="Quit", accelerator="Ctrl+Q", underline=0, command=quit_callback
        )
        file_menu.bind("<KeyRelease-q>", func=quit_callback)
        root.bind_all("<Control-KeyPress-q>", func=impartial(quit_callback))
        menu_frame.add_cascade(label="File", menu=file_menu, underline=0)

        help_menu = tk.Menu(menu_frame, tearoff=False)
        help_menu.add_command(
            label="Open help", accelerator="F1", underline=5, command=help_action
        )
        help_menu.bind("<KeyRelease-h>", func=help_action)
        root.bind_all("<KeyRelease-F1>", func=impartial(help_action))
        # noinspection PyTypeChecker
        help_menu.add_command(
            label="About...", accelerator=None, underline=0, command=about_callback
        )
        help_menu.bind("<KeyRelease-a>", func=about_callback)
        menu_frame.add_cascade(label="Help", menu=help_menu, underline=0)

        trio_parallel.configure_default_context(
            idle_timeout=float("inf"),
            init=calculation.nice_workers,
        )
        # Depending on the system, workers can take up to 30 seconds to finish
        # loading and compiling numba jit stuff. I tried various permutations to
        # warm up the workers and this one seems best for both cached and fresh cases.
        tprs = partial(
            trio_parallel.run_sync,
            limiter=async_tools.cpu_bound_limiter,
            cancellable=True,
        )
        for _ in range(async_tools.cpu_bound_limiter.total_tokens):
            nursery.start_soon(tprs, bool)  # start workers while compiling
        # don't race workers to compile first
        await trio.to_thread.run_sync(calculation.warmup_jit)
        for _ in range(async_tools.cpu_bound_limiter.total_tokens):
            nursery.start_soon(tprs, calculation.warmup_jit)  # only compile cache=false
        await trio.sleep_forever()  # needed if nursery never starts a long running child


def main():
    # make root/parent passing mandatory.
    tk.NoDefaultRoot()
    root = tk.Tk()
    root.wm_resizable(False, False)
    root.wm_minsize(300, 20)
    root.wm_title("Magic AFM")
    # root.wm_iconbitmap("something.ico")
    host = TkHost(root)
    try:
        host.run(main_task, root)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        date = datetime.datetime.now().isoformat().replace(":", ";")
        with open(f"traceback-{date}.dump", "w", encoding="utf8") as file:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=file)
        raise exc
