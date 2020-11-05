"""MagicAFM GUI

This is a trio guest-mode async tkinter graphical interface for AFM users to
calculate indentation ratios and modulus sensitivities for their force curve
data in an intuitive and responsive package. By facilitating these sorts of
calculations, we hope to improve the overall systematic error of reported
modulus maps in the greater AFM nanomechanics community.
"""
__author__ = "Richard J. Sheridan"
__app_name__ = __doc__.split("\n", 1)[0]

import itertools
import sys
from itertools import repeat

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
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

import warnings
import collections
import dataclasses
import datetime
import traceback
import enum
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
from tkinter import ttk
from contextlib import nullcontext
from functools import partial, wraps
from typing import Optional, Callable, ClassVar
from multiprocessing import freeze_support
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except AttributeError:
    pass

import matplotlib
import numpy as np
import outcome
import trio
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg
from matplotlib.contour import ContourSet
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import EngFormatter
from matplotlib.transforms import Bbox, BboxTransform
from tqdm.gui import tqdm_tk
from tqdm.std import TqdmExperimentalWarning
import imageio

from . import calculation, async_tools, data_readers
from .async_tools import trs, LONGEST_IMPERCEPTIBLE_DELAY, tooltip_task, TOOLTIP_CANCEL

warnings.simplefilter("ignore", TqdmExperimentalWarning)

matplotlib.rcParams["savefig.dpi"] = 300
RESAMPLE_NPTS = 512

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


class Box:
    """Hashable mutable objects with mutable content"""

    def __init__(self, content):
        self.content = content


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
    # noinspection NonAsciiCharacters
    δf = enum.auto()


@dataclasses.dataclass
class ForceCurveOptions:
    fit_mode: calculation.FitMode
    disp_kind: DispKind
    k: float
    defl_sens: float
    sync_dist: int
    radius: float
    tau: float


@dataclasses.dataclass
class ForceCurveData:
    z: np.ndarray
    d: np.ndarray
    split: np.ndarray
    f: np.ndarray
    delta: np.ndarray
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


@dataclasses.dataclass(frozen=True)
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
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
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


@dataclasses.dataclass(frozen=True)
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
    def __init__(self, figure, master=None, resize_callback=None):
        self._resize_cancels_pending = set()
        self.draw_send, self.draw_recv = trio.open_memory_channel(float("inf"))

        super().__init__(figure, master, resize_callback)

        self._tkcanvas.configure(background="gray94")  # surely there's a better name...

    def draw_idle(self):
        def null_draw_fn():
            pass

        self.draw_send.send_nowait(null_draw_fn)

    async def idle_draw_task(self, task_status=trio.TASK_STATUS_IGNORED):
        inf = float("inf")
        delay = LONGEST_IMPERCEPTIBLE_DELAY
        task_status.started()
        # One of the slowest processes. Stick everything in a thread.
        while True:
            # Sleep until someone sends artist calls
            draw_fn = await self.draw_recv.receive()
            # Set deadline ASAP so delay scope is accurate
            deadline = trio.current_time() + delay
            async with self.spinner_scope():
                await trs(draw_fn)
                # Batch rapid artist call requests
                # spend roughly equal time building artists and drawing
                with trio.move_on_at(deadline) as delay_scope:
                    async for draw_fn in self.draw_recv:
                        # Cancelling this would drop the fn
                        # would be nicer to prepend back into channel but...
                        delay_scope.deadline = inf
                        await trs(draw_fn)
                        delay_scope.deadline = deadline
                t = trio.current_time()
                await trs(self.draw)
                # previous delay is not great predictor of next delay
                # for now try exponential moving average
                delay = ((trio.current_time() - t) + delay) / 2
            # Funny story, we only want tight layout behavior on resize and
            # a few other special cases, but also we want super().draw()
            # and by extension draw_idle_task to be responsible for calling
            # figure.tight_layout().
            # So everywhere desired, send set_tight_layout(True) in draw_fn
            # and it will be reset here.
            self.figure.set_tight_layout(False)

    def resize(self, event):
        # Swallows initial resize, OK because of change_image_callback
        pass

    async def aresize(self, event):
        for cancel_box in self._resize_cancels_pending:
            cancel_box.content = True
        self._resize_cancels_pending.clear()
        if self._resize_callback is not None:
            self._resize_callback(event)

        # compute desired figure size in inches
        dpival = self.figure.dpi
        width, height = event.width, event.height
        winch = width / dpival
        hinch = height / dpival
        cancel_box = Box(content=False)
        self._resize_cancels_pending.add(cancel_box)

        def draw_fn():
            if cancel_box.content:
                return

            self.figure.set_size_inches(winch, hinch, forward=False)
            self._tkcanvas.delete(self._tkphoto)
            self._tkphoto = tk.PhotoImage(
                master=self._tkcanvas, width=int(width), height=int(height)
            )
            self._tkcanvas.create_image(
                int(width / 2), int(height / 2), image=self._tkphoto,
            )
            self.figure.set_tight_layout(True)
            self.resize_event()  # draw_idle called in here
            self._resize_cancels_pending.discard(cancel_box)

        await self.draw_send.send(draw_fn)

    def teach_canvas_to_use_trio(self, nursery, spinner_scope, async_motion_pick_fn):
        self._tkcanvas.bind("<Configure>", partial(nursery.start_soon, self.aresize))
        self.spinner_scope = spinner_scope
        self.mpl_connect(
            "motion_notify_event", partial(nursery.start_soon, async_motion_pick_fn),
        )
        self.mpl_connect("pick_event", partial(nursery.start_soon, async_motion_pick_fn))


class AsyncNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        self.toolitems += (("Export", "Export calculated maps", "filesave", "export_calculations"),)
        self._prev_filename = ""
        super().__init__(canvas, window)

    def teach_navbar_to_use_trio(self, nursery, get_image_names, get_image_by_name):
        self._parent_nursery = nursery
        self._get_image_names = get_image_names
        self._get_image_by_name = get_image_by_name
        self._wait_cursor_for_draw_cm = nullcontext

    def export_calculations(self):
        # This method is bound early in init so can't use the usual trick of swapping
        # the callback function during the teaching step
        self._parent_nursery.start_soon(self._aexport_calculations)

    async def _aexport_calculations(self):
        ### it's also possible to export image stacks but need a way to indicate that
        import os

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
        fname = await trs(
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
        for image_name in self._get_image_names():
            if image_name.startswith("Calc"):
                exporter = exporter_map.get(ext, imageio.imwrite)
                image, scandown = await self._get_image_by_name(image_name)
                if not scandown:
                    image = image[::-1]
                try:
                    await trio.to_thread.run_sync(
                        exporter, root + "_" + image_name[4:] + ext, image,
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
    def __init__(self, root):
        self.root = root
        self._q = collections.deque()
        self._tk_func_name = root.register(self._tk_func)

    def _tk_func(self):
        self._q.popleft()()

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
        # self.root.after(0, func) # does a fairly intensive wrapping to each func
        self._q.append(func)
        self.root.call("after", "idle", self._tk_func_name)

    def run_sync_soon_not_threadsafe(self, func):
        """Use Tcl "after" command to schedule a function call from the main thread

        If .call is called from the Tcl thread, the locking and sending are optimized away
        so it should be fast enough.

        The incantation `"after idle after 0" <https://wiki.tcl-lang.org/page/after#096aeab6629eae8b244ae2eb2000869fbe377fa988d192db5cf63defd3d8c061>`_ avoids blocking the normal event queue with
        an unending stream of idle tasks.
        """
        self._q.append(func)
        self.root.call("after", "idle", "after", 0, self._tk_func_name)

    def done_callback(self, outcome_):
        """End the Tk app.

        Only really ends if self.root is truly the parent of all other Tk objects!
        thus tk.NoDefaultRoot
        """
        print(f"Trio shutdown. Outcome: {outcome_}")
        if isinstance(outcome_, outcome.Error):
            date = datetime.datetime.now().isoformat().replace(":", ";")
            with open(f"traceback-{date}.dump", "w") as file:
                exc = outcome_.error
                traceback.print_exception(
                    type(exc), exc, exc.__traceback__, file=file,
                )
                traceback.print_exception(
                    type(exc), exc, exc.__traceback__,
                )
        self.root.destroy()


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
    def __init__(self, root, name, initial_values: data_readers.ForceVolumeParams, **kwargs):
        self._nursery = None
        self._prev_defl_sens = 1.0
        self._prev_k = 1.0
        self._prev_sync_dist = 0
        self._prev_radius = 20.0
        self._prev_tau = 0.0
        self.tkwindow = window = tk.Toplevel(root, **kwargs)
        window.wm_title(name)

        # Build figure
        self.fig = Figure((9, 2.75), frameon=False)
        self.canvas = AsyncFigureCanvasTkAgg(self.fig, window)
        self.navbar = AsyncNavigationToolbar2Tk(self.canvas, window)
        self.img_ax, self.plot_ax = img_ax, plot_ax = self.fig.subplots(
            1, 2, gridspec_kw=dict(width_ratios=[1, 1.5])
        )
        self.fig.subplots_adjust(top=0.85)
        img_ax.set_anchor("W")
        img_ax.set_facecolor((0.8, 0, 0))  # scary red for NaN values of images
        # Need to pre-load something into these labels for change_image_callback
        plot_ax.set_xlabel(" ")
        plot_ax.set_ylabel(" ")
        plot_ax.set_ylim([-1000, 1000])
        plot_ax.set_xlim([-1000, 1000])

        # Options and buttons
        self.options_frame = ttk.Frame(root)

        image_opts_frame = ttk.Frame(self.options_frame)
        image_name_labelframe = ttk.Labelframe(image_opts_frame, text="Image")
        self.image_name_strvar = tk.StringVar(image_name_labelframe, value="Choose an image...")
        self.image_name_menu = ttk.Combobox(
            image_name_labelframe, width=12, state="readonly", textvariable=self.image_name_strvar,
        )
        unbind_mousewheel(self.image_name_menu)
        self.image_name_menu.pack(fill="x")
        self.reset_image_name_menu(initial_values.image_names)
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
        self.disp_kind_intvar.trace_add("write", self.change_disp_kind_callback)
        disp_zd_button = ttk.Radiobutton(
            disp_labelframe,
            text="d vs. z",
            value=DispKind.zd.value,
            variable=self.disp_kind_intvar,
            padding=4,
        )
        disp_zd_button.pack(side="left")
        disp_deltaf_button = ttk.Radiobutton(
            disp_labelframe,
            text="f vs. δ",
            value=DispKind.δf.value,
            variable=self.disp_kind_intvar,
            padding=4,
        )
        disp_deltaf_button.pack(side="left")
        disp_labelframe.grid(row=0, column=0)

        preproc_labelframe = ttk.Labelframe(self.options_frame, text="Preprocessing")
        self.defl_sens_strvar = tk.StringVar(preproc_labelframe)
        self.defl_sens_strvar.trace_add("write", self.defl_sens_callback)
        defl_sens_sbox = ttk.Spinbox(
            preproc_labelframe,
            from_=0,
            to=1e3,
            increment=0.1,
            format="%0.1f",
            width=6,
            textvariable=self.defl_sens_strvar,
        )
        defl_sens_sbox.set(initial_values.defl_sens)
        defl_sens_sbox.grid(row=0, column=2, sticky="E")
        defl_sens_label = ttk.Label(preproc_labelframe, text="Deflection Sens.", justify="left")
        defl_sens_label.grid(row=0, column=0, columnspan=2, sticky="W")
        self.spring_const_strvar = tk.StringVar(preproc_labelframe)
        self.spring_const_strvar.trace_add("write", self.spring_const_callback)
        spring_const_sbox = ttk.Spinbox(
            preproc_labelframe,
            from_=0,
            to=1e3,
            increment=0.1,
            format="%0.1f",
            width=6,
            textvariable=self.spring_const_strvar,
        )
        spring_const_sbox.set(initial_values.k)
        spring_const_sbox.grid(row=1, column=2, sticky="E")
        spring_const_label = ttk.Label(preproc_labelframe, text="Spring Constant", justify="left")
        spring_const_label.grid(row=1, column=0, columnspan=2, sticky="W")
        self.sync_dist_strvar = tk.StringVar(preproc_labelframe)
        self.sync_dist_strvar.trace_add("write", self.sync_dist_callback)
        sync_dist_sbox = ttk.Spinbox(
            preproc_labelframe,
            from_=-initial_values.npts,
            to=initial_values.npts,
            increment=1,
            format="%0.0f",
            width=6,
            textvariable=self.sync_dist_strvar,
        )
        sync_dist_sbox.set(initial_values.sync_dist)
        sync_dist_sbox.grid(row=2, column=2, sticky="E")
        sync_dist_label = ttk.Label(preproc_labelframe, text="Sync Distance", justify="left")
        sync_dist_label.grid(row=2, column=0, columnspan=2, sticky="W")
        preproc_labelframe.grid(row=0, column=1, sticky="EW")
        preproc_labelframe.grid_columnconfigure(1, weight=1)

        fit_labelframe = ttk.Labelframe(self.options_frame, text="Fit parameters")
        self.fit_intvar = tk.IntVar(fit_labelframe, value=calculation.FitMode.SKIP.value)
        self.fit_intvar.trace_add("write", self.change_fit_kind_callback)
        fit_skip_button = ttk.Radiobutton(
            fit_labelframe,
            text="Skip",
            value=calculation.FitMode.SKIP.value,
            variable=self.fit_intvar,
        )
        fit_skip_button.grid(row=0, column=0)
        fit_ext_button = ttk.Radiobutton(
            fit_labelframe,
            text="Extend",
            value=calculation.FitMode.EXTEND.value,
            variable=self.fit_intvar,
        )
        fit_ext_button.grid(row=0, column=1)
        fit_ret_button = ttk.Radiobutton(
            fit_labelframe,
            text="Retract",
            value=calculation.FitMode.RETRACT.value,
            variable=self.fit_intvar,
        )
        fit_ret_button.grid(row=0, column=2)

        fit_radius_label = ttk.Label(fit_labelframe, text="Tip radius (nm)")
        fit_radius_label.grid(row=1, column=0, columnspan=2, sticky="W")
        self.radius_strvar = tk.StringVar(fit_labelframe)
        self.radius_strvar.trace_add("write", self.radius_callback)
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
        self.fit_radius_sbox.grid(row=1, column=2, sticky="E")
        fit_tau_label = ttk.Label(fit_labelframe, text="DMT-JKR (0-1)", justify="left")
        fit_tau_label.grid(row=2, column=0, columnspan=2, sticky="W")
        self.tau_strvar = tk.StringVar(fit_labelframe)
        self.tau_strvar.trace_add("write", self.tau_callback)
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
        self.fit_tau_sbox.grid(row=2, column=2, sticky="E")

        self.calc_props_button = ttk.Button(
            fit_labelframe, text="Calculate Property Maps", state="disabled"
        )
        self.calc_props_button.grid(row=3, column=0, columnspan=3)
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
        self.colorbar_ax_tip_text = c(
            """Colorbar
            
            Middle or alt-left click: toggle edit mode (default: off)
            Left click: set color scale maximum
            Right click: set color scale minimum
            Control: set continuously"""
        )
        self.plot_ax_tip_text = c(
            """Force Curve
        
            No actions yet!"""
        )
        self.tipwindow = tipwindow = tk.Toplevel(window)
        tipwindow.wm_withdraw()
        tipwindow.wm_overrideredirect(1)
        try:
            # For Mac OS
            tipwindow.tk.call(
                "::tk::unsupported::MacWindowStyle", "style", tipwindow._w, "help", "noActivates"
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

    def destroy(self):
        self.options_frame.destroy()
        self.tkwindow.withdraw()  # weird navbar hiccup on close
        self.tkwindow.destroy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    @property
    def options(self):
        return ForceCurveOptions(
            fit_mode=self.fit_intvar.get(),
            disp_kind=self.disp_kind_intvar.get(),
            k=float(self.spring_const_strvar.get()),
            defl_sens=float(self.defl_sens_strvar.get()),
            radius=float(self.fit_radius_sbox.get()),
            tau=float(self.fit_tau_sbox.get()),
            sync_dist=int(self.sync_dist_strvar.get()),
        )

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

    async def teach_display_to_use_trio(
        self,
        nursery,
        redraw_existing_points,
        redraw_existing_points_tight,
        calc_prop_map_callback,
        change_cmap_callback,
        manipulate_callback,
        change_image_callback,
    ):
        self.tkwindow.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        self.calc_props_button.configure(
            command=partial(nursery.start_soon, calc_prop_map_callback,)
        )
        self.colormap_strvar.trace_add(
            "write", impartial(partial(nursery.start_soon, change_cmap_callback))
        )
        self.manipulate_strvar.trace_add(
            "write", impartial(partial(nursery.start_soon, manipulate_callback))
        )
        self.image_name_strvar.trace_add(
            "write", impartial(partial(nursery.start_soon, change_image_callback))
        )
        self.replot = partial(nursery.start_soon, redraw_existing_points)
        self.replot_tight = partial(nursery.start_soon, redraw_existing_points_tight)

        await nursery.start(self.canvas.idle_draw_task)

    def defl_sens_callback(self, *args):
        try:
            self._prev_defl_sens = float(self.defl_sens_strvar.get())
        except ValueError:
            self.defl_sens_strvar.set(str(self._prev_defl_sens))
        else:
            self.replot()

    def spring_const_callback(self, *args):
        try:
            self._prev_k = float(self.spring_const_strvar.get())
        except ValueError:
            self.spring_const_strvar.set(str(self._prev_k))
        else:
            self.replot()

    def sync_dist_callback(self, *args):
        try:
            self._prev_sync_dist = int(self.sync_dist_strvar.get())
        except ValueError:
            self.sync_dist_strvar.set(str(self._prev_sync_dist))
        else:
            self.replot()

    def radius_callback(self, *args):
        try:
            self._prev_radius = float(self.radius_strvar.get())
        except ValueError:
            self.radius_strvar.set(str(self._prev_radius))
        else:
            self.replot()

    def tau_callback(self, *args):
        try:
            self._prev_tau = float(self.tau_strvar.get())
        except ValueError:
            self.tau_strvar.set(str(self._prev_tau))
        else:
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


async def force_volume_task(display, opened_fvol):
    # plot_curve_event_response
    plot_curve_cancels_pending = set()
    artists = []
    existing_points = set()

    # set in change_image_callback
    colorbar: Optional[Colorbar] = None
    axesimage: Optional[AxesImage] = None
    scandown: Optional[bool] = None
    image_stats: Optional[ImageStats] = None
    unit: Optional[str] = None

    async def calc_prop_map_callback():
        options = display.options
        optionsdict = dataclasses.asdict(options)
        img_shape = opened_fvol.shape
        ncurves = img_shape[0] * img_shape[1]
        if not options.fit_mode:
            raise ValueError("Property map button should have been disabled")

        async with spinner_scope() as cancel_scope:
            # assign pbar and progress_image ASAP in case of cancel
            progress_image: matplotlib.image.AxesImage = display.img_ax.imshow(
                np.zeros(img_shape + (4,), dtype="f4"), extent=axesimage.get_extent()
            )  # transparent initial image, no need to draw
            pbar = tqdm_tk(
                total=ncurves,
                desc="Loading and resampling force curves...",
                smoothing_time=2,
                mininterval=LONGEST_IMPERCEPTIBLE_DELAY * 2,
                unit=" curves",
                tk_parent=display.tkwindow,
                grab=False,
                leave=False,
                cancel_callback=cancel_scope.cancel,
            )

            npts, split = opened_fvol.npts, opened_fvol.split
            resample = npts > RESAMPLE_NPTS
            if resample:
                split = split * RESAMPLE_NPTS // npts
                npts = RESAMPLE_NPTS
            if options.fit_mode == calculation.FitMode.EXTEND:
                sl = slice(split)
                segment_npts = split
            elif options.fit_mode == calculation.FitMode.RETRACT:
                sl = slice(split, None)
                segment_npts = npts - split
            else:
                raise ValueError("Unknown fit_mode: ", options.fit_mode)

            to_resample = []
            delta = np.empty((ncurves, segment_npts), np.float32)
            f = np.empty((ncurves, segment_npts), np.float32)

            @async_tools.spawn_limit(async_tools.cpu_bound_limiter)
            async def resample_helper(chunk):
                i, *chunk = zip(*chunk)
                chunk = np.concatenate(chunk, 0)
                chunk = await trio.to_thread.run_sync(
                    calculation.resample_dset,
                    chunk,
                    npts,
                    True,
                    cancellable=True,
                    limiter=trio.CapacityLimiter(1),
                )
                z, d = np.split(chunk[:, sl], 2)
                del chunk
                delta[i, :] = z - d
                f[i, :] = d * options.k
                pbar.update(len(i))

            async with trio.open_nursery() as nursery:
                for i in range(ncurves):
                    z, d = await opened_fvol.get_force_curve(*np.unravel_index(i, img_shape))
                    if resample:
                        to_resample.append((i, z, d))
                        if len(to_resample) == 128:
                            async with async_tools.cpu_bound_limiter:
                                nursery.start_soon(resample_helper, to_resample.copy())
                            to_resample.clear()
                    else:
                        z = z[sl]
                        d = d[sl]
                        delta[i, :] = z - d
                        f[i, :] = d * options.k
                        pbar.update()
                if to_resample:
                    nursery.start_soon(resample_helper, to_resample)
            del to_resample, i, z, d

            pbar.set_description_str("Fitting force curves...")
            pbar.unit = " fits"
            pbar.avg_time = None
            pbar.reset(total=ncurves)
            pbar.smoothing_time = 0.1
            property_map = np.empty(
                ncurves, dtype=np.dtype([(name, "f4") for name in calculation.PROPERTY_UNITS_DICT]),
            )
            progress_array = progress_image.get_array()

            def draw_fn():
                progress_image.changed()
                progress_image.pchanged()

            async with trio.open_nursery() as nursery:
                property_aiter = await nursery.start(
                    async_tools.to_process_map_unordered,
                    calculation.calc_properties_imap,
                    zip(delta, f, range(ncurves), repeat(optionsdict)),
                    16,  # chunksize, but sadly can't use kwarg
                )
                t0 = trio.current_time()
                check_time = True
                async for i, properties in property_aiter:
                    if check_time and (trio.current_time() - t0 > 10):
                        check_time = False
                        pbar.smoothing_time = 5
                    if properties is None:
                        property_map[i] = np.nan
                        color = (1, 0, 0, 0.5)
                    else:
                        property_map[i] = properties
                        color = (0, 1, 0, 0.5)

                    pbar.update()
                    r, c = np.unravel_index(i, img_shape)
                    progress_array[r, c, :] = color
                    display.canvas.draw_send.send_nowait(draw_fn)

        def draw_fn():
            progress_image.remove()

        await display.canvas.draw_send.send(draw_fn)
        pbar.close()

        if cancel_scope.cancelled_caught:
            return

        combobox_values = list(display.image_name_menu.cget("values"))
        property_map = property_map.reshape((*img_shape, -1,))

        # Actually write out results to external world
        if options.fit_mode == calculation.FitMode.EXTEND:
            extret = "Ext"
        else:
            extret = "Ret"
        for name in calculation.PROPERTY_UNITS_DICT:
            opened_fvol.add_image(
                image_name="Calc" + extret + name,
                units=calculation.PROPERTY_UNITS_DICT[name],
                scandown=scandown,
                image=property_map[name].squeeze(),
            )
            if "Calc" + extret + name not in combobox_values:
                combobox_values.append("Calc" + extret + name)

        display.reset_image_name_menu(combobox_values)

    async def change_cmap_callback():
        colormap_name = display.colormap_strvar.get()
        # save old clim
        clim = axesimage.get_clim()

        def draw_fn():
            # prevent cbar from getting expanded
            axesimage.set_clim(image_stats.min, image_stats.max)
            # actually change cmap
            axesimage.set_cmap(colormap_name)
            # reset everything
            customize_colorbar(colorbar, *clim, unit=unit)
            if colorbar.frozen:
                expand_colorbar(colorbar)

        await display.canvas.draw_send.send(draw_fn)

    async def change_image_callback():
        nonlocal scandown, image_stats, unit
        image_name = display.image_name_strvar.get()
        cmap = display.colormap_strvar.get()
        image_array, scandown = await opened_fvol.get_image(image_name)
        image_stats = ImageStats.from_array(image_array)
        unit = opened_fvol.get_image_units(image_name)

        scansize = opened_fvol.scansize
        s = (scansize + scansize / len(image_array)) // 2

        def draw_fn():
            nonlocal axesimage, colorbar
            if colorbar is not None:
                colorbar.remove()
            if axesimage is not None:
                axesimage.remove()
            display.img_ax.set_title(image_name)
            display.img_ax.set_ylabel("Y piezo (nm)")
            display.img_ax.set_xlabel("X piezo (nm)")

            axesimage = display.img_ax.imshow(
                image_array,
                origin="lower" if scandown else "upper",
                extent=(-s, s, -s, s,),
                picker=True,
                cmap=cmap,
            )
            axesimage.get_array().fill_value = np.nan

            colorbar = display.fig.colorbar(axesimage, ax=display.img_ax, use_gridspec=True,)
            display.navbar.update()  # let navbar catch new cax in fig for tooltips
            customize_colorbar(colorbar, image_stats.q01, image_stats.q99, unit)

            # start frozen
            colorbar.frozen = True
            expand_colorbar(colorbar)

            # noinspection PyTypeChecker
            display.fig.set_tight_layout(True)

        await display.canvas.draw_send.send(draw_fn)

    @async_tools.spawn_limit(trio.CapacityLimiter(1))
    async def redraw_existing_points():
        def draw_fn():
            for artist in artists:
                artist.remove()
            display.plot_ax.relim()
            display.plot_ax.set_prop_cycle(None)
            artists.clear()

        await display.canvas.draw_send.send(draw_fn)
        async with spinner_scope():
            for point in existing_points:
                await plot_curve_response(point, False)

    async def redraw_existing_points_tight():
        await redraw_existing_points()

        def draw_fn():
            # noinspection PyTypeChecker
            display.fig.set_tight_layout(True)

        await display.canvas.draw_send.send(draw_fn)

    async def manipulate_callback():
        manip_name = display.manipulate_strvar.get()
        current_name = display.image_name_strvar.get()
        current_names = list(display.image_name_menu.cget("values"))
        name = "Calc" + manip_name + current_name
        if name not in opened_fvol.image_names:
            manip_fn = calculation.MANIPULATIONS[manip_name]
            async with spinner_scope():
                manip_img = await trs(manip_fn, axesimage.get_array())
            opened_fvol.add_image(
                name, unit, scandown, manip_img,
            )
            if name not in current_names:
                current_names.append(name)
            display.reset_image_name_menu(current_names)
        display.image_name_menu.set(name)

    async def colorbar_state_response():
        if colorbar.frozen:
            colorbar.frozen = False
            clim = axesimage.get_clim()

            def draw_fn():
                # make full range colorbar especially solids
                axesimage.set_clim(image_stats.min, image_stats.max)
                # reset customizations
                customize_colorbar(colorbar, *clim)

        else:
            colorbar.frozen = True

            def draw_fn():
                expand_colorbar(colorbar)

        await display.canvas.draw_send.send(draw_fn)

    async def plot_curve_response(point: ImagePoint, clear_previous):
        existing_points.add(point)  # should be before 1st checkpoint

        # XXX: only needed on first plot. Maybe later make optional?
        display.plot_ax.set_autoscale_on(True)

        if clear_previous:
            for cancel_scope in plot_curve_cancels_pending:
                cancel_scope.cancel()
            plot_curve_cancels_pending.clear()
        async with spinner_scope():
            with trio.CancelScope() as cancel_scope:
                plot_curve_cancels_pending.add(cancel_scope)

                # Calculation phase
                # Do a few long-running jobs, likely to be canceled
                options = display.options
                opened_fvol.sync_dist = options.sync_dist
                opened_fvol.defl_sens = options.defl_sens
                force_curve = await opened_fvol.get_force_curve(point.r, point.c)
                data = await trs(
                    calculate_force_data,
                    *force_curve,
                    opened_fvol.split,
                    opened_fvol.npts,
                    options,
                    async_tools.make_cancel_poller(),
                )
                del force_curve  # contained in data
                # XXX: Race condition
                # cancel can occur after the last call to the above cancel poller
                # checkpoint make sure to raise Cancelled if it just happened
                # await trio.sleep(0)

            plot_curve_cancels_pending.discard(cancel_scope)

            if cancel_scope.cancelled_caught:
                existing_points.discard(point)
                return

            def draw_fn():
                # Clearing Phase
                # Clear previous artists and reset plots (faster than .clear()?)
                if clear_previous:
                    for artist in artists:
                        artist.remove()
                    artists.clear()
                    existing_points.clear()

                    existing_points.add(point)
                    display.plot_ax.relim()
                    display.plot_ax.set_prop_cycle(None)

                # Drawing Phase
                # Based options choose plots and collect artists for deletion
                new_artists, color = draw_force_curve(data, display.plot_ax, options)
                artists.extend(new_artists)
                artists.extend(
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
                    table = draw_data_table(data, display.plot_ax,)
                    artists.append(table)

            await display.canvas.draw_send.send(draw_fn)

    async def mpl_pick_motion_event_callback(event):
        mouseevent = getattr(event, "mouseevent", event)
        control_held = event.guiEvent.state & TkState.CONTROL
        shift_held = mouseevent.guiEvent.state & TkState.SHIFT
        if mouseevent.inaxes is None:
            await tooltip_send_chan.send(TOOLTIP_CANCEL)
            return
        elif mouseevent.inaxes is display.img_ax:
            if event.name == "motion_notify_event":
                await tooltip_send_chan.send(
                    (event.guiEvent.x_root, event.guiEvent.y_root, display.img_ax_tip_text)
                )
                if not control_held:
                    return
            if mouseevent.button != MouseButton.LEFT:
                return
            point = ImagePoint.from_data(mouseevent.xdata, mouseevent.ydata, axesimage)
            if shift_held and point in existing_points:
                return
            await plot_curve_response(point, not shift_held)
        elif mouseevent.inaxes is colorbar.ax:
            if event.name == "motion_notify_event":
                await tooltip_send_chan.send(
                    (event.guiEvent.x_root, event.guiEvent.y_root, display.colorbar_ax_tip_text)
                )
                if not control_held:
                    return
            if mouseevent.button == MouseButton.MIDDLE or (
                mouseevent.guiEvent.state & TkState.ALT and mouseevent.button == MouseButton.LEFT
            ):
                await colorbar_state_response()
                return
            if colorbar.frozen:
                return
            if mouseevent.button == MouseButton.LEFT:
                vmin = colorbar.norm.vmin
                vmax = max(mouseevent.ydata, vmin)
            elif mouseevent.button == MouseButton.RIGHT:
                vmax = colorbar.norm.vmax
                vmin = min(mouseevent.ydata, vmax)
            else:
                # discard all others
                return

            # fallthrough for left and right clicks
            def draw_fn():
                colorbar.norm.vmin = vmin
                colorbar.norm.vmax = vmax
                colorbar.solids.set_clim(vmin, vmax)

            await display.canvas.draw_send.send(draw_fn)
        elif mouseevent.inaxes is display.plot_ax:
            if event.name == "motion_notify_event":
                await tooltip_send_chan.send(
                    (event.guiEvent.x_root, event.guiEvent.y_root, display.plot_ax_tip_text)
                )

    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:

        spinner_scope = await nursery.start(
            async_tools.spinner_task, display.spinner_start, display.spinner_stop,
        )

        tooltip_send_chan = await nursery.start(
            tooltip_task, display.show_tooltip, display.hide_tooltip, 1, 2
        )

        display.navbar.teach_navbar_to_use_trio(
            nursery=nursery,
            get_image_names=partial(display.image_name_menu.cget, "values"),
            get_image_by_name=opened_fvol.get_image,
        )
        display.canvas.teach_canvas_to_use_trio(
            nursery=nursery,
            spinner_scope=spinner_scope,
            async_motion_pick_fn=mpl_pick_motion_event_callback,
        )
        # XXX: This is a hacky bugfix, maybe find some way to get this logic into ^^^
        display.canvas.mpl_connect(
            "figure_leave_event",
            impartial(partial(nursery.start_soon, tooltip_send_chan.send, TOOLTIP_CANCEL)),
        )

        await display.teach_display_to_use_trio(
            nursery,
            redraw_existing_points,
            redraw_existing_points_tight,
            calc_prop_map_callback,
            change_cmap_callback,
            manipulate_callback,
            change_image_callback,
        )
        # This causes the initial plotting of figures after next checkpoint
        display.image_name_strvar.set(opened_fvol.initial_image_name)


def customize_colorbar(colorbar, vmin=None, vmax=None, unit=None):
    """Central function to reset colorbar as MPL keeps stomping on our settings"""
    # for coordinate/value display on hover
    colorbar.ax.set_navigate(True)
    # for receiving events
    colorbar.solids.set_picker(True)
    # set range of colors and ticks without changing range of colorbar axes
    if vmin is not None and vmax is not None:
        # other permutations handled internally
        colorbar.solids.set_clim(vmin, vmax)
    # pretty engineering units display
    if unit is not None:
        colorbar.formatter = EngFormatter(unit, places=1)
        colorbar.update_ticks()


def expand_colorbar(colorbar):
    """This is the last part of Colorbar.update_normal()"""
    colorbar.draw_all()
    if isinstance(colorbar.mappable, ContourSet):
        CS = colorbar.mappable
        if not CS.filled:
            colorbar.add_lines(CS)
    colorbar.stale = True
    colorbar.solids.set_picker(True)


def draw_data_table(data, ax):
    exp = np.log10(data.beta[0])
    prefix, fac = {0: ("G", 1), 1: ("M", 1e3), 2: ("k", 1e6),}.get((-exp + 2.7) // 3, ("", 1))
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
                "{:.2f}±{:.2f}".format(data.beta[0] * fac, data.beta_err[0] * fac,),
                "{:.2e}".format(data.sens[0]),
                "{:.2f}±{:.2f}".format(data.beta[1], data.beta_err[1]),
                "{:.2f}".format(data.defl),
                "{:.2f}".format(data.ind),
                "{:.2f}".format(data.defl / data.ind),
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


def draw_force_curve(data, plot_ax, options):
    artists = []
    aex = artists.extend
    aap = artists.append
    if options.disp_kind == DispKind.zd:
        plot_ax.set_xlabel("Z piezo (nm)")
        plot_ax.set_ylabel("Cantilever deflection (nm)")
        aex(plot_ax.plot(data.z[: data.split], data.d[: data.split], label="Extend"))
        aex(plot_ax.plot(data.z[data.split :], data.d[data.split :], label="Retract"))
        if options.fit_mode:
            aex(plot_ax.plot(data.z[data.sl], data.d_fit, "--", label="Model"))
            aap(plot_ax.axvline(data.z_tru, ls=":", c=artists[0].get_color(), label="Surface Z"))
    elif options.disp_kind == DispKind.δf:
        plot_ax.set_xlabel("Indentation depth (nm)")
        plot_ax.set_ylabel("Indentation force (nN)")
        if options.fit_mode:
            delta = data.delta - data.beta[2]
            f = data.f - data.beta[3]
            f_fit = data.f_fit - data.beta[3]
            aex(plot_ax.plot(delta[: data.split], f[: data.split], label="Extend"))
            aex(plot_ax.plot(delta[data.split :], f[data.split :], label="Retract"))
            aex(plot_ax.plot(delta[data.sl], f_fit, "--", label="Model"))
            mopts = dict(
                marker="X", markersize=8, linestyle="", markeredgecolor="k", markerfacecolor="k",
            )
            aex(
                plot_ax.plot(
                    data.ind + data.mindelta - data.beta[2],
                    data.defl * options.k - data.beta[1],
                    label="Max/Crit",
                    **mopts,
                )
            )
            aex(plot_ax.plot(data.mindelta - data.beta[2], -data.beta[1], **mopts,))
        else:
            aex(plot_ax.plot(data.delta[: data.split], data.f[: data.split], label="Extend"))
            aex(plot_ax.plot(data.delta[data.split :], data.f[data.split :], label="Retract"))
    else:
        raise ValueError("Unknown DispKind: ", data.disp_kind)
    aap(plot_ax.legend())
    return artists, artists[0].get_color()


def calculate_force_data(z, d, split, npts, options, cancel_poller=lambda: None):
    cancel_poller()
    if npts > RESAMPLE_NPTS:
        split = split * RESAMPLE_NPTS // npts
        z = calculation.resample_dset(z, RESAMPLE_NPTS, True)
        d = calculation.resample_dset(d, RESAMPLE_NPTS, True)
    # Transform data to model units
    f = d * options.k
    delta = z - d

    if not options.fit_mode:
        return ForceCurveData(split=split, z=z, d=d, f=f, delta=delta,)

    if options.fit_mode == calculation.FitMode.EXTEND:
        sl = slice(split)
    elif options.fit_mode == calculation.FitMode.RETRACT:
        sl = slice(split, None)
    else:
        raise ValueError("Unknown fit_mode: ", options.fit_mode)

    cancel_poller()
    optionsdict = dataclasses.asdict(options)
    beta, beta_err, calc_fun = calculation.fitfun(
        delta[sl], f[sl], **optionsdict, cancel_poller=cancel_poller,
    )
    f_fit = calc_fun(delta[sl], *beta)
    d_fit = f_fit / options.k
    cancel_poller()

    eps = 1e-3
    delta_new, f_new, k_new = calculation.perturb_k(delta, f, eps, options.k)
    optionsdict.pop("k")
    beta_perturb, _, _ = calculation.fitfun(
        delta_new[sl], f_new[sl], k_new, **optionsdict, cancel_poller=cancel_poller,
    )
    sens = (beta_perturb - beta) / beta / eps
    if np.all(np.isfinite(beta)):
        deflection, indentation, z_true_surface, mindelta = calculation.calc_def_ind_ztru(
            f[sl], beta, **dataclasses.asdict(options)
        )
    else:
        deflection, indentation, z_true_surface, mindelta = np.nan, np.nan, np.nan, np.nan
    return ForceCurveData(
        z=z,
        d=d,
        split=split,
        f=f,
        delta=delta,
        sl=sl,
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
    )


async def open_task(root):
    """Open a file using a dialog box, then create a window for data analysis

    """
    # Choose file
    path = await trs(
        partial(
            tk.filedialog.askopenfilename,
            master=root,
            filetypes=[
                ("AFM Data", "*.h5 *.ARDF *.spm *.pfc"),
                ("AR HDF5", "*.h5"),
                ("ARDF", "*.ARDF"),
                ("Nanoscope", "*.spm *.pfc"),
            ],
        )
    )
    if not path:
        return  # Cancelled
    path = trio.Path(path)

    # choose handler based on file suffix
    if path.suffix == ".ARDF":
        with trio.CancelScope() as cscope:
            pbar = tqdm_tk(
                tk_parent=root,
                cancel_callback=cscope.cancel,
                total=100,
                unit="%",
                leave=False,
                smoothing_time=0.5,
                miniters=1,
            )
            try:
                path = await data_readers.convert_ardf(path, pbar=pbar)
            except FileNotFoundError as e:
                await trio.to_thread.run_sync(
                    partial(
                        tkinter.messagebox.showerror,
                        master=root,
                        title="Converter not found",
                        message=str(e),
                    )
                )
                return
        if cscope.cancelled_caught:
            return

    async with data_readers.SUFFIX_FVFILE_MAP[path.suffix](path) as opened_fv:
        with ForceVolumeTkDisplay(root, path.name, opened_fv.parameters) as display:
            await force_volume_task(display, opened_fv)


async def demo_task(root):
    async with data_readers.DemoForceVolumeFile("Demo") as opened_fv:
        with ForceVolumeTkDisplay(root, "Demo", opened_fv.parameters) as display:
            await force_volume_task(display, opened_fv)


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

    interval = 20

    async def pbar_runner():
        while True:
            trio_pbar.step()
            await trio.sleep(interval / 1000)

    async def pbar_runner_timely():
        t = trio.current_time()
        t0 = t
        while True:
            v = (trio.current_time() - t0) * 1000 / interval
            timely_trio_pbar["value"] = int(round(v))
            t = t + interval / 1000
            await trio.sleep_until(t)

    async def state_poller_task():
        while True:
            t = trio.current_time()
            task_stats = trio.lowlevel.current_statistics()
            task.set(
                f"Tasks living: {task_stats.tasks_living}\n"
                f"Tasks runnable: {task_stats.tasks_runnable}\n"
                f"Unprocessed callbacks: {task_stats.run_sync_soon_queue_size}"
            )
            thread_cpu.set(
                "CPU-bound threads:" + repr(async_tools.cpu_bound_limiter).split(",")[1][:-1]
            )
            thread_default.set(
                "Default threads:"
                + repr(trio.to_thread.current_default_thread_limiter()).split(",")[1][:-1]
            )
            process.set(
                "Worker Processes: "
                + str(0 if async_tools.EXECUTOR is None else len(async_tools.EXECUTOR._processes))
            )
            await trio.sleep_until(t + interval / 1000)

    # run using tcl event loop
    tk_pbar.start(interval)
    # run using trio
    async with trio.open_nursery() as nursery:
        top.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        nursery.start_soon(state_poller_task)
        nursery.start_soon(pbar_runner)
        nursery.start_soon(pbar_runner_timely)
    top.destroy()


def open_with_os_default(file_url_etc):
    """Open a string using the OS's default program

    from https://stackoverflow.com/a/61968360/4504950 CC BY-SA 4.0
    """
    import subprocess
    import os

    try:  # should work on Windows
        os.startfile(file_url_etc)
    except AttributeError:
        try:  # should work on MacOS and most linux versions
            subprocess.call(["open", file_url_etc])
        except:
            print("Could not open with OS")


async def main_task(root):
    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        # local names of actions
        quit_callback = nursery.cancel_scope.cancel
        open_callback = partial(nursery.start_soon, open_task, root)
        demo_callback = partial(nursery.start_soon, demo_task, root)
        about_callback = partial(nursery.start_soon, about_task, root)
        help_action = partial(open_with_os_default, "https://github.com/richardsheridan/magic-afm")

        # calls root.destroy by default
        root.protocol("WM_DELETE_WINDOW", quit_callback)

        # Build menus
        menu_frame = tk.Menu(root, relief="groove", tearoff=False)
        root.config(menu=menu_frame)

        file_menu = tk.Menu(menu_frame, tearoff=False)
        file_menu.add_command(
            label="Open...", accelerator="Ctrl+O", underline=0, command=open_callback,
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
            label="Open help", accelerator="F1", underline=5, command=help_action,
        )
        help_menu.bind("<KeyRelease-h>", func=help_action)
        root.bind_all("<KeyRelease-F1>", func=impartial(help_action))
        help_menu.add_command(
            label="About...", accelerator=None, underline=0, command=about_callback,
        )
        help_menu.bind("<KeyRelease-a>", func=about_callback)
        menu_frame.add_cascade(label="Help", menu=help_menu, underline=0)

        await trio.sleep_forever()  # needed if nursery never starts a long running child


def main():
    # make root/parent passing mandatory.
    tk.NoDefaultRoot()
    root = tk.Tk()
    root.wm_resizable(False, False)
    root.wm_minsize(300, 20)
    root.wm_title("Magic AFM")
    try:
        dpi = ctypes.windll.user32.GetDpiForWindow(root.winfo_id())
    except AttributeError:
        pass
    else:
        matplotlib.rcParams["figure.dpi"] = dpi
    # root.wm_iconbitmap("something.ico")
    # sabotage update command so that we crash instead of deadlocking
    # breaks ttk.Combobox, maybe others
    # root.tk.call('rename', 'update', 'never_update')
    host = TkHost(root)
    trio.lowlevel.start_guest_run(
        main_task,
        root,
        run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
        run_sync_soon_not_threadsafe=host.run_sync_soon_not_threadsafe,
        done_callback=host.done_callback,
    )
    outcome_ = outcome.capture(root.mainloop)
    print("Tk shutdown. Outcome:", outcome_)
    if isinstance(outcome_, outcome.Error):
        date = datetime.datetime.now().isoformat().replace(":", ";")
        with open(f"traceback-{date}.dump", "w") as file:
            exc = outcome_.error
            traceback.print_exception(
                type(exc), exc, exc.__traceback__, file=file,
            )
            traceback.print_exception(
                type(exc), exc, exc.__traceback__,
            )


if __name__ == "__main__":
    freeze_support()
    main()
