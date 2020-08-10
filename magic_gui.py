"""MagicAFM GUI

"""
__author__ = "Richard J. Sheridan"
__app_name__ = __doc__.split("\n", 1)[0]

import sys
from itertools import repeat

import data_readers

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    from _version import __version__
else:
    import make_version

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

import collections
import dataclasses
import datetime
import traceback
import enum
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from contextlib import nullcontext
from functools import partial, wraps
from typing import Optional, Callable
from multiprocessing import Pool, freeze_support
import threading

import matplotlib
import numpy as np
import outcome
import trio
from matplotlib.backend_bases import MouseButton
from matplotlib.backends._backend_tk import FigureCanvasTk, blit as tk_blit
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.contour import ContourSet
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import EngFormatter
from matplotlib.transforms import Bbox, BboxTransform
from matplotlib.widgets import SubplotTool
from tqdm.gui import tqdm_tk

import async_tools
import magic_calculation
from magic_calculation import MANIPULATIONS
from async_tools import trs, LONGEST_IMPERCEPTIBLE_DELAY

matplotlib.rcParams["savefig.dpi"] = 300

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
    fit_mode: magic_calculation.FitMode
    disp_kind: DispKind
    k: float
    radius: float
    tau: float


@dataclasses.dataclass
class ForceCurveData:
    z: np.ndarray
    d: np.ndarray
    s: np.ndarray
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


@dataclasses.dataclass(frozen=True)
class ImagePoint:
    r: int
    c: int
    x: float
    y: float

    @staticmethod
    def construct_transforms(axesimage):
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

        return data_coords_to_array_index, array_index_to_data_coords

    @classmethod
    def from_index(cls, r, c, transforms):
        return cls(r, c, *transforms[1](r, c))

    @classmethod
    def from_data(cls, x, y, transforms):
        r, c = transforms[0](x, y)
        # center x, y
        x, y = transforms[1](r, c)
        return cls(r, c, x, y)


class AsyncFigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def __init__(self, figure, master=None, resize_callback=None):
        self._resize_cancels_pending = set()
        self.draw_send, self.draw_recv = trio.open_memory_channel(0)
        self._idle_task_running = False
        super(FigureCanvasTk, self).__init__(figure)
        t1, t2, w, h = self.figure.bbox.bounds
        w, h = int(w), int(h)
        self._tkcanvas = tk.Canvas(
            master=master, width=w, height=h, borderwidth=0, highlightthickness=0,
        )
        self._tkphoto = tk.PhotoImage(master=self._tkcanvas, width=w, height=h)
        self._tkcanvas.create_image(w // 2, h // 2, image=self._tkphoto)
        self._resize_callback = resize_callback
        # Late bound in teach_canvas_to_use_trio
        # Swallows initial resize, OK because of change_image_callback
        # self._tkcanvas.bind("<Configure>", self.resize)
        self._tkcanvas.bind("<Key>", self.key_press)
        self._tkcanvas.bind("<Motion>", self.motion_notify_event)
        self._tkcanvas.bind("<Enter>", self.enter_notify_event)
        self._tkcanvas.bind("<Leave>", self.leave_notify_event)
        self._tkcanvas.bind("<KeyRelease>", self.key_release)
        for name in "<Button-1>", "<Button-2>", "<Button-3>":
            self._tkcanvas.bind(name, self.button_press_event)
        for name in "<Double-Button-1>", "<Double-Button-2>", "<Double-Button-3>":
            self._tkcanvas.bind(name, self.button_dblclick_event)
        for name in "<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>":
            self._tkcanvas.bind(name, self.button_release_event)

        # Mouse wheel on Linux generates button 4/5 events
        for name in "<Button-4>", "<Button-5>":
            self._tkcanvas.bind(name, self.scroll_event)
        # Mouse wheel for windows goes to the window with the focus.
        # Since the canvas won't usually have the focus, bind the
        # event to the window containing the canvas instead.
        # See http://wiki.tcl.tk/3893 (mousewheel) for details
        root = self._tkcanvas.winfo_toplevel()
        root.bind("<MouseWheel>", self.scroll_event_windows, "+")

        # Can't get destroy events by binding to _tkcanvas. Therefore, bind
        # to the window and filter.
        def filter_destroy(evt):
            if evt.widget is self._tkcanvas:
                # self._master.update_idletasks()
                self.close_event()

        root.bind("<Destroy>", filter_destroy, "+")

        self._master = master
        self._tkcanvas.focus_set()

    def draw_idle(self):
        def null_draw_fn():
            pass

        try:
            self.draw_send.send_nowait(null_draw_fn)
        except trio.WouldBlock:
            pass

    async def idle_draw_task(self, task_status=trio.TASK_STATUS_IGNORED):
        delay = LONGEST_IMPERCEPTIBLE_DELAY
        task_status.started()
        # One of the slowest processes. Stick everything in a thread.
        while True:
            # Sleep until someone sends artist calls
            fn = await self.draw_recv.receive()
            async with self.spinner_scope():
                await trs(fn)
                # Batch rapid artist call requests
                # spend roughly equal time building artists and drawing
                with trio.move_on_after(delay):
                    async for fn in self.draw_recv:
                        with trio.CancelScope(shield=True):
                            # Cancelling this would drop the fn
                            # would be nicer to prepend back into channel but...
                            await trs(fn)
                t = trio.current_time()
                await trs(super().draw)  # XXX: NOT self.draw()!
                self.blit()  # blit() can't be in a thread.
                # previous delay is not great predictor of next delay
                # for now try exponential moving average
                delay = ((trio.current_time() - t) + delay * 1) / 2
            # Funny story, we want tight layout behavior on resize and
            # a few other special cases, but also we want super().draw()
            # and by extension draw_idle_task to be responsible for calling
            # figure.tight_layout().
            # So everywhere desired, send set_tight_layout(True)
            # and it will be reset here.
            self.figure.set_tight_layout(False)

    def draw(self):
        super().draw()
        tk_blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))

    def blit(self, bbox=None):
        tk_blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3), bbox=bbox)

    async def resize(self, event):
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
            else:
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

    def teach_canvas_to_use_trio(self, nursery, spinner_scope):
        self._tkcanvas.bind("<Configure>", partial(nursery.start_soon, self.resize))
        self.spinner_scope = spinner_scope


class AsyncNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        self.toolitems += (("Export", "Export calculated maps", "filesave", "export_calculations"),)
        self._prev_filename = ""
        super().__init__(canvas, window)

    def teach_navbar_to_use_trio(self, nursery):
        self._parent_nursery = nursery
        self._wait_cursor_for_draw_cm = nullcontext

    def configure_subplots(self):
        self._parent_nursery.start_soon(self._aconfigure_subplots)

    async def _aconfigure_subplots(self):
        window = tk.Toplevel(self.canvas.get_tk_widget().master)
        toolfig = Figure(figsize=(6, 3))
        toolfig.subplots_adjust(top=0.9)
        canvas = type(self.canvas)(toolfig, master=window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        async with trio.open_nursery() as nursery:
            await nursery.start(canvas.idle_draw_task)

            def spinner_start():
                self.window.configure(cursor="watch")

            def spinner_stop():
                self.window.configure(cursor="arrow")

            self.spinner_scope = await nursery.start(
                async_tools.spinner_task, spinner_start, spinner_stop,
            )
            canvas.teach_canvas_to_use_trio(nursery, self.spinner_scope)
            SubplotTool(self.canvas.figure, toolfig)
            window.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        window.destroy()

    def export_calculations(self):
        self._parent_nursery.start_soon(self._aexport_calculations)

    async def _aexport_calculations(self):
        ### it's also possible to export image stacks but need a way to indicate that
        import os
        import imageio

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
            ".tsv": partial(np.savetxt, delimiter="\t", fmt='%.8g'),
            ".csv": partial(np.savetxt, delimiter=",", fmt='%.8g'),
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
                master=self.canvas.get_tk_widget().master,
                title="Export calculated images",
                filetypes=export_filetypes,
                defaultextension=defaultextension,
                initialdir=initialdir,
                initialfile=initialfile,
            )
        )

        if fname in ["", ()]:
            return
        # Save dir for next time, unless empty str (i.e., use cwd).
        if initialdir != "":
            matplotlib.rcParams["savefig.directory"] = os.path.dirname(str(fname))
        self._prev_filename = fname
        root, ext = os.path.splitext(fname)
        for image_name in self.window._host.image_name_menu.cget("values"):
            if image_name.startswith("Calc"):
                exporter = exporter_map.get(ext, imageio.imwrite)
                image = await self.window._host.opened_fvol.get_image(image_name)
                await trs(
                    exporter, root + "_" + image_name[4:] + ext, image,
                )


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

        Compare to `tkthread <https://github.com/serwy/tkthread/blob/1f612e1dd46e770bd0d0bb64d7ecb6a0f04875a3/tkthread/__init__.py#L163>`_
        where definitely thread unsafe `eval <https://github.com/python/cpython/blob/a5d6aba318ead9cc756ba750a70da41f5def3f8f/Modules/_tkinter.c#L1567-L1585>`_
        is used to send thread safe signals between tcl interpreters.

        If .call is called from the Tcl thread, the locking and sending are optimized away
        so it should be fast enough that the run_sync_soon_not_threadsafe version is unnecessary.
        """
        # self.root.after(0, func) # does a fairly intensive wrapping to each func
        self._q.append(func)
        self.root.call("after", "idle", self._tk_func_name)

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


class ForceVolumeWindow:
    default_figsize = (9, 2.75)

    def __init__(self, root, opened_fvol, figsize=default_figsize, **kwargs):
        self.opened_fvol = opened_fvol
        self.tkwindow = window = tk.Toplevel(root, **kwargs)
        window.wm_title(self.opened_fvol.path.name)
        window._host = self

        # Build figure
        self.fig = Figure(figsize, frameon=False)
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
        self.image_name_menu.pack(fill="x")
        image_name_labelframe.pack(fill="x")
        colormap_labelframe = ttk.Labelframe(image_opts_frame, text="Colormap")
        self.colormap_strvar = tk.StringVar(colormap_labelframe, value="viridis")
        self.colormap_menu = ttk.Combobox(
            colormap_labelframe,
            state="readonly",
            textvariable=self.colormap_strvar,
            values=COLORMAPS,
            width=max(map(len, COLORMAPS)) - 1,
        )
        self.colormap_menu.pack(fill="x")
        colormap_labelframe.pack(fill="x")
        manipulate_labelframe = ttk.Labelframe(image_opts_frame, text="Manipulations")
        self.manipulate_strvar = tk.StringVar(
            manipulate_labelframe, value=next(iter(MANIPULATIONS))
        )
        self.manipulate_menu = ttk.Combobox(
            manipulate_labelframe,
            state="readonly",
            textvariable=self.manipulate_strvar,
            values=list(MANIPULATIONS),
            # width=max(map(len, COLORMAPS)) - 1,
        )
        self.manipulate_menu.pack(fill="x")
        manipulate_labelframe.pack(fill="x")

        image_opts_frame.grid(row=1, column=0, rowspan=2)

        disp_labelframe = ttk.Labelframe(self.options_frame, text="Force curve display")
        self.disp_kind_intvar = tk.IntVar(disp_labelframe, value=DispKind.zd.value)
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

        fit_labelframe = ttk.Labelframe(self.options_frame, text="Fit parameters")
        self.fit_intvar = tk.IntVar(fit_labelframe, value=magic_calculation.FitMode.SKIP.value)
        fit_skip_button = ttk.Radiobutton(
            fit_labelframe,
            text="Skip",
            value=magic_calculation.FitMode.SKIP.value,
            variable=self.fit_intvar,
        )
        fit_skip_button.grid(row=0, column=0)
        fit_ext_button = ttk.Radiobutton(
            fit_labelframe,
            text="Extend",
            value=magic_calculation.FitMode.EXTEND.value,
            variable=self.fit_intvar,
        )
        fit_ext_button.grid(row=0, column=1)
        fit_ret_button = ttk.Radiobutton(
            fit_labelframe,
            text="Retract",
            value=magic_calculation.FitMode.RETRACT.value,
            variable=self.fit_intvar,
        )
        fit_ret_button.grid(row=0, column=2)

        fit_radius_label = ttk.Label(fit_labelframe, text="Tip radius (nm)")
        fit_radius_label.grid(row=1, column=0, columnspan=2, sticky="W")
        self.fit_radius_sbox = ttk.Spinbox(
            fit_labelframe,
            from_=1,
            to=10000,
            increment=0.1,
            format="%0.1f",
            width=6,
            validate="all",
            validatecommand="string is double -strict %S",
            invalidcommand="%W set %s",
        )
        self.fit_radius_sbox.set(20.0)
        self.fit_radius_sbox.grid(row=1, column=2, sticky="W")
        fit_tau_label = ttk.Label(fit_labelframe, text="DMT-JKR (0-1)", justify="left")
        fit_tau_label.grid(row=2, column=0, columnspan=2, sticky="W")
        self.fit_tau_sbox = ttk.Spinbox(
            fit_labelframe,
            from_=0,
            to=1,
            increment=0.05,
            format="%0.2f",
            width=6,
            validatecommand="string is double -strict %S",
            invalidcommand="%W set %s",
        )
        self.fit_tau_sbox.set(0.0)
        self.fit_tau_sbox.grid(row=2, column=2, sticky="E")

        self.calc_props_button = ttk.Button(
            fit_labelframe, text="Calculate Property Maps", state="disabled"
        )
        self.calc_props_button.grid(row=3, column=0, columnspan=3)
        fit_labelframe.grid(row=0, column=1, rowspan=3)

        self.options_frame.grid(row=1, column=0, sticky="nsew")

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

        # Finalize pure ARDFWindow stuff
        self.reset_image_name_menu(self.opened_fvol.image_names)

        # change_image_callback stuff
        self.colorbar: Optional[Colorbar] = None
        self.axesimage: Optional[AxesImage] = None

        # plot_curve_event_response stuff
        self.cancels_pending = set()
        self.artists = []
        self.existing_points = set()

        # mpl_resize_event_callback
        self.resize_cancels_pending = set()

    def reset_image_name_menu(self, names):
        names = list(names)
        longest = max(map(len, names))
        self.image_name_menu.configure(values=names, width=min(longest - 1, 20))

    async def calc_prop_map_callback(self):
        optionsdict = dict(
            fit_mode=self.fit_intvar.get(),
            disp_kind=self.disp_kind_intvar.get(),
            k=self.opened_fvol.k,
            radius=float(self.fit_radius_sbox.get()),
            tau=float(self.fit_tau_sbox.get()),
        )
        options = ForceCurveOptions(**optionsdict)
        img_shape = self.opened_fvol.shape
        ncurves = img_shape[0] * img_shape[1]
        resample_npts = 512
        if not options.fit_mode:
            raise ValueError("Property map button should have been disabled")

        async with self.spinner_scope() as cancel_scope:
            # assign pbar and progress_image ASAP in case of cancel
            pbar = tqdm_tk(
                total=ncurves,
                desc="Loading and resampling force curves...",
                smoothing_time=0.25,
                mininterval=LONGEST_IMPERCEPTIBLE_DELAY * 2,
                unit=" curves",
                tk_parent=self.tkwindow,
                grab=False,
                leave=False,
                cancel_callback=cancel_scope.cancel,
            )
            progress_image: matplotlib.image.AxesImage = self.img_ax.imshow(
                np.zeros(img_shape + (4,), dtype="f4"), extent=self.axesimage.get_extent()
            )  # transparent initial image, no need to draw

            _, _, s = await self.opened_fvol.get_force_curve(0, 0)
            npts = len(_)
            resample = npts > resample_npts
            if resample:
                s = s * resample_npts // npts
            else:
                resample_npts = npts
            if options.fit_mode == magic_calculation.FitMode.EXTEND:
                sl = slice(s)
                segment_npts = s
            elif options.fit_mode == magic_calculation.FitMode.RETRACT:
                sl = slice(s, None)
                segment_npts = resample_npts - s
            else:
                raise ValueError("Unknown fit_mode: ", options.fit_mode)

            pbar_lock = threading.Lock()  # for thread_map
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)

            delta = np.empty((ncurves, segment_npts), np.float32)
            f = np.empty((ncurves, segment_npts), np.float32)

            def resample_helper(i):
                r, c = np.unravel_index(i, img_shape)
                z, d, _ = trio.from_thread.run(self.opened_fvol.get_force_curve, r, c)
                z = magic_calculation.resample_dset(z, resample_npts, True)[sl]
                d = magic_calculation.resample_dset(d, resample_npts, True)[sl]
                delta[i, :] = z - d
                f[i, :] = d * options.k
                with pbar_lock:
                    pbar.update()

            if resample:
                await async_tools.thread_map(resample_helper, range(ncurves))
            else:
                for i in range(ncurves):
                    r, c = np.unravel_index(i, img_shape)
                    z, d, _ = await self.opened_fvol.get_force_curve(r, c)
                    z = z[sl]
                    d = d[sl]
                    delta[i, :] = z - d
                    f[i, :] = d * options.k
                    pbar.update()

            pbar.set_description_str("Fitting force curves...")
            pbar.unit = " fits"
            pbar.avg_time = None
            pbar.reset(total=ncurves)
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)

            property_names_units = {
                "CalcIndentationModulus": "Pa",
                "CalcAdhesionForce": "N",
                "CalcDeflection": "m",
                "CalcIndentation": "m",
                "CalcTrueHeight": "m",  # Hi z -> lo h
                "CalcIndentationRatio": "",
            }
            property_map = np.empty(
                ncurves, dtype=np.dtype([(name, "f4") for name in property_names_units]),
            )
            progress_array = progress_image.get_array()

            def draw_fn():
                progress_image.changed()
                progress_image.pchanged()

            async with pool_lock:
                global pool
                try:
                    pool.__enter__()
                except ValueError:
                    pool = await trs(Pool)
                pool_iter = async_tools.asyncify_iterator(
                    pool.imap_unordered(
                        magic_calculation.calc_properties_imap,
                        zip(delta, f, range(ncurves), repeat(optionsdict)),
                        chunksize=8,
                    )
                )
                try:
                    first = True
                    async for i, properties in pool_iter:
                        if first:
                            pbar.unpause()
                            first = False

                        if properties is None:
                            property_map[i] = np.nan
                            color = (1, 0, 0, 0.5)
                        else:
                            property_map[i] = properties
                            color = (0, 1, 0, 0.5)

                        pbar.update()
                        r, c = np.unravel_index(i, img_shape)
                        progress_array[r, c, :] = color
                        try:
                            self.canvas.draw_send.send_nowait(draw_fn)
                        except trio.WouldBlock:
                            pass
                except:
                    pool.terminate()
                    raise

        def draw_fn():
            progress_image.remove()

        await self.canvas.draw_send.send(draw_fn)
        pbar.close()

        if cancel_scope.cancelled_caught:
            return

        combobox_values = list(self.image_name_menu.cget("values"))
        property_map = property_map.reshape((*img_shape, -1,))

        # Actually write out results to external world
        for name in property_names_units:
            self.opened_fvol.add_image(
                image_name=name,
                units=property_names_units[name],
                scandown=True,
                image=property_map[name].squeeze(),
            )
            if name not in combobox_values:
                combobox_values.append(name)
        self.reset_image_name_menu(combobox_values)

    async def change_cmap_callback(self):
        colormap_name = self.colormap_strvar.get()
        # save old clim
        clim = self.axesimage.get_clim()

        def draw_fn():
            # prevent cbar from getting expanded
            self.axesimage.set_clim(self.image_min, self.image_max)
            # actually change cmap
            self.axesimage.set_cmap(colormap_name)
            # reset everything
            customize_colorbar(self.colorbar, self.unit, clim)

        await self.canvas.draw_send.send(draw_fn)

    async def change_image_callback(self):
        image_name = self.image_name_strvar.get()
        cmap = self.colormap_strvar.get()
        image_array, self.scandown = await self.opened_fvol.get_image(image_name)
        clim = np.nanquantile(image_array, [0.01, 0.99])
        self.unit = self.opened_fvol.get_image_units(image_name)
        self.image_min = float(np.nanmin(image_array))
        self.image_max = float(np.nanmax(image_array))

        if self.colorbar is not None:
            self.colorbar.remove()
        if self.axesimage is not None:
            self.axesimage.remove()

        scansize = self.opened_fvol.scansize
        s = (scansize + scansize / len(image_array)) // 2

        def draw_fn():
            self.img_ax.set_title(image_name)
            self.img_ax.set_ylabel("Y piezo (nm)")
            self.img_ax.set_xlabel("X piezo (nm)")

            self.axesimage = self.img_ax.imshow(
                image_array,
                origin="lower" if self.scandown else "upper",
                extent=(-s, s, -s, s,),
                picker=True,
                cmap=cmap,
            )
            self.axesimage.get_array().fill_value = np.nanmedian(image_array)

            self.transforms = ImagePoint.construct_transforms(self.axesimage)

            self.colorbar = self.fig.colorbar(self.axesimage, ax=self.img_ax, use_gridspec=True,)
            self.navbar.update()  # let navbar catch new cax in fig
            customize_colorbar(self.colorbar, self.unit, clim)
            self.fig.set_tight_layout(True)

        await self.canvas.draw_send.send(draw_fn)

    async def change_fit_callback(self):
        self.calc_props_button.configure(state="normal" if self.fit_intvar.get() else "disabled")
        await self.redraw_existing_points()

        def draw_fn():
            self.fig.set_tight_layout(True)

        await self.canvas.draw_send.send(draw_fn)

    async def change_disp_kind_callback(self):
        await self.redraw_existing_points()

        def draw_fn():
            self.fig.set_tight_layout(True)

        await self.canvas.draw_send.send(draw_fn)

    async def redraw_existing_points(self):
        def draw_fn():
            for artist in self.artists:
                artist.remove()
            self.plot_ax.relim()
            self.plot_ax.set_prop_cycle(None)
            self.artists.clear()

        await self.canvas.draw_send.send(draw_fn)
        async with self.spinner_scope():
            for point in self.existing_points:
                await self.plot_curve_response(point, False)

    async def manipulate_callback(self):
        manip_name = self.manipulate_strvar.get()
        current_name = self.image_name_strvar.get()
        current_names = list(self.image_name_menu.cget("values"))
        name = "Calc" + manip_name + current_name
        if name not in self.opened_fvol.image_names:
            manip_fn = MANIPULATIONS[manip_name]
            manip_img = await trs(manip_fn, self.axesimage.get_array())
            self.opened_fvol.add_image(
                name, self.unit, self.scandown, manip_img,
            )
            if name not in current_names:
                current_names.append(name)
            self.reset_image_name_menu(current_names)
        self.image_name_menu.set(name)

    async def colorbar_freeze_response(self):
        def draw_fn():
            self.colorbar.draw_all()
            if isinstance(self.colorbar.mappable, ContourSet):
                CS = self.colorbar.mappable
                if not CS.filled:
                    self.colorbar.add_lines(CS)
            self.colorbar.stale = True

        await self.canvas.draw_send.send(draw_fn)

    async def colorbar_limits_response(self, vmin, vmax):
        def draw_fn():
            self.colorbar.norm.vmin = vmin
            self.colorbar.norm.vmax = vmax
            self.colorbar.solids.set_clim(vmin, vmax)

        await self.canvas.draw_send.send(draw_fn)

    async def plot_curve_response(self, point: ImagePoint, clear_previous):
        self.existing_points.add(point)  # should be before 1st checkpoint
        self.plot_ax.set_autoscale_on(
            True
        )  # XXX: only needed on first plot. Maybe later make optional?
        options = ForceCurveOptions(
            fit_mode=self.fit_intvar.get(),
            disp_kind=self.disp_kind_intvar.get(),
            k=self.opened_fvol.k,
            radius=float(self.fit_radius_sbox.get()),
            tau=float(self.fit_tau_sbox.get()),
        )
        if clear_previous:
            for cancel_scope in self.cancels_pending:
                cancel_scope.cancel()
            self.cancels_pending.clear()
        async with self.spinner_scope():
            with trio.CancelScope() as cancel_scope:
                self.cancels_pending.add(cancel_scope)

                # Calculation phase
                # Do a few long-running jobs, likely to be canceled
                force_curve = await self.opened_fvol.get_force_curve(point.r, point.c)
                data = await trs(
                    calculate_force_data, *force_curve, options, async_tools.make_cancel_poller()
                )
                del force_curve  # contained in data
                # XXX: Race condition
                # cancel can occur after the last call to the above cancel poller
                # checkpoint make sure to raise Cancelled if it just happened
                # await trio.sleep(0)

            self.cancels_pending.discard(cancel_scope)

            if cancel_scope.cancelled_caught:
                self.existing_points.discard(point)
                return

            def draw_fn():
                # Clearing Phase
                # Clear previous artists and reset plots (faster than .clear()?)
                if clear_previous:
                    for artist in self.artists:
                        artist.remove()
                    self.artists.clear()
                    self.existing_points.clear()

                    self.existing_points.add(point)
                    self.plot_ax.relim()
                    self.plot_ax.set_prop_cycle(None)

                # Drawing Phase
                # Based options choose plots and collect artists for deletion
                new_artists, color = draw_force_curve(data, self.plot_ax, options)
                self.artists.extend(new_artists)
                self.artists.extend(
                    self.img_ax.plot(
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
                    table = draw_data_table(data, self.plot_ax,)
                    self.artists.append(table)

            await self.canvas.draw_send.send(draw_fn)

    async def mpl_pick_motion_event_callback(self, event):
        mouseevent = getattr(event, "mouseevent", event)
        control_held = event.guiEvent.state & TkState.CONTROL
        if event.name == "motion_notify_event" and not control_held:
            return
        if mouseevent.inaxes is None:
            return
        elif mouseevent.inaxes is self.img_ax:
            if mouseevent.button != MouseButton.LEFT:
                return
            point = ImagePoint.from_data(mouseevent.xdata, mouseevent.ydata, self.transforms)
            shift_held = mouseevent.guiEvent.state & TkState.SHIFT
            if shift_held and point in self.existing_points:
                return
            await self.plot_curve_response(point, not shift_held)
        elif mouseevent.inaxes is self.colorbar.ax:
            if mouseevent.button == MouseButton.MIDDLE or (
                mouseevent.guiEvent.state & TkState.ALT and mouseevent.button == MouseButton.LEFT
            ):
                await self.colorbar_freeze_response()
            elif mouseevent.button == MouseButton.LEFT:
                vmin = self.colorbar.norm.vmin
                vmax = max(mouseevent.ydata, vmin)
                await self.colorbar_limits_response(vmin, vmax)
            elif mouseevent.button == MouseButton.RIGHT:
                vmax = self.colorbar.norm.vmax
                vmin = min(mouseevent.ydata, vmax)
                await self.colorbar_limits_response(vmin, vmax)

    async def window_task(self):
        nursery: trio.Nursery
        async with trio.open_nursery() as nursery:

            def spinner_start():
                self.tkwindow.configure(cursor="watch")
                self.options_frame.configure(cursor="watch")

            def spinner_stop():
                self.tkwindow.configure(cursor="arrow")
                self.options_frame.configure(cursor="arrow")

            self.spinner_scope = await nursery.start(
                async_tools.spinner_task, spinner_start, spinner_stop,
            )

            # Teach MPL to use trio
            self.canvas.mpl_connect(
                "motion_notify_event",
                partial(nursery.start_soon, self.mpl_pick_motion_event_callback),
            )
            self.canvas.mpl_connect(
                "pick_event", partial(nursery.start_soon, self.mpl_pick_motion_event_callback)
            )
            self.navbar.teach_navbar_to_use_trio(nursery)
            self.canvas.teach_canvas_to_use_trio(nursery, self.spinner_scope)

            # Teach tkinter to use trio
            self.tkwindow.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
            self.colormap_strvar.trace_add(
                "write", impartial(partial(nursery.start_soon, self.change_cmap_callback))
            )
            self.manipulate_strvar.trace_add(
                "write", impartial(partial(nursery.start_soon, self.manipulate_callback))
            )
            self.fit_intvar.trace_add(
                "write", impartial(partial(nursery.start_soon, self.change_fit_callback))
            )
            self.disp_kind_intvar.trace_add(
                "write", impartial(partial(nursery.start_soon, self.change_disp_kind_callback))
            )
            self.calc_props_button.configure(
                command=partial(nursery.start_soon, self.calc_prop_map_callback,)
            )
            self.image_name_strvar.trace_add(
                "write", impartial(partial(nursery.start_soon, self.change_image_callback))
            )

            # Finalize initial window state
            await nursery.start(self.canvas.idle_draw_task)
            # This causes the initial plotting of figures after next checkpoint
            # StringVar.set() won't be effective to plot unless it happens after the
            # trace_add AND start(idle_draw_task). accidentally, the plot will be drawn later
            # due to resize, but let's not rely on that!
            for name in ("MapHeight", "ZSensorTrace", "Demo", "Height Sensor"):
                if name in self.opened_fvol.image_names:
                    self.image_name_strvar.set(name)
                    break
            await trio.sleep_forever()

        # Close phase
        self.options_frame.destroy()
        self.tkwindow.withdraw()  # weird navbar hiccup on close
        self.tkwindow.destroy()


def customize_colorbar(colorbar, unit, clim):
    """MPL keeps stomping on our settings so reset EVERYTHING"""
    colorbar.ax.set_navigate(True)
    colorbar.solids.set_picker(True)
    if unit:
        colorbar.formatter = EngFormatter(unit, places=1)
    colorbar.update_ticks()
    colorbar.solids.set_clim(*clim)


def draw_data_table(data, ax):
    exp = np.log10(data.beta[0])
    prefix, fac = {0: ("G", 1), 1: ("M", 1e3), 2: ("k", 1e6),}.get((-exp + 2.7) // 3, ("", 1))
    table = ax.table(
        [
            [
                "{:.2f}±{:.2f}".format(data.beta[0] * fac, data.beta_err[0] * fac,),
                "{:.2f}±{:.2f}".format(-data.beta[1], -data.beta_err[1]),
                "{:.2f}".format(data.defl),
                "{:.2f}".format(data.ind),
                "{:.2f}".format(data.defl / data.ind),
            ],
        ],
        loc="top",
        colLabels=[f"$M$ ({prefix}Pa)", "$F_{adh}$ (nN)", "d (nm)", "δ (nm)", "d/δ",],
        colWidths=[0.6 / 2, 0.6 / 2, 0.4 / 3, 0.4 / 3, 0.4 / 3],
        colLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    return table


def draw_force_curve(data, plot_ax, options):
    artists = []
    if options.disp_kind == DispKind.zd:
        plot_ax.set_xlabel("Z piezo (nm)")
        plot_ax.set_ylabel("Cantilever deflection (nm)")
        artists.extend(plot_ax.plot(data.z[: data.s], data.d[: data.s]))
        artists.extend(plot_ax.plot(data.z[data.s :], data.d[data.s :]))
        if options.fit_mode:
            artists.extend(plot_ax.plot(data.z[data.sl], data.d_fit, "--"))
            artists.append(plot_ax.axvline(data.z_tru, ls=":", c=artists[0].get_color()))
    elif options.disp_kind == DispKind.δf:
        plot_ax.set_xlabel("Indentation depth (nm)")
        plot_ax.set_ylabel("Indentation force (nN)")
        if options.fit_mode:
            delta = data.delta - data.beta[2]
            f = data.f - data.beta[3]
            f_fit = data.f_fit - data.beta[3]
            artists.extend(plot_ax.plot(delta[: data.s], f[: data.s]))
            artists.extend(plot_ax.plot(delta[data.s :], f[data.s :]))
            artists.extend(plot_ax.plot(delta[data.sl], f_fit, "--"))
            mopts = dict(
                marker="X", markersize=8, linestyle="", markeredgecolor="k", markerfacecolor="k",
            )
            artists.extend(
                plot_ax.plot(
                    data.ind + data.mindelta - data.beta[2],
                    data.defl * options.k + data.beta[1],
                    **mopts,
                )
            )
            artists.extend(plot_ax.plot(data.mindelta - data.beta[2], data.beta[1], **mopts,))
        else:
            artists.extend(plot_ax.plot(data.delta[: data.s], data.f[: data.s]))
            artists.extend(plot_ax.plot(data.delta[data.s :], data.f[data.s :]))
    else:
        raise ValueError("Unknown DispKind: ", data.disp_kind)
    return artists, artists[0].get_color()


def calculate_force_data(z, d, s, options, cancel_poller=lambda: None):
    cancel_poller()
    resample_npts = 512
    s = s * resample_npts // len(z)
    z, d = magic_calculation.resample_dset([z, d], resample_npts, True)
    # Transform data to model units
    f = d * options.k
    delta = z - d

    if not options.fit_mode:
        return ForceCurveData(s=s, z=z, d=d, f=f, delta=delta,)

    if options.fit_mode == magic_calculation.FitMode.EXTEND:
        sl = slice(s)
    elif options.fit_mode == magic_calculation.FitMode.RETRACT:
        sl = slice(s, None)
    else:
        raise ValueError("Unknown fit_mode: ", options.fit_mode)

    cancel_poller()
    beta, beta_err, calc_fun = magic_calculation.fitfun(
        delta[sl], f[sl], **dataclasses.asdict(options), cancel_poller=cancel_poller,
    )
    f_fit = calc_fun(delta[sl], *beta)
    d_fit = f_fit / options.k
    cancel_poller()
    if np.all(np.isfinite(beta)):
        deflection, indentation, z_true_surface, mindelta = magic_calculation.calc_def_ind_ztru(
            f[sl], beta, **dataclasses.asdict(options)
        )
    else:
        deflection, indentation, z_true_surface, mindelta = np.nan, np.nan, np.nan, np.nan
    return ForceCurveData(
        s=s,
        z=z,
        d=d,
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
            path = await data_readers.convert_ardf(path, force=True, pbar=pbar)
        if cscope.cancelled_caught:
            return

    async with data_readers.SUFFIX_FVFILE_MAP[path.suffix](path) as opened_fv:
        window = await trs(ForceVolumeWindow, root, opened_fv)
        await window.window_task()


async def demo_task(root):
    async with data_readers.DemoForceVolumeFile("Demo") as opened_arh5:
        window = await trs(ForceVolumeWindow, root, opened_arh5)
        await window.window_task()


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
    timely_trio_pbar = ttk.Progressbar(top, **opts)
    timely_trio_pbar.pack()
    tk_pbar = ttk.Progressbar(top, **opts)
    tk_pbar.pack()
    trio_pbar = ttk.Progressbar(top, **opts)
    trio_pbar.pack()

    interval = 10

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
            thread_cpu.set(
                "CPU-bound threads:" + repr(async_tools.cpu_bound_limiter).split(",")[1][:-1]
            )
            thread_default.set(
                "Default threads:"
                + repr(trio.to_thread.current_default_thread_limiter()).split(",")[1][:-1]
            )
            process.set("Processes: " + " ".join(repr(pool).split()[1:])[:-1])
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
        help_action = partial(open_with_os_default, "help_page.html")

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
    # root.wm_iconbitmap("something.ico")
    # sabotage update command so that we crash instead of deadlocking
    # breaks ttk.Combobox, maybe others
    # root.tk.call('rename', 'update', 'never_update')
    host = TkHost(root)
    trio.lowlevel.start_guest_run(
        main_task,
        root,
        run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
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


class FirstPool:
    """Allows check for existing and open Pool instance in one try"""

    def __enter__(self):
        raise ValueError()

    def __repr__(self):
        return "nothing state=None pool_size=None>"


pool = FirstPool()
pool_lock = trio.Lock()
if __name__ == "__main__":
    freeze_support()
    main()
