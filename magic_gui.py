"""MagicAFM GUI

"""
__author__ = "Richard J. Sheridan"
__app_name__ = __doc__.split("\n", 1)[0]

try:
    from _version import __version__
except ImportError:
    import make_version

    __version__ = make_version.get()
import datetime

__short_license__ = f"""{__app_name__} {__version__}
Copyright (C) {datetime.datetime.now().year} {__author__}

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
import enum
import tkinter as tk
import traceback
from functools import partial, wraps
from tkinter import ttk, filedialog
from typing import Optional, Callable

import matplotlib
import numpy as np
import outcome
import trio
import trio.testing
from matplotlib.backend_bases import MouseButton
from matplotlib.backends._backend_tk import FigureCanvasTk, blit as tk_blit
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox, BboxTransform
from matplotlib.widgets import SubplotTool
from tqdm import tqdm_gui

import async_tools
import magic_calculation
from async_tools import trs, ctrs, LONGEST_IMPERCEPTIBLE_DELAY

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
    radius: float = 20
    tau: float = 0


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


@dataclasses.dataclass(frozen=True)
class ImagePoint:
    r: int
    c: int
    x: float
    y: float


class AsyncFigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def __init__(self, figure, master=None, resize_callback=None):
        self._trio_draw_event = trio.Event()
        self.trio_draw_lock = trio.Lock()
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
        self._tkcanvas.bind("<Configure>", self.resize)
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
        assert self._idle_task_running, "start idle_draw_task first"
        self._trio_draw_event.set()

    async def idle_draw_task(self, task_status=trio.TASK_STATUS_IGNORED):
        assert not self._idle_task_running
        self._idle_task_running = True
        task_status.started()
        try:
            while True:
                await self._trio_draw_event.wait()
                await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
                async with self.trio_draw_lock:
                    self._trio_draw_event = trio.Event()
                    await trs(super().draw, cancellable=False)
                    tk_blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))
        finally:
            self._idle_task_running = False

    def draw(self):
        super().draw()
        tk_blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))

    def blit(self, bbox=None):
        tk_blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3), bbox=bbox)

    def resize(self, event):
        width, height = event.width, event.height
        if self._resize_callback is not None:
            self._resize_callback(event)

        # compute desired figure size in inches
        dpival = self.figure.dpi
        winch = width / dpival
        hinch = height / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)

        self._tkcanvas.delete(self._tkphoto)
        self._tkphoto = tk.PhotoImage(master=self._tkcanvas, width=int(width), height=int(height))
        self._tkcanvas.create_image(int(width / 2), int(height / 2), image=self._tkphoto)
        self.resize_event()


class ImprovedNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        self.toolitems += (("Export", "Export calculated maps", "filesave", "export_calculations"),)
        self._prev_filename = ""
        super().__init__(canvas, window)

    def teach_navbar_to_use_trio(self, nursery):
        self._parent_nursery = nursery

    def configure_subplots(self):
        self._parent_nursery.start_soon(self._aconfigure_subplots)

    async def _aconfigure_subplots(self):
        window = tk.Toplevel(self.canvas.get_tk_widget().master)
        toolfig = Figure(figsize=(6, 3))
        toolfig.subplots_adjust(top=0.9)
        canvas = type(self.canvas)(toolfig, master=window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        async with trio.open_nursery() as idle_draw_nursery:
            await idle_draw_nursery.start(canvas.idle_draw_task)
            SubplotTool(self.canvas.figure, toolfig)
            window.protocol("WM_DELETE_WINDOW", idle_draw_nursery.cancel_scope.cancel)
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
                exporter_map.get(ext, imageio.imwrite)(
                    root + "_" + image_name[4:] + ext,
                    await self.window._host.opened_arh5.get_image(image_name),
                )


class tqdm_tk(tqdm_gui):
    monitor_interval = 0

    def __init__(
        self, *args, cancel_callback=None, grab=False, tk_parent=None, bar_format=None, **kwargs
    ):
        kwargs["gui"] = True
        self._cancel_callback = cancel_callback
        if tk_parent is None:
            # this will error if tkinter.NoDefaultRoot() called
            try:
                tkparent = tk._default_root
            except AttributeError:
                raise ValueError("tk_parent required when using NoDefaultRoot")
            if tkparent is None:
                # use new default root window as display
                self.tk_window = tk.Tk()
            else:
                # some other windows already exist
                self.tk_window = tk.Toplevel()
        else:
            self.tk_window = tk.Toplevel(tk_parent)
        if bar_format is None:
            kwargs["bar_format"] = (
                "{n_fmt}/{total_fmt}, {rate_noinv_fmt}\n"
                "{elapsed} elapsed, {remaining} ETA\n\n"
                "{percentage:3.0f}%"
            )
        super(tqdm_gui, self).__init__(*args, **kwargs)

        if self.disable:
            return

        self.tk_dispatching = self.tk_dispatching_helper()
        if not self.tk_dispatching:
            # leave is problematic if the mainloop is not running
            self.leave = False
        self.tk_window.protocol("WM_DELETE_WINDOW", self.cancel)
        self.tk_window.wm_title("tqdm_tk")
        self.tk_n_var = tk.DoubleVar(self.tk_window, value=0)
        self.tk_desc_var = tk.StringVar(self.tk_window)
        self.tk_desc_var.set(self.desc)
        self.tk_text_var = tk.StringVar(self.tk_window)
        pbar_frame = ttk.Frame(self.tk_window, padding=5)
        pbar_frame.pack()
        self.tk_desc_frame = ttk.Frame(pbar_frame)
        self.tk_desc_frame.pack()
        self.tk_desc_label = None
        self.tk_label = ttk.Label(
            pbar_frame,
            textvariable=self.tk_text_var,
            wraplength=600,
            anchor="center",
            justify="center",
        )
        self.tk_label.pack()
        self.tk_pbar = ttk.Progressbar(pbar_frame, variable=self.tk_n_var, length=450)
        if self.total is not None:
            self.tk_pbar.configure(maximum=self.total)
        else:
            self.tk_pbar.configure(mode="indeterminate")
        self.tk_pbar.pack()
        if self._cancel_callback is not None:
            self.tk_button = ttk.Button(pbar_frame, text="Cancel", command=self.cancel)
            self.tk_button.pack()
        if grab:
            self.tk_window.grab_set()

    def display(self):
        self.tk_n_var.set(self.n)
        if self.desc:
            if self.tk_desc_label is None:
                self.tk_desc_label = ttk.Label(
                    self.tk_desc_frame,
                    textvariable=self.tk_desc_var,
                    wraplength=600,
                    anchor="center",
                    justify="center",
                )
                self.tk_desc_label.pack()
            self.tk_desc_var.set(self.desc)
        else:
            if self.tk_desc_label is not None:
                self.tk_desc_label.destroy()
                self.tk_desc_label = None
        self.tk_text_var.set(
            self.format_meter(
                n=self.n,
                total=self.total,
                elapsed=self._time() - self.start_t,
                ncols=None,
                prefix=self.desc,
                ascii=self.ascii,
                unit=self.unit,
                unit_scale=self.unit_scale,
                rate=1 / self.avg_time if self.avg_time else None,
                bar_format=self.bar_format,
                postfix=self.postfix,
                unit_divisor=self.unit_divisor,
            )
        )
        if not self.tk_dispatching:
            self.tk_window.update()

    def cancel(self):
        if self._cancel_callback is not None:
            self._cancel_callback()
        self.close()

    def reset(self, total=None):
        if total is not None:
            self.tk_pbar.configure(maximum=total)
        super().reset(total)

    def close(self):
        if self.disable:
            return

        self.disable = True

        with self.get_lock():
            self._instances.remove(self)

        def _close():
            self.tk_window.after(0, self.tk_window.destroy)
            if not self.tk_dispatching:
                self.tk_window.update()

        self.tk_window.protocol("WM_DELETE_WINDOW", _close)
        if not self.leave:
            _close()

    def tk_dispatching_helper(self):
        try:
            return self.tk_window.dispatching()
        except AttributeError:
            pass

        import tkinter, sys

        codes = {tkinter.mainloop.__code__, tkinter.Misc.mainloop.__code__}
        for frame in sys._current_frames().values():
            while frame:
                if frame.f_code in codes:
                    return True
                frame = frame.f_back
        return False


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
            exc = outcome_.error
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        self.root.destroy()


def impartial(fn):
    @wraps(fn)
    def impartial_wrapper(*a, **kw):
        return fn()

    return impartial_wrapper


class ARDFWindow:
    default_figsize = (9, 2.75)

    def __init__(self, root, opened_arh5, figsize=default_figsize, **kwargs):
        self.opened_arh5 = opened_arh5
        self.tkwindow = window = tk.Toplevel(root, **kwargs)
        window.wm_title(self.opened_arh5.h5file_path.name)
        window._host = self

        # Build figure
        self.fig = Figure(figsize, frameon=False)
        self.canvas = AsyncFigureCanvasTkAgg(self.fig, window)
        self.navbar = ImprovedNavigationToolbar2Tk(self.canvas, window)
        self.img_ax, self.plot_ax = img_ax, plot_ax = self.fig.subplots(
            1, 2, gridspec_kw=dict(width_ratios=[1, 1.5])
        )
        img_ax.set_anchor("W")
        img_ax.set_facecolor((0.8, 0, 0))  # scary red for NaN values of images
        # Need to pre-load something into these labels for change_image_callback->tight_layout
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

        image_opts_frame.grid(row=0, column=0)

        disp_labelframe = ttk.Labelframe(self.options_frame, text="Display type")
        self.disp_kind_intvar = tk.IntVar(disp_labelframe, value=DispKind.zd.value)
        disp_zd_button = ttk.Radiobutton(
            disp_labelframe, text="z/d", value=DispKind.zd.value, variable=self.disp_kind_intvar
        )
        disp_zd_button.pack(side="left")
        disp_deltaf_button = ttk.Radiobutton(
            disp_labelframe, text="δ/f", value=DispKind.δf.value, variable=self.disp_kind_intvar
        )
        disp_deltaf_button.pack(side="left")
        disp_labelframe.grid(row=0, column=1)

        fit_labelframe = ttk.Labelframe(self.options_frame, text="Fit type")
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
        self.calc_props_button = ttk.Button(
            fit_labelframe, text="Calculate Property Maps", state="disabled"
        )
        self.calc_props_button.grid(row=1, column=0, columnspan=3)
        fit_labelframe.grid(row=0, column=2)

        self.options_frame.grid(row=1, column=0, sticky="nsew")

        # yes, cheating on trio here
        window.bind("<FocusIn>", impartial(self.options_frame.lift))

        def button_helper(*a, **kw):
            self.calc_props_button.configure(
                state="normal" if self.fit_intvar.get() else "disabled"
            )

        self.fit_intvar.trace_add("write", button_helper)

        # Window widgets
        size_grip = ttk.Sizegrip(window)

        self.navbar.grid(row=0, sticky="we")
        window.grid_rowconfigure(0, weight=0)

        self.canvas.get_tk_widget().grid(row=1, sticky="wens")
        window.grid_rowconfigure(1, weight=1)
        window.grid_columnconfigure(0, weight=1)

        size_grip.grid(row=2, column=1, sticky="es")
        window.grid_columnconfigure(1, weight=0)

        # Finalize pure ARDFWindow stuff
        self.set_image_names(self.opened_arh5.image_names)

        # change_image_callback stuff
        self.cb_fmt = ScalarFormatter()
        self.cb_fmt.set_powerlimits((-2, 2))
        self.colorbar: Optional[Colorbar] = None
        self.axesimage: Optional[AxesImage] = None

        # plot_curve_event_response stuff
        self.cancels_pending = set()
        self.artists = []
        self.existing_points = set()

        # mpl_resize_event_callback
        self.resize_cancels_pending = set()

    def set_image_names(self, image_names):
        self.image_name_menu.configure(
            values=list(image_names), width=max(map(len, image_names)) - 1
        )

    async def arh5_prop_map_callback(self):
        # pbar.set_description("Calculating force maps")
        options = ForceCurveOptions(
            fit_mode=self.fit_intvar.get(),
            disp_kind=self.disp_kind_intvar.get(),
            k=self.opened_arh5.params["k"],
        )
        if not options.fit_mode:
            raise ValueError("Property map button should have been disabled")
        async with self.spinner_scope() as cancel_scope:
            pbar = tqdm_tk(
                total=1,
                desc="Loading force curves into memory...",
                mininterval=None,
                tk_parent=self.tkwindow,
                grab=False,
                leave=False,
                cancel_callback=cancel_scope.cancel,
            )
            pbar.tk_window.wm_title("Loading...")
            pbar.update(0)
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
            z, d, s = await self.opened_arh5.get_all_curves()
            pbar.update()
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
            pbar.close()

            img_shape = z.shape[:2]
            npts = z.shape[-1]
            z = z.reshape((-1, npts))
            d = d.reshape((-1, npts))

            resample_npts = 512
            s = s * resample_npts // npts
            pbar = tqdm_tk(
                total=2,
                desc=f"Resampling force curves to {resample_npts} points...",
                mininterval=None,
                tk_parent=self.tkwindow,
                grab=False,
                leave=False,
                cancel_callback=cancel_scope.cancel,
            )
            pbar.tk_window.wm_title("Calculating...")
            pbar.update(0)
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
            z = magic_calculation.resample_dset(z, resample_npts, True)
            pbar.update()
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
            d = magic_calculation.resample_dset(d, resample_npts, True)
            pbar.update()
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
            pbar.close()

            # Transform data to model units
            f = d * options.k
            delta = z - d
            del z, d

            if options.fit_mode == magic_calculation.FitMode.EXTEND:
                sl = slice(s)
            elif options.fit_mode == magic_calculation.FitMode.RETRACT:
                sl = slice(s, None)
            else:
                raise ValueError("Unknown fit_mode: ", options.fit_mode)

            property_names_units = {
                "CalcIndentationModulus": "GPa",
                "CalcAdhesionForce": "nN",
                "CalcDeflection": "nm",
                "CalcIndentation": "nm",
                "CalcTrueHeight": "nm",  # Hi z -> lo h
                "CalcIndentationRatio": "",
            }
            properties = np.empty(
                len(f), dtype=np.dtype([(name, "f4") for name in property_names_units]),
            )

            pbar = tqdm_tk(
                total=len(properties),
                desc="Fitting force curves...",
                smoothing=1 / 20 / 3,
                unit="fits",
                mininterval=None,
                tk_parent=self.tkwindow,
                grab=False,
                leave=False,
                cancel_callback=cancel_scope.cancel,
            )
            pbar.tk_window.wm_title("Calculating...")
            pbar.update(0)
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)

            progress_image: matplotlib.image.AxesImage = self.img_ax.imshow(
                np.zeros(img_shape + (4,), dtype="f4"), extent=self.axesimage.get_extent()
            )
            progress_array = progress_image.get_array()

            def draw_helper():
                progress_image.changed()
                progress_image.pchanged()
                self.canvas.draw_idle()

            def calc_properties(i, cancel_poller):
                """This task has essentially private access to delta, f, and properties
                so it's totally cool to do this side-effect-full stuff in threads
                as long as they don't step on each other"""
                cancel_poller()
                beta, beta_err, calc_fun = magic_calculation.fitfun(
                    delta[i, sl],
                    f[i, sl],
                    **dataclasses.asdict(options),
                    cancel_poller=cancel_poller,
                )
                cancel_poller()

                if np.all(np.isfinite(beta)):
                    deflection, indentation, z_true_surface = magic_calculation.calc_def_ind_ztru(
                        f[sl], beta, **dataclasses.asdict(options)
                    )
                    properties[i] = (
                        beta[0],
                        -beta[1],
                        deflection,
                        indentation,
                        -z_true_surface,
                        deflection / indentation,
                    )
                    color = (0, 1, 0, 0.5)
                else:
                    properties[i] = np.nan
                    color = (1, 0, 0, 0.5)
                pbar.update()
                r, c = np.unravel_index(i, img_shape)
                progress_array[r, c, :] = color
                trio.from_thread.run_sync(draw_helper)

            await async_tools.thread_map(
                calc_properties, range(len(f)), async_tools.make_cancel_poller()
            )
            await trio.sleep(0)  # check for race condition cancel at end of pool

        progress_image.remove()
        pbar.close()
        if cancel_scope.cancelled_caught:
            return

        combobox_values = list(self.image_name_menu.cget("values"))
        properties = properties.reshape((*img_shape, -1,))

        # Actually write out results to external world
        for name in property_names_units:
            self.opened_arh5.add_image(name, property_names_units[name], properties[name].squeeze())
            combobox_values.append(name)
        self.image_name_menu.configure(
            values=combobox_values, width=max(map(len, combobox_values)) - 1
        )

    async def change_cmap_callback(self):
        colormap_name = self.colormap_strvar.get()
        # save old clim
        clim = self.axesimage.get_clim()
        async with self.canvas.trio_draw_lock:
            # prevent cbar from getting expanded
            self.axesimage.set_clim(self.image_min, self.image_max)
            # actually change cmap
            self.axesimage.set_cmap(colormap_name)
            # reset everything
            self.customize_colorbar(clim)
            self.canvas.draw_idle()

    async def change_image_callback(self):
        image_name = self.image_name_strvar.get()
        cmap = self.colormap_strvar.get()
        image_array = await self.opened_arh5.get_image(image_name)
        self.image_min = float(np.nanmin(image_array))
        self.image_max = float(np.nanmax(image_array))

        if self.colorbar is not None:
            self.colorbar.remove()
        if self.axesimage is not None:
            self.axesimage.remove()

        scansize = self.opened_arh5.params["scansize"]
        s = (scansize + scansize / len(image_array)) // 2

        async with self.canvas.trio_draw_lock:
            self.axesimage = self.img_ax.imshow(
                image_array,
                origin="lower" if self.opened_arh5.scandown else "upper",
                extent=(-s, s, -s, s,),
                picker=True,
                cmap=cmap,
            )

        xmin, xmax, ymin, ymax = self.axesimage.get_extent()
        rows, cols = self.axesimage.get_size()
        if self.axesimage.origin == "upper":
            ymin, ymax = ymax, ymin
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        array_extent = Bbox([[-0.5, -0.5], [rows - 0.5, cols - 0.5]])
        self.trans = BboxTransform(boxin=data_extent, boxout=array_extent)
        self.invtrans = self.trans.inverted()

        self.img_ax.set_ylabel("Y piezo (nm)")
        self.img_ax.set_xlabel("X piezo (nm)")

        async with self.canvas.trio_draw_lock:
            self.colorbar = self.fig.colorbar(
                self.axesimage, ax=self.img_ax, use_gridspec=True, format=self.cb_fmt
            )
            self.navbar.update()  # let navbar catch new cax in fig
            self.customize_colorbar()

            self.fig.tight_layout()
            self.canvas.draw_idle()

    def customize_colorbar(self, clim=None):
        """MPL keeps stomping on our settings so reset EVERYTHING"""
        self.colorbar.formatter = self.cb_fmt
        self.colorbar.ax.set_navigate(True)
        self.colorbar.solids.set_picker(True)
        self.label_colorbar()
        if clim is None:
            clim = np.nanquantile(self.axesimage.get_array().data, [0.01, 0.99])
        self.colorbar.solids.set_clim(*clim)

    def label_colorbar(self):
        """Surprisingly needed often"""
        image_name = self.image_name_strvar.get()
        self.colorbar.ax.set_ylabel(
            image_name + " (" + self.opened_arh5.get_image_units(image_name) + ")"
        )

    def data_coords_to_array_index(self, x, y):
        return self.trans.transform_point([y, x]).round().astype(int)  # row, column

    def array_index_to_data_coords(self, r, c):
        y, x = self.invtrans.transform_point([r, c])  # row, column
        return x, y

    async def plot_curve_event_response(self, point: ImagePoint, shift_held):
        if shift_held and point in self.existing_points:
            return
        self.existing_points.add(point)  # should be before 1st await
        self.plot_ax.set_autoscale_on(
            True
        )  # XXX: only needed on first plot. Maybe later make optional?
        options = ForceCurveOptions(
            fit_mode=self.fit_intvar.get(),
            disp_kind=self.disp_kind_intvar.get(),
            k=self.opened_arh5.params["k"],
        )

        # Calculation phase
        # Do a few long-running jobs, likely to be canceled
        if not shift_held:
            for cancel_scope in self.cancels_pending:
                cancel_scope.cancel()
            self.cancels_pending.clear()
        async with self.spinner_scope():
            with trio.CancelScope() as cancel_scope:
                self.cancels_pending.add(cancel_scope)
                force_curve = await self.opened_arh5.get_force_curve(point.r, point.c)
                data = await ctrs(
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

            async with self.canvas.trio_draw_lock:
                # Clearing Phase
                # Clear previous artists and reset plots (faster than .clear()?)
                if not shift_held:
                    for artist in self.artists:
                        artist.remove()
                    self.artists.clear()
                    # confusing set stuff because of ASAP addition above
                    self.existing_points.intersection_update({point})
                    self.plot_ax.relim()
                    self.plot_ax.set_prop_cycle(None)

                # Drawing Phase
                # Based options choose plots and collect artists for deletion
                new_artists, color = await trs(draw_force_curve, data, self.plot_ax, options)
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
                    table = await ctrs(
                        partial(
                            self.plot_ax.table,
                            [
                                [
                                    "{:.2f}±{:.2f}".format(data.beta[0], data.beta_err[0],),
                                    "{:.2f}±{:.2f}".format(-data.beta[1], -data.beta_err[1]),
                                    "{:.2f}".format(data.defl),
                                    "{:.2f}".format(data.ind),
                                    "{:.2f}".format(data.defl / data.ind),
                                ],
                            ],
                            loc="top",
                            colLabels=["$M$ (GPa)", "$F_{adh}$ (nN)", "d (nm)", "δ (nm)", "d/δ"],
                            colLoc="center",
                        )
                    )

                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    self.artists.append(table)
                    await trs(partial(self.fig.subplots_adjust, top=0.85))
                self.canvas.draw_idle()

    def freeze_colorbar_response(self):
        self.colorbar.draw_all()
        from matplotlib.contour import ContourSet

        if isinstance(self.colorbar.mappable, ContourSet):
            CS = self.colorbar.mappable
            if not CS.filled:
                self.colorbar.add_lines(CS)
        self.colorbar.stale = True
        self.label_colorbar()
        self.canvas.draw_idle()

    async def mpl_pick_motion_event_callback(self, event):
        mouseevent = getattr(event, "mouseevent", event)
        control_held = event.guiEvent.state & TkState.CONTROL
        if event.name == "motion_notify_event" and not control_held:
            return
        shift_held = mouseevent.guiEvent.state & TkState.SHIFT
        if mouseevent.inaxes is None:
            return
        elif mouseevent.inaxes is self.img_ax:
            if mouseevent.button != MouseButton.LEFT:
                return
            r, c = self.data_coords_to_array_index(mouseevent.xdata, mouseevent.ydata)
            x, y = self.array_index_to_data_coords(r, c)
            point = ImagePoint(r, c, x, y)
            await self.plot_curve_event_response(point, shift_held)
        elif mouseevent.inaxes is self.colorbar.ax:
            if mouseevent.button == MouseButton.LEFT:
                self.colorbar.norm.vmax = max(mouseevent.ydata, self.colorbar.norm.vmin)
            elif mouseevent.button == MouseButton.RIGHT:
                self.colorbar.norm.vmin = min(mouseevent.ydata, self.colorbar.norm.vmax)
            elif mouseevent.button == MouseButton.MIDDLE:
                self.freeze_colorbar_response()
            else:
                return
            # Adjust colorbar scale
            # simple enough, so keep inline
            self.colorbar.solids.set_clim(self.colorbar.norm.vmin, self.colorbar.norm.vmax)
            self.canvas.draw_idle()

    async def mpl_resize_event_callback(self):
        for cancel_scope in self.resize_cancels_pending:
            cancel_scope.cancel()
        self.resize_cancels_pending.clear()
        with trio.CancelScope() as cancel_scope:
            self.resize_cancels_pending.add(cancel_scope)
            async with self.canvas.trio_draw_lock:
                await trs(self.fig.tight_layout, cancellable=False)
        self.resize_cancels_pending.discard(cancel_scope)

    def spinner_start(self):
        self.tkwindow.configure(cursor="watch")
        self.options_frame.configure(cursor="watch")

    def spinner_stop(self):
        self.tkwindow.configure(cursor="arrow")
        self.options_frame.configure(cursor="arrow")

    async def arh5_task(self):

        async with trio.open_nursery() as nursery:
            self.tkwindow.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
            self.canvas.mpl_connect(
                "motion_notify_event",
                partial(nursery.start_soon, self.mpl_pick_motion_event_callback),
            )
            self.canvas.mpl_connect(
                "pick_event", partial(nursery.start_soon, self.mpl_pick_motion_event_callback)
            )
            self.canvas.mpl_connect(
                "resize_event",
                impartial(partial(nursery.start_soon, self.mpl_resize_event_callback)),
            )

            self.colormap_strvar.trace_add(
                "write", impartial(partial(nursery.start_soon, self.change_cmap_callback))
            )

            await nursery.start(self.canvas.idle_draw_task)
            self.image_name_strvar.trace_add(
                "write", impartial(partial(nursery.start_soon, self.change_image_callback))
            )
            # StringVar.set() won't be effective to plot unless it happens after the
            # trace_add AND start(idle_draw_task). accidentally, the plot will be drawn later
            # due to resize, but let's not rely on that!
            for name in ("MapHeight", "ZSensorTrace"):
                if name in self.opened_arh5.image_names:
                    self.image_name_strvar.set(name)
                    break

            self.navbar.teach_navbar_to_use_trio(nursery)
            self.calc_props_button.configure(
                command=partial(nursery.start_soon, self.arh5_prop_map_callback,)
            )
            self.spinner_scope = await nursery.start(
                async_tools.spinner_task, self.spinner_start, self.spinner_stop,
            )
            await trio.sleep_forever()

        # Close phase
        self.options_frame.destroy()
        self.tkwindow.withdraw()  # weird navbar hiccup on close
        self.tkwindow.destroy()


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
            artists.extend(plot_ax.plot(data.ind, data.defl * options.k + data.beta[1], **mopts,))
            artists.extend(plot_ax.plot(0, data.beta[1], **mopts,))
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
    deflection, indentation, z_true_surface = magic_calculation.calc_def_ind_ztru(
        f[sl], beta, **dataclasses.asdict(options)
    )
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
    )


async def ardf_converter(filename, root):
    """Convert ARDF file to ARH5"""
    with trio.CancelScope() as cscope:
        pbar = tqdm_tk(
            tk_parent=root, cancel_callback=cscope.cancel, total=100, unit="%", leave=False
        )
        filename = await async_tools.convert_ardf(filename, "ARDFtoHDF5.exe", True, pbar)
        async with async_tools.AsyncARH5File(filename) as opened_arh5:
            window = ARDFWindow(root, opened_arh5)
            await window.arh5_task()


async def open_callback(root):
    """Open a file using a dialog box, then create a window for data analysis

    """
    # Choose file
    filename = await trs(
        partial(
            filedialog.askopenfilename,
            master=root,
            filetypes=[("AFM Data", "*.h5 *.ARDF"), ("AR HDF5", "*.h5"), ("ARDF", "*.ARDF"),],
        )
    )
    if not filename:
        return  # Cancelled
    filename = trio.Path(filename)

    # choose handler based on file suffix
    suffix = filename.suffix
    if suffix == ".ARDF":
        await ardf_converter(filename, root)
    elif suffix == ".h5":
        async with async_tools.AsyncARH5File(filename) as opened_arh5:
            window = ARDFWindow(root, opened_arh5)
            await window.arh5_task()
    else:
        raise ValueError("Unknown filename suffix: ", suffix)


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
    gc_count = tk.StringVar(top)
    gc_count_label = ttk.Label(top, textvariable=gc_count)
    gc_count_label.pack()
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


async def main_task(root):
    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        # calls root.destroy by default
        root.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)

        # Build menus
        menu_frame = tk.Menu(root, relief="groove", tearoff=False)
        root.config(menu=menu_frame)

        file_menu = tk.Menu(menu_frame, tearoff=False)
        file_menu.add_command(
            label="Open...",
            accelerator="Ctrl+O",
            underline=0,
            command=partial(nursery.start_soon, open_callback, root),
        )
        file_menu.bind("<KeyRelease-o>", func=partial(nursery.start_soon, open_callback, root))
        root.bind_all(
            "<Control-KeyPress-o>", func=impartial(partial(nursery.start_soon, open_callback, root))
        )
        file_menu.add_command(
            label="Quit", accelerator="Ctrl+Q", underline=0, command=nursery.cancel_scope.cancel
        )
        file_menu.bind("<KeyRelease-q>", func=nursery.cancel_scope.cancel)
        root.bind_all("<Control-KeyPress-q>", func=impartial(nursery.cancel_scope.cancel))
        menu_frame.add_cascade(label="File", menu=file_menu, underline=0)

        help_menu = tk.Menu(menu_frame, tearoff=False)
        help_menu.add_command(
            label="About...",
            accelerator=None,
            underline=0,
            command=partial(nursery.start_soon, about_task, root),
        )
        help_menu.bind("<KeyRelease-a>", func=partial(nursery.start_soon, about_task, root))
        root.bind_all(
            "<Control-KeyPress-F1>", func=impartial(partial(nursery.start_soon, about_task, root))
        )
        menu_frame.add_cascade(label="Help", menu=help_menu, underline=0)

        toolbar_frame = ttk.Frame(root)
        # Eventually some buttons go here!
        toolbar_frame.grid(row=0, column=0, sticky="ew")

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
        exc = outcome_.error
        traceback.print_exception(type(exc), exc, exc.__traceback__)


if __name__ == "__main__":
    main()
