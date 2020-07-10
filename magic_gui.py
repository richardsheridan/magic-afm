"""MagicAFM GUI

"""
__author__ = "Richard J. Sheridan"
__short_license__ = f"""Copyright (C) 2020  {__author__}

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
import tkinter as tk
import traceback
import enum
from contextlib import asynccontextmanager
from functools import partial, wraps
from tkinter import ttk, filedialog
from typing import Optional, Callable

import numpy as np
import outcome
import trio
import trio.testing
from matplotlib.backend_bases import MouseButton
from matplotlib.backends import _backend_tk
from matplotlib.backends._backend_tk import FigureCanvasTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox, BboxTransform
from matplotlib.widgets import SubplotTool

import async_tools
import magic_calculation
from async_tools import trs, ctrs

LONGEST_IMPERCEPTIBLE_DELAY = 0.016  # seconds


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
                    _backend_tk.blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))
        finally:
            self._idle_task_running = False

    def draw(self):
        super().draw()
        _backend_tk.blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))

    def blit(self, bbox=None):
        _backend_tk.blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3), bbox=bbox)

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


class PBar:
    def __init__(self, root=None, maximum=100, grab=True, cancel_callback=None):
        self._cancel_callback = cancel_callback
        self._top = tk.Toplevel(root)
        self._top.wm_title("Loading...")
        self._top.protocol("WM_DELETE_WINDOW", lambda: None)
        self._n_var = tk.DoubleVar(self._top, value=0)
        self._text_var = tk.StringVar(self._top)
        self._label = ttk.Label(self._top, textvariable=self._text_var, padding=5, wraplength=600)
        self._label.pack()
        self._pbar = ttk.Progressbar(self._top, maximum=maximum, variable=self._n_var, length=450)
        self._pbar.pack()
        if self._cancel_callback is not None:
            self._butt = ttk.Button(self._top, text="Cancel", command=self.cancel)
            self._butt.pack()
        if grab:
            self._top.grab_set()

    def set_description(self, desc):
        self._text_var.set(desc)

    def update(self, value):
        self._pbar.step(value)

    @property
    def n(self):
        return self._n_var.get()

    def cancel(self):
        if self._cancel_callback is not None:
            self._cancel_callback()

    def close(self):
        self._top.destroy()


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


def embed_figure(root, fig, title, image_names):
    window = tk.Toplevel(root)
    window.wm_title(title)
    canvas = AsyncFigureCanvasTkAgg(fig, window,)
    navbar = ImprovedNavigationToolbar2Tk(canvas, window)

    options_frame = ttk.Frame(root)
    image_name_labelframe = ttk.Labelframe(options_frame, text="Current image")
    image_name_strvar = tk.StringVar(image_name_labelframe, value="Choose an image...")
    image_name_menu = ttk.Combobox(
        image_name_labelframe, width=12, state="readonly", textvariable=image_name_strvar,
    )
    image_name_menu.configure(values=image_names, width=max(map(len, image_names)))

    image_name_menu.pack(side="left")
    image_name_labelframe.pack(side="left")

    disp_labelframe = ttk.Labelframe(options_frame, text="Display type")
    disp_kind_intvar = tk.IntVar(disp_labelframe, value=DispKind.zd.value)
    disp_zd_button = ttk.Radiobutton(
        disp_labelframe, text="z/d", value=DispKind.zd.value, variable=disp_kind_intvar
    )
    disp_zd_button.pack(side="left")
    disp_deltaf_button = ttk.Radiobutton(
        disp_labelframe, text="δ/f", value=DispKind.δf.value, variable=disp_kind_intvar
    )
    disp_deltaf_button.pack(side="left")
    disp_labelframe.pack(side="left")

    fit_labelframe = ttk.Labelframe(options_frame, text="Fit type")
    fit_intvar = tk.IntVar(fit_labelframe, value=magic_calculation.FitMode.SKIP.value)
    fit_skip_button = ttk.Radiobutton(
        fit_labelframe, text="Skip", value=magic_calculation.FitMode.SKIP.value, variable=fit_intvar
    )
    fit_skip_button.pack(side="left")
    fit_ext_button = ttk.Radiobutton(
        fit_labelframe,
        text="Extend",
        value=magic_calculation.FitMode.EXTEND.value,
        variable=fit_intvar,
    )
    fit_ext_button.pack(side="left")
    fit_ret_button = ttk.Radiobutton(
        fit_labelframe,
        text="Retract",
        value=magic_calculation.FitMode.RETRACT.value,
        variable=fit_intvar,
    )
    fit_ret_button.pack(side="left")
    fit_labelframe.pack(side="left")

    options_frame.grid(row=1, column=0, sticky="nsew")
    window.bind("<FocusIn>", impartial(options_frame.lift))

    size_grip = ttk.Sizegrip(window)

    navbar.grid(row=0, sticky="we")
    window.grid_rowconfigure(0, weight=0)

    canvas.get_tk_widget().grid(row=1, sticky="wens")
    window.grid_rowconfigure(1, weight=1)
    window.grid_columnconfigure(0, weight=1)

    size_grip.grid(row=2, column=1, sticky="es")
    window.grid_columnconfigure(1, weight=0)
    return window, canvas, navbar, options_frame, image_name_strvar, disp_kind_intvar, fit_intvar


ARH5_FIGURE_SIZE = (8, 2.75)


def create_arh5_figure(figsize=ARH5_FIGURE_SIZE):
    fig = Figure(figsize, frameon=False)
    img_ax, plot_ax = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.35]))
    img_ax.set_anchor("W")
    # Need to pre-load something into these labels for change_image_callback->tight_layout
    plot_ax.set_xlabel(" ")
    plot_ax.set_ylabel(" ")
    plot_ax.set_ylim([-1000, 1000])
    plot_ax.set_xlim([-1000, 1000])
    return fig, img_ax, plot_ax


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
        delta[sl], f[sl], **dataclasses.asdict(options), _poll_for_cancel=cancel_poller,
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


async def spinner_task(set_spinner, set_normal, task_status):
    spinner_starter_sendchan, spinner_starter_recvchan = trio.open_memory_channel(0)
    spinner_stopper_sendchan, spinner_stopper_recvchan = trio.open_memory_channel(0)

    @asynccontextmanager
    async def spinner_scope():
        with trio.CancelScope(shield=True):
            with trio.fail_after(1):  # assert this should never really block
                await spinner_starter_sendchan.send(None)
                stopper = await spinner_stopper_recvchan.receive()
        try:
            yield
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


async def arh5_task(opened_arh5, root):
    k = float(opened_arh5.notes["SpringConstant"])
    scansize = float(opened_arh5.notes["ScanSize"]) * async_tools.NANOMETER_UNIT_CONVERSION

    fig, img_ax, plot_ax = create_arh5_figure()
    (
        window,
        canvas,
        navbar,
        options_frame,
        image_name_strvar,
        disp_kind_intvar,
        fit_intvar,
    ) = embed_figure(root, fig, opened_arh5.h5file_path.name, opened_arh5.image_names)

    fmt = ScalarFormatter()
    fmt.set_powerlimits((-2, 2))
    colorbar: Optional[Colorbar] = None
    axesimage: Optional[AxesImage] = None
    data_coords_to_array_index: Optional[Callable] = None
    array_index_to_data_coords: Optional[Callable] = None

    async def change_image_callback():
        nonlocal colorbar, axesimage
        nonlocal data_coords_to_array_index
        nonlocal array_index_to_data_coords
        image_name = image_name_strvar.get()
        image_array = await opened_arh5.get_image(image_name)

        if colorbar is not None:
            colorbar.remove()
        if axesimage is not None:
            axesimage.remove()

        s = (scansize + scansize / len(image_array)) // 2
        axesimage = img_ax.imshow(
            image_array,
            origin="upper" if opened_arh5.scandown else "lower",
            extent=(-s, s, -s, s,),
            picker=True,
        )

        xmin, xmax, ymin, ymax = axesimage.get_extent()
        rows, cols = axesimage.get_size()
        if axesimage.origin == "upper":
            ymin, ymax = ymax, ymin
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        array_extent = Bbox([[-0.5, -0.5], [rows - 0.5, cols - 0.5]])
        trans = BboxTransform(boxin=data_extent, boxout=array_extent)
        invtrans = trans.inverted()

        def data_coords_to_array_index(x, y):
            return trans.transform_point([y, x]).round().astype(int)  # row, column

        def array_index_to_data_coords(r, c):
            y, x = invtrans.transform_point([r, c])  # row, column
            return x, y

        img_ax.set_ylabel("Y piezo (nm)")
        img_ax.set_xlabel("X piezo (nm)")

        colorbar = fig.colorbar(axesimage, ax=img_ax, use_gridspec=True, format=fmt)
        navbar.update()  # let navbar catch new cax in fig
        # colorbar.ax.set_navigate(True)
        colorbar.solids.set_picker(True)
        colorbar.ax.set_ylabel(opened_arh5.units_map.get(image_name, "Volts"))

        fig.tight_layout()
        canvas.draw_idle()

    cancels_pending = set()
    artists = []
    existing_points = set()

    async def plot_curve_event_response(point: ImagePoint, shift_held):
        if shift_held and point in existing_points:
            return
        existing_points.add(point)  # should be before 1st await
        plot_ax.set_autoscale_on(True)  # XXX: only needed on first plot. Maybe later make optional?
        options = ForceCurveOptions(
            fit_mode=fit_intvar.get(), disp_kind=disp_kind_intvar.get(), k=k
        )

        # Calculation phase
        # Do a few long-running jobs, likely to be canceled
        if not shift_held:
            for cancel_scope in cancels_pending:
                cancel_scope.cancel()
            cancels_pending.clear()
        async with spinner_scope():
            with trio.CancelScope() as cancel_scope:
                cancels_pending.add(cancel_scope)
                force_curve = await opened_arh5.get_force_curve(point.r, point.c)
                data = await ctrs(
                    calculate_force_data, *force_curve, options, async_tools.make_cancel_poller()
                )
                del force_curve  # contained in data
                # XXX: Race condition
                # cancel can occur after the last call to the above cancel poller
                # checkpoint make sure to raise Cancelled if it just happened
                # await trio.sleep(0)

            cancels_pending.discard(cancel_scope)

            if cancel_scope.cancelled_caught:
                existing_points.discard(point)
                return

            async with canvas.trio_draw_lock:
                # Clearing Phase
                # Clear previous artists and reset plots (faster than .clear()?)
                if not shift_held:
                    for artist in artists:
                        artist.remove()
                    artists.clear()
                    # confusing set stuff because of ASAP addition above
                    existing_points.intersection_update({point})
                    plot_ax.relim()
                    plot_ax.set_prop_cycle(None)

                # Drawing Phase
                # Based options choose plots and collect artists for deletion
                new_artists, color = draw_force_curve(data, plot_ax, options)
                artists.extend(new_artists)
                artists.extend(
                    img_ax.plot(
                        point.x,
                        point.y,
                        marker="X",
                        markersize=8,
                        linestyle="",
                        markeredgecolor="k",
                        markerfacecolor=color,
                    )
                )
                canvas.draw_idle()

    async def mpl_pick_motion_event_callback(event):
        mouseevent = getattr(event, "mouseevent", event)
        control_held = event.guiEvent.state & TkState.CONTROL
        if event.name == "motion_notify_event" and not control_held:
            return
        shift_held = mouseevent.guiEvent.state & TkState.SHIFT
        if mouseevent.inaxes is None:
            return
        elif mouseevent.inaxes is img_ax:
            if mouseevent.button != MouseButton.LEFT:
                return
            r, c = data_coords_to_array_index(mouseevent.xdata, mouseevent.ydata)
            x, y = array_index_to_data_coords(r, c)
            point = ImagePoint(r, c, x, y)
            await plot_curve_event_response(point, shift_held)
        elif mouseevent.inaxes is colorbar.ax:
            if mouseevent.button == MouseButton.LEFT:
                colorbar.norm.vmax = max(mouseevent.ydata, colorbar.norm.vmin)
            elif mouseevent.button == MouseButton.RIGHT:
                colorbar.norm.vmin = min(mouseevent.ydata, colorbar.norm.vmax)
            else:
                return
            # Adjust colorbar scale
            # simple enough, so keep inline
            colorbar.solids.set_clim(colorbar.norm.vmin, colorbar.norm.vmax)
            canvas.draw_idle()

    resize_cancels_pending = set()

    async def mpl_resize_event_callback():
        for cancel_scope in resize_cancels_pending:
            cancel_scope.cancel()
        resize_cancels_pending.clear()
        with trio.CancelScope() as cancel_scope:
            resize_cancels_pending.add(cancel_scope)
            async with canvas.trio_draw_lock:
                await trs(fig.tight_layout, cancellable=False)
        resize_cancels_pending.discard(cancel_scope)

    async with trio.open_nursery() as nursery:
        window.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        canvas.mpl_connect(
            "motion_notify_event", partial(nursery.start_soon, mpl_pick_motion_event_callback)
        )
        canvas.mpl_connect(
            "pick_event", partial(nursery.start_soon, mpl_pick_motion_event_callback)
        )
        canvas.mpl_connect(
            "resize_event", impartial(partial(nursery.start_soon, mpl_resize_event_callback))
        )

        await nursery.start(canvas.idle_draw_task)
        image_name_strvar.trace_add(
            "write", impartial(partial(nursery.start_soon, change_image_callback))
        )
        # StringVar.set() won't be effective to plot unless it happens after the
        # trace_add AND start(idle_draw_task). accidentally, the plot will be drawn later
        # due to resize, but let's not rely on that!
        for name in ("MapHeight", "ZSensorTrace"):
            if name in opened_arh5.image_names:
                image_name_strvar.set(name)
                break

        navbar.teach_navbar_to_use_trio(nursery)
        spinner_scope = await nursery.start(
            spinner_task,
            partial(window.configure, cursor="watch"),
            partial(window.configure, cursor="arrow"),
        )
        await trio.sleep_forever()

    # Close phase
    window.withdraw()  # weird navbar hiccup on close
    window.destroy()
    options_frame.destroy()


async def ardf_converter(filename, root):
    """Convert ARDF file to ARH5"""
    with trio.CancelScope() as cscope:
        pbar = PBar(root, cancel_callback=cscope.cancel)
        filename = await async_tools.convert_ardf(filename, "ARDFtoHDF5.exe", True, pbar)
        async with async_tools.AsyncARH5File(filename) as opened_arh5:
            await arh5_task(opened_arh5, root)


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
            await arh5_task(opened_arh5, root)
    else:
        raise ValueError("Unknown filename suffix: ", suffix)


async def about_task(root):
    """Display and control the About menu

    ☒ Make new Toplevel window
    ☒ Show copyright and version info and maybe something else
    ☒ display cute progress bar spinners to diagnose event loops

    """
    top = tk.Toplevel(root)
    app_name = __doc__.split("\n", 1)[0]
    top.wm_title(f"About {app_name}")
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
        temp_text = ttk.Label(toolbar_frame, text="Eventually some buttons go here!")
        temp_text.pack()
        toolbar_frame.grid(row=0, column=0, sticky="ew")

        await trio.sleep_forever()  # needed if nursery never starts a long running child


def main():
    # make root/parent passing mandatory.
    tk.NoDefaultRoot()
    root = tk.Tk()
    root.wm_resizable(False, False)
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
