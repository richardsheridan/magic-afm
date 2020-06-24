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
import tkinter as tk
import traceback
from enum import IntEnum
from functools import partial, wraps
from tkinter import ttk, filedialog

import trio
import trio.testing
from matplotlib.backends import _backend_tk
from matplotlib.backends._backend_tk import FigureCanvasTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from outcome import Error

import async_tools
import magic_calculation
from async_tools import trs


class TKSTATE(IntEnum):
    """AND/OR these with a tk.Event to see which keys were held down during it"""
    SHIFT = 0x0001
    CAPSLOCK = 0x0002
    CONTROL = 0x0004
    LALT = 0x0008
    NUMLOCK = 0x0010
    RALT = 0x0080
    MB1 = 0x0100
    MB2 = 0x0200
    MB3 = 0x0400


class DISP_TYPE(IntEnum):
    zd = 0
    δf = 1


def impartial(fn):
    @wraps(fn)
    def impartial_wrapper(*a, **kw):
        return fn()

    return impartial_wrapper


class AsyncFigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def __init__(self, figure, master=None, resize_callback=None):
        self._parking_lot = trio.lowlevel.ParkingLot()
        self.draw_idle = self._parking_lot.unpark_all
        super(FigureCanvasTk, self).__init__(figure)
        t1, t2, w, h = self.figure.bbox.bounds
        w, h = int(w), int(h)
        self._tkcanvas = tk.Canvas(
            master=master, background="white",
            width=w, height=h, borderwidth=0, highlightthickness=0)
        self._tkphoto = tk.PhotoImage(
            master=self._tkcanvas, width=w, height=h)
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
        assert False, "this should be overridden in __init__"

    async def idle_draw_task(self, task_status=trio.TASK_STATUS_IGNORED):
        task_status.started()
        while True:
            await self._parking_lot.park()
            with trio.fail_after(1):  # assert nothing is chewing up the event loop
                await trio.testing.wait_all_tasks_blocked()
            self.draw()

    def draw(self):
        super().draw()
        _backend_tk.blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))

    def blit(self, bbox=None):
        _backend_tk.blit(
            self._tkphoto, self.renderer._renderer, (0, 1, 2, 3), bbox=bbox)

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
        self._tkphoto = tk.PhotoImage(
            master=self._tkcanvas, width=int(width), height=int(height))
        self._tkcanvas.create_image(
            int(width / 2), int(height / 2), image=self._tkphoto)
        self.resize_event()
        self._update_pointer_position(event)


class PBar:
    def __init__(self, root=None, maximum=100, grab=True, cancel_callback=None):
        self._cancel_callback = cancel_callback
        self._top = tk.Toplevel(root)
        self._top.wm_title('Loading...')
        self._top.protocol("WM_DELETE_WINDOW", lambda: None)
        self._n_var = tk.DoubleVar(self._top, value=0)
        self._text_var = tk.StringVar(self._top)
        self._label = ttk.Label(self._top, textvariable=self._text_var, padding=5, wraplength=600)
        self._label.pack()
        self._pbar = ttk.Progressbar(self._top, maximum=maximum, variable=self._n_var, length=450)
        self._pbar.pack()
        if self._cancel_callback is not None:
            self._butt = ttk.Button(self._top, text='Cancel', command=self.cancel)
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
        self.root.call('after', 'idle', self._tk_func_name)

    def done_callback(self, outcome):
        """End the Tk app.
        """
        print(f"Outcome: {outcome}")
        if isinstance(outcome, Error):
            exc = outcome.error
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        self.root.destroy()


class ARH5Window(tk.Toplevel):
    def embed_figure(self, fig):
        self.canvas = AsyncFigureCanvasTkAgg(fig, self, resize_callback=impartial(fig.tight_layout))
        self.navbar = NavigationToolbar2Tk(self.canvas, self)

        options_frame = ttk.Frame(self)
        image_name_labelframe = ttk.Labelframe(options_frame, text='Current image')
        self.image_name_strvar = tk.StringVar(image_name_labelframe, value='Choose an image...')
        self.image_name_menu = ttk.Combobox(image_name_labelframe, width=12, state='readonly',
                                            textvariable=self.image_name_strvar, )
        self.image_name_menu.pack(side='left')
        image_name_labelframe.pack(side='left')

        disp_labelframe = ttk.Labelframe(options_frame, text='Display type')
        self.disp_type_var = tk.IntVar(disp_labelframe, value=DISP_TYPE.zd.value)
        self.disp_zd_button = ttk.Radiobutton(disp_labelframe, text='z/d',
                                              value=DISP_TYPE.zd.value,
                                              variable=self.disp_type_var)
        self.disp_zd_button.pack(side='left')
        self.disp_deltaf_button = ttk.Radiobutton(disp_labelframe, text='δ/f',
                                                  value=DISP_TYPE.δf.value,
                                                  variable=self.disp_type_var)
        self.disp_deltaf_button.pack(side='left')
        disp_labelframe.pack(side='left')

        fit_labelframe = ttk.Labelframe(options_frame, text='Fit type')
        self.fit_intvar = tk.IntVar(fit_labelframe, value=magic_calculation.FIT_MODE.skip.value)
        self.fit_skip_button = ttk.Radiobutton(fit_labelframe, text='Skip',
                                               value=magic_calculation.FIT_MODE.skip.value,
                                               variable=self.fit_intvar)
        self.fit_skip_button.pack(side='left')
        self.fit_ext_button = ttk.Radiobutton(fit_labelframe, text='Extend',
                                              value=magic_calculation.FIT_MODE.extend.value,
                                              variable=self.fit_intvar)
        self.fit_ext_button.pack(side='left')
        self.fit_ret_button = ttk.Radiobutton(fit_labelframe, text='retract',
                                              value=magic_calculation.FIT_MODE.retract.value,
                                              variable=self.fit_intvar)
        self.fit_ret_button.pack(side='left')
        fit_labelframe.pack(side='left')

        self.navbar.grid(row=0, sticky='we')
        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.canvas.get_tk_widget().grid(row=1, sticky='wens')
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        options_frame.grid(row=2, sticky='we')
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)


async def arh5_task(filename, root):
    # open and parse key variables
    opened_arh5 = async_tools.AsyncARH5File(filename)
    await opened_arh5.ainitialize()
    k = float(opened_arh5.notes['SpringConstant'])

    fig = Figure(figsize=(7, 2.5))
    img_ax, plot_ax = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.35]))
    img_ax.set_anchor('W')
    fmt = ScalarFormatter()
    fmt.set_powerlimits((-2, 2))

    # Build window
    window = ARH5Window(root)
    window.embed_figure(fig)
    window.wm_title(filename.name)
    window.image_name_menu.configure(values=opened_arh5.image_names, width=max(map(len, opened_arh5.image_names)))

    cax = None

    async def change_image_callback():
        image = await opened_arh5.get_image(window.image_name_strvar.get())

        nonlocal cax
        if cax is not None:
            cax.clear()
        img_ax.clear()
        axesimage = img_ax.imshow(image, picker=True)
        colorbar = fig.colorbar(axesimage, cax=cax, ax=img_ax, use_gridspec=True, format=fmt)
        cax = colorbar.ax

        fig.tight_layout()
        fig.canvas.draw_idle()

    plot_pick_cancel = lambda: None
    clear_lot = trio.lowlevel.ParkingLot()
    clear_lock = trio.Lock()

    async def plot_pick_callback(event):
        # Event Phase
        # Unpack, filter event and get local copies of nonlocal state
        with trio.testing.assert_no_checkpoints():
            mouseevent = getattr(event, 'mouseevent', event)
            if mouseevent.inaxes is not img_ax:
                return
            if event.name == 'pick_event' and mouseevent.button != 1:
                return  # have to click left mouse button to see curve
            if event.name == 'motion_notify_event' and not event.guiEvent.state & TKSTATE.CONTROL:
                return  # Have to hold down ctrl to see curves on mouse move
            x, y = int(round(mouseevent.xdata)), int(round(mouseevent.ydata))
            fit_mode = window.fit_intvar.get()
            disp_type = window.disp_type_var.get()

        # Calculation phase
        # Do a few long-running jobs, likely to be canceled
        nonlocal plot_pick_cancel
        plot_pick_cancel()
        with trio.CancelScope() as cancel_scope:
            plot_pick_cancel = cancel_scope.cancel

            z, d, s = await opened_arh5.get_force_curve(y, x)
            # Transform data to model units
            f = d * k
            delta = z - d
            if fit_mode:
                if fit_mode == magic_calculation.FIT_MODE.extend:
                    sl = slice(None, s)
                elif fit_mode == magic_calculation.FIT_MODE.retract:
                    sl = slice(s, None)
                else:
                    raise ValueError('Unknown fit_mode: ', fit_mode)

                beta, beta_err, calc_fun = await trs(magic_calculation.fitfun, delta[sl], f[sl], k, 20, 0,
                                                     fit_mode,
                                                     async_tools.make_cancel_poller())
                f_fit = calc_fun(delta[sl], *beta)
                d_fit = f_fit / k

        if cancel_scope.cancelled_caught:
            return

        # Clearing Phase
        # Clear previous artists and reset plots (faster than .clear()?)
        if not mouseevent.guiEvent.state & TKSTATE.SHIFT:
            async with clear_lock:
                clear_lot.unpark_all()
                # wait for artist removals, then relim
                with trio.fail_after(1):  # assert nothing is chewing up the event loop
                    await trio.testing.wait_all_tasks_blocked()
                plot_ax.relim()
                plot_ax.set_prop_cycle(None)
                plot_ax.set_autoscale_on(True)

        # Drawing Phase
        # Based on local state choose plots and collect artists for deletion
        with trio.testing.assert_no_checkpoints():
            artists = []
            if disp_type == DISP_TYPE.zd:
                artists.extend(plot_ax.plot(z[:s], d[:s]))
                artists.extend(plot_ax.plot(z[s:], d[s:]))
                if fit_mode:
                    artists.extend(plot_ax.plot(z[sl], d_fit, '--'))
            elif disp_type == DISP_TYPE.δf:
                artists.extend(plot_ax.plot(delta[:s], f[:s]))
                artists.extend(plot_ax.plot(delta[s:], f[s:]))
                if fit_mode:
                    artists.extend(plot_ax.plot(delta[sl], f_fit, '--'))
            else:
                raise ValueError('Unknown DISP_TYPE: ', disp_type)
            artists.extend(img_ax.plot(x, y,
                                       marker='X',
                                       markersize=8,
                                       linestyle='',
                                       markeredgecolor='k',
                                       markerfacecolor=artists[0].get_color()))
            fig.canvas.draw_idle()

        # Waiting Phase
        # effectively waiting for a non-shift event in any new task
        await clear_lot.park()
        for artist in artists:
            artist.remove()

    async with trio.open_nursery() as nursery:
        window.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        fig.canvas.mpl_connect('motion_notify_event', partial(nursery.start_soon, plot_pick_callback))
        fig.canvas.mpl_connect('pick_event', partial(nursery.start_soon, plot_pick_callback))

        await nursery.start(fig.canvas.idle_draw_task)
        window.image_name_strvar.trace_add('write', impartial(partial(nursery.start_soon, change_image_callback)))
        # StringVar.set() won't be effective to plot unless it happens after the trace add AND idle_draw_task
        # accidentally, the plot will be drawn later due to resize, but let's not rely on that
        for name in ('MapHeight', 'ZSensorTrace'):
            if name in opened_arh5.image_names:
                window.image_name_strvar.set(name)
                break

        await trio.sleep_forever()

    # open_task close phase
    window.withdraw()  # weird navbar hiccup on close
    window.destroy()
    await opened_arh5.aclose()


async def ardf_converter(filename, nursery, root):
    """Convert ARDF file to ARH5"""
    with trio.CancelScope() as cscope:
        pbar = PBar(root, cancel_callback=cscope.cancel)
        filename = await async_tools.convert_ardf(filename, 'ARDFtoHDF5.exe', True, pbar)
    if cscope.cancel_called:
        return
    nursery.start_soon(arh5_task, filename, root)


async def open_callback(nursery, root):
    """Open a file using a dialog box, then create a window for data analysis

    """
    # Choose file
    filename = await trs(partial(filedialog.askopenfilename,
                                 master=root,
                                 filetypes=[('AFM Data', '*.h5 *.ARDF'),
                                            ('AR HDF5', '*.h5'),
                                            ('ARDF', '*.ARDF')]))
    if not filename:
        return  # Cancelled
    filename = trio.Path(filename)

    # choose handler based on file suffix
    nursery.start_soon(*
                       {'.ARDF': (ardf_converter, filename, nursery, root),
                        '.h5': (arh5_task, filename, root),
                        }[filename.suffix]
                       )


async def about_task(root):
    """Display and control the About menu

    ☒ Make new Toplevel window
    ☒ Show copyright and version info and maybe something else
    ☒ display cute progress bar spinners to diagnose event loops

    """
    top = tk.Toplevel(root)
    app_name = __doc__.split('\n', 1)[0]
    top.wm_title(f'About {app_name}')
    message = tk.Message(top, text=__short_license__)
    message.pack()
    opts = dict(mode='indeterminate', maximum=80, length=300)
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
        while True:
            t = t + interval / 1000
            timely_trio_pbar.step()
            await trio.sleep_until(t)

    # run using tcl event loop
    tk_pbar.start(interval)
    # run using trio
    async with trio.open_nursery() as nursery:
        top.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        nursery.start_soon(pbar_runner)
        nursery.start_soon(pbar_runner_timely)
    top.destroy()


async def main_task(root):
    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        # calls root.destroy by default
        root.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)

        # Build menus
        menu_frame = tk.Menu(root, relief='groove', tearoff=False)
        root.config(menu=menu_frame)

        file_menu = tk.Menu(menu_frame, tearoff=False)
        file_menu.add_command(label='Open...', accelerator='Ctrl+O', underline=0,
                              command=partial(nursery.start_soon, open_callback, nursery, root))
        file_menu.bind('<KeyRelease-o>',
                       func=partial(nursery.start_soon, open_callback, nursery, root))
        root.bind_all('<Control-KeyPress-o>',
                      func=impartial(partial(nursery.start_soon, open_callback, nursery, root)))
        file_menu.add_command(label='Quit', accelerator='Ctrl+Q', underline=0,
                              command=nursery.cancel_scope.cancel)
        file_menu.bind('<KeyRelease-q>',
                       func=nursery.cancel_scope.cancel)
        root.bind_all('<Control-KeyPress-q>',
                      func=impartial(nursery.cancel_scope.cancel))
        menu_frame.add_cascade(label='File', menu=file_menu, underline=0)

        help_menu = tk.Menu(menu_frame, tearoff=False)
        help_menu.add_command(label='About...', accelerator=None, underline=0,
                              command=partial(nursery.start_soon, about_task, root))
        help_menu.bind('<KeyRelease-a>',
                       func=partial(nursery.start_soon, about_task, root))
        root.bind_all('<Control-KeyPress-F1>',
                      func=impartial(partial(nursery.start_soon, about_task, root)))
        menu_frame.add_cascade(label='Help', menu=help_menu, underline=0)

        await trio.sleep_forever()  # needed if nursery never starts a long running child


def main():
    root = tk.Tk()
    # sabotage update command so that we crash instead of deadlocking
    root.tk.call('rename', 'update', 'never_update')
    host = TkHost(root)
    trio.lowlevel.start_guest_run(
        main_task,
        root,
        run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
        done_callback=host.done_callback,
    )
    root.mainloop()


if __name__ == '__main__':
    main()
