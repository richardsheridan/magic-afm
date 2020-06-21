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

import collections
import tkinter as tk
import traceback
from functools import partial, wraps
from tkinter import ttk, filedialog

import trio
from matplotlib.backends import _backend_tk
from matplotlib.backends._backend_tk import FigureCanvasTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from outcome import Error

import async_tools
from async_tools import trs


def NavFigure(parent):
    fig = Figure()
    canvas = FigureCanvasTkAgg(fig, parent, )
    navbar = NavigationToolbar2Tk(canvas, parent)
    navbar.grid(row=0, sticky='we')
    parent.grid_rowconfigure(0, weight=0)
    parent.grid_columnconfigure(0, weight=1)
    canvas.get_tk_widget().grid(row=1, sticky='wens')
    parent.grid_rowconfigure(1, weight=1)
    parent.grid_columnconfigure(0, weight=1)
    return fig


def impartial(fn):
    @wraps(fn)
    def impartial_wrapper(*a, **kw):
        return fn()

    return impartial_wrapper


class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def draw(self):
        super(FigureCanvasTkAgg, self).draw()
        _backend_tk.blit(self._tkphoto, self.renderer._renderer, (0, 1, 2, 3))
        # self._master.update_idletasks()

    def blit(self, bbox=None):
        _backend_tk.blit(
            self._tkphoto, self.renderer._renderer, (0, 1, 2, 3), bbox=bbox)
        # self._master.update_idletasks()

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
        # self.draw() # resize event calls self.draw_idle()...

        # a resizing will in general move the pointer position
        # relative to the canvas, so process it as a motion notify
        # event.  An intended side effect of this call is to allow
        # window raises (which trigger a resize) to get the cursor
        # position to the mpl event framework so key presses which are
        # over the axes will work w/o clicks or explicit motion
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
        self.root.call('after', 0, self._tk_func_name)

    def done_callback(self, outcome):
        """End the Tk app.
        """
        print(f"Outcome: {outcome}")
        if isinstance(outcome, Error):
            exc = outcome.error
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        self.root.destroy()


class MagicGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.wm_title(self.__class__.__qualname__)

    async def open_task(self):
        """Open a file using a dialog box, then create a window for data analysis

        """
        filename = await trs(partial(filedialog.askopenfilename,
                                     master=self.root,
                                     filetypes=[('AFM Data', '*.h5 *.ARDF'),
                                                ('AR HDF5', '*.h5'),
                                                ('ARDF', '*.ARDF')]))
        if not filename:
            return  # Cancelled

        filename = trio.Path(filename)
        if filename.suffix == '.ARDF':
            with trio.CancelScope() as cscope:
                pbar = PBar(self.root, cancel_callback=cscope.cancel)
                filename = await async_tools.convert_ardf(filename, 'ARDFtoHDF5.exe', True, pbar)
            if cscope.cancel_called:
                return  # Cancelled
            del pbar

        top = tk.Toplevel(self.root)
        top.wm_title(filename.name)
        fig = NavFigure(top)
        img_ax, plot_ax = fig.subplots(1, 2)
        async with async_tools.AsyncARH5File(filename) as opened_arh5:
            z, d = await opened_arh5.get_force_curve(0, 0)
            plot_ax.plot(z, d)
            async with trio.open_nursery() as nursery:
                top.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
                await trio.sleep_forever()
        top.withdraw()  # weird navbar hiccup on close
        top.destroy()

    async def about_task(self):
        """Display and control the About menu

        ☒ Make new Toplevel window
        ☒ Show copyright and version info and maybe something else
        ☒ display cute progress bar spinners to diagnose event loops

        """
        top = tk.Toplevel(self.root)
        top.wm_title(f'About {self.__class__.__qualname__}')
        shortlicense = """Copyright (C) 2020  Richard J. Sheridan

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
        message = tk.Message(top, text=shortlicense)
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

    async def main(self):
        nursery: trio.Nursery
        async with trio.open_nursery() as nursery:
            # calls root.destroy by default
            self.root.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)

            # Build menus
            menu_frame = tk.Menu(self.root, relief='groove', tearoff=False)
            self.root.config(menu=menu_frame)

            file_menu = tk.Menu(menu_frame, tearoff=False)
            file_menu.add_command(label='Open...', accelerator='Ctrl+O', underline=0,
                                  command=partial(nursery.start_soon, self.open_task), )
            file_menu.bind('<KeyRelease-o>',
                           func=partial(nursery.start_soon, self.open_task))
            self.root.bind_all('<Control-KeyPress-o>',
                               func=impartial(partial(nursery.start_soon, self.open_task)))
            file_menu.add_command(label='Quit', accelerator='Ctrl+Q', underline=0,
                                  command=nursery.cancel_scope.cancel, )
            file_menu.bind('<KeyRelease-q>',
                           func=nursery.cancel_scope.cancel)
            self.root.bind_all('<Control-KeyPress-q>',
                               func=impartial(nursery.cancel_scope.cancel))
            menu_frame.add_cascade(label='File', menu=file_menu, underline=0)

            help_menu = tk.Menu(menu_frame, tearoff=False)
            help_menu.add_command(label='About...', accelerator=None, underline=0,
                                  command=partial(nursery.start_soon, self.about_task), )
            help_menu.bind('<KeyRelease-a>',
                           func=partial(nursery.start_soon, self.about_task))
            self.root.bind_all('<Control-KeyPress-F1>',
                               func=impartial(partial(nursery.start_soon, self.about_task)))
            menu_frame.add_cascade(label='Help', menu=help_menu, underline=0)

            await trio.sleep_forever()  # needed if nursery never starts a long running child


def main():
    root = tk.Tk()
    root.update_idletasks()
    host = TkHost(root)
    app = MagicGUI(root)
    trio.lowlevel.start_guest_run(
        app.main,
        run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
        done_callback=host.done_callback,
    )
    root.mainloop()


if __name__ == '__main__':
    main()
