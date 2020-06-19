import collections
import tkinter as tk
import traceback
from functools import partial, wraps
from tkinter import ttk, filedialog

import trio
from outcome import Error

import async_tools
from async_tools import trs


def impartial(fn):
    @wraps(fn)
    def impartial_wrapper(*a, **kw):
        return fn()

    return impartial_wrapper


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


async def pbar_runner(pbar, interval):
    interval = interval / 1000
    while True:
        pbar.step()
        await trio.sleep(interval)


async def pbar_runner_timely(pbar, interval):
    interval = interval / 1000
    t = trio.current_time()
    while True:
        t = t + interval
        pbar.step()
        await trio.sleep_until(t)


class MagicGUI:
    def __init__(self, root: tk.Tk):
        self.root = root

    def _build_menus(self, nursery):
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

    async def open_task(self):
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

        print(filename)
        data = async_tools.AsyncARH5File(filename)
        async with data as opened_data:
            curve = await opened_data.get_force_curve(0, 0)
        print(curve)

    async def about_task(self):
        """Display and control the About menu

        Show copyright and version info and maybe something else
        display cute progress bars to diagnose event loops

        """

    async def main(self):
        nursery: trio.Nursery
        async with trio.open_nursery() as nursery:
            # calls root.destroy by default
            self.root.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)

            self._build_menus(nursery)

            pbar = ttk.Progressbar(self.root, mode='indeterminate', maximum=20)
            pbar.pack()
            pbar2 = ttk.Progressbar(self.root, mode='indeterminate', maximum=20)
            pbar2.pack()
            pbar3 = ttk.Progressbar(self.root, mode='indeterminate', maximum=20)
            pbar3.pack()

            # run using tcl event loop
            pbar.start(20)
            # run using trio
            nursery.start_soon(pbar_runner, pbar2, 20)
            nursery.start_soon(pbar_runner_timely, pbar3, 20)

            await trio.sleep_forever()  # needed if nursery never starts a long running child


def main():
    root = tk.Tk()
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
