import collections
import threading
import time
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
        self._tk_func_name = root.register(self._tk_func)
        self._q = collections.deque()

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


class button_scope:
    cancel = lambda: None


async def button_task():
    global button_scope
    button_scope.cancel()
    with trio.CancelScope() as button_scope:
        print('task going to sleep')
        await trio.sleep(2)
        print('task slept well!')
        return
    print('task rudely awoken..')


thread_flag = [False]


def button_thread():
    global thread_flag
    thread_flag[0] = True
    local_flag = thread_flag = [False]
    print('thread going to sleep')
    for _ in range(100):
        if local_flag[0]:
            break
        time.sleep(.02)
    else:
        print('thread slept poorly...')
        return
    print('thread rudely awoken..')


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


async def open_task():
    filename = await trs(partial(filedialog.askopenfilename, filetypes=[('AFM Data', '*.h5 *.ARDF'),
                                                                        ('AR HDF5', '*.h5'),
                                                                        ('ARDF', '*.ARDF')]))
    if not filename:
        return  # Cancelled

    filename = trio.Path(filename)
    if filename.suffix == '.ARDF':
        with trio.CancelScope() as cscope:
            filename = await async_tools.convert_ardf(filename, 'ARDFtoHDF5.exe', force=True,
                                                      pbar=PBar(cancel_callback=cscope.cancel))
        if cscope.cancel_called:
            return  # Cancelled

    print(filename)
    data = async_tools.AsyncARH5File(filename)
    async with data as opened_data:
        curve = await opened_data.get_force_curve(0, 0)
    print(curve)


async def amain(root: tk.Tk):
    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        # convenience aliases
        quit_app = nursery.cancel_scope.cancel
        launch = nursery.start_soon
        open_dialog = partial(launch, open_task)

        # calls root.destroy by default
        root.protocol("WM_DELETE_WINDOW", quit_app)
        root.bind_all('<Control-KeyPress-q>', impartial(quit_app))
        root.bind_all('<Control-KeyPress-o>', impartial(open_dialog))

        menuframe = tk.Menu(root, relief='groove', tearoff=False)
        root.config(menu=menuframe)
        file_menu = tk.Menu(menuframe, tearoff=False)
        file_menu.add_command(label='Open...', accelerator='Ctrl+O', underline=0, command=open_dialog, )
        file_menu.bind('<KeyRelease-o>', quit_app)
        file_menu.add_command(label='Quit', accelerator='Ctrl+Q', underline=0, command=quit_app, )
        file_menu.bind('<KeyRelease-q>', quit_app)
        menuframe.add_cascade(label='File', menu=file_menu, underline=0)

        pbar = ttk.Progressbar(root, mode='indeterminate', maximum=20)
        pbar.pack()
        pbar2 = ttk.Progressbar(root, mode='indeterminate', maximum=20)
        pbar2.pack()
        pbar3 = ttk.Progressbar(root, mode='indeterminate', maximum=20)
        pbar3.pack()
        task_button = ttk.Button(root, text='start task', command=partial(launch, button_task))
        task_button.pack()
        thread_button = ttk.Button(root, text='start thread',
                                   command=lambda: threading.Thread(target=button_thread, daemon=True).start())
        thread_button.pack()
        # run using tcl event loop
        pbar.start(20)
        # await trio.sleep_forever()
        # run using trio
        nursery.start_soon(pbar_runner, pbar2, 20)
        nursery.start_soon(pbar_runner_timely, pbar3, 20)


def main():
    root = tk.Tk()
    host = TkHost(root)
    trio.lowlevel.start_guest_run(
        amain,
        root,
        run_sync_soon_threadsafe=host.run_sync_soon_threadsafe,
        done_callback=host.done_callback,
    )
    root.mainloop()


if __name__ == '__main__':
    main()
