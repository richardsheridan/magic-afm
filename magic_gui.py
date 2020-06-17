import collections
import tkinter as tk
import traceback
from tkinter import ttk

import trio
from outcome import Error


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
        # self.master.after(0, func) # does a fairly intensive wrapping to each func
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


buttonscope = None


async def afn():
    global buttonscope
    if buttonscope is not None:
        buttonscope.cancel()
    with trio.CancelScope() as buttonscope:
        await trio.sleep(0.01)  # give someone a chance to cancel us
        print('going to sleep')
        await trio.sleep(2)
        print('slept well!')
        return
    print('rudely awoken..')


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


async def amain(root: tk.Tk):
    nursery: trio.Nursery
    async with trio.open_nursery() as nursery:
        # calls root.destroy by default
        root.protocol("WM_DELETE_WINDOW", nursery.cancel_scope.cancel)
        pbar = ttk.Progressbar(root, mode='indeterminate', maximum=20)
        pbar.pack()
        pbar2 = ttk.Progressbar(root, mode='indeterminate', maximum=20)
        pbar2.pack()
        pbar3 = ttk.Progressbar(root, mode='indeterminate', maximum=20)
        pbar3.pack()
        button = tk.Button(root, text='button text', command=lambda: nursery.start_soon(afn))
        button.pack()
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
