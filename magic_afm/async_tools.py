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
import os
from contextlib import asynccontextmanager
from functools import partial, wraps
from multiprocessing import Process, Lock, Semaphore, Pipe
from itertools import islice, count
# noinspection PyUnresolvedReferences
from multiprocessing.reduction import ForkingPickler

import outcome
import psutil
import trio

TOOLTIP_CANCEL = None, None, None
LONGEST_IMPERCEPTIBLE_DELAY = 0.032  # seconds

cpu_bound_limiter = trio.CapacityLimiter(os.cpu_count())
trs = partial(trio.to_thread.run_sync, cancellable=True)


# How long a process will idle waiting for new work before gives up and exits.
# This should be longer than a thread timeout proportionately to startup time.
IDLE_TIMEOUT = 60 * 10
IDLE_PROC_CACHE = []
proc_counter = count()
# This semaphore counts total worker procs
PROC_SEMAPHORE = Semaphore(0)

try:
    PROC_NICE = psutil.BELOW_NORMAL_PRIORITY_CLASS
except AttributeError:
    PROC_NICE = 3

# TODO
# if os.name == "posix":
#     from trio.lowlevel import FdStream as PipeSendStream
#     PipeReceiveStream = PipeSendStream
if os.name == "nt":
    from trio._windows_pipes import PipeReceiveStream, PipeSendStream


class WorkerProc:
    def __init__(self):
        self._worker_lock = Lock()
        self._worker_lock.acquire()
        self._recv_pipe, self._send_pipe = Pipe()
        self._proc = Process(
            target=self._work,
            args=(self._worker_lock, self._recv_pipe, self._send_pipe, PROC_SEMAPHORE),
            name=f"WorkerProc-{next(proc_counter)}",
        )
        self._proc.start()
        self._recv_stream = None
        self._send_stream = None

    @staticmethod
    def _work(lock: Lock, recv_pipe, send_pipe, semaphore: Semaphore):
        semaphore.release()
        psutil.Process().nice(PROC_NICE)
        while lock.acquire(timeout=IDLE_TIMEOUT):
            # We got a job
            fn, args = recv_pipe.recv()
            result = outcome.capture(fn, *args)
            # Tell the cache that we're done and available for a job
            # Unlike the thread cache, it's impossible to deliver the
            # result from the worker process. So shove it onto the queue
            # and hope the receiver delivers the result and marks us idle
            send_pipe.send(result)

            del fn
            del args
            del result
        # Timeout acquiring lock, so we can probably exit.
        # Unlike thread cache, the race condition of someone trying to
        # assign a job as we quit must be checked by the assigning task.
        semaphore.acquire(timeout=0.0)

    # async def send_all(self, buf):
    #     """Implement multiprocessing framing"""
    #     n = len(buf)
    #     if n > 0x7fffffff:
    #         pre_header = struct.pack("!i", -1)
    #         header = struct.pack("!Q", n)
    #         await self._send_stream.send_all(pre_header)
    #         await self._send_stream.send_all(header)
    #         await self._send_stream.send_all(buf)
    #     else:
    #         # For wire compatibility with 3.7 and lower
    #         header = struct.pack("!i", n)
    #         if n > 16384:
    #             # The payload is large so Nagle's algorithm won't be triggered
    #             # and we'd better avoid the cost of concatenation.
    #             await self._send_stream.send_all(header)
    #             await self._send_stream.send_all(buf)
    #         else:
    #             # Issue #20540: concatenate before sending, to avoid delays due
    #             # to Nagle's algorithm on a TCP socket.
    #             # Also note we want to avoid sending a 0-length buffer separately,
    #             # to avoid "broken pipe" errors if the other end closed the pipe.
    #             await self._send_stream.send_all(header + buf)
    #
    # async def receive_some(self):
    #     """Implement multiprocessing framing"""
    #     buf = await self._recv_stream.receive_some(4)
    #     size, = struct.unpack("!i", buf.getvalue())
    #     if size == -1:
    #         buf = await self._recv_stream.receive_some(8)
    #         size, = struct.unpack("!Q", buf.getvalue())
    #     return await self._recv_stream.receive_some(size)

    async def run_sync(self, sync_fn, *args):
        # Rehabilitate pipes on our side to use trio
        if self._send_stream is None:
            self._recv_stream = PipeReceiveStream(self._recv_pipe.fileno())
            self._send_stream = PipeSendStream(self._send_pipe.fileno())
        try:
            self._worker_lock.release()
            await self._send_stream.send_all(ForkingPickler.dumps((sync_fn, args)))
            result = ForkingPickler.loads(await self._recv_stream.receive_some())
        except trio.Cancelled:
            # Cancellation leaves the process in an unknown state so
            # there is no choice but to kill
            self._proc.kill()
            self._proc.join()
            PROC_SEMAPHORE.acquire()
            raise
        return result.unwrap()

    def is_alive(self):
        return self._proc.is_alive()


async def to_process_run_sync(sync_fn, *args, cancellable=False, limiter=cpu_bound_limiter):
    """Run sync_fn in a separate process

    This is a wrapping of multiprocessing.Process that follows the API of
    trio.to_thread.run_sync. The intended use of this function is limited:

    - Circumvent the GIL for CPU-bound functions
    - Make blocking APIs or infinite loops truly cancellable through
      SIGKILL/TerminateProcess without leaking resources
    - TODO: Protect main process from untrusted/crashy code without leaks

    Anything else that works is gravy, normal multiprocessing caveats apply.

    If submitting many sync_fn calls, a slightly more efficient method may be
    to_process_map_unordered."""

    async with limiter:
        proc: WorkerProc
        try:
            while True:
                proc = IDLE_PROC_CACHE.pop()
                # Under normal circumstances workers are waiting on lock.acquire
                # for a new job, but if they time out, they die immediately.
                if proc.is_alive():
                    break
        except IndexError:
            proc = await trio.to_thread.run_sync(WorkerProc)

        try:
            with trio.CancelScope(shield=not cancellable):
                return await proc.run_sync(sync_fn, *args)
        finally:
            if proc.is_alive():
                IDLE_PROC_CACHE.append(proc)


def _chunk_producer(fn, job_items, chunksize):
    while x := tuple(islice(job_items, chunksize)):
        yield fn, x


async def _async_chunk_producer(fn, job_items, chunksize):
    x = []
    async for job_item in job_items:
        x.append(job_item)
        if len(x) == chunksize:
            yield fn, tuple(x)
            x.clear()
    if x:
        yield fn, tuple(x)


def _chunk_consumer(chunk):
    return tuple(map(*chunk))


async def to_sync_runner_map_unordered(
    sync_runner,
    sync_fn,
    job_items,
    chunksize=1,
    cancellable=False,
    limiter=cpu_bound_limiter,
    task_status=trio.TASK_STATUS_IGNORED,
):
    if task_status is trio.TASK_STATUS_IGNORED:
        buffer = float("inf")
    else:
        buffer = 0
    send_chan, recv_chan = trio.open_memory_channel(buffer)
    task_status.started(recv_chan)

    try:
        job_items = iter(job_items)  # Duck type any iterable
    except TypeError as e:
        if not str(e).endswith("object is not iterable"):
            raise e
        job_items = job_items.__aiter__()  # Duck type any async iterable
        if chunksize != 1:
            job_items = _async_chunk_producer(sync_fn, job_items, chunksize)
            sync_fn = _chunk_consumer
    else:
        if chunksize != 1:
            job_items = _chunk_producer(sync_fn, job_items, chunksize)
            sync_fn = _chunk_consumer
        job_items = asyncify_iterator(job_items)

    async def send(item):
        # https://gitter.im/python-trio/general?at=5f7776b9cfe2f9049a1c67f9
        try:
            await send_chan.send(item)
        except trio.Cancelled:
            if cancellable:
                raise
            else:
                send_chan._state.max_buffer_size += 1
                send_chan.send_nowait(item)

    async def worker(job_item):
        # Backpressure: hold limiter for entire task to avoid spawning too many workers
        async with limiter:
            result = await sync_runner(
                sync_fn, job_item, cancellable=cancellable, limiter=trio.CapacityLimiter(1)
            )
            if chunksize == 1:
                await send(result)
            else:
                for r in result:
                    await send(r)

    async with send_chan, trio.open_nursery() as nursery:
        async for job_item in job_items:
            async with limiter:
                nursery.start_soon(worker, job_item)

    if task_status is trio.TASK_STATUS_IGNORED:
        # internal details version
        return recv_chan._state.data

        # Public API version
        # results = []
        # try:
        #     while True:
        #         results.append(recv_chan.receive_nowait())
        # except trio.EndOfChannel:
        #     pass
        # return results


async def to_process_map_unordered(
    sync_fn,
    job_items,
    chunksize=1,
    cancellable=False,
    limiter=cpu_bound_limiter,
    task_status=trio.TASK_STATUS_IGNORED,
):
    """Run many job items in separate processes

    This imitates the multiprocessing.Pool.imap_unordered API, but using
    trio to make the behavior properly nonblocking and cancellable.

    Job items can be any iterable, sync or async, finite or infinite,
    blocking or non-blocking.

    Awaiting this function produces an in-memory iterable of the map results.
    Using nursery.start() on this function produces a MemoryRecieveChannel
    with no buffer to receive results as-completed."""
    return await to_sync_runner_map_unordered(
        to_process_run_sync,
        sync_fn,
        job_items,
        chunksize=chunksize,
        cancellable=cancellable,
        limiter=limiter,
        task_status=task_status,
    )


async def to_thread_map_unordered(
    sync_fn,
    job_items,
    cancellable=False,
    limiter=cpu_bound_limiter,
    task_status=trio.TASK_STATUS_IGNORED,
):
    """Run many job items in separate threads

    This imitates the multiprocessing.dummy.Pool.imap_unordered API, but using
    trio to make the behavior properly nonblocking and cancellable.

    Job items can be any iterable, sync or async, finite or infinite,
    blocking or non-blocking.

    Awaiting this function produces an in-memory iterable of the map results.
    Using nursery.start() on this function produces a MemoryRecieveChannel
    with no buffer to receive results as-completed."""

    return await to_sync_runner_map_unordered(
        trio.to_thread.run_sync,
        sync_fn,
        job_items,
        chunksize=1,
        cancellable=cancellable,
        limiter=limiter,
        task_status=task_status,
    )


async def spinner_task(set_spinner, set_normal, task_status):
    # Invariant: number of open+opening spinner_scopes equals outstanding scopes
    outstanding_scopes = 0
    spinner_start = trio.Event()
    spinner_pending_or_active = trio.Event()

    @asynccontextmanager
    async def spinner_scope():
        nonlocal outstanding_scopes
        nonlocal spinner_start, spinner_pending_or_active
        spinner_start.set()
        outstanding_scopes += 1
        with trio.CancelScope() as cancel_scope:
            try:
                await spinner_pending_or_active.wait()
                # Invariant: spinner_pending_or_active set while any scopes entered
                # Invariant: pending_or_active_cscope entered while any scopes entered
                # (the former ensures the latter)
                yield cancel_scope
            finally:
                assert outstanding_scopes > 0
                outstanding_scopes -= 1
                # Invariant: if zero, event states are equivalent to initial state
                # just after calling task_status.started
                if not outstanding_scopes:
                    # these actions must occur atomically to satisfy the invariant
                    pending_or_active_cscope.cancel()
                    spinner_pending_or_active = trio.Event()
                    spinner_start = trio.Event()

    task_status.started(spinner_scope)
    while True:
        # Invariant: set the spinner once after a delay after spinner_start.set
        await spinner_start.wait()

        spinner_pending_or_active.set()
        with trio.CancelScope() as pending_or_active_cscope:
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY * 5)
            set_spinner()
            await trio.sleep_forever()

        # Allow a short delay after a final scope exits before resetting
        await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
        # Another spinner_scope may have opened during this time.
        # If so, don't change the cursor to avoid blinking unnecessarily
        if not spinner_start.is_set():
            set_normal()


async def asyncify_iterator(iter):
    sentinel = object()

    while True:
        result = await trs(next, iter, sentinel)
        if result is sentinel:
            return
        yield result


def spawn_limit(limiter):
    def actual_decorator(async_fn):
        @wraps(async_fn)
        async def spawn_limit_wrapper(*args, **kwargs):
            async with limiter:
                return await async_fn(*args, **kwargs)

        return spawn_limit_wrapper

    return actual_decorator


async def tooltip_task(show_tooltip, hide_tooltip, show_delay, hide_delay, task_status):
    """Manage a tooltip window visibility, position, and text."""

    send_chan, recv_chan = trio.open_memory_channel(0)
    cancel_scope = trio.CancelScope()  # dummy starter object

    async def single_show_hide(task_status):
        with cancel_scope:
            task_status.started()
            if text is None:
                return
            await trio.sleep(show_delay)
            show_tooltip(x, y, text)
            await trio.sleep(hide_delay)
        hide_tooltip()

    async with trio.open_nursery() as nursery:
        task_status.started(send_chan)
        async for x, y, text in recv_chan:
            cancel_scope.cancel()
            cancel_scope = trio.CancelScope()
            await nursery.start(single_show_hide)


def make_cancel_poller():
    """Uses internal undocumented bits so probably super fragile"""
    _cancel_status = trio.lowlevel.current_task()._cancel_status

    def poll_for_cancel(*args, **kwargs):
        if _cancel_status.effectively_cancelled:
            raise trio.Cancelled._create()

    return poll_for_cancel
