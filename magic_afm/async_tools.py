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
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from itertools import islice

import trio
from threadpoolctl import threadpool_limits

LONGEST_IMPERCEPTIBLE_DELAY = 0.032  # seconds
EXECUTOR: Optional[ProcessPoolExecutor] = None

internal_threadpool_limiters = threadpool_limits(1)
cpu_bound_limiter = trio.CapacityLimiter(os.cpu_count())
trs = partial(trio.to_thread.run_sync, cancellable=True)


def make_cancel_poller():
    """Uses internal undocumented bits so probably super fragile"""
    _cancel_status = trio.lowlevel.current_task()._cancel_status

    def poll_for_cancel(*args, **kwargs):
        if _cancel_status.effectively_cancelled:
            raise trio.Cancelled._create()

    return poll_for_cancel


async def start_global_executor():
    global EXECUTOR
    if EXECUTOR is None:
        EXECUTOR = ProcessPoolExecutor(cpu_bound_limiter.total_tokens)
        with trio.CancelScope(shield=True):
            # submit trash to start processes
            cf_fut = await trs(EXECUTOR.submit, int)
            cf_fut.cancel()
            import psutil

            def nicifier(pid):
                psutil.Process(pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

            for pid in EXECUTOR._processes:
                await trs(nicifier, pid)


async def to_process_run_sync(sync_fn, *args, limiter=cpu_bound_limiter):
    """Run sync_fn in a separate process

    This is a simple wrapping of concurrent.futures.ProcessPoolExecutor.submit
    that follows the API of trio.to_thread.run_sync.

    If submitting many sync_fn calls a slightly more efficient method may be
    to_process_map_unordered."""
    await start_global_executor()
    done_event = trio.Event()
    trio_token = trio.lowlevel.current_trio_token()

    def done_callback(cf_fut):
        trio_token.run_sync_soon(done_event.set)

    async with limiter:
        cf_fut = EXECUTOR.submit(sync_fn, *args)
        try:
            cf_fut.add_done_callback(done_callback)
            await done_event.wait()
            return cf_fut.result(timeout=0)
        except:  # noqa: E722
            cf_fut.cancel()
            raise


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


async def to_process_map_unordered(
    sync_fn,
    job_items,
    chunksize=1,
    limiter=cpu_bound_limiter,
    task_status=trio.TASK_STATUS_IGNORED,
):
    """Run many job items in separate processes

    This imitates the multiprocessing.Pool.imap_unordered API, but using
    minimal threads to make the behavior properly nonblocking and cancellable.

    Job items can be any iterable, sync or async, finite or infinite,
    blocking or non-blocking.

    Awaiting this function produces an in-memory iterable of the map results.
    Using nursery.start() on this function produces a MemoryRecieveChannel
    with no buffer to receive results as-completed."""
    await start_global_executor()

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

    async def worker(job_item):
        # Backpressure: hold limiter for entire task to avoid spawning too many workers
        async with limiter:
            result = await to_process_run_sync(sync_fn, job_item, limiter=trio.CapacityLimiter(1))
            if chunksize == 1:
                await send_chan.send(result)
            else:
                for r in result:
                    await send_chan.send(r)

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


async def to_thread_map_unordered(
    sync_fn,
    job_items,
    cancellable=True,
    limiter=cpu_bound_limiter,
    task_status=trio.TASK_STATUS_IGNORED,
):
    """Run many job items in separate threads

    This imitates the multiprocessing.dummy.Pool.imap_unordered API, but using
    trio threads to make the behavior properly nonblocking and cancellable.

    Job items can be any iterable, sync or async, finite or infinite,
    blocking or non-blocking.

    Awaiting this function produces an in-memory iterable of the map results.
    Using nursery.start() on this function produces a MemoryRecieveChannel
    with no buffer to receive results as-completed."""
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
    else:
        job_items = asyncify_iterator(job_items)

    async def thread_worker(job_item):
        # Backpressure: hold limiter for entire task to avoid spawning too many workers
        async with limiter:
            result = await trio.to_thread.run_sync(
                sync_fn, job_item, cancellable=cancellable, limiter=trio.CapacityLimiter(1)
            )
            await send_chan.send(result)

    async with send_chan, trio.open_nursery() as nursery:
        async for job_item in job_items:
            async with limiter:
                nursery.start_soon(thread_worker, job_item)

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
                yield cancel_scope
            finally:
                assert outstanding_scopes > 0
                outstanding_scopes -= 1
                # Invariant: if zero, task state is equivalent to initial state
                # just after calling task_status.started
                if not outstanding_scopes:
                    set_normal()
                    # these actions must occur atomically
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


async def asyncify_iterator(iter):
    sentinel = object()

    while True:
        result = await trs(next, iter, sentinel)
        if result is sentinel:
            return
        yield result
