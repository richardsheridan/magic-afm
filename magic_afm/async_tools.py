"""Magic AFM Async Tools

A collection of standalone tools created to serve the async needs of the GUI
but that have no dependency on any other part of the package.

It is meant to be easy to lift individual items out into other projects.
"""

# Copyright (C) Richard J. Sheridan
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
from itertools import islice

import trio
import trio_parallel

TOOLTIP_CANCEL = None, None, None
LONGEST_IMPERCEPTIBLE_DELAY = 0.032  # seconds

cpu_bound_limiter = trio.CapacityLimiter(os.cpu_count())
trs = partial(trio.to_thread.run_sync, cancellable=True)


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

    async def worker(job_item, task_status):
        # Backpressure: hold limiter for entire task to avoid spawning too many workers
        async with limiter:
            task_status.started()
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
            await nursery.start(worker, job_item)

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
        trio_parallel.run_sync,
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
