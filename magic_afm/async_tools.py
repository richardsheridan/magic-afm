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
from contextlib import asynccontextmanager, nullcontext
from functools import partial
from itertools import islice
from math import inf

import trio

TOOLTIP_CANCEL = None, None, None
LONGEST_IMPERCEPTIBLE_DELAY = 0.032  # seconds

cpu_bound_limiter = trio.CapacityLimiter(os.cpu_count() or 1)
trs = partial(trio.to_thread.run_sync, cancellable=True)


def _chunk_producer(fn, job_items, chunksize):
    while x := tuple(islice(job_items, chunksize)):
        yield fn, x


async def _async_chunk_producer(fn, job_items, chunksize, *, task_status):
    x = []
    send_chan, recv_chan = trio.open_memory_channel(0)
    task_status.started(recv_chan)
    async with send_chan:
        async for job_item in job_items:
            x.append(job_item)
            if len(x) == chunksize:
                await send_chan.send((fn, tuple(x)))
                x.clear()
        if x:
            await send_chan.send((fn, tuple(x)))


def _chunk_consumer(chunk):
    return tuple(map(*chunk))


async def _asyncify_iterator(iter, limiter=None, *, task_status):
    sentinel = object()
    send_chan, recv_chan = trio.open_memory_channel(0)
    task_status.started(recv_chan)
    with send_chan:
        while (
            result := await trs(next, iter, sentinel, limiter=limiter)
        ) is not sentinel:
            await send_chan.send(result)


async def to_sync_runner_map_unordered(
    sync_runner,
    sync_fn,
    job_items,
    chunksize=1,
    cancellable=False,
    limiter=cpu_bound_limiter,
    task_status=trio.TASK_STATUS_IGNORED,
):
    """Run many job items in workers.

    This imitates the multiprocessing.Pool.imap_unordered API, but using
    trio to make the behavior properly nonblocking and cancellable.

    First argument should be trio.to_thread.run_sync or compatible.

    Job items can be any iterable, sync or async, finite or infinite,
    blocking or non-blocking.

    Awaiting this function produces an in-memory iterable of the map results.
    Using nursery.start() on this function produces a MemoryReceiveChannel
    with no buffer to receive results as-completed."""
    chunky = chunksize > 1
    buffer = []
    if task_status is trio.TASK_STATUS_IGNORED:

        async def send(item):
            buffer.append(item)

        send_chan = nullcontext
    else:
        send_chan, recv_chan = trio.open_memory_channel(0)
        task_status.started(recv_chan)
        send = send_chan.send

    async def worker():
        async for job_item in job_items:
            result = await sync_runner(
                sync_fn, job_item, cancellable=cancellable, limiter=limiter
            )
            if chunky:
                for r in result:
                    await send(r)
            else:
                await send(result)

    async with send_chan, trio.open_nursery() as nursery:
        try:
            job_items = iter(job_items)  # Duck type any iterable
        except TypeError as e:
            if not str(e).endswith("object is not iterable"):
                raise e
            job_items = aiter(job_items)  # Duck type any async iterable
            if chunky:
                job_items = await nursery.start(
                    _async_chunk_producer, sync_fn, job_items, chunksize
                )
                sync_fn = _chunk_consumer
        else:
            if chunky:
                job_items = _chunk_producer(sync_fn, job_items, chunksize)
                sync_fn = _chunk_consumer
            job_items = await nursery.start(_asyncify_iterator, job_items, limiter)

        for _ in range(limiter.total_tokens):
            nursery.start_soon(worker)

    return buffer


async def spinner_task(set_spinner, set_normal, task_status):
    # Invariant: number of open+opening spinner_scopes equals outstanding_scopes
    outstanding_scopes = 0
    ending_or_inactive_cscope = trio.CancelScope()
    pending_or_active_cscope = trio.CancelScope()
    spinner_pending_or_active = trio.Event()

    @asynccontextmanager
    async def spinner_scope():
        nonlocal outstanding_scopes, pending_or_active_cscope
        nonlocal ending_or_inactive_cscope, spinner_pending_or_active
        ending_or_inactive_cscope.cancel()
        outstanding_scopes += 1
        try:
            await spinner_pending_or_active.wait()
            # Invariant: spinner_pending_or_active set while any scopes entered
            # Invariant: pending_or_active_cscope entered while any scopes entered
            # (the former ensures the latter)
            yield
        finally:
            assert outstanding_scopes > 0
            outstanding_scopes -= 1
            # Invariant: if zero, event states are equivalent to initial state
            # just after calling task_status.started
            if not outstanding_scopes:
                # these actions must occur atomically to satisfy the invariant
                # because the very next task scheduled may open a spinner_scope
                pending_or_active_cscope.cancel()
                spinner_pending_or_active = trio.Event()
                ending_or_inactive_cscope = trio.CancelScope()
                pending_or_active_cscope = trio.CancelScope()

    task_status.started(spinner_scope)
    while True:
        # always fresh scope thanks to atomic state reset
        with ending_or_inactive_cscope:
            # Allow a short delay after a final scope exits before resetting
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY)
            # Another spinner_scope may have opened during this time.
            # If so, don't change the cursor to avoid blinking unnecessarily.
            set_normal()
            await trio.sleep_forever()

        spinner_pending_or_active.set()

        # always fresh scope thanks to atomic state reset.
        with pending_or_active_cscope:
            # Allow a short delay after a first scope enters before setting.
            await trio.sleep(LONGEST_IMPERCEPTIBLE_DELAY * 5)
            # spinner_scope may exit quickly during this time.
            # If so, don't change the cursor to avoid blinking unnecessarily.
            set_spinner()
            await trio.sleep_forever()


# noinspection PyUnreachableCode
async def tooltip_task(show_tooltip, hide_tooltip, show_delay, hide_delay, task_status):
    """Manage a tooltip window visibility, position, and text."""

    send_chan, recv_chan = trio.open_memory_channel(inf)
    task_status.started(send_chan)
    tooltip_command = TOOLTIP_CANCEL  # a tuple of x, y, text. start hidden

    while True:
        if tooltip_command != TOOLTIP_CANCEL:
            with trio.move_on_after(show_delay):
                tooltip_command = await recv_chan.receive()
                continue
            show_tooltip(*tooltip_command)
            with trio.move_on_after(hide_delay):
                tooltip_command = await recv_chan.receive()
                hide_tooltip()
                continue
            hide_tooltip()
        tooltip_command = await recv_chan.receive()


def make_cancel_poller():
    """Uses internal undocumented bits so probably super fragile"""
    _cancel_status = trio.lowlevel.current_task()._cancel_status

    def poll_for_cancel(*args, **kwargs):
        if _cancel_status.effectively_cancelled:
            raise trio.Cancelled._create()

    return poll_for_cancel
