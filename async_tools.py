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

import trio
from threadpoolctl import threadpool_limits

LONGEST_IMPERCEPTIBLE_DELAY = 0.032  # seconds

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


async def thread_map(sync_fn, job_items, *args, cancellable=True, limiter=cpu_bound_limiter):

    async def thread_worker(item, i, send_chan):
        async with send_chan:
            result = await trio.to_thread.run_sync(
                sync_fn, item, *args, cancellable=cancellable, limiter=limiter
            )
            await send_chan.send((i, result))

    send_chan, recv_chan = trio.open_memory_channel(0)
    async with trio.open_nursery() as nursery:
        for i, item in enumerate(job_items):
            async with limiter:
                nursery.start_soon(thread_worker, item, i, send_chan.clone())
        await send_chan.aclose()
        results = []
        async for i, result in recv_chan:
            results.append(result)

    return results


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

    def sentinel_next():
        """Returns a sentinel object instead of raising StopIteration"""
        try:
            return next(iter)
        except StopIteration:
            return sentinel

    while True:
        result = await trs(sentinel_next)
        if result is sentinel:
            return
        yield result
