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
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
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
        # submit trash to start processes
        cf_fut = await trs(EXECUTOR.submit, int)
        cf_fut.cancel()


async def to_process_run_sync(sync_fn, *args, cancellable=True, limiter=cpu_bound_limiter):
    """Run sync_fn in a separate process

    This is a simple wrapping of concurrent.futures.ProcessPoolExecutor.submit
    that follows the API of trio.to_thread.run_sync, using one thread per call
    up to the limiter total_tokens.

    If submitting many sync_fn calls a slightly more efficient method may be
    to_process_map_unordered."""
    await start_global_executor()

    async with limiter:
        cf_fut = EXECUTOR.submit(sync_fn, *args)
        try:
            return await trs(
                cf_fut.result, cancellable=cancellable, limiter=trio.CapacityLimiter(1)
            )
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


def _release_obo(limiter, future_token):
    trio_token = trio.lowlevel.current_trio_token()

    def limiter_release_callback(cf_fut):
        trio_token.run_sync_soon(limiter.release_on_behalf_of, future_token)

    return limiter_release_callback


async def _top_up_not_done(not_done, sync_fn, job_items, limiter):
    """Top up set of not_done futures while respecting the limiter"""
    if not_done and not limiter.available_tokens:
        # Can't top up now, just bail and wait on cf.wait
        return not_done

    # need to acquire limiter before submitting to executor
    future_token = object()
    # wait here in case of limiter contention
    await limiter.acquire_on_behalf_of(future_token)

    try:
        async for job_item in job_items:
            cf_fut = EXECUTOR.submit(sync_fn, job_item)
            not_done.add(cf_fut)
            # tokens are effectively paired with futures via this callback
            cf_fut.add_done_callback(_release_obo(limiter, future_token))

            future_token = object()
            try:
                limiter.acquire_on_behalf_of_nowait(future_token)
            except trio.WouldBlock:
                # prefer waiting on cf.wait
                break
        else:
            # iterator exhausted, need to release the limiter manually
            limiter.release_on_behalf_of(future_token)
    except trio.Cancelled as e:
        # cancelled while waiting for a job, release limiter manually
        limiter.release_on_behalf_of(future_token)
        raise e

    # not_done can be empty here iff it was empty and no job_items left
    return not_done


async def to_process_map_unordered(
    sync_fn,
    job_items,
    chunksize=1,
    cancellable=True,
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

    not_done = set()
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

    try:
        # use walrus helper to submit jobs lazily to the pool
        while not_done := await _top_up_not_done(not_done, sync_fn, job_items, limiter):
            # use thread to do nonblocking wait
            done, not_done = await trs(
                wait, not_done, None, FIRST_COMPLETED, cancellable=cancellable
            )

            # Send off results one-by-one
            for cf_fut in done:
                # Yes, crash hard if TimeoutError or CancelledError
                # The above waiting logic SHOULD only produce done futures
                # For errors raised in the pool, all bets are off
                res = cf_fut.result(timeout=0)
                if chunksize == 1:
                    await send_chan.send(res)
                else:
                    for r in res:
                        await send_chan.send(r)
    finally:
        for cf_fut in not_done:
            cf_fut.cancel()
        await send_chan.aclose()

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

    Awaiting this function produces an in-memory iterable of the map results.
    Using nursery.start() on this function produces a MemoryRecieveChannel
    with no buffer to receive results as-completed."""
    if task_status is trio.TASK_STATUS_IGNORED:
        buffer = float("inf")
    else:
        buffer = 0
    send_chan, recv_chan = trio.open_memory_channel(buffer)
    task_status.started(recv_chan)

    async def thread_worker(item, send_chan_clone):
        async with send_chan_clone:
            result = await trio.to_thread.run_sync(
                sync_fn, item, cancellable=cancellable, limiter=limiter
            )
            await send_chan_clone.send(result)

    async with send_chan, trio.open_nursery() as nursery:
        for job_item in job_items:
            async with limiter:
                nursery.start_soon(thread_worker, job_item, send_chan.clone())

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
