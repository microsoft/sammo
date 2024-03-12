# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import asyncio
import threading
import time

import pytest
from pytest import approx

from sammo.throttler import Throttler, AtMost


async def simple_job(job_id, throttler, fail=False, delay=0):
    scheduled = time.perf_counter()
    job = await throttler.wait_in_line()
    run = time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    end = time.perf_counter()
    throttler.update_job_stats(job, cost=0, failed=fail)
    return {
        "scheduled": scheduled,
        "start": run,
        "duration": end - scheduled,
        "net_duration": end - run,
        "job_id": job_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("n_jobs,completion_time", [(10, 0.1), (11, 0.2), (20, 0.3)])
async def test_basic_call_limit(n_jobs, completion_time):
    throttler = Throttler([AtMost(10, "calls", 0.1)])

    async with asyncio.TaskGroup() as g:
        jobs = [g.create_task(simple_job(i, throttler)) for i in range(n_jobs)]
    jobs = [j.result() for j in jobs]
    durations = [j["duration"] for j in jobs]
    # provide a relaxed upper bound for max duration to account for differences
    # in executors across test environments
    assert max(durations) <= (completion_time * 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("n_jobs,completion_time", [(10, 0.06), (12, 0.11)])
async def test_basic_running_limit(n_jobs, completion_time, job_duration=0.05):
    throttler = Throttler([AtMost(10, "running")], sleep_interval=0.001)

    async with asyncio.TaskGroup() as g:
        jobs = [g.create_task(simple_job(i, throttler, delay=job_duration)) for i in range(n_jobs)]
    jobs = [j.result() for j in jobs]

    durations = [j["duration"] for j in jobs]
    # provide a relaxed upper bound for max duration to account for differences
    # in executors across test environments
    assert max(durations) <= (completion_time * 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("jobs_with_flags,completion_time", [([True] * 2 + [False] * 5, 0.21)])
async def test_basic_failed_limit(jobs_with_flags, completion_time):
    throttler = Throttler([AtMost(1, "failed", 0.1)], rejection_window=-1, sleep_interval=0.001)

    async with asyncio.TaskGroup() as g:
        jobs = [g.create_task(simple_job(i, throttler, fail=j)) for i, j in enumerate(jobs_with_flags)]
    jobs = [j.result() for j in jobs]

    durations = [j["duration"] for j in jobs]
    # provide a relaxed upper bound for max duration to account for differences
    # in executors across test environments
    assert max(durations) <= (completion_time * 2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "jobs_with_flags,completion_time",
    [([True, False] * 1, 0.11), ([True] * 2 + [False] * 1, 0.21), ([True] * 1 + [False] * 5, 0.11)],
)
async def test_basic_rejected_limit(jobs_with_flags, completion_time):
    throttler = Throttler([AtMost(1, "rejected", 0.1, 0.1)], rejection_window=1, sleep_interval=0.001)

    async with asyncio.TaskGroup() as g:
        jobs = [g.create_task(simple_job(i, throttler, fail=j)) for i, j in enumerate(jobs_with_flags)]
    jobs = [j.result() for j in jobs]
    durations = [j["duration"] for j in jobs]
    # provide a relaxed upper bound for max duration to account for differences
    # in executors across test environments
    assert max(durations) <= (completion_time * 2)
