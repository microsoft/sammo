# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Provides a context manager for throttling async jobs. The throttling is defined by a list of AtMost instance and
can be used to limit the number of concurrent jobs, the number of jobs per time period, the total cost of jobs per
time period, or the number of failed jobs per time period. The context manager will block until there is capacity
to run the job. The jobs are run in order of priority, breaking ties with creation time.
"""
import asyncio
import bisect
from collections import deque
from dataclasses import dataclass, field
import enum
import logging
import threading
import time

from beartype import beartype
from beartype.typing import Literal

__all__ = ["Throttler", "AtMost"]

logger = logging.getLogger(__name__)


class JobStatus(enum.Enum):
    NEW = 0
    RUNNING = 1
    REJECTED = 2
    FAILED = 3
    SUCCESSFUL = 4


@dataclass(order=True)
class Job:
    """Class for keeping track of an async job."""

    priority: int
    id: int
    created: float = field(default_factory=time.perf_counter, compare=False)
    start: float = field(default=0, compare=False)
    end: float = field(default=0, compare=False)
    cost: int = field(default=0, compare=False)
    status: JobStatus = field(default=JobStatus.NEW, compare=False)

    def __str__(self) -> str:
        return f"{self.priority}|{self.id}"

    def get_value(self, property: str) -> float | int | bool:
        if property == "calls":
            return True
        elif property == "cost":
            return self.cost
        else:
            return self.status == JobStatus[property.upper()]

    def time_since_start(self) -> float:
        return time.perf_counter() - self.start

    def time_since_end(self) -> float:
        if self.status in [JobStatus.NEW, JobStatus.RUNNING]:
            return -1
        return time.perf_counter() - self.end


@beartype
@dataclass
class AtMost:
    """Class for defining a throttling limit."""

    value: float | int
    type: Literal["calls", "running", "failed", "rejected"]
    period: float | int = 1
    pause_for: float | int = 0


@beartype
class Throttler:
    """Class that provides flexible throttling for async jobs.

    :param limits: A list of :class:`sammo.throttler.AtMost` instances that define the throttling limits.
    :param  sleep_interval: The time (in s) between checks for capacity.
    :param impute_pending_costs: Whether to estimate the cost of pending jobs with the running average.
    :param n_cost_samples: The number of samples to use when calculating the running average.
    :param rejection_window: The time (in s) within which a job is considered rejected instead of failed.
    """

    DEBUG_INTERVAL_SECONDS = 3

    def __init__(
        self,
        limits: list[AtMost],
        sleep_interval: float = 0.01,
        impute_pending_costs: bool = True,
        n_cost_samples: int = 10,
        rejection_window: int | float = 0.5,
    ):
        self._limits = limits
        self._max_history_window = max([x.period for x in limits] + [60])
        self._sleep_interval = sleep_interval
        self._lock = threading.Lock()
        self._task_logs = deque()
        self._wait_list = deque()
        self._id_counter = 0
        self._cost_samples = list()
        self._running_avg = 0
        self._n_cost_samples = n_cost_samples
        self._impute_pending_costs = impute_pending_costs
        self._rejection_limit_start = None
        self._last_log = 0
        self._daemon_active = None
        self._rejection_window = rejection_window
        rejected = [x for x in limits if x.type == "rejected"]

        if len(rejected) > 1:
            raise ValueError("Only one rejected limit can be specified.")
        elif len(rejected) == 1:
            self._rejection_limit = rejected[0]
        else:
            self._rejection_limit = None

    def _collect_garbage(self) -> None:
        while self._task_logs:
            job = self._task_logs[0]
            if job.time_since_end() > self._max_history_window:
                with self._lock:
                    self._task_logs.popleft()
            else:
                break

    @staticmethod
    async def sleep(delay: float):
        """A more precise sleep function on Windows"""
        await asyncio.get_running_loop().run_in_executor(None, time.sleep, delay)

    def update_job_stats(self, job: Job, cost: float | int = 0, failed: bool = False) -> None:
        """Update the stats for a job. Needs to be called when a job is finished.

        :param job: Job instance to update.
        :param cost: The cost of the job.
        :param failed: Whether the job failed, default is False.
        """
        job.cost = cost
        job.end = time.perf_counter()
        with self._lock:
            n_running = sum([x.get_value("running") for x in self._task_logs])
        if self._daemon_active is not None and n_running <= 1:
            # last job finished, cancel the daemon
            self._daemon_active.cancel()
            self._daemon_active = None
        if failed:
            if job.time_since_start() < self._rejection_window:
                job.status = JobStatus.REJECTED
                if self._rejection_limit is not None:
                    self._rejection_limit_start = job.end
            else:
                job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.SUCCESSFUL
            if self._rejection_limit_start and job.start > self._rejection_limit_start:
                self._rejection_limit_start = None

            self._cost_samples = self._cost_samples[-self._n_cost_samples :] + [cost]
            self._running_avg = sum(self._cost_samples) / (1.0 if self._cost_samples == 0 else len(self._cost_samples))

    async def wait_in_line(self, priority: int = 0) -> Job:
        """Wait async until there is capacity to run a job. The jobs are run in order of priority,
        breaking ties with creation time.

        :param priority: The priority of the job. Lower numbers are higher priority.
        """
        try:
            with self._lock:
                if self._daemon_active is None:
                    self._daemon_active = asyncio.get_event_loop().create_task(self._log_stats())
                my_id = self._id_counter
                self._id_counter += 1
                this_job = Job(priority=priority, id=my_id)
                bisect.insort(self._wait_list, this_job)

            while True:
                if self._has_capacity() and self._wait_list[0].id == my_id:
                    break

                await self.sleep(self._sleep_interval)
            self._wait_list.remove(this_job)
            this_job.cost = self._running_avg if self._impute_pending_costs else 0
            this_job.start = time.perf_counter()
            this_job.status = JobStatus.RUNNING
            self._task_logs.append(this_job)

            self._collect_garbage()
            return this_job

        except asyncio.CancelledError:
            logger.debug(f"Canceling {my_id}")
            self._wait_list.remove(this_job)
            raise

    async def _log_stats(self) -> None:
        while True:
            with self._lock:
                completed = [x for x in self._task_logs if x.time_since_end() <= 60]
                running = [x for x in self._task_logs if x.status == JobStatus.RUNNING]
            successful = [x for x in completed if x.status == JobStatus.SUCCESSFUL]
            failed = [x for x in completed if x.status == JobStatus.FAILED]
            costs = sum([x.cost for x in completed])
            rejected = [x for x in completed if x.status == JobStatus.REJECTED]

            logger.info(
                f"{len(running)} running, "
                f"last minute: {len(successful)} successful ({costs} total costs), "
                f"{len(rejected)} rejected, {len(failed)} failed later"
            )
            await asyncio.sleep(self.DEBUG_INTERVAL_SECONDS)

    def _active_rejection_limit(self) -> list[AtMost]:
        if self._rejection_limit_start:
            return [AtMost(1, "calls", self._rejection_limit.pause_for)]
        else:
            return list()

    def _has_capacity(self) -> bool:
        individual_limits_okay = list()
        with self._lock:
            for limit in self._limits + self._active_rejection_limit():
                if limit.type == "calls":
                    relevant_jobs = [x for x in self._task_logs if x.time_since_start() < limit.period]
                    individual_limits_okay.append(len(relevant_jobs) < limit.value)
                elif limit.type == "running":
                    individual_limits_okay.append(sum([x.get_value("running") for x in self._task_logs]) < limit.value)
                elif limit.type in ["cost", "failed", "rejected"]:
                    relevant_jobs = [x for x in self._task_logs if x.time_since_end() < limit.period]
                    job_count = sum([x.get_value(limit.type) for x in relevant_jobs])
                    individual_limits_okay.append(job_count < limit.value)

        return all(individual_limits_okay)
