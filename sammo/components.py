# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import asyncio
import logging
import math
import warnings

import beartype
from beartype.typing import Callable, Literal
from frozendict import frozendict

import sammo.utils as utils
from sammo.base import (
    Component,
    LLMResult,
    ListComponent,
    ScalarComponent,
    EmptyResult,
    Runner,
    TimeoutResult,
)
from sammo.data import DataTable

from sammo.compactbars import CompactProgressBars
from sammo.scheduler import Scheduler


logger = logging.getLogger(__name__)

__all__ = ["GenerateText", "Output", "Union", "ForEach"]


class GenerateText(ScalarComponent):
    """Call the LLM to generate a response.

    :param child: The child component to run.
    :param name: The name of the component for later querying.
    :param system_prompt: A system prompt to use.
    :param history: A previous chat conversation to continue. Cannot be used with system_prompt.
    :param seed: The local seed to use for caching. Needs to be changed if sampling from LLM multiple times.
    :param randomness: The how deterministic the LLM output should be (typically corresponds to temperature).
    :param max_tokens: The maximum number of tokens to generate, defaulting to max length supported by `Runner`.
    :param on_error: What to do in case the text cannot be generated.
    """

    NEEDS_SCHEDULING = True

    def __init__(
        self,
        child: ScalarComponent,
        name=None,
        system_prompt: str | None = None,
        history: ScalarComponent | None = None,
        seed=0,
        randomness: float = 0,
        max_tokens=None,
        on_error: Literal["raise", "empty_result"] = "empty_result",
    ):
        super().__init__(child, name)

        if history and system_prompt:
            raise ValueError("Cannot specify both history and system_prompt.")
        self._history = history
        self._system_prompt = system_prompt
        self._randomness = randomness
        self._seed = seed
        self._max_tokens = max_tokens
        self._on_error = on_error
        self.dependencies = [self._child, self._history] if self._history else [self._child]

        if seed > 0 and randomness == 0:
            warnings.warn("Seed is being used but randomness is 0.")

    async def _call(
        self,
        runner: Runner,
        context: dict,
        dynamic_context: frozendict | None,
        priority: int = 0,
    ) -> LLMResult:
        y = await self._child(runner, context, dynamic_context)
        if self._history:
            history = (await self._history(runner, context, dynamic_context)).history
        else:
            history = None
        try:
            result = await runner.generate_text(
                y.value,
                priority=priority,
                system_prompt=self._system_prompt,
                history=history,
                randomness=self._randomness,
                seed=self._seed,
                max_tokens=self._max_tokens,
            )
            return result.with_parent(y)
        except Exception as e:
            logger.warning(f"Failed to generate text: {repr(e)}")
            if self._on_error == "empty_result":
                if isinstance(e, asyncio.TimeoutError):
                    return TimeoutResult(repr(e), parent=y)
                else:
                    return EmptyResult(repr(e), parent=y)
            else:
                raise e


class Union(ListComponent):
    """Union of multiple components. Runs all components and returns the union of their results as list.

    :param children: The child components to run.
    :param name: The name of the component for later querying.
    """

    def __init__(self, *children: Component, name: str | None = None):
        if len(children) == 0:
            raise ValueError("Must be given at least one component.")

        super().__init__(children, name)
        self.dependencies = list(children)

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> list[LLMResult]:
        results = list()
        for c in self._child:
            results.append(await c(runner, context, dynamic_context))
        return self._flatten(results)


class ForEach(ListComponent):
    """Run a component for each output element of a child.
    The operator is run in parallel and the results are flattened.

    :param loop_variable: The name of the variable to use for the loop.
    :param child: The child component whose results are looped over.
    :param operator: The operator to run for each element.
    :param name: The name of the component for later querying.
    """

    NEEDS_SCHEDULING = True

    def __init__(
        self,
        loop_variable: str,
        child: ListComponent,
        operator: Component,
        name: str | None = None,
    ):
        super().__init__(operator, name)
        self._loop_variable = loop_variable
        self._collection = child
        self._operator = operator
        self.dependencies = [self._collection] + self._operator.dependencies

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> list[LLMResult]:
        collection = await self._collection(runner, context, dynamic_context)
        tasks = []
        if dynamic_context is None:
            dynamic_context = {}
        if not isinstance(collection, list):
            collection = [collection]

        async with asyncio.TaskGroup() as tg:
            for x in collection:
                tasks.append(
                    tg.create_task(
                        self._operator(
                            runner,
                            context,
                            frozendict({**dynamic_context, self._loop_variable: x}),
                        )
                    )
                )

        return self._flatten([t.result() for t in tasks])


class Minibatch:
    NEEDS_SCHEDULING = True
    __slots__ = (
        "_context",
        "_child",
        "progress_callback",
        "dependencies",
        "original_idx",
        "_runner",
    )

    def __init__(self, child, data, runner, progress_callback=None, original_idx=None):
        self.original_idx = original_idx
        self.dependencies = [child]
        if data is not None:
            self._context = dict(
                data=frozendict(
                    inputs=data.inputs.values,
                    constants=data.constants,
                )
            )
        else:
            self._context = dict()
        self._child = child
        self.progress_callback = progress_callback
        self._runner = runner

    async def __call__(self, *args, **kwargs):
        if "output" not in self._context:
            self._context["output"] = await self._child(self._runner, self._context)
            if self.progress_callback is not None:
                self.progress_callback()
        return self._context["output"]


@beartype.beartype
class Output(Component):
    """Final output of a prompt pipeline. Runs the child component on each batch in the input and has an
    optional extraction step.

    :param child: The child component to run.
    :param minibatch_size: Number of rows to pack into a single prompt.
    """

    def __init__(
        self,
        child: Component,
        minibatch_size=1,
        on_error: Literal["raise", "empty_result", "backoff"] = "raise",
    ):
        super().__init__(child)
        self.row_batch_size = minibatch_size
        self.reshaping_needed = minibatch_size > 1
        self._on_error = on_error

    def run(
        self,
        runner: Runner,
        data: DataTable | list | None = None,
        progress_callback: Callable | bool = True,
        priority: int = 0,
        on_error: Literal["raise", "empty_result", "backoff"] | None = None,
    ) -> DataTable:
        """Synchronous version of `arun`."""
        return utils.sync(self.arun(runner, data, progress_callback, priority, on_error))

    def n_minibatches(self, table: DataTable) -> int:
        """Return the number of minibatches that will be run on the given table.

        :param table: The DataTable to estimate minibatches on.
        """
        n_rows = len(table)
        return math.ceil(n_rows / self.row_batch_size)

    async def arun(
        self,
        runner: Runner,
        data: DataTable | list | None = None,
        progress_callback: Callable | bool = True,
        priority: int = 0,
        on_error: Literal["raise", "empty_result", "backoff"] | None = None,
    ):
        """
        Run the component asynchronously and return a DataTable with the results.

        :param runner: The runner to use.
        :param data: The input data to run on. Can be empty.
        :param progress_callback: Called after each minibatch. If True, shows default progress bar.
                                  If False, shows nothing.
        :param priority: The priority to use for scheduling (highest by default).
        :param on_error: The error handling strategy to use. Backoff re-runs the prompt with minibatch size 1.
        """
        if isinstance(data, list):
            table = DataTable(data)
        elif isinstance(data, DataTable):
            table = data
        elif data is None:
            table = DataTable([None])

        if on_error is None:
            on_error = self._on_error
        if progress_callback is False or (progress_callback is True and len(table) == 1):
            progress_callback = lambda: None
        elif progress_callback is True:
            colbar = CompactProgressBars()
            progress_callback = colbar.get("minibatches", total=self.n_minibatches(table)).update

        jobs = self._create_minibatch_jobs(runner, table, progress_callback)

        scheduler = Scheduler(runner, jobs, base_priority=priority)

        await scheduler.arun()

        results = table.copy()
        rerun = list()
        for job in jobs:
            minibatch_idx = job.original_idx
            result = await job()
            result_len = len(result) if hasattr(result, "__len__") else 1
            if self.reshaping_needed and result_len != len(minibatch_idx):
                if on_error == "raise":
                    raise ValueError(
                        f"Minibatch results do not have right length (need: {len(minibatch_idx)}, got: {result_len})"
                    )
                elif on_error == "empty_result":
                    result = EmptyResult("Number of returned results was inconsistent.", parent=result)
                elif on_error == "backoff":
                    for j in minibatch_idx:
                        rerun.append(Minibatch(self._child, table[j], runner, None, j))
                    result = None
            results.outputs[minibatch_idx] = result

        if rerun:
            colbar = CompactProgressBars()
            progress_callback = colbar.get("reruns", total=len(rerun)).update
            for j in rerun:
                j.progress_callback = progress_callback
            scheduler = Scheduler(runner, rerun, base_priority=priority)
            await scheduler.arun()
            for job in rerun:
                minibatch_idx = job.original_idx
                result = await job()
                results.outputs[minibatch_idx] = result

        return results

    def _create_minibatch_jobs(self, runner: Runner, table: DataTable, progress_callback: Callable):
        jobs = list()
        batches = table.get_minibatch_iterator(self.row_batch_size)
        for _, row_batch in enumerate(batches):
            job = Minibatch(self._child, table[row_batch], runner, progress_callback, row_batch)
            jobs.append(job)
        return jobs
