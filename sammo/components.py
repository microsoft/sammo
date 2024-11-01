# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import asyncio
import quattro
import logging
import math
import warnings

import beartype
from beartype.typing import Callable, Literal
from beartype.typing import Union as TUnion
from frozendict import frozendict

import sammo.utils as utils
from sammo.base import (
    Component,
    LLMResult,
    EmptyResult,
    Runner,
    TimeoutResult,
    Result,
    NonEmptyResult,
)
from sammo.data import DataTable

from sammo.compactbars import CompactProgressBars
from sammo.scheduler import Scheduler


logger = logging.getLogger(__name__)

__all__ = ["GenerateText", "Output", "Union", "ForEach"]


class GenerateText(Component):
    """Call the LLM to generate a response.

    :param child: The content component to run.
    :param reference_id: The reference_id of the component for later querying.
    :param system_prompt: A system prompt to use.
    :param history: A previous chat conversation to continue. Cannot be used with system_prompt.
    :param seed: The local seed to use for caching. Needs to be changed if sampling from LLM multiple times.
    :param randomness: The how deterministic the LLM output should be (typically corresponds to temperature).
    :param max_tokens: The maximum number of tokens to generate, defaulting to max length supported by `Runner`.
    :param on_error: What to do in case the text cannot be generated.
    :param runner: If supplied, the given runner will be used for this generation rather than the Runner passed to _call
    """

    NEEDS_SCHEDULING = True

    def __init__(
        self,
        child: Component,
        reference_id=None,
        system_prompt: TUnion[str, None] = None,
        history: TUnion[Component, None] = None,
        seed=0,
        randomness: float = 0,
        max_tokens=None,
        json_mode: bool = False,
        on_error: Literal["raise", "empty_result"] = "empty_result",
        *,  # the following arguments are keyword only
        runner: Runner | None = None,
    ):
        super().__init__(child, reference_id)

        if history and system_prompt:
            raise ValueError("Cannot specify both history and system_prompt.")
        self._history = history
        self._system_prompt = system_prompt
        self._randomness = randomness
        self._seed = seed
        self._max_tokens = max_tokens
        self._on_error = on_error
        self._json_mode = json_mode
        self.dependencies = [self._child, self._history] if self._history else [self._child]
        self._override_runner = runner

        if seed != 0 and randomness == 0:
            warnings.warn("Seed is being used but randomness is 0.")

    async def _call(
        self,
        runner: Runner,
        context: dict,
        dynamic_context: frozendict | None,
        priority: int = 0,
    ) -> LLMResult:
        y = await self._child(runner, context, dynamic_context)
        parents = [y]
        if self._history:
            previous_turn = await self._history(runner, context, dynamic_context)
            parents.append(previous_turn)
            history = previous_turn.history
        else:
            history = None

        try:
            if self._override_runner is not None:
                runner_for_generation = self._override_runner
            else:
                runner_for_generation = runner

            result = await runner_for_generation.generate_text(
                y.value,
                priority=priority,
                system_prompt=self._system_prompt,
                history=history,
                randomness=self._randomness,
                seed=self._seed,
                max_tokens=self._max_tokens,
                json_mode=self._json_mode,
            )
            return result.with_parent(parents).with_op(self)
        except Exception as e:
            logger.warning(f"Failed to generate text: {repr(e)}")
            if self._on_error == "empty_result":
                if isinstance(e, asyncio.TimeoutError):
                    return TimeoutResult(repr(e), parent=parents, op=self)
                else:
                    return EmptyResult(repr(e), parent=parents, op=self)
            else:
                raise e


class Union(Component):
    """Union of multiple components. Runs all components and returns the union of their results as list.

    :param children: The content components to run.
    :param reference_id: The reference_id of the component for later querying.
    """

    def __init__(self, *children: Component, reference_id: str | None = None):
        if len(children) == 0:
            raise ValueError("Must be given at least one component.")

        super().__init__(children, reference_id)
        self.dependencies = self._child

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> Result:
        results = [await c(runner, context, dynamic_context) for c in self._child]
        return NonEmptyResult(self._flatten(results), parent=results, op=self)

    def run(
        self,
        runner: Runner,
        progress_callback: TUnion[Callable, bool] = True,
        priority: int = 0,
    ):
        return utils.sync(self.arun(runner, progress_callback, priority))

    async def arun(
        self,
        runner: Runner,
        progress_callback: TUnion[Callable, bool] = True,
        priority: int = 0,
    ):
        if progress_callback is False:
            progress_callback = lambda: None
        elif progress_callback is True:
            colbar = CompactProgressBars()
            progress_callback = colbar.get("tasks", total=len(self.dependencies)).update

        jobs = [Minibatch(c, None, runner, progress_callback) for c in self.dependencies]
        scheduler = Scheduler(runner, jobs, base_priority=priority)
        await scheduler.arun()
        return [await job() for job in jobs]


class JoinStrings(Union):
    """Join the results of multiple components into a single string.

    :param children: The content components to run.
    :param separator: The separator to use between the strings.
    :param reference_id: The reference_id of the component for later querying.
    """

    def __init__(self, *children: Component, separator: str, reference_id: str | None = None):
        super().__init__(*children, reference_id=reference_id)
        self._separator = separator

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> Result:
        results = [await c(runner, context, dynamic_context) for c in self._child]
        joined = self._separator.join(self._flatten(results))
        return NonEmptyResult(joined, parent=results, op=self)


class ForEach(Component):
    """Run a component for each output element of a content.
    The operator is run in parallel and the results are flattened.

    :param loop_variable: The reference_id of the variable to use for the loop.
    :param child: The content component whose results are looped over.
    :param operator: The operator to run for each element.
    :param reference_id: The reference_id of the component for later querying.
    """

    NEEDS_SCHEDULING = True

    def __init__(
        self,
        loop_variable: str,
        child: Component,
        operator: Component,
        reference_id: str | None = None,
    ):
        super().__init__(operator, reference_id)
        self._loop_variable = loop_variable
        self._collection = child
        self._operator = operator
        self.dependencies = [self._collection] + self._operator.dependencies

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> Result:
        collection = await self._collection(runner, context, dynamic_context)
        tasks = []
        if dynamic_context is None:
            dynamic_context = {}
        assert isinstance(collection, Result)

        async with quattro.TaskGroup() as tg:
            for x in collection.values_as_list():
                tasks.append(
                    tg.create_task(
                        self._operator(
                            runner,
                            context,
                            frozendict({**dynamic_context, self._loop_variable: x}),
                        )
                    )
                )
        results = [t.result() for t in tasks]
        return NonEmptyResult(self._flatten(results), parent=[collection] + results, op=self)


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
    """Final output of a prompt pipeline. Runs the content component on each batch in the input and has an
    optional extraction step.

    :param child: The content component to run.
    :param minibatch_size: Number of rows to pack into a single prompt.
    """

    def __init__(
        self,
        child: Component,
        minibatch_size=1,
        on_error: Literal["raise", "empty_result"] = "raise",
    ):
        super().__init__(child)
        self.row_batch_size = minibatch_size
        self.reshaping_needed = minibatch_size > 1
        self._on_error = on_error

    def run(
        self,
        runner: Runner,
        data: TUnion[DataTable, list, None] = None,
        progress_callback: TUnion[Callable, bool] = True,
        priority: int = 0,
        on_error: TUnion[Literal["raise", "empty_result"], None] = None,
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
        data: TUnion[DataTable, list, None] = None,
        progress_callback: TUnion[Callable, bool] = True,
        priority: int = 0,
        on_error: TUnion[Literal["raise", "empty_result"], None] = None,
    ):
        """
        Run the component asynchronously and return a DataTable with the results.

        :param runner: The runner to use.
        :param data: The input data to run on. Can be empty.
        :param progress_callback: Called after each minibatch. If True, shows default progress bar.
                                  If False, shows nothing.
        :param priority: The priority to use for scheduling (highest by default).
        :param on_error: The error handling strategy to use.
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
        for job in jobs:
            minibatch_idx = job.original_idx
            result = await job()
            assert isinstance(result, Result)
            values = result.values_as_list()

            result_len = len(values) if hasattr(values, "__len__") else 1
            if not self.reshaping_needed:
                results.outputs[minibatch_idx] = NonEmptyResult(result.value, parent=result, op=self)
            elif result_len != len(minibatch_idx):
                if on_error == "raise":
                    raise ValueError(
                        f"Minibatch results do not have right length (need: {len(minibatch_idx)}, got: {result_len})"
                    )
                elif on_error == "empty_result":
                    results.outputs[minibatch_idx] = EmptyResult(
                        "Number of returned results was inconsistent.", parent=result
                    )
            else:
                results.outputs[minibatch_idx] = [NonEmptyResult(v, parent=result, op=self) for v in values]

        return results

    def _create_minibatch_jobs(self, runner: Runner, table: DataTable, progress_callback: Callable):
        jobs = list()
        batches = table.get_minibatch_iterator(self.row_batch_size)
        for _, row_batch in enumerate(batches):
            job = Minibatch(self._child, table[row_batch], runner, progress_callback, row_batch)
            jobs.append(job)
        return jobs
