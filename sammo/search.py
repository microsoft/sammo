# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import asyncio
import collections
import copy
import logging
import random
from collections.abc import Callable
import datetime
from pathlib import Path

from beartype import beartype
from beartype.typing import Literal
from orjson import orjson
import pyglove as pg
from tabulate import tabulate

import sammo.base
import sammo.runners
from sammo.base import EvaluationScore
from sammo.components import Output
from sammo.data import DataTable
from sammo.mutators import Mutator
from sammo.compactbars import CompactProgressBars
import sammo.utils as utils

logger = logging.getLogger(__name__)


@beartype
class Optimizer:
    REPORT_COLUMNS = ("objective", "costs")

    def __init__(
        self,
        runner: sammo.base.Runner,
        search_space: Callable[[], Output] | None,
        objective: Callable[[DataTable, DataTable, bool], float],
        maximize: bool = False,
    ):
        self._runner = runner
        self._maximize = maximize
        self._search_space = search_space
        self._objective = objective
        self._state = None
        self._reset()

    def argbest(self, x, key="objective"):
        if not x:
            return None
        return self.argsort(x, key=key)[0]

    def argsort(self, x, key="objective"):
        return sorted(x, key=lambda x: x[key], reverse=self._maximize)

    def break_even(self, baseline_costs, weights=None):
        if weights is None:
            weights = dict()
        search_costs, improvement = 0, 0
        for k in ["input", "output"]:
            improvement += weights.get(k, 1) * (baseline_costs[k] - self.best["costs"][k])
            search_costs += self.fit_costs[k]
        return search_costs * len(self.best["predictions"]) / (improvement or 1)

    def fit(self, dataset: DataTable):
        self.fit_transform(dataset)

    def fit_transform(self, dataset: DataTable) -> DataTable:
        return utils.sync(self.afit_transform(dataset))

    def score(self, dataset: DataTable, **kwargs) -> dict:
        best = self.best_prompt
        pbar = CompactProgressBars().get("inference", total=best.n_minibatches(dataset), show_rate=False)
        y_pred = best.run(self._runner, dataset, progress_callback=pbar.update, **kwargs)
        record = self._candidate_record(best, dataset, y_pred)
        self._state["transform"].append(record)
        return record

    def transform(self, dataset: DataTable, **kwargs) -> DataTable:
        return self.score(dataset, **kwargs)["predictions"]

    @property
    def best(self) -> dict:
        if self._state is None:
            raise ValueError("Need to fit model first.")
        else:
            return self._state["best"]

    @property
    def best_score(self):
        return self.best["objective"]

    @property
    def best_prompt(self) -> Output:
        return self.best["candidate"]

    @property
    def fit_costs(self):
        return self._state["fit_costs"]

    def save(self, fname: str | Path | None = None, **extra_info):
        def default(obj):
            if hasattr(obj, "to_json"):
                return obj.to_json()
            raise TypeError

        if fname is None:
            fname = f"models/{self.__class__.__name__.lower()}_{datetime.date.today():%b_%d_%y}.json"
        fpath = Path(fname)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        state = pg.to_json({**self._state, **extra_info})
        with open(fpath, "wb") as f:
            f.write(orjson.dumps(state, default=default))

    def _candidate_record(self, candidate, dataset, predictions, objective=None):
        if objective is None:
            objective = self._objective
        scored = objective(dataset, predictions)
        return {
            "candidate": candidate,
            **scored.to_dict("objective"),
            "predictions": predictions,
            "costs": dict(input=predictions.outputs.input_cost, output=predictions.outputs.output_cost),
            "parse_errors": predictions.outputs.empty_rate,
        }

    def _reset(self):
        self._state = {
            "fit": list(),
            "transform": list(),
            "fit_costs": None,
            "best": None,
            "priors": None,
            "validation": False,
            "posteriors": None,
        }
        self._runner.reset_costs()
        if hasattr(self, "_action_stats"):
            self._state["priors"] = copy.deepcopy(self._action_stats)

    def show_report(self):
        if self._state is None or not self._state["fit"]:
            raise ValueError("Need to fit model first.")

        table_data = list()
        for x in self._state["fit"]:
            row = {k: x.get(k, "") for k in self.REPORT_COLUMNS} | x["details"]
            if self._state["validation"]:
                row["validation_objective"] = x.get("validation", {"objective": ""})["objective"]
            table_data.append(row)
        print(f"\nFitting log ({len(table_data)} entries):")
        print(tabulate(table_data, headers="keys", maxcolwidths=50))
        self._show_extra_report()

    def _show_extra_report(self):
        pass

    def _updated_best(self):
        self._state["best"] = self.argbest(self._state["fit"])
        logger.info(f"Best: {self._state['best']['objective']}")
        return self._state["best"]["predictions"]

    async def evaluate(
        self,
        candidates: list[Output],
        runner: sammo.base.Runner,
        objective: Callable[[DataTable, DataTable], EvaluationScore],
        dataset: DataTable,
        colbar: CompactProgressBars | None = None,
    ) -> list[dict]:
        if not candidates:
            return list()

        if colbar is None:
            colbar = CompactProgressBars()
        update_when_done = colbar.get("eval", total=len(candidates), position=1, show_time=False).update
        subtasks_total = sum([m.n_minibatches(dataset) for m in candidates])
        subtasks_cb = colbar.get("tasks", total=subtasks_total).update

        evaluation_tasks = list()
        async with asyncio.TaskGroup() as g:
            for i, candidate in enumerate(candidates):
                task = g.create_task(candidate.arun(runner, dataset, subtasks_cb, i))
                task.add_done_callback(update_when_done)
                evaluation_tasks.append(task)

        scored_mutations = list()
        for candidate, y_pred in zip(candidates, evaluation_tasks):
            scored_mutations.append(self._candidate_record(candidate, dataset, y_pred.result(), objective=objective))

        return scored_mutations

    def validate(self, dataset: DataTable, k_best=5):
        k_best_candidates = sorted(self._state["fit"], key=lambda x: x["objective"], reverse=self._maximize)[:k_best]
        validation_scores = utils.sync(
            self.evaluate([x["candidate"] for x in k_best_candidates], self._runner, self._objective, dataset)
        )
        for candidate, score in zip(k_best_candidates, validation_scores):
            candidate["validation"] = score
        self._state["best"] = self.argbest(validation_scores)
        self._state["validation"] = True
        logger.info(f"Best on val: {self._state['best']['objective']}")


@beartype
class BeamSearch(Optimizer):
    REPORT_COLUMNS = ("iteration", "action", "objective", "costs", "parse_errors", "prev_actions")

    def __init__(
        self,
        runner: sammo.base.Runner,
        mutator: Mutator,
        objective: Callable[[DataTable, DataTable, bool], float],
        maximize: bool = True,
        beam_width: int = 4,
        depth: int = 6,
        mutations_per_beam: int = 8,
        n_initial_candidates: int = 1,
        add_previous: bool = False,
        priors: Literal["uniform"] | dict = "uniform",
        max_evals: int | None = None,
    ):
        super().__init__(runner, None, objective, maximize)
        self._mutator = mutator
        self._beam_width = beam_width
        self._depth = depth
        self._n_mutations = mutations_per_beam
        self._n_initial_candidates = n_initial_candidates
        self._add_previous = add_previous
        self._action_stats = collections.defaultdict(lambda: collections.defaultdict(int))
        self._max_evals = max_evals
        if priors != "uniform":
            for k, v in priors.items():
                for k2, v2 in v.items():
                    self._action_stats[k][k2] = v2

    def log(self, depth, items):
        for item in items:
            self._state["fit"].append({"iteration": depth, **item})
        logger.info(f"Best at depth={depth}: {items[0]['objective']}")

    async def afit_transform(
        self,
        dataset: DataTable,
    ) -> DataTable:
        self._reset()
        self._mutator.update_priors(self._action_stats)
        self._mutator.objective = self._objective
        initial_candidates = await self._mutator.get_initial_candidates(self._runner, self._n_initial_candidates)

        colbar = CompactProgressBars()
        depth_pbar = colbar.get("search depth", total=self._depth, show_rate=False)

        active_set = await self.evaluate(
            [c.candidate for c in initial_candidates], self._runner, self._objective, dataset, colbar
        )
        active_set = self.argsort(
            [{**x, "action": c.action, "prev_actions": [c.action]} for c, x in zip(initial_candidates, active_set)]
        )
        self.log(-1, active_set)
        active_set = self._update_active_set(active_set, active_set)
        rng = random.Random(42)

        for d in range(self._depth):
            # Mutate candidates in parallel
            mutation_tasks = list()
            update_pbar = colbar.get("mutate", total=len(active_set), show_time=False, position=1).update

            candidates_for_mutation = self._pick_candidates_for_mutation(active_set, rng)
            async with asyncio.TaskGroup() as g:
                for i, x in enumerate(candidates_for_mutation):
                    task = g.create_task(
                        self._mutator.mutate(
                            x["candidate"],
                            dataset,
                            self._runner,
                            n_mutations=self._n_mutations,
                            random_state=d * self._beam_width * self._n_mutations + i,
                        )
                    )
                    task.add_done_callback(update_pbar)
                    mutation_tasks.append(task)

            # Prune mutation set if necessary
            mutations = list()
            for parent, mutation_task in zip(candidates_for_mutation, mutation_tasks):
                offspring = mutation_task.result()
                if len(offspring) > self._n_mutations:
                    logger.warning(f"Mutate() exceeded max mutations ({self._n_mutations}) with len {len(offspring)}.")
                mutations += [x.with_parent(parent) for x in offspring]

            if self._max_evals:
                n_evals = len(self._state["fit"])
                if len(mutations) + n_evals > self._max_evals:
                    mutations = mutations[: self._max_evals - n_evals]
                    logger.warning(f"Max iterations reached. Truncating mutations to {len(mutations)}.")

            if not mutations:
                break

            # Evaluate candidates in parallel
            scored_mutations = await self.evaluate(
                [m.candidate for m in mutations], self._runner, self._objective, dataset, colbar
            )
            scored_mutations = [
                {**m_scored, "prev_actions": [m.action] + m.parent["prev_actions"], "action": m.action}
                for m, m_scored in zip(mutations, scored_mutations)
            ]
            self.log(d, scored_mutations)
            if self._add_previous:
                scored_mutations += active_set

            active_set = self._update_active_set(active_set, scored_mutations)

            depth_pbar.update()

        colbar.finalize()
        self._update_priors()
        self._state["fit_costs"] = self._runner.costs.to_dict()
        return self._updated_best()

    def _pick_candidates_for_mutation(self, active_set, rng):
        return active_set

    def _update_active_set(self, active_set, scored_mutations):
        return self.argsort(scored_mutations)[: self._beam_width]

    def _candidates_at_depth(self, depth):
        return [x for x in self._state["fit"] if x["iteration"] == depth]

    def _update_priors(self):
        posterior = copy.deepcopy(self._action_stats)
        for d in range(self._depth):
            previous_best = self.argbest(self._candidates_at_depth(d - 1))
            for c in self._candidates_at_depth(d):
                if self._maximize:
                    has_improved = c["objective"] > previous_best["objective"]
                else:
                    has_improved = c["objective"] < previous_best["objective"]
                posterior[c["action"]]["chosen"] += 1
                posterior[c["action"]]["improved"] += int(has_improved)
        self._state["posteriors"] = posterior

    def _show_extra_report(self):
        print("Action stats:")
        print(tabulate([(k, dict(v)) for k, v in self._state["posteriors"].items()], headers=["action", "stats"]))


@beartype
class RegularizedEvolution(BeamSearch):
    def _pick_candidates_for_mutation(self, active_set, rng):
        picked = list()
        for t in range(self._beam_width):
            subset = rng.sample(active_set, k=len(active_set) // 4)
            picked.append(self.argbest(subset))
        return picked

    def _update_active_set(self, active_set, scored_mutations):
        best_offspring = self.argsort(scored_mutations)[: self._beam_width]
        # keep most recent candidates in active set
        return active_set[self._beam_width :] + reversed(best_offspring)


@beartype
class SequentialSearch(BeamSearch):
    def __init__(
        self,
        runner: sammo.base.Runner,
        mutator: Mutator,
        objective: Callable[[DataTable, DataTable, bool], float],
        maximize: bool = True,
        depth: int = 25,
    ):
        super().__init__(
            runner,
            mutator,
            objective,
            maximize=maximize,
            depth=depth,
            beam_width=1,
            add_previous=False,
            n_initial_candidates=1,
        )

    def _updated_best(self):
        self._state["best"] = self._state["fit"][-1]
        logger.info(f"Best: {self._state['best']['objective']}")
        return self._state["best"]["predictions"]


class EnumerativeSearch(Optimizer):
    REPORT_COLUMNS = ("iteration", "action", "objective", "costs", "parse_errors")

    def __init__(
        self,
        runner: sammo.base.Runner,
        search_space: Callable[[], Output],
        objective: Callable[[DataTable, DataTable], EvaluationScore],
        maximize: bool = True,
        algorithm: Literal["grid", "random"] = "grid",
        max_candidates: int | None = None,
        n_evals_parallel: int = 2,
        mutate_from: Output | None = None,
        random_state: int = 42,
    ):
        super().__init__(runner, search_space, objective, maximize)
        self._algorithm = algorithm
        self._max_trials = max_candidates
        self._n_evals_parallel = n_evals_parallel
        self._mutate_from = mutate_from
        self._random_state = random_state

    async def afit_transform(
        self,
        dataset: DataTable,
    ) -> DataTable:
        self._reset()
        traced_search_space = pg.hyper.trace(self._search_space)

        if self._algorithm == "grid":
            total = traced_search_space.dna_spec.space_size
            if self._max_trials and total > self._max_trials:
                total = self._max_trials
        else:
            total = self._max_trials

        pbar = CompactProgressBars()
        candidate_progress = pbar.get("candidate", total, show_rate=False)
        minibatch_progress_callback = None

        semaphore = asyncio.Semaphore(self._n_evals_parallel)

        async def evaluate_point(candidate, iteration, action):
            async with semaphore:
                if self._mutate_from is not None:
                    evolved = self._mutate_from
                    for mutator in candidate:
                        # todo: think of a better interface for this
                        evolved = (await mutator.mutate(evolved, dataset, self._runner))[0].candidate
                    candidate = evolved
                y_pred = await candidate.arun(self._runner, dataset, minibatch_progress_callback)
                candidate_progress.update()
                return {
                    "iteration": iteration,
                    "action": action,
                    **self._candidate_record(candidate, dataset, y_pred),
                }

        running_tasks = list()
        total_minibatches = 0
        async with asyncio.TaskGroup() as tg:
            for i, search_context in enumerate(
                pg.iter(
                    traced_search_space,
                    num_examples=self._max_trials,
                    algorithm=pg.geno.Random(self._random_state) if self._algorithm == "random" else None,
                ),
            ):
                with search_context():
                    current_point = self._search_space()
                # todo: fix this in case of mutators that change the number of minibatches
                total_minibatches += (current_point if self._mutate_from is None else self._mutate_from).n_minibatches(
                    dataset
                )
                minibatch_progress_callback = pbar.get(
                    "minibatches (total)",
                    total_minibatches,
                    show_rate=False,
                ).update
                decisions = search_context.__closure__[0].cell_contents.to_dict("name_or_id", "literal")
                running_tasks.append(tg.create_task(evaluate_point(current_point, i, decisions)))
        self._state["fit"] += [t.result() for t in running_tasks]
        self._state["fit_costs"] = self._runner.costs.to_dict()
        pbar.finalize()
        return self._updated_best()
