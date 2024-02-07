# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
import asyncio
import collections
import logging
import random
import re
import textwrap

from beartype.typing import Callable, Literal, Any
import numpy as np
import pyglove as pg
import spacy

import sammo.base
import sammo.runners
from sammo.base import Component, Template, Runner, CompiledQuery
from sammo.components import Output, GenerateText, Union, ForEach
from sammo.instructions import FewshotExamples
from sammo.data import DataTable
from sammo.extractors import ExtractRegex
from sammo.search_op import get_points_from_search_space

logger = logging.getLogger(__name__)


class MutatedCandidate:
    __slots__ = ["action", "candidate", "parent", "extra_info"]

    def __init__(self, action, candidate, **kwargs):
        self.action = action
        self.candidate = candidate
        self.parent = None
        self.extra_info = kwargs

    def with_parent(self, parent):
        self.parent = parent
        return self


@pg.symbolize(eq=True)
class Mutator(abc.ABC):
    def __init__(
        self, starting_prompt: Output | Callable | None = None, seed: int = 42, sample_for_init_candidates: bool = True
    ):
        self._starting_prompt = starting_prompt
        self._seed = seed
        self._priors = None
        self._objective = None
        self._sample_for_init_candidates = sample_for_init_candidates

    def applicable(self, candidate: Output) -> bool:
        """Returns True if this mutator can be applied to a candidate."""
        return True

    @property
    def priors(self):
        return self._priors

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        self._objective = value

    def update_priors(self, priors: dict[dict]):
        self._priors = {k: (v["improved"], v["chosen"] - v["improved"]) for k, v in priors.items()}

    @abc.abstractmethod
    async def mutate(
        self,
        candidate: Output,
        data: DataTable,
        runner: sammo.base.Runner,
        n_mutations: int = 1,
        random_state: int = 42,
    ) -> list[MutatedCandidate]:
        """Returns a list of at most n_mutations new candidates."""
        pass

    async def get_initial_candidates(
        self, runner: sammo.base.Runner | None, n_initial_candidates: int
    ) -> list[MutatedCandidate]:
        if self._starting_prompt is None:
            raise ValueError("In order to use search, this method needs to be overriden or starting prompt passed.")

        candidates, names = get_points_from_search_space(
            self._starting_prompt, n_initial_candidates, self._sample_for_init_candidates, self._seed, return_names=True
        )
        if not names:
            names = ["init"] * len(candidates)
        return [MutatedCandidate(name, c) for name, c in zip(names, candidates)]


class SyntaxTreeMutator(Mutator):
    syntax_parser = None

    def __init__(
        self,
        path_descriptor: str | dict,
        starting_prompt: Output | Callable,
        cache: collections.abc.MutableMapping | None = None,
    ):
        super().__init__(starting_prompt)
        self._path_descriptor = CompiledQuery.from_path(path_descriptor)
        self._full_performance = None
        self._cache = {} if cache is None else cache
        self._state = dict()

    @staticmethod
    def parse_document(raw_doc):
        if SyntaxTreeMutator.syntax_parser is None:
            import spacy
            import benepar

            SyntaxTreeMutator.syntax_parser = spacy.load("en_core_web_sm")
            SyntaxTreeMutator.syntax_parser.add_pipe("benepar", config={"model": "benepar_en3"})
        doc = SyntaxTreeMutator.syntax_parser(raw_doc)
        return doc

    @classmethod
    def _merge_subtree(cls, node, merge_threshold=0.5):
        n_non_phrase_leafs = 0
        children = list(node._.children)
        for child in children:
            n_non_phrase_leafs += int(len(child._.labels) == 0)
        return n_non_phrase_leafs / len(children) >= merge_threshold

    @classmethod
    def split_into_phrases(cls, sent):
        queue = [sent]
        phrases = list()
        while queue:
            node = queue.pop(-1)
            children = list(node._.children)
            if not children or cls._merge_subtree(node):
                phrases = [node.text_with_ws] + phrases
            else:
                queue += children
        return phrases

    @classmethod
    def get_phrases(cls, raw_doc):
        phrases = [cls.split_into_phrases(s) for s in cls.parse_document(raw_doc).sents]
        return sum(phrases, [])

    def applicable(self, candidate: Output):
        return candidate.query(self._path_descriptor) is not None

    @staticmethod
    def replace_all_elements(l, needle, replacement):
        return [replacement if e.lower() == needle.lower() else e for e in l]

    @staticmethod
    def delete_all_elements(l, needle):
        return [e for e in l if e.lower() != needle.lower()]

    @staticmethod
    def _join_all_elements(l):
        return re.sub(r"\s+", " ", " ".join(l))

    @staticmethod
    def swap_all_elements(lst, x, y):
        result = list()
        for e in lst:
            if e.lower() == x.lower():
                result.append(y)
            elif e.lower() == y.lower():
                result.append(x)
            else:
                result.append(e)
        return result

    async def mutate(
        self,
        candidate: Output,
        data: DataTable,
        runner: sammo.base.Runner,
        n_mutations: int = 1,
        random_state: int = 42,
    ) -> list[MutatedCandidate]:
        if id(candidate) not in self._state:
            instructions = candidate.query(self._path_descriptor)
            if hasattr(instructions, "static_text"):
                instructions = instructions.static_text()
            else:
                instructions = str(instructions)
            if instructions not in self._cache:
                self._cache[instructions] = self.get_phrases(instructions)
            self._state[id(candidate)] = dict(phrases=self._cache[instructions], deleted=list())

        current_phrases = self._state[id(candidate)]["phrases"].copy()
        non_punctuation = [x for x in self._state[id(candidate)]["phrases"] if len(x.strip()) > 1]

        deleted = self._state[id(candidate)]["deleted"].copy()
        actions_to_sample = (
            ["par", "del"] * (len(non_punctuation) > 0)
            + ["add"] * (len(deleted) > 0)
            + ["swap"] * (len(current_phrases) > 1)
        )
        rng = random.Random(random_state)
        if len(actions_to_sample) == 0:
            logger.debug("SyntaxTree: No actions to sample")
            return list()

        actions = rng.choices(actions_to_sample, k=n_mutations)

        new_candidates = [None] * len(actions)

        for i, action in enumerate(actions):
            if action == "del":
                target = rng.choice(non_punctuation)
                new_candidates[i] = {
                    "action": action,
                    "phrases": self.delete_all_elements(current_phrases, target),
                    "deleted": deleted + [target],
                }
            elif action == "par":
                target = rng.choice(non_punctuation)
                new_candidates[i] = {"action": action, "phrases": current_phrases, "deleted": deleted, "target": target}
            elif action == "swap":
                p_1, p_2 = rng.sample(current_phrases, 2)
                new_candidates[i] = {
                    "action": action,
                    "phrases": self.swap_all_elements(current_phrases, p_1, p_2),
                    "deleted": deleted,
                }
            elif action == "add":
                target = rng.choice(deleted)
                new_deleted = [d for d in deleted if d != target]
                new_phrases = current_phrases.copy()
                new_phrases.insert(rng.randint(0, len(current_phrases) + 1), target)
                new_candidates[i] = {"action": action, "phrases": new_phrases, "deleted": new_deleted}
            else:
                raise ValueError(f"Unknown action {action}")

        to_pharaphrase = [x | {"i": i} for i, x in enumerate(new_candidates) if x["action"] == "par"]
        if to_pharaphrase:
            paraphrased = (
                await Output(
                    Union(
                        *(
                            GenerateText(
                                Template(
                                    "Generate a variation of the following text while keeping the semantic meaning."
                                    "\nInput: {{{text}}}\nOutput: ",
                                    text=x["target"],
                                ),
                                randomness=0.9,
                                seed=random_state,
                            )
                            for x in to_pharaphrase
                        )
                    )
                ).arun(runner)
            ).outputs.raw_values[0]
            for o, x in zip(paraphrased, to_pharaphrase):
                new_candidates[x["i"]] = x | {
                    "phrases": self.replace_all_elements(current_phrases, x["target"], o.value)
                }

        candidates = list()
        for new_cand in new_candidates:
            mutated_cand = candidate.replace_static_text(
                self._path_descriptor, self._join_all_elements(new_cand["phrases"])
            )
            self._state[id(mutated_cand)] = new_cand
            candidates.append(MutatedCandidate(new_cand["action"], mutated_cand))
        return candidates


class PruneSyntaxTree(SyntaxTreeMutator):
    def __init__(
        self,
        path_descriptor: str | dict,
        starting_prompt: Output | Callable,
        prune_metric: Callable,
        cache: collections.abc.MutableMapping | None = None,
    ):
        super().__init__(path_descriptor, starting_prompt, cache)
        self._current_prompt = ["0"]
        self._queue = ["0"]
        self._full_performance = None
        self._prune_metric = prune_metric
        self._cache = {} if cache is None else cache
        self._syntax_tree = None

    @classmethod
    def _json_parse_tree(cls, raw_doc: str) -> dict:
        """Construct JSON representation of a parse tree."""
        doc = cls.parse_document(raw_doc)
        n_sents = len(list(doc.sents))
        queue = [(i + 1, sent) for i, sent in enumerate(doc.sents)]
        d = dict([("0", {"val": doc.text_with_ws, "children": [str(i) for i in range(1, n_sents + 1)]})])
        count = n_sents + 1
        # run BFS to construct the tree
        while queue:
            pos, node = queue.pop()
            record = {"val": node.text_with_ws}
            if list(node._.children):
                record["children"] = [str(count + i) for i, _ in enumerate(node._.children)]
                queue += [(count + i, child) for i, child in enumerate(node._.children)]
                count += len(record["children"])
            d[str(pos)] = record
        return dict(sorted(d.items(), key=lambda x: int(x[0])))

    def _patch(self, candidate, prompt=None):
        if prompt is None:
            prompt = self._current_prompt
        return candidate.replace_static_text(
            self._path_descriptor, "".join(self._syntax_tree[k]["val"] for k in prompt)
        )

    def _load_syntax_tree(self, candidate):
        instructions = candidate.query(self._path_descriptor)
        if hasattr(instructions, "static_text"):
            instructions = instructions.static_text()
        if instructions not in self._cache:
            self._cache[instructions] = self._json_parse_tree(instructions)
        return self._cache[instructions]

    async def mutate(
        self,
        candidate: Output,
        data: DataTable,
        runner: sammo.base.Runner,
        n_mutations: int = 1,
        random_state: int = 42,
    ) -> list[MutatedCandidate]:
        if self._syntax_tree is None:
            self._syntax_tree = self._load_syntax_tree(candidate)

        current_prompt = self._current_prompt
        if not current_prompt or not self._queue:
            return [MutatedCandidate("noop", self._patch(candidate))]
        node = self._queue.pop(0)
        prompt_with_node_removed = [c for c in current_prompt if c != node]
        if self._full_performance is None:
            y_pred = await self._patch(candidate).arun(runner, data, progress_callback=False)
            self._full_performance = self._prune_metric(data, y_pred).score

        y_pred = await self._patch(candidate, prompt_with_node_removed).arun(runner, data, progress_callback=False)
        ablation_performance = self._prune_metric(data, y_pred).score

        if ablation_performance >= self._full_performance:
            current_prompt = prompt_with_node_removed
            action = "drop"
        else:
            pos = self._current_prompt.index(node)
            current_prompt = (
                current_prompt[:pos] + self._syntax_tree[node].get("children", [node]) + current_prompt[pos + 1 :]
            )
            self._queue += self._syntax_tree[node].get("children", [])
            action = "keep"
        self._current_prompt = current_prompt
        return [MutatedCandidate(action, self._patch(candidate))]


class ShortenSegment(Mutator):
    def __init__(self, path_descriptor: str | dict, reduction_factor: float = 0.5):
        super().__init__()
        self._path_descriptor = CompiledQuery.from_path(path_descriptor)
        self._reduction_factor = reduction_factor

    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state: int = 42
    ) -> list[MutatedCandidate]:
        segment_path, segment_content = candidate.query(self._path_descriptor, return_path=True)
        segment_path += ".content"
        segment_content = segment_content.static_text()
        rewritten = await self._rewrite(runner, segment_content, n_mutations, random_state)
        return [
            MutatedCandidate(self.__class__.__name__, pg.clone(candidate, override={segment_path: r}))
            for r in rewritten
        ]

    def applicable(self, candidate: Output):
        return candidate.query(self._path_descriptor) is not None

    async def _rewrite(self, runner, segment_content, n_mutations, random_state):
        n_words = len(str(segment_content).split()) * self._reduction_factor
        rewritten = (
            await Output(
                GenerateText(
                    Template(
                        f"Summarize the text below in {n_words:0.0f} words or less. \n\n" "{{{content}}}",
                        content=segment_content,
                    )
                )
            ).arun(runner)
        ).outputs.raw_values[0]
        return [rewritten.value]


class APO(Mutator):
    def __init__(
        self,
        path_descriptor: str | dict,
        starting_prompt: Output | Callable | None,
        num_rewrites=2,
        num_sampled_errors=4,
        num_gradients=4,
        steps_per_gradient=2,
        minibatch_size=None,
        seed=42,
    ):
        super().__init__(starting_prompt, seed)
        self._path_descriptor = CompiledQuery.from_path(path_descriptor)
        self._n_rewrites = num_rewrites
        self._n_sampled_errors = num_sampled_errors
        self._n_gradients = num_gradients
        self._n_steps_per_gradient = steps_per_gradient
        self._minibatch_size = minibatch_size

    async def mutate(
        self,
        candidate: Output,
        data: DataTable,
        runner: sammo.base.Runner,
        n_mutations: int = 1,
        random_state: int = 42,
    ) -> list[MutatedCandidate]:
        segment_path, segment_content = candidate.query(self._path_descriptor, return_path=True)
        if hasattr(segment_content, "static_text"):
            current_instructions = segment_content.static_text()
        else:
            current_instructions = str(segment_content)
        rng = random.Random(random_state)
        minibatch = data.sample(self._minibatch_size, seed=random_state) if self._minibatch_size else data

        # collect errors
        minibatch_pred = await candidate.arun(runner, minibatch, progress_callback=False)
        wrong_examples = self.objective(minibatch, minibatch_pred).mistakes

        # if no errors are found, just return current candidate
        if len(wrong_examples) == 0:
            return [candidate]

        # choose random examples from the minibatch
        sampled_errors_idx = rng.sample(sorted(wrong_examples), min(self._n_sampled_errors, len(wrong_examples)))
        sampled_errors = list()
        for yrow, ytrue, ypred in zip(
            minibatch[sampled_errors_idx].inputs.values,
            minibatch[sampled_errors_idx].outputs.normalized_values(),
            minibatch_pred[sampled_errors_idx].outputs.normalized_values(),
        ):
            sampled_errors += [(yrow, ytrue, ypred)]

        # generate "gradients" - 1 call
        example_formatter = candidate.query(".*data_formatter")
        intro = Template(
            "I'm trying to write a zero-shot classifier prompt.\n"
            "My current prompt is: <PROMPT>{{{current_instructions}}}</PROMPT>\n"
            "But this prompt gets the following examples wrong:\n{{{errors}}}",
            current_instructions=current_instructions,
            errors=example_formatter.format_batch(*zip(*sampled_errors)),
        )
        gradients = ExtractRegex(
            GenerateText(
                Template(
                    "{{{intro}}}\n"
                    "Give {{{num_gradients}}} reasons why the prompt could have gotten these examples wrong."
                    "Wrap each reason with <START> and </START>.",
                    intro=intro,
                    num_gradients=self._n_gradients,
                )
            ),
            r"<START>(.*?)</START>",
        )

        # edit prompt - num_gradients calls
        edited_prompts = ForEach(
            "gradient",
            gradients,
            ExtractRegex(
                GenerateText(
                    Template(
                        "{{{intro}}}\nBased on these examples the problem with this prompt is that:\n{{{gradient}}}"
                        "\n\nBased on the above information, I wrote {{{steps_per_gradient}}} different improved prompts. "
                        "Each prompt is wrapped with <START> and </START>."
                        "\nThe {{{steps_per_gradient}}} new prompts are:",
                        intro=intro,
                        steps_per_gradient=self._n_steps_per_gradient,
                    )
                ),
                r"<START>\s*(.*?)</START>",
                max_matches=self._n_steps_per_gradient,
            ),
        )

        # generate prompt variants - num_gradients * num_rewrites * steps_per_gradient calls
        if self._n_rewrites:
            rewritten_prompts = ForEach(
                "prompt",
                edited_prompts,
                GenerateText(
                    Template(
                        "Generate a variation of the following instruction while keeping the semantic meaning.\n"
                        "Input: {{{prompt}}}"
                        "\nOutput: "
                    ),
                    randomness=0.7,
                ),
            )
            prompt_variants = Union(edited_prompts, rewritten_prompts)
        else:
            prompt_variants = edited_prompts

        output = (await Output(prompt_variants).arun(runner)).outputs.raw_values[0]
        res = rng.sample(output, min(n_mutations, len(output)))

        return [
            MutatedCandidate(
                self.__class__.__name__,
                pg.clone(candidate, override={segment_path: r.value}),
                prompt_variants=prompt_variants,
                output=output,
                sampled_errors_idx=sampled_errors_idx,
            )
            for r in res
        ]


class APE(ShortenSegment):
    RESAMPLE = textwrap.dedent(
        """\
    Generate a variation of the following instruction while keeping the semantic meaning.
    Input: {{{instructions}}}
    Output: """
    )

    _FORWARD_GENERATION = textwrap.dedent(
        """\
    I gave a friend an instruction and five inputs. The friend read the instruction and
    wrote an output for every one of the inputs.
    Here are the input-output pairs:
    {{{examples}}}

    Here are the instructions that were given:
    """
    )

    def __init__(
        self,
        path_descriptor: str | dict,
        starting_prompt: Output | Callable,
        d_incontext: DataTable,
        n_incontext_subsamples: int | None = None,
    ):
        super().__init__(path_descriptor)
        self._starting_prompt = starting_prompt
        self._seed = 42
        self._d_incontext = d_incontext
        self._n_incontext_subsamples = n_incontext_subsamples

    async def get_initial_candidates(
        self, runner: sammo.base.Runner, n_initial_candidates: int
    ) -> list[MutatedCandidate]:
        candidate = (await super().get_initial_candidates(runner, 1))[0].candidate
        return await self._induce_instructions(candidate, runner, n_initial_candidates, self._seed)

    async def _induce_instructions(self, candidate, runner, n_mutations, random_state):
        example_formatter = candidate.query(".*data_formatter")
        candidates = list()
        for i in range(n_mutations):
            if self._n_incontext_subsamples is None:
                d_incontext = self._d_incontext
            else:
                d_incontext = self._d_incontext.sample(self._n_incontext_subsamples, seed=i + random_state)
            fewshot_examples = example_formatter.format_datatable(d_incontext)
            induced_instructions = GenerateText(
                Template(self._FORWARD_GENERATION, examples=fewshot_examples), randomness=0.9, seed=i + random_state
            )
            candidates.append(induced_instructions)
        induced = (await Output(Union(*candidates)).arun(runner)).outputs.raw_values[0]
        return [
            MutatedCandidate(self.__class__.__name__, candidate.replace_static_text(self._path_descriptor, r.value))
            for r in induced
        ][:n_mutations]

    async def _rewrite(self, runner, segment_content, n_mutations, random_state):
        output = (
            await Output(
                Union(
                    *[
                        GenerateText(Template(self.RESAMPLE, instructions=segment_content), randomness=0.9, seed=i)
                        for i in range(n_mutations)
                    ]
                )
            ).arun(runner)
        ).outputs.raw_values[0]
        return [o.value for o in output]


class InduceInstructions(APE):
    def __init__(self, path_descriptor: str | dict, d_incontext: DataTable):
        super().__init__(path_descriptor, None, d_incontext)

    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state: int = 42
    ) -> list[MutatedCandidate]:
        return await self._induce_instructions(candidate, runner, n_mutations, random_state)


class SegmentToBulletPoints(ShortenSegment):
    async def _rewrite(self, runner, segment_content, n_mutations, random_seed):
        rewritten = (
            await Output(
                GenerateText(
                    Template(
                        f"Rewrite the text below as a bullet list with at most 10 words per bullet point. \n\n"
                        "{{{content}}}",
                        content=segment_content,
                    )
                )
            ).arun(runner)
        ).outputs.raw_values[0]
        return [rewritten.value]


class Paraphrase(ShortenSegment):
    async def _rewrite(self, runner, segment_content, n_mutations, random_state):
        rewrites = [
            GenerateText(
                Template(
                    "Paraphrase the text inside the tags. Do not output the tags. \n\n" "<TEXT>{{{content}}}</TEXT>",
                    content=segment_content,
                ),
                randomness=0.9,
                seed=random_state + i,
            )
            for i in range(n_mutations)
        ]
        rewritten = (await Output(Union(*rewrites)).arun(runner)).outputs.raw_values[0]
        return [r.value for r in rewritten]


class ParaphraseStatic(ShortenSegment):
    def __init__(self, path_descriptor: str | dict, static_content: str):
        self._path_descriptor = CompiledQuery.from_path(path_descriptor)
        self._static_content = static_content

    async def _rewrite(self, runner, segment_content, n_mutations, random_state):
        rewrites = [
            GenerateText(
                Template(
                    "Paraphrase the text inside the tags. Do not output the tags. \n\n" "<TEXT>{{{content}}}</TEXT>",
                    content=self._static_content,
                ),
                randomness=0.9,
                seed=random_state + i,
            )
            for i in range(n_mutations)
        ]
        rewritten = (await Output(Union(*rewrites)).arun(runner)).outputs.raw_values[0]
        return [r.value for r in rewritten]


class RemoveStopWordsFromSegment(ShortenSegment):
    def __init__(self, path_descriptor: str | dict, choices: Any):
        self._path_descriptor = CompiledQuery.from_path(path_descriptor)
        self._choices = choices

    async def _rewrite(self, runner, segment_content, n_mutations, random_seed):
        if not isinstance(self._choices, list):
            return [self._choices.compress(segment_content)]
        else:
            return [proc.compress(segment_content) for proc in self._choices[:n_mutations]]


class DropParameter(Mutator):
    def __init__(self, path_descriptor: str | dict):
        self._path_descriptor = CompiledQuery.from_path(path_descriptor)

    def applicable(self, candidate: Output):
        return candidate.query(self._path_descriptor) is not None

    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state: int = 42
    ) -> list[MutatedCandidate]:
        current_path, current_value = candidate.query(self._path_descriptor, return_path=True)
        dropped = pg.clone(candidate).rebind({current_path: pg.MISSING_VALUE})
        return [MutatedCandidate(self.__class__.__name__, dropped)]


class RepeatSegment(DropParameter):
    def __init__(self, path_descriptor: str | dict, after: str | dict):
        super().__init__(path_descriptor)
        self._after = CompiledQuery.from_path(after)

    def applicable(self, candidate: Output):
        return candidate.query(self._path_descriptor) is not None and candidate.query(self._after) is not None

    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state: int = 42
    ) -> list[MutatedCandidate]:
        repeated = candidate.clone()
        current_value = repeated.query(self._path_descriptor)
        after_value = repeated.query(self._after)
        index = after_value.sym_path.key
        if not isinstance(index, int):
            raise ValueError("RepeatSegment can only be used with lists.")
        after_value.sym_parent.insert(index + 1, current_value)
        return [MutatedCandidate(self.__class__.__name__, repeated)]


class DropExamples(DropParameter):
    pass


class DropIntro(DropParameter):
    pass


class ReplaceParameter(DropParameter):
    def __init__(self, path_descriptor: str | dict, choices: Any):
        super().__init__(path_descriptor)
        self._choices = choices

    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state: int = 42
    ) -> list[MutatedCandidate]:
        current_path, new_values = self._sample_new_values(candidate, n_mutations, random_state)
        mutations = list()

        for new_value in new_values:
            mutations.append(
                MutatedCandidate(self.__class__.__name__, pg.clone(candidate, override={current_path: new_value}))
            )
        return mutations

    def _sample_new_values(self, candidate, n_mutations, random_state):
        current_path, current_value = candidate.query(self._path_descriptor, return_path=True)
        if not isinstance(self._choices, list):
            # if we only have a single value, deterministically replace it
            new_values = [self._choices]
        else:
            valid_values = [v for v in self._choices if v != current_value]
            rng = random.Random(random_state)
            new_values = rng.sample(valid_values, min(n_mutations, len(valid_values)))
        return current_path, new_values


class ChangeDataFormat(ReplaceParameter):
    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state: int = 42
    ) -> list[MutatedCandidate]:
        current_path, new_values = self._sample_new_values(candidate, n_mutations, random_state)
        mutations = list()
        for new_value in new_values:
            mutation = pg.clone(candidate, override={current_path: new_value})
            parser = new_value.get_extractor(mutation.query({"type": GenerateText}), on_error="empty_result")
            mutation.rebind({"child": parser})
            mutations.append(MutatedCandidate(self.__class__.__name__, mutation))
        return mutations


class ChangeSectionsFormat(ReplaceParameter):
    pass


class DecreaseInContextExamples(Mutator):
    def __init__(self, d_incontext, reduction_factor=0.8, min_examples=1):
        self._d_incontext = d_incontext
        self._reduction_factor = reduction_factor
        self._min_examples = min_examples

    def applicable(self, candidate: Output):
        return candidate.query({"type": FewshotExamples}) is not None

    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state=42
    ) -> list[MutatedCandidate]:
        mutated = list()
        current_path, current_value = candidate.query({"type": FewshotExamples, "_child": "data"}, return_path=True)
        n_examples = max(self._min_examples, int(len(current_value) * self._reduction_factor))

        for i in range(n_mutations):
            new_data = self._d_incontext.sample(n_examples, random_state)
            mutated.append(
                MutatedCandidate(self.__class__.__name__, pg.clone(candidate, override={current_path: new_data}))
            )

        return mutated


class BagOfMutators(Mutator):
    def __init__(
        self, starting_prompt: Output | Callable, *bag, seed: int = 42, sample_for_init_candidates: bool = True
    ):
        super().__init__(starting_prompt, seed=seed, sample_for_init_candidates=sample_for_init_candidates)
        self._bag = bag

    def applicable(self, candidate: Output):
        return any(m.applicable(candidate) for m in self._bag)

    @staticmethod
    def draw_beta_bernoulli(n_samples: int, success_failure_pairs: list[tuple[int]], priors=(2, 6), seed=42):
        success_failure_pairs = np.asarray(success_failure_pairs).T
        if success_failure_pairs.shape[0] != 2:
            raise ValueError("success_failure_pairs must be of shape (*, 2)")
        rng = np.random.default_rng(seed)
        successes, failures = success_failure_pairs
        alphas, betas = np.asarray(successes) + priors[0], np.asarray(failures) + priors[1]
        posterior_samples = rng.beta(alphas, betas, size=(n_samples, len(alphas)))
        return np.argmax(posterior_samples, axis=1).tolist()

    @staticmethod
    def draw_map_beta_bernoulli(n_samples: int, success_failure_pairs: list[tuple[int]], priors=(5, 5), seed=None):
        success_failure_pairs = np.asarray(success_failure_pairs).T
        if success_failure_pairs.shape[0] != 2:
            raise ValueError("success_failure_pairs must be of shape (*, 2)")
        rng = np.random.default_rng(seed)
        successes, failures = success_failure_pairs
        alphas, betas = np.asarray(successes) + priors[0], np.asarray(failures) + priors[1]
        mode = (alphas - 1) / (alphas + betas - 2)
        p = mode / mode.sum()
        return np.argmax(rng.multinomial(1, p, size=n_samples), axis=1).tolist()

    async def mutate(
        self, candidate: Output, data: DataTable, runner: Runner, n_mutations: int = 1, random_state=None
    ) -> list[MutatedCandidate]:
        bag = [m for m in self._bag if m.applicable(candidate)]
        if not bag:
            return list()
        if not self.priors:
            selected_mutators = random.Random(random_state).choices(bag, k=n_mutations)
        else:
            action_stats = [self.priors.get(m.__class__.__name__, (0, 0)) for m in bag]
            idx = self.draw_map_beta_bernoulli(n_mutations, action_stats, seed=random_state)
            selected_mutators = [bag[i] for i in idx]

        if n_mutations == 1:
            return await selected_mutators[0].mutate(candidate, data, runner, random_state)
        else:
            selected_mutators = collections.Counter(selected_mutators)
            tasks = list()
            async with asyncio.TaskGroup() as tg:
                for i, (mut, n_mut) in enumerate(selected_mutators.items()):
                    mut.objective = self._objective
                    tasks.append(tg.create_task(mut.mutate(candidate, data, runner, n_mut, random_state + i)))
            return sum([t.result() for t in tasks], [])


class StopwordsCompressor(Component):
    REUTERS_STOPLIST = [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "it",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    ]

    def __init__(
        self,
        filter_stopwords: Literal["reuters", "spacy", "none"],
        remove_punctuation=False,
        remove_whitespace=False,
    ):
        self._remove_punctuation = remove_punctuation
        self._filter_stopwords = filter_stopwords
        self._remove_whitespace = remove_whitespace
        self._nlp = spacy.load("en_core_web_sm")

    def compress(self, x: str):
        tokenized = self._nlp(x)
        filters = list()
        if self._remove_punctuation:
            filters += [lambda x: x.is_punct]
        if self._filter_stopwords == "spacy":
            filters += [lambda x: x.is_stop]
        elif self._filter_stopwords == "reuters":
            filters += [lambda x: x.text.lower() in self.REUTERS_STOPLIST]
        if filters:
            tokenized = [v for v in tokenized if all(not filtered(v) for filtered in filters)]
        return "".join(
            map(
                lambda y: y.text if self._remove_whitespace else y.text_with_ws,
                tokenized,
            )
        )

    def __call__(self, x: str | dict | list):
        if isinstance(x, str):
            return self.compress(x)
        elif isinstance(x, dict):
            return {k: self(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self(v) for v in x]
        else:
            return x
