# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import hashlib
import math

import numpy as np
from beartype.typing import Literal, MutableMapping
from frozendict import frozendict
import pyglove as pg

from sammo.compactbars import CompactProgressBars
from sammo.utils import sync
from sammo.base import Component, LLMResult, Runner, TextResult, ScalarComponent
from sammo.components import GenerateText
from sammo.data import DataTable
from sammo.dataformatters import DataFormatter


class Renderer(Component):
    def __init__(self, content, name: str | None = None):
        super().__init__(content, name)
        if not isinstance(content, list):
            content = [content]
        if not isinstance(self, Paragraph):
            content = [Paragraph(c) if isinstance(content, str) else c for c in content]
        self.content = content

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> dict:
        depth = dynamic_context.get("depth", -1)
        updated_context = frozendict({**dynamic_context, "depth": depth + 1})
        children = self._unwrap_results(
            [
                await child(runner, context, updated_context) if isinstance(child, Component) else child
                for child in self.content
            ]
        )

        render_as = dynamic_context["render_as"]
        if render_as == "xml":
            return TextResult(self.render_as_xml(children, depth=depth))
        elif render_as == "raw":
            return TextResult(self.render_as_raw(children))
        else:
            return TextResult(self.render_as_markdown(children, depth=depth, alternative_headings="alt" in render_as))


class Section(Renderer):
    def __init__(self, name, content, id=None):
        super().__init__(content, name)
        self.id = id

    def static_text(self, sep="\n"):
        return "\n".join([v.static_text(sep) if hasattr(v, "static_text") else str(v) for v in self.content])

    def set_static_text(self, text):
        return self.rebind({"content": text})

    def render_as_markdown(self, data, alternative_headings=False, depth=0, **kwargs):
        UNDERLINES = ["=", "-"]
        md_string = ""
        title = self._name
        if alternative_headings:
            if depth > 2:
                raise ValueError("Alternative headings are only supported up to depth 2.")
            md_string += f"{title}\n{UNDERLINES[depth] * len(title)}\n"
        else:
            md_string += f"{'#' * (depth + 1)} {title}\n"
        return md_string + "\n".join(data) + "\n"

    def render_as_xml(self, content, depth=0):
        kind = self.__class__.__name__.lower()
        outer_element = lambda x: f"\n<{kind}>{x}\n</{kind}>"
        xml_element_w_tag = lambda x, tag: f"\n<{tag}>{x}</{tag}>"

        inner = ""
        if hasattr(self, "id"):
            inner += xml_element_w_tag(self.id, "id")

        inner += xml_element_w_tag(self._name, "name")
        return outer_element(inner + "\n".join(content))

    def render_as_raw(self, content):
        return "".join(content)


class Paragraph(Section):
    def __init__(self, content, id=None):
        super().__init__(None, content, id)
        self._name = None

    def render_as_markdown(self, data, alternative_headings=False, depth=0):
        return "\n".join(data) + "\n\n"


class MetaPrompt(Renderer):
    def __init__(
        self,
        child: list[Paragraph | Section] | Paragraph | Section,
        render_as: Literal["raw", "xml", "markdown", "markdown-alt"] = "markdown",
        data_formatter: DataFormatter | None = None,
        name: str | None = None,
        seed: int = 0,
    ):
        super().__init__(child, name)

        self._render_as = render_as
        self._data_formatter = data_formatter
        self._seed = seed

    def render_as_markdown(self, data, alternative_headings=False, depth=0):
        return "\n".join(data).strip()

    def render_as_xml(self, data, depth=0):
        return "\n".join(data).strip()

    def render_as_raw(self, data):
        return "".join(data).strip()

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        context["data_formatter"] = self._data_formatter
        if dynamic_context is None:
            dynamic_context = dict()
        dynamic_context = frozendict({**dynamic_context, "render_as": self._render_as})
        return await super()._call(runner, context, dynamic_context)

    def with_extractor(self, on_error: Literal["raise", "empty_result"] = "raise"):
        if self._data_formatter is not None:
            return self._data_formatter.get_extractor(GenerateText(self), on_error=on_error)
        else:
            raise ValueError("Without a given data_formatter, responses must be parsed manually.")


class FewshotExamples(ScalarComponent):
    def __init__(
        self,
        data: DataTable,
        n_examples: int | None = None,
        name: str | None = None,
    ):
        super().__init__(None, name)
        self._data = data
        if n_examples is None:
            n_examples = len(data)
        self._n_examples = n_examples
        self._formatted_data = None

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        if self._formatted_data is None:
            self._formatted_data = context["data_formatter"].format_datatable(self._data[: self._n_examples])
        return LLMResult(self._formatted_data)


class RandomFewshotExamples(FewshotExamples):
    def __init__(
        self,
        data: DataTable,
        n_examples: int | None = None,
        name: str | None = None,
        seed: int = 0,
    ):
        super().__init__(data, n_examples, name)
        self._data = data.sample(len(data), seed=seed)


class EmbeddingFewshotExamples(FewshotExamples):
    MAX_BATCH_SIZE = 1000

    def __init__(
        self,
        embedder: Runner,
        data: DataTable,
        n_examples: int | None = None,
        name: str | None = None,
        aggregate: Literal["roundrobin", "max"] = "roundrobin",
        cache: MutableMapping | None = None,
        filter_exact_matches: bool = True,
        budget: Literal["absolute", "relative"] = "absolute",
    ):
        super().__init__(data, n_examples, name)
        self._embedder = embedder
        self._fingerprint = hashlib.md5(pg.to_json_str(embedder).encode("utf-8")).hexdigest()
        self._aggregate = aggregate
        self._index = cache or dict()
        self._data = data
        rendered = self._render(data)
        self._filter_exact = filter_exact_matches
        self._train_ids = dict(zip(rendered, range(len(rendered))))
        self._train = sync(self._embed(rendered))
        self._budget = budget

    def _render(self, data: DataTable | list[dict]):
        if isinstance(data, DataTable):
            data = data.inputs.raw_values
        return [str(x) for x in data]

    async def _embed(self, rendered: list[str]) -> np.ndarray:
        missing = [x for x in rendered if (self._fingerprint, x) not in self._index]
        if missing:
            if len(missing) > self.MAX_BATCH_SIZE:
                embeddings = list()
                pbar = CompactProgressBars().get("embedding minibatches", math.ceil(len(missing) / self.MAX_BATCH_SIZE))

                for i in range(0, len(missing), self.MAX_BATCH_SIZE):
                    embeddings += (await self._embedder.generate_embedding(missing[i : i + self.MAX_BATCH_SIZE])).value
                    pbar.update()
            else:
                embeddings = (await self._embedder.generate_embedding(missing)).value

            for key, value in zip(missing, embeddings):
                self._index[(self._fingerprint, key)] = value
        return np.asarray([self._index[(self._fingerprint, key)] for key in rendered])

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        input_rendered = self._render(context["data"]["inputs"])
        input_embeddings = await self._embed(input_rendered)

        scores = input_embeddings.dot(self._train.T)

        if self._aggregate == "roundrobin":
            idx = np.argsort(-scores, axis=1).flatten("F")
        elif self._aggregate == "max":
            idx = np.argsort(-scores.max(axis=0))
        deduped_idx = idx[np.sort(np.unique(idx, return_index=True)[1])]
        if self._filter_exact:
            invalid = {self._train_ids[x] for x in input_rendered if x in self._train_ids}
            deduped_idx = np.setdiff1d(deduped_idx, list(invalid), assume_unique=True)
        if self._budget == "absolute":
            budget = self._n_examples
        else:
            budget = self._n_examples * len(input_rendered)

        top_k = deduped_idx[:budget].tolist()
        formatted_data = context["data_formatter"].format_datatable(self._data[top_k])
        return LLMResult(formatted_data)


class InputData(ScalarComponent):
    def __init__(
        self,
        id_offset: int = 0,
        name: str | None = None,
    ):
        super().__init__(None, name)
        self.id_offset = id_offset

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        return LLMResult(context["data_formatter"].format_batch(context["data"]["inputs"], offset=self.id_offset))
