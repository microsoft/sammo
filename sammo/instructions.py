# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from abc import abstractmethod, ABC, ABCMeta

import numpy as np
from beartype.typing import Literal
from frozendict import frozendict
from sammo.utils import sync
from sammo.base import Component, LLMResult, Runner, TextResult
from sammo.components import GenerateText
from sammo.data import DataTable
from sammo.dataformatters import DataFormatter


class Renderer(metaclass=ABCMeta):
    @abstractmethod
    def render_section(self, content, reference_id=None, reference_classes=None, title=None, depth=0):
        pass

    @abstractmethod
    def render_paragraph(self, content, reference_id=None, reference_classes=None, depth=0):
        pass

    @abstractmethod
    def render_metaprompt(self, content, depth=0, **kwargs):
        pass


class BaseRenderer(Renderer, ABC):
    def render_metaprompt(self, content, depth=0, **kwargs):
        return "\n".join(content).strip()

    def render_paragraph(self, content, reference_id=None, reference_classes=None, depth=0):
        return self.render_section(content, reference_id=reference_id, reference_classes=reference_classes, depth=depth)


class MarkdownRenderer(BaseRenderer):
    UNDERLINES = ["=", "-"]

    def __init__(self, alternative_headings=False):
        self._alternative_headings = alternative_headings

    def render_section(self, content, reference_id=None, reference_classes=None, title=None, depth=0):
        md_string = ""
        if self._alternative_headings:
            if depth > 2:
                raise ValueError("Alternative headings are only supported up to depth 2.")
            md_string += f"{title}\n{self.UNDERLINES[depth] * len(title)}\n"
        else:
            md_string += f"{'#' * (depth + 1)} {title}\n"
        return md_string + "\n".join(content) + "\n"

    def render_paragraph(self, data, **kwargs):
        return "\n".join(data) + "\n\n"


class MarkdownRendererAlt(MarkdownRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, alternative_headings=True)


class RawRenderer(BaseRenderer):
    def render_section(self, content, reference_id=None, reference_classes=None, title=None, depth=0):
        title = title if title is not None else ""
        return title + "".join([str(c) for c in content])

    def render_metaprompt(self, content, depth=0, **kwargs):
        return "".join(content).strip()


class XmlRenderer(BaseRenderer):
    def _render_section(self, content, reference_id=None, reference_classes=None, title=None, depth=0, kind="section"):
        outer_element = lambda x: f"\n<{kind}>{x}\n</{kind}>"
        xml_element_w_tag = lambda x, tag: f"\n<{tag}>{x}</{tag}>"

        inner = ""
        if reference_id is not None:
            inner += xml_element_w_tag(reference_id, "id")
        if title is not None:
            inner += xml_element_w_tag(title, "title")
        return outer_element(inner + "\n".join(content))

    def render_section(self, content, reference_id=None, reference_classes=None, title=None, depth=0):
        return self._render_section(content, reference_id, reference_classes, title, depth, kind="section")

    def render_paragraph(self, content, reference_id=None, reference_classes=None, title=None, depth=0):
        return self._render_section(content, reference_id, reference_classes, None, depth, kind="paragraph")


class DocumentComponent(Component):
    def __init__(self, content, reference_id: str | None = None, reference_classes: list[str] | None = None):
        super().__init__(content, reference_id)
        if not isinstance(content, list):
            content = [content]
        if not isinstance(self, Paragraph):
            content = [Paragraph(c) if isinstance(content, str) else c for c in content]
        self.content = content
        self._attributes = dict(reference_classes=reference_classes, reference_id=reference_id)

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> Component:
        depth = dynamic_context.get("depth", -1)
        updated_context = frozendict({**dynamic_context, "depth": depth + 1})
        children_results = [
            await child(runner, context, updated_context) if isinstance(child, Component) else child
            for child in self.content
        ]
        children = self._unwrap_results(children_results)
        rendered = getattr(dynamic_context["renderer"], f"render_{self.__class__.__name__.lower()}")(
            children, depth=depth, **self._attributes
        )
        return TextResult(rendered, op=self, parent=children_results)


class Section(DocumentComponent):
    def __init__(self, title, content, reference_id=None, reference_classes=None):
        super().__init__(content, reference_id, reference_classes)
        if title is not None:
            self._attributes["title"] = title

    def static_text(self, sep="\n"):
        return "\n".join([v.static_text(sep) if hasattr(v, "static_text") else str(v) for v in self.content])

    def set_static_text(self, text):
        return self.rebind({"content": text})


class Paragraph(Section):
    def __init__(self, content, reference_id=None, reference_classes=None):
        super().__init__(None, content, reference_id, reference_classes)


class MetaPrompt(DocumentComponent):
    RENDERERS = {
        "xml": XmlRenderer,
        "raw": RawRenderer,
        "markdown": MarkdownRenderer,
        "markdown-alt": MarkdownRendererAlt,
    }

    def __init__(
        self,
        child: list[Paragraph | Section] | Paragraph | Section,
        render_as: Literal["raw", "xml", "markdown", "markdown-alt"] = "markdown",
        data_formatter: DataFormatter | None = None,
        reference_id: str | None = None,
        seed: int = 0,
    ):
        super().__init__(child, reference_id)

        self._renderer = self.RENDERERS[render_as]()
        self._data_formatter = data_formatter
        self._seed = seed

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        context["data_formatter"] = self._data_formatter
        if dynamic_context is None:
            dynamic_context = dict()
        dynamic_context = frozendict({**dynamic_context, "renderer": self._renderer})
        return await super()._call(runner, context, dynamic_context)

    def with_extractor(self, on_error: Literal["raise", "empty_result"] = "raise"):
        if self._data_formatter is not None:
            return self._data_formatter.get_extractor(GenerateText(self), on_error=on_error)
        else:
            raise ValueError("Without a given data_formatter, responses must be parsed manually.")


class FewshotExamples(Component):
    def __init__(
        self,
        data: DataTable,
        n_examples: int | None = None,
        reference_id: str | None = None,
    ):
        super().__init__(None, reference_id)
        self._data = data
        if n_examples is None:
            n_examples = len(data)
        self._n_examples = n_examples
        self._formatted_data = None

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        if self._formatted_data is None:
            self._formatted_data = context["data_formatter"].format_datatable(self._data[: self._n_examples])
        return TextResult(self._formatted_data, op=self)


class RandomFewshotExamples(FewshotExamples):
    def __init__(
        self,
        data: DataTable,
        n_examples: int | None = None,
        reference_id: str | None = None,
        seed: int = 0,
    ):
        super().__init__(data, n_examples, reference_id)
        self._data = data.sample(len(data), seed=seed)


class EmbeddingFewshotExamples(FewshotExamples):
    MAX_BATCH_SIZE = 1000

    def __init__(
        self,
        embedder: Runner,
        data: DataTable,
        n_examples: int | None = None,
        reference_id: str | None = None,
        aggregate: Literal["roundrobin", "max"] = "roundrobin",
        filter_exact_matches: bool = True,
        budget: Literal["absolute", "relative"] = "absolute",
    ):
        super().__init__(data, n_examples, reference_id)
        self._embedder = embedder
        self._aggregate = aggregate
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
        return np.asarray((await self._embedder.generate_embedding(rendered)).value)

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
        return TextResult(formatted_data, op=self)


class InputData(Component):
    def __init__(
        self,
        id_offset: int = 0,
        reference_id: str | None = None,
    ):
        super().__init__(None, reference_id)
        self.id_offset = id_offset

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        return TextResult(
            context["data_formatter"].format_batch(context["data"]["inputs"], offset=self.id_offset), op=self
        )
