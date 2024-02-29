# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import hashlib
import json
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


class Section(Component):
    def __init__(self, name, content, id=None):
        super().__init__(content, name)
        self.id = id
        self.name = name
        if not isinstance(content, list):
            content = [content]
        if not isinstance(self, Paragraph):
            content = [Paragraph(c) if isinstance(content, str) else c for c in content]
        self.content = content

    def static_text(self, sep="\n"):
        return "\n".join([v.static_text(sep) if hasattr(v, "static_text") else str(v) for v in self.content])

    def set_static_text(self, text):
        return self.rebind({"content": text})

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> dict:
        return {
            "type": self.__class__.__name__.lower(),
            "id": self.id,
            "name": self.name,
            "content": self._unwrap_results(
                [
                    await child(runner, context, dynamic_context) if isinstance(child, Component) else child
                    for child in self.content
                ]
            ),
        }


class Paragraph(Section):
    def __init__(self, content, id=None):
        super().__init__(None, content, id)


class MetaPrompt(Component):
    def __init__(
        self,
        structure: list[Paragraph | Section],
        render_as: Literal["raw", "json", "xml", "markdown", "markdown-alt"] = "markdown",
        data_formatter: DataFormatter | None = None,
        name: str | None = None,
        seed: int = 0,
    ):
        super().__init__(structure, name)

        self._render_as = render_as
        self._data_formatter = data_formatter
        self._structure = structure
        self._seed = seed

    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None) -> LLMResult:
        context["data_formatter"] = self._data_formatter
        filled_out_structure = [await child(runner, context, dynamic_context) for child in self._structure]
        return TextResult(self._render(filled_out_structure))

    def with_extractor(self, on_error: Literal["raise", "empty_result"] = "raise"):
        if self._data_formatter is not None:
            return self._data_formatter.get_extractor(GenerateText(self), on_error=on_error)
        else:
            raise ValueError("Without a given data_formatter, responses must be parsed manually.")

    def _render(self, structure):
        if self._render_as == "raw":
            return "".join(["".join(v["content"]) for v in structure]).strip()
        elif self._render_as == "json":
            return self.render_as_json(structure).strip()
        elif self._render_as.startswith("xml"):
            return self.render_as_xml(structure, use_attr=self._render_as == "xml-attr").strip()
        elif self._render_as.startswith("markdown"):
            return self.render_as_markdown(structure, alternative_headings="alt" in self._render_as).strip()
        else:
            return NotImplementedError()

    @classmethod
    def render_as_json(cls, data, is_key=False):
        """
        If it detects JSON in a string, it tries to output it unescaped
        :param data:
        :param is_key:
        :return:
        """
        if isinstance(data, dict):
            return (
                "{"
                + ", ".join([f"{cls.render_as_json(k, is_key=True)}:{cls.render_as_json(v)}" for k, v in data.items()])
                + "}"
            )
        elif isinstance(data, list):
            return f"[{', '.join([cls.render_as_json(x) for x in data])}]"
        elif isinstance(data, str) and data.startswith("{") or data.startswith("["):
            return data
        elif is_key:
            return json.dumps(str(data))
        else:
            return json.dumps(data)

    @classmethod
    def render_as_markdown(cls, data, alternative_headings=False, depth=0):
        UNDERLINES = ["=", "-"]
        md_string = ""
        if isinstance(data, dict):
            kind, attributes = data["type"], [k for k in data.keys() if k not in ["type", "content"]]
            content = data["content"]
            if kind == "section":
                title = data["name"]
                if alternative_headings:
                    if depth > 2:
                        raise ValueError("Alternative headings are only supported up to depth 2.")
                    md_string += f"{title}\n{UNDERLINES[depth] * len(title)}\n"
                else:
                    md_string += f"{'#' * (depth + 1)} {title}\n"
            if isinstance(content, str):
                md_string += content
            else:
                md_string += cls.render_as_markdown(content, alternative_headings, depth=depth + 1)
            if kind == "paragraph":
                md_string += "\n"
        elif isinstance(data, list):
            md_string += "\n".join([cls.render_as_markdown(v, alternative_headings, depth=depth) for v in data])
        else:
            md_string += str(data) + "\n"
        return md_string

    @classmethod
    def render_as_xml(cls, data, depth=0, use_attr=True):
        if isinstance(data, dict):
            kind, attributes = data["type"], [k for k in data.keys() if k not in ["type", "content"]]
            content = data["content"]
            xml_element = lambda x: f"\n<{kind}>{x}\n</{kind}>"
            xml_element_w_tag = lambda x, tag: f"\n<{tag}>{x}</{tag}>"
            attri = lambda y: " ".join([f'{k}="{v}"' for k, v in y.items() if v is not None])
            xml_element_w_attr = lambda x, y: f"\n<{kind} {attri(y)}>{x}\n</{kind}>"

            if isinstance(content, str):
                return xml_element(content)
            elif use_attr:
                attr = {k: v for k, v in data.items() if k not in ["type", "content"]}
                return xml_element_w_attr(cls.render_as_xml(content, depth + 1, use_attr), attr)
            else:
                nested = list()
                inner = ""
                for key, value in data.items():
                    if key == "content":
                        inner = cls.render_as_xml(content, depth + 1, use_attr)
                    elif key not in ["type"]:
                        nested.append(xml_element_w_tag(value, key))
                return xml_element("".join(nested + [inner]))
        elif isinstance(data, list):
            return "\n".join([cls.render_as_xml(x, depth, use_attr) for x in data])
        else:
            return str(data)


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
