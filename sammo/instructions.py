import json

from beartype.typing import Literal
from frozendict import frozendict

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
