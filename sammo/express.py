import json
import re
from collections import namedtuple
from mistletoe.block_token import List
from mistletoe.markdown_renderer import MarkdownRenderer

from mistletoe import Document, block_token

from sammo.base import Template
from sammo.instructions import MetaPrompt, Section, Paragraph

HTML_COMMENT = re.compile(r"<!--(.*?)-->")
HTML_IDS = re.compile(r"#(\w+)($|\s)")
HTML_CLASSES = re.compile(r"\.([\w-]+)($|\s)")


def _extract_html_comment(text):
    rest = text
    inner_comment = ""

    if HTML_COMMENT.search(text) is not None:
        inner_comment = HTML_COMMENT.search(text).group(1)
        rest = HTML_COMMENT.sub("", text)

    return inner_comment, rest


def _get_ids_and_classes(text):
    comment, rest = _extract_html_comment(text)
    ids = HTML_IDS.findall(comment) or list()
    ids = [i[0] for i in ids]

    classes = HTML_CLASSES.findall(comment) or list()
    classes = [c[0] for c in classes]

    return {"text": rest, "ids": ids, "classes": classes}


class MarkdownParser:
    def __init__(self, input_text: str):
        self._input_text = input_text
        self._sammo_tree, self._sammo_config = None, None

    def _parse(self):
        if self._sammo_tree is None:
            json_tree, config = self._parse_annotated_markdown(self._input_text)
            self._sammo_tree = self._json_to_sammo(json_tree)
            self._sammo_config = config

    def get_sammo_program(self):
        self._parse()
        return self._sammo_tree

    def get_sammo_config(self):
        self._parse()
        return self._sammo_config

    @staticmethod
    def from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return MarkdownParser(file.read())

    @staticmethod
    def _parse_annotated_markdown(text):
        doc = Document(text)
        sammo_config = dict()
        State = namedtuple("State", ["current", "parent", "level"])
        with MarkdownRenderer() as mrender:
            processed = list()
            stack = [State(processed, processed, 0)]
            for element in doc.children:
                last = stack[-1]
                if isinstance(element, List):
                    list_elements = list()
                    classes = set()
                    ids = set()

                    for c in element.children:
                        d = _get_ids_and_classes(mrender.render(c))
                        classes.update(d["classes"])
                        ids.update(d["ids"])
                        list_elements.append(d["text"])

                    last.current.append(
                        {"type": "list", "children": list_elements, "class": list(classes), "id": list(ids)}
                    )
                elif isinstance(element, block_token.Heading):
                    d = _get_ids_and_classes(mrender.render(element))
                    new = {
                        "type": "section",
                        "title": d["text"],
                        "children": list(),
                        "id": d["ids"],
                        "class": d["classes"],
                    }
                    if element.level < last.level:
                        while stack[-1].level >= element.level:
                            stack.pop()
                        scope = stack[-1].current
                    elif element.level == last.level:
                        scope = last.parent
                    else:
                        scope = last.current
                    stack.append(State(new["children"], scope, element.level))
                    scope.append(new)
                elif isinstance(element, block_token.CodeFence) and element.language.lower() == "{sammo/mutators}":
                    sammo_config = json.loads(element.children[0].content)
                else:
                    last.current.append(
                        {"type": element.__class__.__name__.lower(), "children": [mrender.render(element)]}
                    )
        return {"type": "root", "children": processed}, sammo_config

    @classmethod
    def _json_to_sammo(cls, node):
        def _empty_to_none(x):
            return None if len(x) == 0 else x

        def _unwrap_list(x):
            if not isinstance(x, list) or len(x) > 1:
                return ValueError(f"Expected list of length 0 or 1, got {len(x)}")
            elif len(x) == 1:
                return x[0]
            return x

        def _get_annotations(x):
            return dict(
                reference_id=_empty_to_none(_unwrap_list(x.get("id", []))),
                reference_classes=_empty_to_none(x.get("class", [])),
            )

        if isinstance(node, str) and "{{" in node:
            return Template(node)
        elif not isinstance(node, dict):
            return node
        elif node["type"] == "root":
            return MetaPrompt([cls._json_to_sammo(child) for child in node["children"]], render_as="raw")
        elif node["type"] == "section":
            return Section(
                title=node["title"],
                content=[cls._json_to_sammo(child) for child in node["children"]],
                **_get_annotations(node),
            )
        elif node["type"] in ["paragraph", "list", "blockcode", "codefence", "quote"]:
            return Paragraph(
                content=[cls._json_to_sammo(child) for child in node["children"]], **_get_annotations(node)
            )
        else:
            raise ValueError(f"Unsupported type: {type(node)} with node: {node}")
