# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module contains components that extract data from the output of other components, typically from LLM calls.
Common formats such as JSON, XML, or Markdown are supported and require no data format specification. If validation is
required, it should happen downstream of the extraction step.
"""
import abc
import ast
import fractions
import json
import logging
import re

from beartype.typing import Literal
from frozendict import frozendict
import jsonpath_ng
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
import xmltodict
import yaml

from sammo.base import ListComponent, Component, LLMResult, EmptyResult, ParseResult, Runner, NonEmptyResult

logger = logging.getLogger(__name__)


class Extractor(ListComponent):
    """Base class for all extractors. Extractors take the output of a child component and extract data from it.

    :param child: The child component whose output gets processed.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param flatten: Whether to flatten the output.
    """

    def __init__(self, child: Component, on_error: Literal["raise", "empty_result"] = "raise", flatten=True):
        super().__init__(child)
        self._on_error = on_error
        self.dependencies = [self._child]
        self._flatten = flatten

    async def _call(
        self, runner: Runner, context: dict, dynamic_context: frozendict | None = None
    ) -> list[ParseResult | EmptyResult]:
        x = await self._child(runner, context, dynamic_context)

        if not isinstance(x, list):
            x = [x]
        processed = list()
        for y in x:
            if isinstance(y, EmptyResult):
                processed.append(y)
            else:
                try:
                    if isinstance(y, NonEmptyResult):
                        vals = y.value
                    else:
                        vals = y
                    result = [ParseResult(v, parent=y) for v in self._extract_from_single_value(vals)]
                    if self._flatten:
                        processed.extend(result)
                    else:
                        processed.append(result)
                except Exception as e:
                    logger.warning(f"Error extracting from {str(x)[:100]}: {e}")
                    if self._on_error == "raise":
                        raise e
                    else:
                        processed.append(EmptyResult(parent=y))

        return processed

    @abc.abstractmethod
    def _extract_from_single_value(self, x: str) -> list:
        pass


class DefaultExtractor(Extractor):
    """Performs no processing. Passes value(s) through.

    :param child: The child component whose output gets processed.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param flatten: Whether to flatten the output.
    """

    def _extract_from_single_value(self, x: str) -> list:
        return [x]


class SplitLines(Extractor):
    """Splits a string into lines.

    :param child: The child component whose output gets processed.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param flatten: Whether to flatten the output.
    """

    def _extract_from_single_value(self, x: str) -> list:
        return x.splitlines()


class StripWhitespace(Extractor):
    """Strips whitespace from a string.

    :param child: The child component whose output gets processed.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param flatten: Whether to flatten the output.
    """

    def _extract_from_single_value(self, x: str) -> list:
        return [x.strip()]


class LambdaExtractor(Extractor):
    """Extracts data from the output of a child component using a lambda expression.
    This is useful if minor processing is required, e.g. to lowercase the output.
    The lambda expression need to be a string so it can be serialized. It must take exactly one argument.

    :param child: The child component whose output gets processed.
    :param lambda_src_code: The lambda expression as a string.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param flatten: Whether to flatten the output.
    """

    def __init__(
        self,
        child: Component,
        lambda_src_code: str,
        on_error: Literal["raise", "empty_result"] = "raise",
        flatten=True,
    ):
        super().__init__(child, on_error, flatten)
        tree = ast.parse(lambda_src_code, mode="eval")
        if not isinstance(tree.body, ast.Lambda):
            raise ValueError("Source code passed must be a lambda expression.")
        elif len(tree.body.args.args) != 1:
            raise ValueError("Lambda expression must take exactly one argument.")
        self._func = eval(lambda_src_code)

    def _extract_from_single_value(self, x):
        result = self._func(x)
        return [result]


class ParseJSON(Extractor):
    """Extracts all matches of a regular expression. Supports object or list fragments surrounded by text.

    :param child: The child component whose output gets processed.
    :param parse_fragments: Whether to parse all fragments, only the first, or require whole string to be valid JSON.
    :param lowercase_fieldnames: Whether to lowercase dictionary keys.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    """

    OPENING = "[{"
    CLOSING = "]}"

    def __init__(
        self,
        child: Component,
        parse_fragments: Literal["all", "first", "whole"] = "all",
        lowercase_fieldnames: bool = True,
        on_error="raise",
    ):
        super().__init__(child, on_error=on_error)
        self._parse_fragments = parse_fragments
        self._hook = self._lowercase_keys if lowercase_fieldnames else None

    @staticmethod
    def _lowercase_keys(dct) -> str:
        return {k.lower(): v for k, v in dct.items()}

    def _find_json_fragments(self, x: str):
        stack = list()
        for i, char in enumerate(x):
            if char in self.OPENING:
                stack.insert(0, (i, char))
            elif char in self.CLOSING:
                if len(stack) == 0:
                    # Ignore and closing tags before we see open ones
                    continue
                start, prev_char = stack.pop(0)
                if len(stack) == 0:
                    if self.OPENING.index(prev_char) != self.CLOSING.index(char):
                        raise ValueError(
                            f"Malformed JSON: Expected closing '{self.OPENING[self.OPENING.index(prev_char)]}'."
                        )
                    else:
                        yield x[start : i + 1]

    def _extract_from_single_value(self, x: str):
        if self._parse_fragments == "all":
            x = list(self._find_json_fragments(x))
        elif self._parse_fragments == "first":
            x = list(self._find_json_fragments(x))[0]
        if isinstance(x, list):
            return [json.loads(y, object_hook=self._hook) for y in x]
        else:
            return [json.loads(x, object_hook=self._hook)]


class ExtractRegex(Extractor):
    """Extracts all matches of a regular expression.

    :param child: The child component whose output gets processed.
    :param regex: The regular expression string.
    :param max_matches: The maximum number of matches to return.
    :param strip_whitespaces: Whether to strip whitespaces from the matches.
    """

    INT_EOL = r"\d+$"
    FRACTION_EOL = r"[0-9]+/0*[1-9][0-9]*$"
    PERCENTAGE_EOL = r"(\.\d+%$)|(\d+.\d+?%$)"
    LAST_TSV_COL = r"(?:\t)([^\t]*)$"

    def __init__(self, child: Component, regex: str, max_matches: int | None = None, strip_whitespaces: bool = True):
        super().__init__(child)

        self._max_matches = max_matches
        self._strip = strip_whitespaces
        self._regex = re.compile(regex, re.MULTILINE | re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _strip_whitespaces(x: str | tuple[str]) -> str:
        if isinstance(x, tuple):
            return tuple([y.strip() for y in x])
        else:
            return x.strip()

    def _extract_from_single_value(self, x: str) -> list[str] | list[tuple[str]]:
        matches = self._regex.findall(x)
        if self._max_matches is not None:
            matches = matches[: self._max_matches]
        if self._strip:
            matches = [self._strip_whitespaces(m) for m in matches]
        return matches


class MarkdownParser(Extractor):
    """Parses text as simplified Markdown only separating sections and paragraphs.

    :param child: The child component whose output gets processed.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param flatten: Whether to flatten the output.
    """

    @classmethod
    def _get_text(cls, x):
        if x.type == "text":
            return x.content
        else:
            return "".join([cls._get_text(c) for c in x.children])

    def _extract_from_single_value(self, x: str):
        md = MarkdownIt("zero")
        syntax_tree = SyntaxTreeNode(md.enable(["heading", "lheading"]).parse(x))
        queue = [syntax_tree]
        root_section = {"name": None, "type": "root", "content": list()}
        last_level = 0
        last_section_at_level = {0: root_section}

        while queue:
            current = queue.pop(0)
            if current.type == "heading":
                last_level = int(current.tag.replace("h", ""))
                new_section = {"type": "section", "name": self._get_text(current), "content": list()}
                last_section_at_level[last_level - 1]["content"].append(new_section)
                last_section_at_level[last_level] = new_section
            elif current.type == "paragraph":
                last_section_at_level[last_level]["content"].append(
                    {"type": "paragraph", "content": self._get_text(current)}
                )
            queue = queue + current.children

        return root_section["content"]


class YAMLParser(Extractor):
    """Parses text as YAML, but does not support fragments due to lack of delimiter in this format.

    :param child: The child component whose output gets processed.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param flatten: Whether to flatten the output.
    """

    def _extract_from_single_value(self, x: str):
        return [yaml.safe_load(x)]


class ParseXML(Extractor):
    """Parses text as XML, supporting fragments that are surrounded by text.

    :param child: The child component whose output gets processed.
    :param parse_fragments: Whether to parse all fragments, only the first, or none.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param use_attributes_marker: Whether to prefix XML attribute names with '@'.
    :param lowercase_fieldnames: Whether to lowercase any dictionary keys.
    :param ignore_fragments_with_tags: A list of tags to ignore."""

    XML_TAGS = re.compile(r"<(?P<opening>\w+)[^/>]*[^/>]?>|</(?P<closing>\w+)[^>]*[^/>]?>", re.DOTALL)

    def __init__(
        self,
        child: Component,
        parse_fragments: Literal["all", "first", "none"] = "first",
        ignore_fragments_with_tags=tuple(),
        on_error: str = "raise",
        use_attributes_marker=False,
        lowercase_fieldnames=False,
    ):
        super().__init__(child, on_error=on_error)
        self._parse_fragments = parse_fragments
        self._use_attributes_marker = use_attributes_marker
        self._lowercase_fieldnames = lowercase_fieldnames
        self._ignore_tags = {x.lower() for x in ignore_fragments_with_tags}

    def _post_process(self, path, key, value):
        if not self._use_attributes_marker and isinstance(key, str) and key.startswith("@"):
            key = key[1:]
        if self._lowercase_fieldnames and isinstance(key, str):
            key = key.lower()
        return key, value

    def _extract_from_single_value(self, x: str) -> list:
        if self._parse_fragments == "all":
            return [xmltodict.parse(f, postprocessor=self._post_process) for f in self._find_xml_fragments(x)]
        elif self._parse_fragments == "first":
            return [xmltodict.parse(list(self._find_xml_fragments(x))[0], postprocessor=self._post_process)]
        else:
            return [xmltodict.parse(x, postprocessor=self._post_process)]

    def _find_xml_fragments(self, x: str):
        stack = list()
        for match in self.XML_TAGS.finditer(x):
            if match.group("opening") and match.group("opening").lower() not in self._ignore_tags:
                stack.insert(0, match)
            elif match.group("closing") and match.group("closing").lower() not in self._ignore_tags:
                if len(stack) == 0:
                    # Ignore and closing tags before we see open ones
                    continue
                opening = stack.pop(0)
                if len(stack) == 0:
                    if opening.group("opening") != match.group("closing"):
                        raise ValueError("Malformed XML: Closing tag does not match opening tag.")
                    else:
                        yield x[opening.start() : match.end()]


class JSONPath(Extractor):
    """Extracts all matches of a JSONPath expression.

    :param child: The child component whose output gets processed.
    :param path: The JSONPath expression.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param max_results: The maximum number of results to return.
    :param flatten_lists: Whether to flatten matches which are lists.
    """

    def __init__(self, child: Extractor, path: str, on_error="raise", max_results=None, flatten_lists=True):
        super().__init__(child, on_error=on_error)
        self._path = jsonpath_ng.parse(path)
        self._flatten_lists = flatten_lists
        self._max_results = max_results

    def _extract_from_single_value(self, x):
        output = list()
        for match in self._path.find(x):
            if isinstance(match.value, list) and self._flatten_lists:
                output += match.value
            else:
                output += [match.value]

        if self._max_results is not None:
            output = output[: self._max_results]
        return output


class ToNum(Extractor):
    """Converts the output of a child component to a number * factor + offset.

    **Example usage:**

    .. code-block:: python

        ToNum(child, dtype="int", factor=3, offset=4) # explicit specification
        ToNum(child) * 3 + 4 # implicit specification

    :param child: The child component whose output gets processed.
    :param on_error: What to do if an error occurs. Either 'raise' or 'empty_result'.
    :param dtype: The type of number to return. Either 'float', 'int', or 'fraction'.
    :param factor: A factor to multiply the number by.
    :param offset: An offset to add to the number.
    """

    def __init__(
        self,
        child: Component,
        on_error: Literal["raise", "empty_result"] = "raise",
        dtype: Literal["fraction", "int", "float"] = "float",
        factor: float = 1,
        offset: float = 0,
    ):
        super().__init__(child, on_error=on_error)
        self._dtype = dict(float=float, int=int, fraction=lambda x: float(fractions.Fraction(x)))[dtype]
        self._factor = factor
        self._offset = offset

    def __truediv__(self, value: float | int):
        self._factor /= value
        return self

    def __mul__(self, value: float | int):
        self._factor *= value
        return self

    def __add__(self, value: float | int):
        self._offset += value
        return self

    def __sub__(self, value: float | int):
        return self - value

    @staticmethod
    def _to_int_if_possible(x: float | int) -> int | float:
        if isinstance(x, int) or x.is_integer():
            return int(x)
        else:
            return x

    @property
    def factor(self):
        return self._to_int_if_possible(self._factor)

    @property
    def offset(self):
        return self._to_int_if_possible(self._offset)

    def _extract_from_single_value(self, x: str) -> list:
        return [self._dtype(x) * self.factor + self.offset]
