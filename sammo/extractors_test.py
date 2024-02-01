# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
from fractions import Fraction
from sammo.extractors import *
from sammo.base import VerbatimText
from sammo.runners import MockedRunner


# Test DefaultExtractor
def test_default_extractor_process():
    extractor = DefaultExtractor("")
    assert extractor._extract_from_single_value("sample") == ["sample"]


# Test SplitLines
@pytest.mark.asyncio
async def test_split_lines_process():
    split_lines = SplitLines("")
    assert split_lines._extract_from_single_value("line1\nline2") == ["line1", "line2"]

    split_lines = await SplitLines(VerbatimText("line1\nline2"))(MockedRunner, {})
    assert [o.value for o in split_lines] == ["line1", "line2"]


@pytest.mark.parametrize("expr", ["lambda: x.upper()", "lambda x, y: x.upper()", "3"])
def test_lambda_extractor_init(expr):
    with pytest.raises(ValueError):
        extractor = LambdaExtractor("", expr)


def test_lambda_extractor_process():
    extractor = LambdaExtractor("", "lambda x: x.upper()")
    assert extractor._extract_from_single_value("nEveR") == ["NEVER"]


# Test ParseJSON
def test_parse_json_process():
    json_extractor = ParseJSON("")
    assert json_extractor._extract_from_single_value('{"key": "value"}') == [{"key": "value"}]


def test_parse_json_process_fragments():
    json_extractor = ParseJSON("", parse_fragments="all")
    assert json_extractor._extract_from_single_value('surrounding{"key": "value"}text{"num": [1, 2]}is ignored') == [
        {"key": "value"},
        {"num": [1, 2]},
    ]


# Test ExtractRegex
def test_extract_regex_process():
    extractor = ExtractRegex("", r"\d+")
    assert extractor._extract_from_single_value("There are 12 apples and 13 oranges.") == ["12", "13"]


# Test MarkdownParser
def test_markdown_parser_process():
    parser = MarkdownParser("")
    assert parser._extract_from_single_value("# Heading1\nThis is content.") == [
        {"name": "Heading1", "type": "section", "content": [{"type": "paragraph", "content": "This is content."}]}
    ]


# Test YAMLParser
def test_yaml_parser_process():
    parser = YAMLParser("")
    assert parser._extract_from_single_value("key: value") == [{"key": "value"}]


# Test ParseXML
def test_parse_xml_process():
    parser = ParseXML("")
    assert parser._extract_from_single_value("<root><child>value</child></root>") == [{"root": {"child": "value"}}]


def test_parse_xml_process_fragments():
    parser = ParseXML("", parse_fragments="all")
    assert parser._extract_from_single_value("<root><child>value</child></root><root><child>value</child></root>") == [
        {"root": {"child": "value"}},
        {"root": {"child": "value"}},
    ]


# Test JSONPath
def test_jsonpath_process():
    json_path = JSONPath("", "$.key")
    assert json_path._extract_from_single_value({"key": "value"}) == ["value"]


# Test ToNum
@pytest.mark.parametrize(
    "input_val, dtype, factor, offset, expected",
    [("3/2", "fraction", 1, 0, Fraction("3/2")), ("5", "int", 1, 0, 5), ("3.5", "float", 2, 1, 8.0)],
)
def test_to_num_process(input_val, dtype, factor, offset, expected):
    to_num = ToNum(VerbatimText(""), dtype=dtype, factor=factor, offset=offset)
    assert to_num._extract_from_single_value(input_val) == [expected]

    # alternative syntax
    to_num = ToNum(VerbatimText(""), dtype=dtype) * factor + offset
    assert to_num._extract_from_single_value(input_val) == [expected]
