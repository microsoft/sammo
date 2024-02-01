# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
from sammo.dataformatters import DataFormatter, JSONDataFormatter, XMLDataFormatter
from sammo.base import VerbatimText


def test_format_batch_typical_data():
    test = DataFormatter()
    test._dump = lambda x: x.records
    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"label": 1}], [{"label": 0}])
    expected = [
        {"id": 0, "kind": "input", "kind_alias": "input", "kind_order": 0, "value": "value1"},
        {"id": 0, "kind": "gold_label", "kind_alias": "output", "kind_order": 1, "value": 1},
        {"id": 0, "kind": "predicted_label", "kind_alias": "predicted_output", "kind_order": 2, "value": 0},
    ]

    assert returned == expected


def test_format_not_flatten_data():
    test = DataFormatter(flatten_1d_dicts=False)
    test._dump = lambda x: x.records
    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"l": 1}], [{"l": 0}])
    expected = [
        {"id": 0, "kind": "input", "kind_alias": "input", "kind_order": 0, "value": {"key": "value1"}},
        {"id": 0, "kind": "gold_label", "kind_alias": "output", "kind_order": 1, "value": {"l": 1}},
        {"id": 0, "kind": "predicted_label", "kind_alias": "predicted_output", "kind_order": 2, "value": {"l": 0}},
    ]
    assert returned == expected

    returned_single = test.format_single({"key": "value1"}, {"l": 1}, {"l": 0})
    assert returned_single == expected


@pytest.mark.asyncio
async def test_format_flat_json_item_orient():
    test = JSONDataFormatter(newline_delimited=False, flatten_1d_dicts=True, indent=None)

    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"l": 1}], [{"l": 0}])
    expected = '[{"id": 0, "input": "value1", "output": 1, "predicted_output": 0}]'

    assert returned == expected

    result = await test.get_extractor(VerbatimText(returned))(None, {})
    assert test._unwrap_results(result) == [1]


@pytest.mark.asyncio
async def test_format_flat_ndjson_item_orient():
    test = JSONDataFormatter(newline_delimited=True, flatten_1d_dicts=True, indent=None)

    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"l": 1}], [{"l": 0}])
    expected = '{"id": 0, "input": "value1", "output": 1, "predicted_output": 0}'
    assert returned == expected

    result = await test.get_extractor(VerbatimText(returned))(None, {})
    assert test._unwrap_results(result) == [1]


@pytest.mark.asyncio
async def test_format_flat_ndjson_kind_orient():
    test = JSONDataFormatter(newline_delimited=True, flatten_1d_dicts=True, indent=None, orient="kind")

    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"l": 1}], [{"l": 0}])
    expected = (
        'input: [{"id": 0, "value": "value1"}]\n'
        'output: [{"id": 0, "value": 1}]\n'
        'predicted_output: [{"id": 0, "value": 0}]'
    )
    assert returned == expected


def test_format_flat_json_kind_orient():
    test = JSONDataFormatter(newline_delimited=False, flatten_1d_dicts=True, indent=None, orient="kind")

    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"l": 1}], [{"l": 0}])
    expected = (
        '{"input": [{"id": 0, "value": "value1"}], "output": [{"id": 0, "value": 1}], '
        '"predicted_output": [{"id": 0, "value": 0}]}'
    )
    assert returned == expected


@pytest.mark.asyncio
async def test_format_nested_json_item_orient():
    test = JSONDataFormatter(
        newline_delimited=False, flatten_1d_dicts=False, indent=None, include_ids=False, orient="item"
    )

    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"l": 1}], [{"l": 0}])
    expected = '[{"input": {"key": "value1"}, "output": {"l": 1}, "predicted_output": {"l": 0}}]'
    assert returned == expected

    result = await test.get_extractor(VerbatimText(returned))(None, {})
    assert result[0].to_json() == {"l": 1}


@pytest.mark.asyncio
async def test_format_nested_json_item_orient_with_ids():
    test = JSONDataFormatter(
        newline_delimited=False, flatten_1d_dicts=False, indent=None, include_ids=True, orient="item"
    )

    # call the format_batch function
    returned = test.format_batch([{"key": "value1"}], [{"l": 1}], [{"l": 0}])
    expected = '[{"id": 0, "input": {"key": "value1"}, "output": {"l": 1}, "predicted_output": {"l": 0}}]'
    assert returned == expected

    result = await test.get_extractor(VerbatimText(returned))(None, {})
    assert result[0].to_json() == {"l": 1}


def test_xml_flat_item_orient():
    test = XMLDataFormatter(flatten_1d_dicts=True, orient="item")

    # call the format_batch function
    returned = test.format_batch([{"key": "row1"}, {"key": "row2"}], [{"l": [1, 2]}, {"l": [0, 1]}])
    expected = (
        '<input id="0">\n'
        "\t<value>value1</value>\n"
        '</input><output id="0">\n'
        "\t<value>1</value>\n"
        "\t<value>2</value>\n"
        '</output><predicted_output id="0">\n'
        "\t<value>0</value>\n"
        "\t<value>1</value>\n"
        "</predicted_output>"
    )
    returned == expected
