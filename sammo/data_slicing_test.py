# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest

from sammo.data import DataTable


@pytest.mark.parametrize("size", [0, 5])
@pytest.mark.parametrize(
    "key",
    [
        slice(None),
        slice(None, 0),
        slice(-2, None),
        slice(None, -1),
        slice(-100, 100),
        slice(100, None),
        slice(None, None, -1),
        slice(3, 0, -1),
        slice(None, None, 2),
        slice(None, None, -2),
        slice(100, -100, -2),
        slice(3, 1),
    ],
)
def test_slices_match_python_lists(size, key):
    inputs = list(range(size))
    outputs = [f"label-{i}" for i in inputs]
    table = DataTable(inputs, outputs, constants={"task": "classification"})

    sliced = table[key]

    assert sliced.inputs.raw_values == inputs[key]
    assert sliced.outputs.raw_values == outputs[key]
    assert sliced.constants == table.constants
    assert table.inputs[key] == inputs[key]
    assert table.outputs[key] == outputs[key]


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize(
    "key", [slice(None, 0), slice(-2, None), slice(None, None, -1), slice(3, 0, -2), slice(-100, 100)]
)
def test_output_slice_assignment_updates_only_selected_rows(key, broadcast):
    outputs = ["A", "B", "C", "D", "E"]
    table = DataTable(list(range(len(outputs))), outputs.copy())
    replacements = [f"new-{i}" for i in range(len(outputs[key]))]
    expected = outputs.copy()
    expected[key] = ["updated"] * len(replacements) if broadcast else replacements

    table.outputs[key] = "updated" if broadcast else replacements

    assert table.outputs.raw_values == expected
    assert table.inputs.raw_values == list(range(len(outputs)))


def test_slice_reads_reject_zero_step():
    table = DataTable([1, 2, 3], ["A", "B", "C"])
    for sequence in (table, table.inputs, table.outputs):
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            sequence[::0]


def test_output_slice_assignment_rejects_zero_step():
    table = DataTable([1, 2, 3], ["A", "B", "C"])

    with pytest.raises(ValueError, match="slice step cannot be zero"):
        table.outputs[::0] = "updated"

    assert table.outputs.raw_values == ["A", "B", "C"]


@pytest.mark.parametrize("sizes", [(0, 2), (2, 0, 2), (0, 0, 0)])
def test_random_split_keeps_zero_sized_partitions_empty(sizes):
    table = DataTable(list(range(5)), ["A", "B", "C", "D", "E"])

    partitions = table.random_split(*sizes, seed=0)

    assert tuple(map(len, partitions)) == sizes
    selected = [row for partition in partitions for row in partition.inputs.raw_values]
    assert len(selected) == len(set(selected))
    for partition in partitions:
        assert partition.outputs.raw_values == [table.outputs[row] for row in partition.inputs.raw_values]
