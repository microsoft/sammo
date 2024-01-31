# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import pandas as pd
from sammo.data import DataTable, MinibatchIterator, Accessor


@pytest.fixture
def sample_dict_data():
    attributes = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}, {"name": "Charlie", "age": 35}]
    labels = [1, 0, 1]
    return DataTable(attributes, labels)


@pytest.fixture
def sample_scalar_data():
    attributes = ["Alice", "Bob", "Charlie"]
    labels = [1, 0, 1]
    return DataTable(attributes, labels)


def test_datatable_filtered(sample_dict_data):
    filtered = sample_dict_data.inputs.filtered_on(lambda x: x["name"] == "Alice")
    assert len(filtered) == 1


def test_datatable_to_string(sample_dict_data):
    string = sample_dict_data.to_string()
    assert isinstance(string, str)
    DataTable([]).to_string()


def test_from_pandas(sample_dict_data):
    df = pd.DataFrame(sample_dict_data.to_records())
    dt = DataTable.from_pandas(df, input_fields="input", output_fields="output")
    assert dt.to_records() == sample_dict_data.to_records()


def test_datatable_indexing(sample_dict_data):
    sample_dict_data.outputs[0]
    sample_dict_data.outputs[[0, 1]]
    sample_dict_data.outputs[0:1]
    sample_dict_data.inputs[0]
    sample_dict_data.inputs[[0, 1]]
    sample_dict_data.inputs[0:1]


def test_datatable_only_rows():
    DataTable([1, 2, 3])


def test_datatable_to_dicts(sample_dict_data):
    dicts = sample_dict_data.to_records()
    expected = [
        {"input": {"name": "Alice", "age": 25}, "output": 1},
        {"input": {"name": "Bob", "age": 30}, "output": 0},
        {"input": {"name": "Charlie", "age": 35}, "output": 1},
    ]
    assert dicts == expected


def test_datatable_scalars_to_dicts(sample_scalar_data):
    dicts = sample_scalar_data.to_records()
    expected = [
        {"input": "Alice", "output": 1},
        {"input": "Bob", "output": 0},
        {"input": "Charlie", "output": 1},
    ]
    assert dicts == expected


def test_accessor_init():
    parent_data = {"input": [1, 2, 3], "output": [4, 5, 6]}
    accessor = Accessor(parent_data, "input")
    assert hasattr(accessor, "_parent")
    assert hasattr(accessor, "_group")
    assert accessor._group == "input"


def test_accessor_safe_get():
    y = {"test": 1}
    result = Accessor._safe_get(y, "test")
    assert result == 1


def test_datatable_init():
    attributes = [1, 2, 3]
    labels = [4, 5, 6]
    table = DataTable(attributes, labels)
    assert len(table) == 3


def test_datatable_persistent_hash():
    attributes = [1, 2, 3]
    labels = [4, 5, 6]
    table = DataTable(attributes, labels)
    assert isinstance(table.persistent_hash(), int)


def test_hash_changes(sample_dict_data):
    hash_before = sample_dict_data.persistent_hash()
    sample_dict_data.outputs[[0, 1]] = [1, 0]
    hash_after = sample_dict_data.persistent_hash()
    assert hash_before == hash_after
    sample_dict_data.outputs[0] = [2]
    hash_after = sample_dict_data.persistent_hash()
    assert hash_before != hash_after


def test_datatable_from_records():
    records = [{"input": 4, "output": 1}, {"input": 5, "output": 2}, {"input": 6, "output": 3}]
    table = DataTable.from_records(records)
    assert len(table) == 3


def test_datable_set_all(sample_dict_data):
    sample_dict_data.outputs[:] = 3
    sample_dict_data.outputs[:] = "hello"


def test_datatable_set_to_list(sample_dict_data):
    sample_dict_data.outputs[0] = [1, 2, 3]
    assert sample_dict_data.outputs.raw_values[0] == [1, 2, 3]


def test_datatable_getitem_single():
    attributes = [1, 2, 3]
    labels = [4, 5, 6]
    table = DataTable(attributes, labels)
    sliced = table[1]
    assert len(sliced) == 1
    assert sliced.outputs.raw_values == [5]


def test_datatable_getitem_multiple():
    attributes = [1, 2, 3]
    labels = [4, 5, 6]
    table = DataTable(attributes, labels)
    sliced = table[[1, 2]]
    assert len(sliced) == 2
    assert sliced.outputs.raw_values == [5, 6]


def test_datatable_sample():
    attributes = [1, 2, 3, 4, 5]
    labels = [6, 7, 8, 9, 10]
    table = DataTable(attributes, labels)
    sampled = table.sample(3)
    assert len(sampled) == 3


def test_datatable_shuffle():
    attributes = [1, 2, 3, 4, 5]
    labels = [6, 7, 8, 9, 10]
    table = DataTable(attributes, labels)
    shuffled = table.shuffle()
    assert len(shuffled) == 5


def test_datatable_random_split():
    attributes = [1, 2, 3, 4, 5]
    labels = [6, 7, 8, 9, 10]
    table = DataTable(attributes, labels)
    split1, split2 = table.random_split(3, 2)
    assert len(split1) == 3
    assert len(split2) == 2


def test_minibatch_iterator():
    attributes = [1, 2, 3, 4, 5]
    labels = [6, 7, 8, 9, 10]
    table = DataTable(attributes, labels)
    iterator = MinibatchIterator(table, 2)
    batches = list(iterator)
    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[-1]) == 1
    iterator = MinibatchIterator(table, 1)
    batches = list(iterator)
    assert len(batches) == 5
    assert batches[0] == 0


def test_persistent_hashing_consistency(sample_dict_data):
    hash1 = sample_dict_data.persistent_hash()
    hash2 = sample_dict_data.persistent_hash()
    assert hash1 == hash2
