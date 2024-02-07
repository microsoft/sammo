# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from unittest.mock import patch, mock_open, Mock
import pytest
from io import BytesIO
from sammo.store import PersistentDict, InMemoryDict, serialize_json


@pytest.mark.parametrize(
    "expected,data",
    [
        ({b'"test"': "Hello"}, b'"test"\t"Hello"'),
        ({b'"test"': "Hello", b'"int"': 3}, b'"test"\t"Hello"\n"int"\t3'),
        ({b'"test"': "Hello"}, b'"test"\t"Hello"'),
        ({b'"test"': "Hello", b'"int"': None}, b'"test"\t"Hello"\n"int"\t3\n"int"\tnull'),
    ],
)
def test_read(data, expected):
    # patch file lock
    with patch("sammo.store.filelock.FileLock"):
        with patch("builtins.open", mock_open(read_data=data)) as mock_file:
            store = PersistentDict("dummy.txt")
            read = store._load()
            assert read == expected


@pytest.mark.parametrize(
    "data",
    [
        [({"id": 1, "irrelevant": "12"}, "Hello")],
        [({"id": 1, "irrelevant": "12"}, "Hello"), ({"id": 2, "irrelevant": "12"}, "Hello")],
    ],
)
def test_remapping(data):
    with patch("sammo.store.filelock.FileLock"):
        with patch("builtins.open", lambda x, y: BytesIO()) as mock_file:
            store = PersistentDict("test_file.data", project_keys=lambda x: x.get("id"))
            for k, v in data:
                store[k] = v
                assert k in store
                k_prime = k.copy()
                k_prime["irrelevant"] = 23
                assert k_prime in store
                assert store[k_prime] == "Hello"
                store[k_prime] = "World"
                assert store[k] == "World"


@pytest.mark.parametrize(
    "data",
    [
        [("test", "Hello")],
        [("test", "Hello"), ("int", 3)],
    ],
)
def test_delete(data):
    with patch("sammo.store.filelock.FileLock"):
        with patch("builtins.open", lambda x, y: BytesIO()) as mock_file:
            store = PersistentDict("test_file.data")
            for k, v in data:
                store[k] = v
            assert data[0][0] in store
            del store[data[0][0]]
            assert data[0][0] not in store


def test_vaccum():
    with patch("sammo.store.filelock.FileLock"):
        with patch("os.replace"):
            with patch("builtins.open", lambda x, y: BytesIO()) as mock_file:
                store = PersistentDict("test_file.data")

                data = [("test", "Hello"), ("int", 3), ("int", None)]
                for k, v in data:
                    store[k] = v
                store.vacuum()


@pytest.mark.parametrize(
    "data,expected",
    [
        ([("test", "Hello")], b'"test"\t"Hello"'),
        ([("test", "Hello"), ("int", 3)], b'"test"\t"Hello"\n"int"\t3'),
        ([("test", "Hello"), ("int", None)], b'"test"\t"Hello"\n"int"\tnull'),
        ([("test", "Hello"), ("int", 3), ("int", None)], b'"test"\t"Hello"\n"int"\t3\n"int"\tnull'),
        (
            [("test", "Hello"), ("int", 3), ("int", None), ("test", True)],
            b'"test"\t"Hello"\n"int"\t3\n"int"\tnull\n"test"\ttrue',
        ),
    ],
)
def test_write(data, expected):
    with patch("sammo.store.filelock.FileLock"):
        with patch("builtins.open", lambda x, y: BytesIO()) as mock_file:
            store = PersistentDict("test_file.data")
            for k, v in data:
                store[k] = v
            store._fp.seek(0)
            content = store._fp.read()
            assert content == expected


@pytest.mark.parametrize(
    "data,expected",
    [([("test", "Hello")], b'"test"\t"Hello"'), ([("test", "Hello"), ("int", 3)], b'"test"\t"Hello"\n"int"\t3')],
)
def test_persist(data, expected):
    with patch("builtins.open") as mock_file:
        store = InMemoryDict()
        for k, v in data:
            store[k] = v
        store.persist("")
        print(mock_file.mock_calls)
        assert len(mock_file.mock_calls) == len(data) * 3 + 3


@pytest.mark.parametrize(
    "data",
    [3, {"test": "Hello"}, "some string", b"bytes string"],
)
def test_fix_point(data):
    serialize_json(data) == serialize_json(serialize_json(data))
