# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest

from sammo.store import PersistentDict


@pytest.mark.parametrize("second_value", ["x", "a longer cached response than the next write"])
@pytest.mark.parametrize("preexisting", [False, True])
def test_interleaved_writers_preserve_all_committed_entries(tmp_path, second_value, preexisting):
    path = tmp_path / "cache.jsonl"
    if preexisting:
        path.write_bytes(b'"existing"\t"retained"\n')
    first = PersistentDict(path)
    second = PersistentDict(path)
    try:
        first["first"] = "first response"
        second["second"] = second_value
        first["third"] = "third response"
        second["fourth"] = "fourth response"

        restored = PersistentDict(path)
        assert restored["first"] == "first response"
        assert restored["second"] == second_value
        assert restored["third"] == "third response"
        assert restored["fourth"] == "fourth response"
        if preexisting:
            assert restored["existing"] == "retained"
        assert len(restored) == 4 + int(preexisting)
    finally:
        for writer in (first, second):
            if writer._fp is not None:
                writer._fp.close()


def test_delete_appends_after_another_writer(tmp_path):
    path = tmp_path / "cache.jsonl"
    first = PersistentDict(path)
    first["obsolete"] = "old response"
    second = PersistentDict(path)
    try:
        second["retained"] = "another writer's response"
        del first["obsolete"]

        restored = PersistentDict(path)
        assert "obsolete" not in restored
        assert restored["retained"] == "another writer's response"
    finally:
        for writer in (first, second):
            if writer._fp is not None:
                writer._fp.close()
