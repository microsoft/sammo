# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sammo.base import Template, VerbatimText


def test_simple_query():
    a = VerbatimText("Hello", name="1")
    b = VerbatimText("World", name="2")
    nested = Template("{{a}} {{b}}", a=a, b=b)
    assert nested.query({"name": "1"}) == a
    assert nested.query({"name": "2"}) == b
    assert nested.query({"name": "3"}) is None


def test_same_name_query():
    a = VerbatimText("Hello", name="1")
    b = VerbatimText("World", name="1")
    nested = Template("{{a}} {{b}}", a=a, b=b)
    assert nested.query({"name": "1"}) == a
    assert nested.query({"name": "2"}) is None
    assert nested.query({"name": "1"}, max_matches=None) == [a, b]
    assert nested.query({"name": "1"}, return_path=True) == ("a", a)
    assert nested.query({"name": "1"}, max_matches=None, return_path=True) == [("a", a), ("b", b)]
