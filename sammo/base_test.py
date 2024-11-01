# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sammo.base import Template, VerbatimText


def test_simple_query():
    a = VerbatimText("Hello", reference_id="1")
    b = VerbatimText("World", reference_id="2")
    nested = Template("{{a}} {{b}}", a=a, b=b)
    assert nested.find_first("#1").node == a
    assert nested.find_first("#2").node == b
    assert nested.find_first("a").node == a
    assert nested.find_first("b").node == b
    assert nested.find_first("#3") is None
    assert len(nested.find_all("verbatimtext")) == 2
