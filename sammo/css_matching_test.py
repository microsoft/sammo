# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sammo.css_matching import XmlTree
import pytest
import pyglove as pg

from sammo.instructions import MetaPrompt, Section, Paragraph, InputData
from sammo.search_op import one_of, get_first_point


@pytest.fixture
def simple_tree():
    return pg.Dict(name="root", children=[pg.Dict(child1="value1"), pg.Dict(child2="value2")])


@pytest.fixture
def sammo_example():
    return MetaPrompt(
        [
            Section(
                title="T-A",
                content=Section(title="Title of subsection", content="Text in subsection", reference_id="42"),
                reference_id="A",
                reference_classes=["class1"],
            ),
            Section(title="T-B", reference_id="B", content="Other text.", reference_classes=["class1"]),
        ]
    )


def test_from_pyglove(simple_tree):
    xml_tree = XmlTree.from_pyglove(simple_tree)
    assert isinstance(xml_tree, XmlTree)
    assert xml_tree.root.tag == "root"


def test_find_all(simple_tree):
    xml_tree = XmlTree.from_pyglove(simple_tree)
    matches = xml_tree.find_all("child1")
    assert len(matches) == 1
    assert matches[0] == "children[0].child1"


def test_empty_pg_node():
    str(XmlTree.from_pyglove(None)) == "</root>"


def test_list():
    list = pg.List(["yes", "no"])
    xml_tree = XmlTree.from_pyglove(list)
    assert len(xml_tree.find_all("str")) == 2


def test_sammo_integration(sammo_example):
    xml_tree = XmlTree.from_pyglove(sammo_example)
    matches = xml_tree.find_all("#42")
    assert len(matches) == 1
    matches = xml_tree.find_all("#A")
    assert len(matches) == 1
    matches = xml_tree.find_all(".class1")
    assert len(matches) == 2
