# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from unittest.mock import MagicMock

import pybars
import pytest

from sammo.dataformatters import PlainFormatter
from sammo.instructions import MetaPrompt, Section
from sammo.mutators import *
from sammo.runners import MockedRunner


def basic_template():
    return Output(
        MetaPrompt(
            Section(name="test", content="The big ball rolled over the street."),
            data_formatter=PlainFormatter(),
            render_as="markdown",
        )
    )


def duplicate_template():
    return Output(
        MetaPrompt(
            [
                Section(name="test", content="The big ball rolled over the street."),
                Section(name="test", content="The big ball rolled over the street."),
            ],
            data_formatter=PlainFormatter(),
            render_as="markdown",
        )
    )


@pytest.mark.slow
def test_parsing():
    assert ["The big ball ", "rolled over the street", "."] == SyntaxTreeMutator.get_phrases(
        "The big ball rolled over the street."
    )


@pytest.mark.asyncio
async def test_paraphrase():
    runner = MockedRunner(["1", "2"])
    mutator = Paraphrase(path_descriptor={"name": "test"})
    result = await mutator.mutate(basic_template(), MagicMock(), runner, n_mutations=2, random_state=42)
    assert len(result) == 2
    assert result[0].candidate.query({"name": "test", "_child": "content"}) == "1"
    assert result[1].candidate.query({"name": "test", "_child": "content"}) == "2"


@pytest.mark.asyncio
async def test_paraphrase_with_duplicates():
    runner = MockedRunner(["1", "2"])
    mutator = Paraphrase(path_descriptor={"name": "test"})
    result = await mutator.mutate(duplicate_template(), MagicMock(), runner, n_mutations=2, random_state=42)
    assert len(result) == 2
    assert result[0].candidate.query({"name": "test", "_child": "content"}, max_matches=None) == ["1", "1"]
    assert result[1].candidate.query({"name": "test", "_child": "content"}, max_matches=None) == ["2", "2"]


@pytest.mark.asyncio
async def test_to_bulletpoints():
    runner = MockedRunner(["1", "2"])
    mutator = SegmentToBulletPoints(path_descriptor={"name": "test"})
    result = await mutator.mutate(basic_template(), MagicMock(), runner, n_mutations=2, random_state=42)
    assert len(result) == 1
    assert result[0].candidate.query({"name": "test", "_child": "content"}) == "1"
    assert runner.prompt_log[0] == (
        "Rewrite the text below as a bullet list with at most 10 words per bullet "
        "point. \n"
        "\n"
        "The big ball rolled over the street."
    )


@pytest.mark.asyncio
async def test_induce():
    runner = MockedRunner(["1", "2"])
    datatable = DataTable.from_records([{"input": "_a_", "output": "_b_"}] * 5)
    mutator = InduceInstructions({"name": "test"}, datatable)
    result = await mutator.mutate(basic_template(), MagicMock(), runner, n_mutations=2, random_state=42)
    assert len(result) == 2
    assert result[0].candidate.query({"name": "test", "_child": "content"}) == "1"
    assert result[1].candidate.query({"name": "test", "_child": "content"}) == "2"
    assert "_a_" in runner.prompt_log[0]


@pytest.mark.asyncio
async def test_replace_param():
    runner = MockedRunner(["1", "2"])
    mutator = ReplaceParameter(r".*render_as", ["markdown", "xml"])
    result = await mutator.mutate(basic_template(), MagicMock(), runner, n_mutations=2, random_state=42)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_apo(n_data=5, num_gradients=2):
    runner = MockedRunner(
        ["".join([f"<START>Reason {i}</START>" for i in range(num_gradients)])]
        + [f"<START>Improved Prompt {i}</START>" for i in range(num_gradients)]
    )
    datatable = DataTable.from_records([{"input": "_a_", "output": "_b_"}] * n_data)
    mutator = APO({"name": "test", "_child": "content"}, None, num_gradients=2, steps_per_gradient=1, num_rewrites=1)
    mutator.objective = MagicMock()
    mutator.objective.return_value = MagicMock(mistakes=list(range(n_data)))
    result = await mutator.mutate(basic_template(), datatable, runner, n_mutations=2, random_state=42)
    assert result[0].candidate.query({"name": "test", "_child": "content"}) == "Improved Prompt 0"
    assert result[1].candidate.query({"name": "test", "_child": "content"}) == "<START>Improved Prompt 0</START>"
    assert len(result) == 2


@pytest.mark.asyncio
async def test_mutate():
    cache = {"The big ball rolled over the street.": ["The big ball ", "rolled over the street", "."]}
    mutator = SyntaxTreeMutator(starting_prompt=basic_template(), cache=cache, path_descriptor={"name": "test"})
    runner = MockedRunner("LLM response")
    result = await mutator.mutate(basic_template(), MagicMock(), runner, random_state=42)
    assert result[0].action == "del"
    result = await mutator.mutate(basic_template(), MagicMock(), runner, random_state=43)
    assert result[0].action == "par"
    assert result[0].candidate.query(".*content").strip() == "LLM response rolled over the street ."
    result = await mutator.mutate(basic_template(), MagicMock(), runner, random_state=46)
    assert result[0].action == "swap"
    assert result[0].candidate.query(".*content").strip() == "rolled over the street The big ball ."


def test_MultiStepRewrite_constructor_raises_exception_for_no_templates():
    invalid_template = 1
    with pytest.raises(ValueError):
        MultiStepRewrite("descriptor")


def test_MultiStepRewrite_constructor_raises_exception_for_invalid_template():
    invalid_template = 1
    with pytest.raises(TypeError):
        MultiStepRewrite("descriptor", invalid_template)


def test_MultiStepRewrite_constructor_resolves_str_templates():
    template1 = "1 {{{content}}}"
    template2 = "2 {{{content}}}"
    mutator = MultiStepRewrite("descriptor", template1, template2)
    assert len(mutator._templates) == 2
    assert mutator._templates[0]({"content": "test"}) == "1 test"
    assert mutator._templates[1]({"content": "test"}) == "2 test"


def test_MultiStepRewrite_constructor_resolves_path_templates(tmp_path):
    p1 = tmp_path / "template1.txt"
    p1.write_text("path 1 {{{content}}}")

    p2 = tmp_path / "template2.txt"
    p2.write_text("path 2 {{{content}}}")

    mutator = MultiStepRewrite("descriptor", p1, p2)
    assert len(mutator._templates) == 2
    assert mutator._templates[0]({"content": "test"}) == "path 1 test"
    assert mutator._templates[1]({"content": "test"}) == "path 2 test"


def test_MultiStepRewrite_constructor_resolves_file_templates(tmp_path):
    p1 = tmp_path / "template1.txt"
    p1.write_text("file 1 {{{content}}}")

    p2 = tmp_path / "template2.txt"
    p2.write_text("file 2 {{{content}}}")

    with p1.open("r") as f1:
        with p2.open("r") as f2:
            mutator = MultiStepRewrite("descriptor", f1, f2)
            assert len(mutator._templates) == 2
            assert mutator._templates[0]({"content": "test"}) == "file 1 test"
            assert mutator._templates[1]({"content": "test"}) == "file 2 test"


@pytest.mark.asyncio
async def test_MultiStepRewrite_basic_mutation():
    template1 = "Content: {{{content}}}\nParam1: {{{param1}}}"
    template2 = "{{{last_prompt}}}\n{{{last_result}}}\nParam2: {{{param2}}}"
    runner = MockedRunner(["m1result1", "m1result2", "m2result1", "m2result2"])
    mutator = MultiStepRewrite({"name": "test"}, template1, template2, param1="p1", param2="p2")
    result = await mutator.mutate(basic_template(), MagicMock(), runner, n_mutations=2, random_state=42)
    assert len(result) == 2
    assert result[0].candidate.query({"name": "test", "_child": "content"}) == "m1result2"
    assert runner.prompt_log[0] == ("Content: The big ball rolled over the street.\n" "Param1: p1")
    assert runner.prompt_log[1] == (
        "Content: The big ball rolled over the street.\n" "Param1: p1\n" "m1result1\n" "Param2: p2"
    )

    assert result[1].candidate.query({"name": "test", "_child": "content"}) == "m2result2"
    assert runner.prompt_log[2] == ("Content: The big ball rolled over the street.\n" "Param1: p1")
    assert runner.prompt_log[3] == (
        "Content: The big ball rolled over the street.\n" "Param1: p1\n" "m2result1\n" "Param2: p2"
    )


@pytest.mark.asyncio
async def test_MultiStepRewrite_duplicate_mutation():
    template1 = "Content: {{{content}}}\nParam1: {{{param1}}}"
    template2 = "{{{last_prompt}}}\n{{{last_result}}}\nParam2: {{{param2}}}"
    runner = MockedRunner(["m1result1", "m1result2"])
    mutator = MultiStepRewrite({"name": "test"}, template1, template2, param1="p1", param2="p2")
    result = await mutator.mutate(duplicate_template(), MagicMock(), runner, n_mutations=1, random_state=42)
    assert len(result) == 1
    assert result[0].candidate.query({"name": "test", "_child": "content"}, max_matches=None) == [
        "m1result2",
        "m1result2",
    ]


@pytest.mark.asyncio
async def test_MultiStepRewrite_with_helper():
    def _my_helper(this, x):
        return pybars.strlist(["_", x, "_"])

    helpers = {"my_helper": _my_helper}

    template = "Content: {{{content}}}\nParam1: {{{my_helper param1}}}"
    runner = MockedRunner(["result"])
    mutator = MultiStepRewrite({"name": "test"}, template, template_helpers=helpers, param1="p1")
    result = await mutator.mutate(basic_template(), MagicMock(), runner, n_mutations=1, random_state=42)
    assert len(result) == 1
    assert result[0].candidate.query({"name": "test", "_child": "content"}) == "result"
    assert runner.prompt_log[0] == ("Content: The big ball rolled over the street.\n" "Param1: _p1_")


@pytest.mark.asyncio
async def test_MultiStepRewrite_with_partial():
    partials = {"a": "Partial got {{{param1}}}"}

    template = "Content: {{{content}}}\nParam1: {{{param1}}}\n{{> a}}"
    runner = MockedRunner(["result"])
    mutator = MultiStepRewrite({"name": "test"}, template, template_partials=partials, param1="p1")
    result = await mutator.mutate(basic_template(), MagicMock(), runner, n_mutations=1, random_state=42)
    assert len(result) == 1
    assert result[0].candidate.query({"name": "test", "_child": "content"}) == "result"
    assert runner.prompt_log[0] == ("Content: The big ball rolled over the street.\n" "Param1: p1\n" "Partial got p1")
