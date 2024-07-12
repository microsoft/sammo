# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from unittest.mock import MagicMock

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
