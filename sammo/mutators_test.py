# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from unittest.mock import MagicMock

import pytest

from sammo.instructions import MetaPrompt, Section
from sammo.mutators import *
from sammo.runners import MockedRunner


def basic_template():
    return Output(MetaPrompt(Section(name="test", content="The big ball rolled over the street.")))


@pytest.mark.slow
def test_parsing():
    assert ["The big ball ", "rolled over the street", "."] == SyntaxTreeMutator.get_phrases(
        "The big ball rolled over the street."
    )


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
