# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest

from sammo.base import Template
from sammo.components import Output, Union, ForEach, GenerateText
from sammo.runners import MockedRunner


@pytest.mark.asyncio
async def test_union():
    res = await Union("a", "b", "c")(None, dict())
    assert [r.value for r in res] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_for_each():
    res = await ForEach("x", Union("a", "b", "c"), Template(".{{x}}"))(None, dict())
    assert [r.value for r in res] == [".a", ".b", ".c"]


@pytest.mark.asyncio
async def test_generate_text():
    runner = MockedRunner("Return value.")
    res = await GenerateText("This is a simple test.")(runner, dict())
    assert res.value == "Return value."
