# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest

from sammo.base import Template
from sammo.components import Output, Union, ForEach, GenerateText
from sammo.runners import MockedRunner


@pytest.mark.asyncio
async def test_union():
    res = await Union("a", "b", "c")(None, dict())
    assert res.value == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_union_run():
    res = await Union("a", "b", "c").arun(None)
    assert [r.value for r in res] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_for_each():
    res = await ForEach("x", Union("a", "b", "c"), Template(".{{x}}"))(None, dict())
    assert res.value == [".a", ".b", ".c"]


@pytest.mark.asyncio
async def test_generate_text():
    runner = MockedRunner("Return value.")
    res = await GenerateText("This is a simple test.")(runner, dict())
    assert res.value == "Return value."


@pytest.mark.asyncio
async def test_override_runner():
    runner1 = MockedRunner("test1")
    runner2 = MockedRunner("test2")
    res1 = GenerateText("Get test1", runner=runner1)
    res2 = GenerateText(Template("I got {{res1}}", res1=res1))
    res = await res2(runner2, dict())
    assert runner2.prompt_log[0] == "I got test1"
    assert res.value == "test2"


@pytest.mark.asyncio
async def test_child_runner_not_overridden():
    runner1 = MockedRunner("test1")
    runner2 = MockedRunner("test2")
    res2 = GenerateText(Template("I got {{res1}}", res1=GenerateText("Get test1")), runner=runner2)
    res = await res2(runner1, dict())
    assert runner2.prompt_log[0] == "I got test1"
    assert res.value == "test2"
