# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from asyncio import TaskGroup
from unittest.mock import AsyncMock, MagicMock

import pytest

from sammo.runners import BaseRunner, OpenAIChat, OpenAIEmbedding
from sammo.base import Costs
from sammo.store import InMemoryDict

"""Testing the Costs class"""


def test_costs_addition():
    c1 = Costs(input_costs=1, output_costs=2)
    c2 = Costs(input_costs=3, output_costs=4)
    result = c1 + c2
    assert result.input == 4
    assert result.output == 6


def test_costs_subtraction():
    c1 = Costs(input_costs=5, output_costs=6)
    c2 = Costs(input_costs=3, output_costs=4)
    result = c1 - c2
    assert result.input == 2
    assert result.output == 2


def test_costs_to_dict():
    c = Costs(input_costs=1, output_costs=2)
    result = c.to_dict()
    assert result == {"input": 1, "output": 2}


def test_costs_total():
    c = Costs(input_costs=1, output_costs=2)
    assert c.total == 3


@pytest.fixture
def basic():
    mock = MagicMock()
    coro = MagicMock()
    coro.post.return_value.__aenter__.return_value.json = AsyncMock(
        return_value={
            "usage": {"total_tokens": 1, "prompt_tokens": 2, "completion_tokens": 3},
            "choices": [{"message": {"content": "test"}}],
        }
    )
    mock.return_value.__aenter__.return_value = coro
    return mock


@pytest.fixture
def basic_embedding():
    mock = MagicMock()
    coro = MagicMock()
    coro.post.return_value.__aenter__.return_value.json = AsyncMock(
        return_value={
            "usage": {"total_tokens": 1, "prompt_tokens": 2, "completion_tokens": 3},
            "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}],
        }
    )
    mock.return_value.__aenter__.return_value = coro
    return mock


@pytest.mark.asyncio
async def test_generate_text(basic):
    runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, cache=None)
    runner._get_session = basic
    result = await runner.generate_text(prompt="test prompt")
    assert result.value == "test"


@pytest.mark.asyncio
async def test_parallel_identical_calls(basic):
    runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, rate_limit=10, cache=InMemoryDict())
    runner._get_session = basic
    async with TaskGroup() as g:
        for _ in range(10):
            g.create_task(runner.generate_text(prompt="test prompt", seed=0))
    # we expect the backend to be called only once, other values from cache
    assert basic.call_count == 1


@pytest.mark.asyncio
async def test_system_message(basic):
    runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, cache=None)
    runner._get_session = basic
    await runner.generate_text(prompt="test prompt", system_prompt="test system")
    assert basic.mock_calls[2].kwargs["json"]["messages"][0] == {"role": "system", "content": "test system"}


@pytest.mark.asyncio
async def test_cache(basic):
    cache = InMemoryDict()
    runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, cache=cache)
    runner._get_session = basic
    await runner.generate_text(prompt="test prompt")
    assert len(cache) == 1


@pytest.mark.asyncio
async def test_generate_embedding(basic_embedding):
    runner = OpenAIEmbedding(model_id="text-embedding-ada-002", api_config={"api_key": "test"}, cache=None)
    runner._get_session = basic_embedding
    result = await runner.generate_embedding("text")
    assert result.value == [[0.1, 0.2], [0.3, 0.4]]
