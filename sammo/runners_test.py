# Testing the Costs class
from asyncio import TaskGroup
from collections.abc import MutableMapping
from unittest.mock import AsyncMock, patch, Mock, MagicMock

import pytest

from sammo.runners import OpenAIBaseRunner, OpenAIChat
from sammo.base import Costs
from sammo.store import InMemoryDict


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
    return AsyncMock(
        return_value=Mock(
            **{
                "to_dict_recursive.return_value": {
                    "usage": {"total_tokens": 1, "prompt_tokens": 2, "completion_tokens": 3},
                    "choices": [{"message": {"content": "test"}}],
                }
            }
        )
    )


@pytest.mark.asyncio
async def test_chatgpt_generate_text(basic):
    with patch("sammo.runners.openai.ChatCompletion.acreate", basic) as mock_api:
        runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, cache=None)
        result = await runner.generate_text(prompt="test prompt")
        assert result.value == "test"


@pytest.mark.asyncio
async def test_parallel_identical_calls(basic):
    with patch("sammo.runners.openai.ChatCompletion.acreate", basic) as mock_api:
        runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, rate_limit=10, cache=InMemoryDict())
        async with TaskGroup() as g:
            for _ in range(10):
                g.create_task(runner.generate_text(prompt="test prompt", seed=0))
        # we expect the backend to be called only once, other values from cache
        assert mock_api.call_count == 1


@pytest.mark.asyncio
async def test_chatgpt_system_message(basic):
    with patch("sammo.runners.openai.ChatCompletion.acreate", basic) as mock_api:
        runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, cache=None)
        result = await runner.generate_text(prompt="test prompt", system_prompt="test system")
        assert mock_api.call_args[-1]["messages"][0] == {"role": "system", "content": "test system"}


@pytest.mark.asyncio
async def test_chatgpt_cache(basic):
    with patch("sammo.runners.openai.ChatCompletion.acreate", basic) as mock_api:
        cache = InMemoryDict()
        runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, cache=cache)
        await runner.generate_text(prompt="test prompt")
        assert len(cache) == 1
