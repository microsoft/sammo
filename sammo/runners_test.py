# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from aiohttp import ClientConnectorError
from aiohttp.client_reqrep import ConnectionKey
from quattro import TaskGroup
from unittest.mock import AsyncMock, MagicMock

import pytest

from sammo.runners import BaseRunner, OpenAIChat, OpenAIEmbedding, RetriableError, JsonSchema, AzureEmbedding
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
    coro.post.return_value.__aenter__.return_value.status = 200
    coro.post.return_value.__aenter__.return_value.json = AsyncMock(
        return_value={
            "usage": {"total_tokens": 1, "prompt_tokens": 2, "completion_tokens": 3},
            "choices": [{"message": {"content": "test"}}],
        }
    )
    mock.return_value.__aenter__.return_value = coro
    return mock


@pytest.fixture
def connector_error_in_post():
    session_mock = MagicMock()
    post_mock = MagicMock()
    post_mock.post.side_effect = ClientConnectorError(
        ConnectionKey("example.com", 123, False, False, None, None, None), OSError("mock error")
    )
    session_mock.return_value.__aenter__.return_value = post_mock
    return session_mock


@pytest.fixture
def basic_embedding():
    mock = MagicMock()
    coro = MagicMock()
    coro.post.return_value.__aenter__.return_value.status = 200
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
    runner = OpenAIEmbedding(model_id="some_id", api_config={"api_key": "test"}, cache=None)
    runner._get_session = basic_embedding
    result = await runner.generate_embedding(["text", "text2"])
    assert result.value == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_cached_embeddings(basic_embedding):
    cache = InMemoryDict()
    cache[("some_id", "text2")] = [0.3, 0.4]
    runner = OpenAIEmbedding(model_id="some_id", api_config={"api_key": "test"}, cache=cache)
    runner._get_session = basic_embedding
    result = await runner.generate_embedding(["text", "text2"])
    assert len(basic_embedding.mock_calls[2].kwargs["json"]["input"]) == 1
    assert result.value == [[0.1, 0.2], [0.3, 0.4]]
    print(cache._dict)

    # test that backend will not be called again if cached
    result = await runner.generate_embedding(["text", "text2"])
    assert basic_embedding.call_count == 1

    assert result.value == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_retry_connector_errors(connector_error_in_post):
    runner = OpenAIChat(model_id="gpt-4", api_config={"api_key": "test"}, rate_limit=10, cache=InMemoryDict())
    runner._get_session = connector_error_in_post
    with pytest.raises(RetriableError) as excinfo:
        await runner.generate_text(prompt="test prompt")
    assert "Client/server connection error" in str(excinfo)
    assert "example.com" in str(excinfo)


def test_schema_simple_dict():
    data = {"name": "John", "age": 30, "employed": True}
    expected = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "employed": {"type": "boolean"}},
        "required": ["name", "age", "employed"],
        "additionalProperties": False,
    }
    assert JsonSchema.guess_schema(data).schema == expected


def test_instantiate_azure():
    test = AzureEmbedding(
        model_id="dummy",
        api_config={"api_key": "test", "endpoint": "test", "deployment_id": "sth", "api_version": "2023-05-15"},
    )
    assert hasattr(test, "_embeddings_cache")


def test_schema_nested_dict():
    data = {"person": {"name": "Alice", "age": 25}}
    expected = {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False,
            }
        },
        "required": ["person"],
        "additionalProperties": False,
    }
    assert JsonSchema.guess_schema(data).schema == expected


def test_infer_schema_empty_dict():
    data = {}
    expected = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    assert JsonSchema.guess_schema(data).schema == expected


def test_infer_schema_list_of_integers():
    data = [1, 2, 3]
    expected = {"type": "array", "items": {"type": "integer"}}
    assert JsonSchema._guess_schema(data, top_level=False) == expected


def test_infer_schema_empty_list():
    data = []
    with pytest.raises(IndexError):
        JsonSchema._guess_schema(data, top_level=False)


def test_infer_schema_set_of_strings():
    data = {"apple", "banana", "cherry"}
    expected = {"type": "string", "enum": list(data)}
    assert JsonSchema._guess_schema(data, top_level=False) == expected


def test_infer_schema_mixed_type_set():
    data = {1, "two", 3.0}
    expected = {"type": ["integer", "number", "string"]}
    assert JsonSchema._guess_schema(data, top_level=False) == expected


def test_infer_schema_dict_with_description():
    data = {("name", "The user's name"): "Alice", ("age", "The user's age"): 30}
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The user's name"},
            "age": {"type": "integer", "description": "The user's age"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    assert JsonSchema.guess_schema(data).schema == expected


def test_infer_schema_top_level_type_error():
    data = "not a dict"
    with pytest.raises(TypeError):
        JsonSchema.guess_schema(data)
