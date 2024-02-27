# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
import time
from abc import abstractmethod
import asyncio
from collections.abc import MutableMapping
import json
import logging
import os
import pathlib
import httpx
import orjson
from beartype import beartype
from beartype.typing import Literal
import openai

from sammo import PROMPT_LOGGER_NAME
from sammo.base import LLMResult, Costs, Runner
from sammo.store import PersistentDict
from sammo.throttler import AtMost, Throttler
from sammo.utils import serialize_json

logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger(PROMPT_LOGGER_NAME)


class MockedRunner:
    def __init__(self, return_value=""):
        self.return_value = return_value

    @property
    def __class__(self) -> type:
        return Runner

    async def generate_text(self, prompt: str, *args, **kwargs):
        return LLMResult(self.return_value)


@beartype
class BaseRunner(Runner):
    """Base class for OpenAI API runners.

    :param model_id: Model specifier as listed in the API documentation.
    :param cache: A dict-like object to use for storing results.
    :param api_config: The path to the API config file or a dictionary containing the API information.
    :param rate_limit: The rate limit to use. If an integer, it specifies max calls per second.
    :param max_retries: The maximum number of retries to attempt.
    :param debug_mode: Enable debug mode where queries do not get issued.
    :param retry_on: Retry when any of these exceptions happen. Defaults to timeout and connection-based errors.
    :param timeout: The timeout (in s) to use for a query.
    :param max_context_window: The maximum number of tokens to use for the context window. Defaults to None, which
    means that the maximum context window is used.
    :param max_timeout_retries: The maximum number of retries to attempt when a timeout occurs.
    :param use_cached_timeouts: Whether to use cached timeouts.
    """

    RETRY_ERRORS = ()

    def __init__(
        self,
        model_id: str,
        api_config: dict | pathlib.Path,
        cache: None | MutableMapping | str | os.PathLike = None,
        equivalence_class: str | Literal["major", "exact"] = "major",
        rate_limit: AtMost | list[AtMost] | Throttler | int = 2,
        max_retries: int = 50,
        max_context_window: int | None = None,
        retry_on: tuple | str = "default",
        timeout: float | int = 60,
        max_timeout_retries: int = 1,
        use_cached_timeouts: bool = True,
    ):
        super().__init__()

        if isinstance(api_config, dict):
            self._api_config = dict(api_config)
        elif isinstance(api_config, pathlib.Path):
            with api_config.open() as api_config_file:
                self._api_config = json.load(api_config_file)
        if isinstance(rate_limit, Throttler):
            self._throttler = rate_limit
        elif isinstance(rate_limit, AtMost):
            self._throttler = Throttler(limits=[rate_limit])
        else:
            if isinstance(rate_limit, int):
                rate_limit = [AtMost(rate_limit, "calls", period=1)]
            self._throttler = Throttler(limits=rate_limit)

        self._model_id = model_id

        if equivalence_class == "major":
            self._equivalence_class = self._get_equivalence_class(self._model_id)
        elif equivalence_class == "exact":
            self._equivalence_class = self._model_id
        else:
            self._equivalence_class = equivalence_class

        if isinstance(cache, str) or isinstance(cache, os.PathLike):
            self._cache = PersistentDict(cache)
        else:
            self._cache = cache
        self._retry_on = self.RETRY_ERRORS if retry_on == "default" else retry_on
        self._max_retries = max_retries
        self._semaphores = dict()
        self._timeout = timeout
        self._max_timeout_retries = max_timeout_retries
        self._max_context_window = max_context_window
        self._use_cached_timeouts = use_cached_timeouts
        self._post_init()

    async def _execute_request(self, request, fingerprint, priority=0):
        if fingerprint in self._semaphores:
            sem = self._semaphores[fingerprint]
        else:
            sem = asyncio.Semaphore(1)
            self._semaphores[fingerprint] = sem
        async with sem:
            # important: ensure that we do not run the same prompt concurrently
            if self._cache is not None and fingerprint in self._cache:
                record = self._cache[fingerprint]
                if self._use_cached_timeouts and isinstance(record, dict) and "sammo.error.timeout" in record:
                    # re-raise the timeout error if the timeout is the same or higher
                    if self._timeout <= record["sammo.error.timeout"]["timeout"]:
                        raise TimeoutError("Cached timeout")
                else:
                    json = self._cache[fingerprint]
                    response_obj = self._to_llm_result(request, json, fingerprint)
                    self._costs += response_obj.costs
                    return response_obj

            n_timeouts = 0
            for cur_try in range(self._max_retries):
                retry_on = self._retry_on if cur_try < self._max_retries - 1 else tuple()

                try:
                    job_handle = await self._throttler.wait_in_line(priority)
                    async with asyncio.timeout(self._timeout):
                        json = await self._call_backend(request)
                    response_obj = self._llm_result(request, json, fingerprint)
                    response_obj.retries = cur_try
                    self._throttler.update_job_stats(job_handle, cost=response_obj.costs.total)
                    self._costs += response_obj.costs
                    if self._cache is not None:
                        self._cache[fingerprint] = json
                    return response_obj
                except TimeoutError:
                    n_timeouts += 1
                    self._throttler.update_job_stats(job_handle, failed=True, cost=0)
                    logger.error(f"TimeoutError: {request}")
                    if n_timeouts > self._max_timeout_retries:
                        self._cache[fingerprint] = {
                            "sammo.error.timeout": {"retries": cur_try, "timeout": self._timeout}
                        }
                        raise TimeoutError
                    continue
                except retry_on as e:
                    qualified_name = f"{type(e).__module__}.{type(e).__name__}".replace("builtins.", "")
                    self._throttler.update_job_stats(job_handle, failed=True, cost=0)
                    logger.error(f"{qualified_name}: {str(e).split(' Contact us')[0]}")
                    continue

            raise RuntimeError(f"Could not get completion for {request.params}")

    def _llm_result(self, request: dict, json_data: dict, fingerprint: str | bytes) -> LLMResult:
        result = self._to_llm_result(request, json_data, fingerprint)
        result.fingerprint = fingerprint
        result.extra_data = json_data
        return result

    @abstractmethod
    def _to_llm_result(self, request: dict, json_data: dict, fingerprint: str | bytes) -> LLMResult:
        pass

    @abstractmethod
    async def _call_backend(self, request: dict) -> dict:
        pass

    @classmethod
    def _get_equivalence_class(cls, model_id: str) -> str:
        return model_id

    def _post_init(self):
        pass


class OpenAIBaseRunner(BaseRunner):
    RETRY_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)

    @classmethod
    def _get_equivalence_class(cls, model_id: str) -> str:
        if model_id.startswith("gpt-3"):
            return "gpt-3"
        elif model_id.startswith("gpt-4"):
            return "gpt-4"
        else:
            return model_id

    def _post_init(self):
        if self._api_config.get("api_type", "") == "azure":
            self._api_config["deployment_id"] = self._model_id
        else:
            self._api_config["model"] = self._model_id

        self._client = openai.AsyncOpenAI(api_key=self._api_config["api_key"])


class OpenAIChat(OpenAIBaseRunner):
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        randomness: float | None = 0,
        seed: int = 0,
        priority: int = 0,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> LLMResult:
        """Calls the chat endpoint of the OAI model.

        Args:
            prompt: The user prompt.
            max_tokens: The maximum number of tokens to generate. If not set, corresponds to maximum
            available tokens.
            randomness: The randomness to use when generating tokens.
            seed: When using randomness, use this seed for local reproducibility (achieved by caching).
            priority: The priority of the request (used for throttling).

        Returns:
            Dictionary with keys "data" (the generated text), "cost" (the number of tokens used),
            and "retries" (the number of retries).
        """

        messages = []
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                history = [x for x in history if x["role"] != "system"]
        if history is not None:
            messages = messages + history
        messages += [{"role": "user", "content": prompt}]
        request = dict(messages=messages, max_tokens=self._max_context_window or max_tokens, temperature=randomness)
        fingerprint = serialize_json({"seed": seed, "generative_model_id": self._equivalence_class, **request})

        return await self._execute_request(request, fingerprint, priority)

    def _to_llm_result(self, request: dict, json_data: dict, fingerprint: str | bytes) -> LLMResult:
        prompt_logger.debug(f"\n\n\nAPI call:\n->\n\n{json_data['choices'][0]['message']['content']}")
        return LLMResult(
            json_data["choices"][0]["message"]["content"],
            history=request["messages"] + [json_data["choices"][0]["message"]],
            costs=self._extract_costs(json_data),
        )

    @staticmethod
    def _extract_costs(json_data: dict) -> dict:
        return Costs(
            input_costs=json_data["usage"].get("prompt_tokens", 0),
            output_costs=json_data["usage"].get("completion_tokens", 0),
        )

    async def _call_backend(self, request: dict) -> dict:
        return (await self._client.chat.completions.create(**request, model=self._model_id)).model_dump()


class OpenAIEmbedding(OpenAIChat):
    async def generate_embedding(self, text: str | list[str], priority: int = 0) -> LLMResult:
        if isinstance(text, list) and len(text) > 2048:
            raise ValueError("Batch size must be below 2048.")
        fingerprint = serialize_json({"embedding_model_id": self._equivalence_class, "input": text})
        request = dict(input=text)
        return await self._execute_request(request, fingerprint, priority)

    def _to_llm_result(self, request: dict, json_data: dict, fingerprint: str | bytes):
        return LLMResult([x["embedding"] for x in json_data["data"]], costs=self._extract_costs(json_data))

    async def _call_backend(self, request: dict) -> dict:
        return (await self._client.embeddings.create(**request, model=self._model_id)).model_dump()


class DeepInfraEmbedding(BaseRunner):
    BASE_URL = r"https://api.deepinfra.com/v1/inference/"

    def _post_init(self):
        self._client = httpx.AsyncClient()

    def __del__(self):
        # somewhat hacky way to close the client
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        if loop.is_running():
            loop.create_task(self._client.aclose())
        else:
            loop.run_until_complete(self._client.aclose())

    async def generate_embedding(self, text: str | list[str], priority: int = 0) -> LLMResult:
        if isinstance(text, list) and len(text) > 2048:
            raise ValueError("Batch size must be below 2048.")
        elif not isinstance(text, list):
            text = [text]
        fingerprint = serialize_json({"embedding_model_id": self._equivalence_class, "inputs": text})
        request = dict(inputs=text)
        return await self._execute_request(request, fingerprint, priority)

    async def _call_backend(self, request: dict) -> dict:
        response = await self._client.post(
            self.BASE_URL + self._model_id,
            headers=dict(Authorization=f"Bearer {self._api_config['api_key']}"),
            data=orjson.dumps(request),
        )
        return response.raise_for_status().json()

    def _to_llm_result(self, request: dict, json_data: dict, fingerprint: str | bytes):
        return LLMResult(json_data["embeddings"], costs=Costs(json_data["input_tokens"]))
