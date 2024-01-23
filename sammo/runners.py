from abc import abstractmethod
import asyncio
from collections.abc import MutableMapping
import json
import logging
import os
import pathlib

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
class OpenAIBaseRunner(Runner):
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
    """

    ERRORS = (
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.APIError,
        openai.error.Timeout,
        openai.error.TryAgain,
        openai.error.ServiceUnavailableError,
        asyncio.TimeoutError,
    )

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
    ):
        super().__init__()

        if isinstance(api_config, dict):
            self._oai_config = dict(api_config)
        elif isinstance(api_config, pathlib.Path):
            with api_config.open() as api_config_file:
                self._oai_config = json.load(api_config_file)
        if isinstance(rate_limit, Throttler):
            self._throttler = rate_limit
        elif isinstance(rate_limit, AtMost):
            self._throttler = Throttler(limits=[rate_limit])
        else:
            if isinstance(rate_limit, int):
                rate_limit = [AtMost(rate_limit, "calls", period=1)]
            self._throttler = Throttler(limits=rate_limit)

        self._model_id = model_id
        if self._oai_config.get("api_type", "") == "azure":
            self._oai_config["deployment_id"] = self._model_id
        else:
            self._oai_config["model"] = self._model_id
        if equivalence_class == "major":
            self._equivalence_class = self.get_equivalence_class(self._model_id)
        elif equivalence_class == "exact":
            self._equivalence_class = self._model_id
        else:
            self._equivalence_class = equivalence_class

        if isinstance(cache, str) or isinstance(cache, os.PathLike):
            self._cache = PersistentDict(cache)
        else:
            self._cache = cache
        self._retry_on = self.ERRORS if retry_on == "default" else retry_on
        self._max_retries = max_retries
        self._semaphores = dict()
        self._timeout = timeout
        self._max_timeout_retries = max_timeout_retries
        self._max_context_window = max_context_window

    @classmethod
    def get_equivalence_class(cls, model_id: str) -> str:
        if model_id.startswith("gpt-3"):
            return "gpt-3"
        elif model_id.startswith("gpt-4"):
            return "gpt-4"
        else:
            return model_id

    async def _execute_request(self, request):
        fingerprint = request.fingerprint
        if fingerprint in self._semaphores:
            sem = self._semaphores[fingerprint]
        else:
            sem = asyncio.Semaphore(1)
            self._semaphores[fingerprint] = sem
        async with sem:
            # important: ensure that we do not run the same prompt concurrently
            if self._cache is not None and fingerprint in self._cache:
                return request.with_cached_result(json=self._cache[fingerprint])
            else:
                timeout_retries = 0
                for cur_try in range(self._max_retries):
                    retry_on = self._retry_on if cur_try < self._max_retries - 1 else tuple()

                    try:
                        job_handle = await self._throttler.wait_in_line(request.priority)
                        async with asyncio.timeout(self._timeout):
                            value = await request.with_result(retries=cur_try)
                        self._throttler.update_job_stats(job_handle, cost=value.costs.total)
                        if self._cache is not None:
                            self._cache[fingerprint] = value.json
                        return value
                    except TimeoutError:
                        timeout_retries += 1
                        self._throttler.update_job_stats(job_handle, failed=True, cost=0)
                        logger.error(f"TimeoutError: {request.params}")
                        if timeout_retries > self._max_timeout_retries:
                            raise TimeoutError
                        continue
                    except retry_on as e:
                        qualified_name = f"{type(e).__module__}.{type(e).__name__}".replace("builtins.", "")
                        self._throttler.update_job_stats(job_handle, failed=True, cost=0)
                        logger.error(f"{qualified_name}: {str(e).split(' Contact us')[0]}")
                        continue

                raise RuntimeError(f"Could not get completion for {request.params}")

    async def generate_text(
        self, prompt: str, max_tokens: int | None = None, randomness: float | None = 0, seed: int = 0, priority: int = 0
    ) -> dict:
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


class RawApiRequest:
    def __init__(self, params: dict, seed: int, model_id: str, priority: int = 0, extra_params: dict = None):
        self.params = params
        self.seed = seed
        self.model_id = model_id
        self.priority = priority
        self.json = None
        self.retries = None
        self.extra_params = extra_params or dict()

    def with_cached_result(self, json):
        self.json = json
        return self

    async def with_result(self, retries=None):
        self.json = await self._execute()
        self.retries = retries
        return self

    @property
    @abstractmethod
    def fingerprint_obj(self) -> dict:
        pass

    @property
    def fingerprint(self):
        return serialize_json(self.fingerprint_obj)

    @property
    @abstractmethod
    def costs(self) -> Costs:
        pass


class OpenAIChatRequest(RawApiRequest):
    @property
    def fingerprint_obj(self) -> dict:
        return {"seed": self.seed, "generative_model_id": self.model_id, **self.params}

    @property
    def costs(self) -> Costs:
        return Costs(
            input_costs=self.json["usage"].get("prompt_tokens", 0),
            output_costs=self.json["usage"].get("completion_tokens", 0),
        )

    async def _execute(self) -> dict:
        return (await openai.ChatCompletion.acreate(**self.params, **self.extra_params)).to_dict_recursive()


class OpenAIChat(OpenAIBaseRunner):
    """Provides simplified access to the (newer) chat-based OAI models"""

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
        """
        Use the Chat API to generate text.

        :param system_prompt: The prompt to use to prime the system response.
        :type system_prompt: str, optional
        """
        messages = []

        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                history = [x for x in history if x["role"] != "system"]
        if history is not None:
            messages = messages + history
        messages += [{"role": "user", "content": prompt}]
        result = await self._execute_request(
            OpenAIChatRequest(
                params=dict(
                    messages=messages, max_tokens=self._max_context_window or max_tokens, temperature=randomness
                ),
                extra_params=self._oai_config,
                model_id=self._equivalence_class,
                seed=seed,
                priority=priority,
            )
        )
        self._costs += result.costs
        prompt_logger.debug(
            f"\n\n\nAPI call (seed={seed}, randomness={randomness}):\n{prompt} \n->\n\n{result.json['choices'][0]['message']['content']}"
        )

        return LLMResult(
            result.json["choices"][0]["message"]["content"],
            costs=result.costs,
            history=messages + [result.json["choices"][0]["message"]],
            retries=result.retries,
            extra_data=result.json | {"prompt_id": result.fingerprint},
            fingerprint=result.fingerprint,
        )
