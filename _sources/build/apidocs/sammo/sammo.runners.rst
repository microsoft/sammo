:py:mod:`sammo.runners`
=======================

.. py:module:: sammo.runners

.. autodoc2-docstring:: sammo.runners
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`MockedRunner <sammo.runners.MockedRunner>`
     - .. autodoc2-docstring:: sammo.runners.MockedRunner
          :summary:
   * - :py:obj:`OpenAIBaseRunner <sammo.runners.OpenAIBaseRunner>`
     - .. autodoc2-docstring:: sammo.runners.OpenAIBaseRunner
          :summary:
   * - :py:obj:`RawApiRequest <sammo.runners.RawApiRequest>`
     - .. autodoc2-docstring:: sammo.runners.RawApiRequest
          :summary:
   * - :py:obj:`OpenAIChatRequest <sammo.runners.OpenAIChatRequest>`
     - .. autodoc2-docstring:: sammo.runners.OpenAIChatRequest
          :summary:
   * - :py:obj:`OpenAIChat <sammo.runners.OpenAIChat>`
     - .. autodoc2-docstring:: sammo.runners.OpenAIChat
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`logger <sammo.runners.logger>`
     - .. autodoc2-docstring:: sammo.runners.logger
          :summary:
   * - :py:obj:`prompt_logger <sammo.runners.prompt_logger>`
     - .. autodoc2-docstring:: sammo.runners.prompt_logger
          :summary:

API
~~~

.. py:data:: logger
   :canonical: sammo.runners.logger
   :value: None

   .. autodoc2-docstring:: sammo.runners.logger

.. py:data:: prompt_logger
   :canonical: sammo.runners.prompt_logger
   :value: None

   .. autodoc2-docstring:: sammo.runners.prompt_logger

.. py:class:: MockedRunner(return_value='')
   :canonical: sammo.runners.MockedRunner

   .. autodoc2-docstring:: sammo.runners.MockedRunner

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.runners.MockedRunner.__init__

   .. py:method:: generate_text(prompt: str, *args, **kwargs)
      :canonical: sammo.runners.MockedRunner.generate_text
      :async:

      .. autodoc2-docstring:: sammo.runners.MockedRunner.generate_text

.. py:class:: OpenAIBaseRunner(model_id: str, api_config: dict | pathlib.Path, cache: None | collections.abc.MutableMapping | str | os.PathLike = None, equivalence_class: str | typing.Literal[major, exact] = 'major', rate_limit: sammo.throttler.AtMost | list[sammo.throttler.AtMost] | sammo.throttler.Throttler | int = 2, max_retries: int = 50, max_context_window: int | None = None, retry_on: tuple | str = 'default', timeout: float | int = 60, max_timeout_retries: int = 1)
   :canonical: sammo.runners.OpenAIBaseRunner

   Bases: :py:obj:`sammo.base.Runner`

   .. autodoc2-docstring:: sammo.runners.OpenAIBaseRunner

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.runners.OpenAIBaseRunner.__init__

   .. py:attribute:: ERRORS
      :canonical: sammo.runners.OpenAIBaseRunner.ERRORS
      :value: ()

      .. autodoc2-docstring:: sammo.runners.OpenAIBaseRunner.ERRORS

   .. py:method:: get_equivalence_class(model_id: str) -> str
      :canonical: sammo.runners.OpenAIBaseRunner.get_equivalence_class
      :classmethod:

      .. autodoc2-docstring:: sammo.runners.OpenAIBaseRunner.get_equivalence_class

   .. py:method:: generate_text(prompt: str, max_tokens: int | None = None, randomness: float | None = 0, seed: int = 0, priority: int = 0) -> dict
      :canonical: sammo.runners.OpenAIBaseRunner.generate_text
      :async:

      .. autodoc2-docstring:: sammo.runners.OpenAIBaseRunner.generate_text

.. py:class:: RawApiRequest(params: dict, seed: int, model_id: str, priority: int = 0, extra_params: dict = None)
   :canonical: sammo.runners.RawApiRequest

   .. autodoc2-docstring:: sammo.runners.RawApiRequest

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.runners.RawApiRequest.__init__

   .. py:method:: with_cached_result(json)
      :canonical: sammo.runners.RawApiRequest.with_cached_result

      .. autodoc2-docstring:: sammo.runners.RawApiRequest.with_cached_result

   .. py:method:: with_result(retries=None)
      :canonical: sammo.runners.RawApiRequest.with_result
      :async:

      .. autodoc2-docstring:: sammo.runners.RawApiRequest.with_result

   .. py:property:: fingerprint_obj
      :canonical: sammo.runners.RawApiRequest.fingerprint_obj
      :abstractmethod:
      :type: dict

      .. autodoc2-docstring:: sammo.runners.RawApiRequest.fingerprint_obj

   .. py:property:: fingerprint
      :canonical: sammo.runners.RawApiRequest.fingerprint

      .. autodoc2-docstring:: sammo.runners.RawApiRequest.fingerprint

   .. py:property:: costs
      :canonical: sammo.runners.RawApiRequest.costs
      :abstractmethod:
      :type: sammo.base.Costs

      .. autodoc2-docstring:: sammo.runners.RawApiRequest.costs

.. py:class:: OpenAIChatRequest(params: dict, seed: int, model_id: str, priority: int = 0, extra_params: dict = None)
   :canonical: sammo.runners.OpenAIChatRequest

   Bases: :py:obj:`sammo.runners.RawApiRequest`

   .. autodoc2-docstring:: sammo.runners.OpenAIChatRequest

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.runners.OpenAIChatRequest.__init__

   .. py:property:: fingerprint_obj
      :canonical: sammo.runners.OpenAIChatRequest.fingerprint_obj
      :type: dict

      .. autodoc2-docstring:: sammo.runners.OpenAIChatRequest.fingerprint_obj

   .. py:property:: costs
      :canonical: sammo.runners.OpenAIChatRequest.costs
      :type: sammo.base.Costs

      .. autodoc2-docstring:: sammo.runners.OpenAIChatRequest.costs

.. py:class:: OpenAIChat(model_id: str, api_config: dict | pathlib.Path, cache: None | collections.abc.MutableMapping | str | os.PathLike = None, equivalence_class: str | typing.Literal[major, exact] = 'major', rate_limit: sammo.throttler.AtMost | list[sammo.throttler.AtMost] | sammo.throttler.Throttler | int = 2, max_retries: int = 50, max_context_window: int | None = None, retry_on: tuple | str = 'default', timeout: float | int = 60, max_timeout_retries: int = 1)
   :canonical: sammo.runners.OpenAIChat

   Bases: :py:obj:`sammo.runners.OpenAIBaseRunner`

   .. autodoc2-docstring:: sammo.runners.OpenAIChat

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.runners.OpenAIChat.__init__

   .. py:method:: generate_text(prompt: str, max_tokens: int | None = None, randomness: float | None = 0, seed: int = 0, priority: int = 0, system_prompt: str | None = None, history: list[dict] | None = None) -> sammo.base.LLMResult
      :canonical: sammo.runners.OpenAIChat.generate_text
      :async:

      .. autodoc2-docstring:: sammo.runners.OpenAIChat.generate_text
