:py:mod:`sammo`
===============

.. py:module:: sammo

.. autodoc2-docstring:: sammo
   :allowtitles:

Submodules
----------

.. toctree::
   :titlesonly:
   :maxdepth: 1

   sammo.base
   sammo.compactbars
   sammo.components
   sammo.data
   sammo.dataformatters
   sammo.extractors
   sammo.instructions
   sammo.mutators
   sammo.runners
   sammo.scheduler
   sammo.search
   sammo.search_op
   sammo.store
   sammo.throttler
   sammo.utils

Package Contents
----------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`setup_logger <sammo.setup_logger>`
     - .. autodoc2-docstring:: sammo.setup_logger
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PROMPT_LOGGER_NAME <sammo.PROMPT_LOGGER_NAME>`
     - .. autodoc2-docstring:: sammo.PROMPT_LOGGER_NAME
          :summary:

API
~~~

.. py:data:: PROMPT_LOGGER_NAME
   :canonical: sammo.PROMPT_LOGGER_NAME
   :value: 'prompt_logger'

   .. autodoc2-docstring:: sammo.PROMPT_LOGGER_NAME

.. py:function:: setup_logger(default_level: int | str = 'DEBUG', log_prompts_to_file: bool = False, prompt_level: int | str = 'DEBUG', prompt_logfile_name: str = None) -> logging.Logger
   :canonical: sammo.setup_logger

   .. autodoc2-docstring:: sammo.setup_logger
