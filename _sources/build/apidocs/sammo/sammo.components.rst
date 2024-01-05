:py:mod:`sammo.components`
==========================

.. py:module:: sammo.components

.. autodoc2-docstring:: sammo.components
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`GenerateText <sammo.components.GenerateText>`
     - .. autodoc2-docstring:: sammo.components.GenerateText
          :summary:
   * - :py:obj:`Output <sammo.components.Output>`
     - .. autodoc2-docstring:: sammo.components.Output
          :summary:
   * - :py:obj:`Union <sammo.components.Union>`
     - .. autodoc2-docstring:: sammo.components.Union
          :summary:
   * - :py:obj:`ForEach <sammo.components.ForEach>`
     - .. autodoc2-docstring:: sammo.components.ForEach
          :summary:

API
~~~

.. py:class:: GenerateText(child: sammo.base.ScalarComponent, name=None, system_prompt: str | None = None, history: sammo.base.ScalarComponent | None = None, seed=0, randomness: float = 0, max_tokens=None, on_error: typing.Literal[raise, empty_result] = 'empty_result')
   :canonical: sammo.components.GenerateText

   Bases: :py:obj:`sammo.base.ScalarComponent`

   .. autodoc2-docstring:: sammo.components.GenerateText

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.components.GenerateText.__init__

   .. py:attribute:: NEEDS_SCHEDULING
      :canonical: sammo.components.GenerateText.NEEDS_SCHEDULING
      :value: True

      .. autodoc2-docstring:: sammo.components.GenerateText.NEEDS_SCHEDULING

.. py:class:: Output(child: sammo.base.Component, minibatch_size=1, on_error: typing.Literal[raise, empty_result, backoff] = 'raise')
   :canonical: sammo.components.Output

   Bases: :py:obj:`sammo.base.Component`

   .. autodoc2-docstring:: sammo.components.Output

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.components.Output.__init__

   .. py:method:: run(runner: sammo.base.Runner, data: sammo.data.DataTable | list | None = None, progress_callback: typing.Callable | bool = True, priority: int = 0, on_error: typing.Literal[raise, empty_result, backoff] = 'raise') -> sammo.data.DataTable
      :canonical: sammo.components.Output.run

      .. autodoc2-docstring:: sammo.components.Output.run

   .. py:method:: n_minibatches(table: sammo.data.DataTable) -> int
      :canonical: sammo.components.Output.n_minibatches

      .. autodoc2-docstring:: sammo.components.Output.n_minibatches

   .. py:method:: arun(runner: sammo.base.Runner, data: sammo.data.DataTable | list | None = None, progress_callback: typing.Callable | bool = True, priority: int = 0, on_error: typing.Literal[raise, empty_result, backoff] = 'raise')
      :canonical: sammo.components.Output.arun
      :async:

      .. autodoc2-docstring:: sammo.components.Output.arun

.. py:class:: Union(*children: sammo.base.Component, name: str | None = None)
   :canonical: sammo.components.Union

   Bases: :py:obj:`sammo.base.ListComponent`

   .. autodoc2-docstring:: sammo.components.Union

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.components.Union.__init__

.. py:class:: ForEach(loop_variable: str, child: sammo.base.ListComponent, operator: sammo.base.Component, name: str | None = None)
   :canonical: sammo.components.ForEach

   Bases: :py:obj:`sammo.base.ListComponent`

   .. autodoc2-docstring:: sammo.components.ForEach

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.components.ForEach.__init__

   .. py:attribute:: NEEDS_SCHEDULING
      :canonical: sammo.components.ForEach.NEEDS_SCHEDULING
      :value: True

      .. autodoc2-docstring:: sammo.components.ForEach.NEEDS_SCHEDULING
