:py:mod:`sammo.base`
====================

.. py:module:: sammo.base

.. autodoc2-docstring:: sammo.base
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Costs <sammo.base.Costs>`
     - .. autodoc2-docstring:: sammo.base.Costs
          :summary:
   * - :py:obj:`Runner <sammo.base.Runner>`
     - .. autodoc2-docstring:: sammo.base.Runner
          :summary:
   * - :py:obj:`Result <sammo.base.Result>`
     - .. autodoc2-docstring:: sammo.base.Result
          :summary:
   * - :py:obj:`NonEmptyResult <sammo.base.NonEmptyResult>`
     - .. autodoc2-docstring:: sammo.base.NonEmptyResult
          :summary:
   * - :py:obj:`TextResult <sammo.base.TextResult>`
     - .. autodoc2-docstring:: sammo.base.TextResult
          :summary:
   * - :py:obj:`LLMResult <sammo.base.LLMResult>`
     - .. autodoc2-docstring:: sammo.base.LLMResult
          :summary:
   * - :py:obj:`ParseResult <sammo.base.ParseResult>`
     - .. autodoc2-docstring:: sammo.base.ParseResult
          :summary:
   * - :py:obj:`EmptyResult <sammo.base.EmptyResult>`
     - .. autodoc2-docstring:: sammo.base.EmptyResult
          :summary:
   * - :py:obj:`TimeoutResult <sammo.base.TimeoutResult>`
     - .. autodoc2-docstring:: sammo.base.TimeoutResult
          :summary:
   * - :py:obj:`EvaluationScore <sammo.base.EvaluationScore>`
     - .. autodoc2-docstring:: sammo.base.EvaluationScore
          :summary:
   * - :py:obj:`CompiledQuery <sammo.base.CompiledQuery>`
     - .. autodoc2-docstring:: sammo.base.CompiledQuery
          :summary:
   * - :py:obj:`Component <sammo.base.Component>`
     - .. autodoc2-docstring:: sammo.base.Component
          :summary:
   * - :py:obj:`ScalarComponent <sammo.base.ScalarComponent>`
     -
   * - :py:obj:`ListComponent <sammo.base.ListComponent>`
     -
   * - :py:obj:`StoreAs <sammo.base.StoreAs>`
     - .. autodoc2-docstring:: sammo.base.StoreAs
          :summary:
   * - :py:obj:`Template <sammo.base.Template>`
     - .. autodoc2-docstring:: sammo.base.Template
          :summary:
   * - :py:obj:`VerbatimText <sammo.base.VerbatimText>`
     -

API
~~~

.. py:class:: Costs(input_costs=0, output_costs=0)
   :canonical: sammo.base.Costs

   .. autodoc2-docstring:: sammo.base.Costs

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.Costs.__init__

   .. py:property:: total
      :canonical: sammo.base.Costs.total

      .. autodoc2-docstring:: sammo.base.Costs.total

   .. py:method:: to_dict()
      :canonical: sammo.base.Costs.to_dict

      .. autodoc2-docstring:: sammo.base.Costs.to_dict

.. py:class:: Runner()
   :canonical: sammo.base.Runner

   .. autodoc2-docstring:: sammo.base.Runner

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.Runner.__init__

   .. py:method:: reset_costs()
      :canonical: sammo.base.Runner.reset_costs

      .. autodoc2-docstring:: sammo.base.Runner.reset_costs

   .. py:property:: costs
      :canonical: sammo.base.Runner.costs

      .. autodoc2-docstring:: sammo.base.Runner.costs

.. py:class:: Result(value, parent=None, stored_values=None)
   :canonical: sammo.base.Result

   .. autodoc2-docstring:: sammo.base.Result

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.Result.__init__

   .. py:method:: to_json()
      :canonical: sammo.base.Result.to_json

      .. autodoc2-docstring:: sammo.base.Result.to_json

   .. py:method:: bfs(start, match_condition: typing.Callable)
      :canonical: sammo.base.Result.bfs
      :classmethod:

      .. autodoc2-docstring:: sammo.base.Result.bfs

   .. py:method:: with_parent(parent)
      :canonical: sammo.base.Result.with_parent

      .. autodoc2-docstring:: sammo.base.Result.with_parent

   .. py:method:: clone_with_stored_value(name, value)
      :canonical: sammo.base.Result.clone_with_stored_value

      .. autodoc2-docstring:: sammo.base.Result.clone_with_stored_value

   .. py:property:: parents
      :canonical: sammo.base.Result.parents

      .. autodoc2-docstring:: sammo.base.Result.parents

.. py:class:: NonEmptyResult(value, parent=None, stored_values=None)
   :canonical: sammo.base.NonEmptyResult

   Bases: :py:obj:`sammo.base.Result`

   .. autodoc2-docstring:: sammo.base.NonEmptyResult

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.NonEmptyResult.__init__

.. py:class:: TextResult(value, parent=None, stored_values=None)
   :canonical: sammo.base.TextResult

   Bases: :py:obj:`sammo.base.NonEmptyResult`

   .. autodoc2-docstring:: sammo.base.TextResult

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.TextResult.__init__

.. py:class:: LLMResult(value, parent=None, stored_values=None, extra_data=None, history=None, retries=0, costs=None, request_text=None, fingerprint=None)
   :canonical: sammo.base.LLMResult

   Bases: :py:obj:`sammo.base.NonEmptyResult`

   .. autodoc2-docstring:: sammo.base.LLMResult

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.LLMResult.__init__

   .. py:property:: costs
      :canonical: sammo.base.LLMResult.costs

      .. autodoc2-docstring:: sammo.base.LLMResult.costs

.. py:class:: ParseResult(value, parent=None, stored_values=None)
   :canonical: sammo.base.ParseResult

   Bases: :py:obj:`sammo.base.NonEmptyResult`

   .. autodoc2-docstring:: sammo.base.ParseResult

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.ParseResult.__init__

.. py:class:: EmptyResult(value=None, parent=None, stored_values=None)
   :canonical: sammo.base.EmptyResult

   Bases: :py:obj:`sammo.base.Result`

   .. autodoc2-docstring:: sammo.base.EmptyResult

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.EmptyResult.__init__

.. py:class:: TimeoutResult(value=None, parent=None, stored_values=None)
   :canonical: sammo.base.TimeoutResult

   Bases: :py:obj:`sammo.base.EmptyResult`

   .. autodoc2-docstring:: sammo.base.TimeoutResult

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.TimeoutResult.__init__

.. py:class:: EvaluationScore(score, mistakes=None, details=None)
   :canonical: sammo.base.EvaluationScore

   .. autodoc2-docstring:: sammo.base.EvaluationScore

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.EvaluationScore.__init__

   .. py:method:: to_dict(name_score='score')
      :canonical: sammo.base.EvaluationScore.to_dict

      .. autodoc2-docstring:: sammo.base.EvaluationScore.to_dict

.. py:class:: CompiledQuery(query, child_selector=None)
   :canonical: sammo.base.CompiledQuery

   .. autodoc2-docstring:: sammo.base.CompiledQuery

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.CompiledQuery.__init__

   .. py:method:: from_path(path_descriptor: str | dict | typing_extensions.Self)
      :canonical: sammo.base.CompiledQuery.from_path
      :classmethod:

      .. autodoc2-docstring:: sammo.base.CompiledQuery.from_path

.. py:class:: Component(child: typing_extensions.Self | str, name: str | None = None)
   :canonical: sammo.base.Component

   .. autodoc2-docstring:: sammo.base.Component

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.Component.__init__

   .. py:attribute:: NEEDS_SCHEDULING
      :canonical: sammo.base.Component.NEEDS_SCHEDULING
      :value: False

      .. autodoc2-docstring:: sammo.base.Component.NEEDS_SCHEDULING

   .. py:method:: query(regex_or_query=None, return_path=False, max_matches=1)
      :canonical: sammo.base.Component.query

      .. autodoc2-docstring:: sammo.base.Component.query

   .. py:method:: replace_static_text(regex_or_query: str | dict | sammo.base.CompiledQuery, new_text: str)
      :canonical: sammo.base.Component.replace_static_text

      .. autodoc2-docstring:: sammo.base.Component.replace_static_text

   .. py:property:: text
      :canonical: sammo.base.Component.text

      .. autodoc2-docstring:: sammo.base.Component.text

   .. py:method:: print_structure()
      :canonical: sammo.base.Component.print_structure

      .. autodoc2-docstring:: sammo.base.Component.print_structure

   .. py:method:: store_as(name: str)
      :canonical: sammo.base.Component.store_as

      .. autodoc2-docstring:: sammo.base.Component.store_as

.. py:class:: ScalarComponent(child: typing_extensions.Self | str, name: str | None = None)
   :canonical: sammo.base.ScalarComponent

   Bases: :py:obj:`sammo.base.Component`

.. py:class:: ListComponent(child: typing_extensions.Self | str, name: str | None = None)
   :canonical: sammo.base.ListComponent

   Bases: :py:obj:`sammo.base.Component`

.. py:class:: StoreAs
   :canonical: sammo.base.StoreAs

   .. autodoc2-docstring:: sammo.base.StoreAs

.. py:class:: Template(template_text: str, name: str | None = None, **dependencies: dict)
   :canonical: sammo.base.Template

   Bases: :py:obj:`sammo.base.ScalarComponent`

   .. autodoc2-docstring:: sammo.base.Template

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.base.Template.__init__

   .. py:property:: text
      :canonical: sammo.base.Template.text

      .. autodoc2-docstring:: sammo.base.Template.text

.. py:class:: VerbatimText(template: str, name: str | None = None)
   :canonical: sammo.base.VerbatimText

   Bases: :py:obj:`sammo.base.Template`
