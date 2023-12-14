:py:mod:`sammo.search`
======================

.. py:module:: sammo.search

.. autodoc2-docstring:: sammo.search
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Optimizer <sammo.search.Optimizer>`
     - .. autodoc2-docstring:: sammo.search.Optimizer
          :summary:
   * - :py:obj:`BeamSearch <sammo.search.BeamSearch>`
     - .. autodoc2-docstring:: sammo.search.BeamSearch
          :summary:
   * - :py:obj:`RegularizedEvolution <sammo.search.RegularizedEvolution>`
     - .. autodoc2-docstring:: sammo.search.RegularizedEvolution
          :summary:
   * - :py:obj:`SequentialSearch <sammo.search.SequentialSearch>`
     - .. autodoc2-docstring:: sammo.search.SequentialSearch
          :summary:
   * - :py:obj:`EnumerativeSearch <sammo.search.EnumerativeSearch>`
     - .. autodoc2-docstring:: sammo.search.EnumerativeSearch
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`logger <sammo.search.logger>`
     - .. autodoc2-docstring:: sammo.search.logger
          :summary:

API
~~~

.. py:data:: logger
   :canonical: sammo.search.logger
   :value: None

   .. autodoc2-docstring:: sammo.search.logger

.. py:class:: Optimizer(runner: sammo.base.Runner, search_space: collections.abc.Callable[[], sammo.components.Output] | None, objective: collections.abc.Callable[[sammo.data.DataTable, sammo.data.DataTable, bool], float], maximize: bool = False)
   :canonical: sammo.search.Optimizer

   .. autodoc2-docstring:: sammo.search.Optimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.search.Optimizer.__init__

   .. py:attribute:: REPORT_COLUMNS
      :canonical: sammo.search.Optimizer.REPORT_COLUMNS
      :value: ('objective', 'costs')

      .. autodoc2-docstring:: sammo.search.Optimizer.REPORT_COLUMNS

   .. py:method:: argbest(x, key='objective')
      :canonical: sammo.search.Optimizer.argbest

      .. autodoc2-docstring:: sammo.search.Optimizer.argbest

   .. py:method:: argsort(x, key='objective')
      :canonical: sammo.search.Optimizer.argsort

      .. autodoc2-docstring:: sammo.search.Optimizer.argsort

   .. py:method:: break_even(baseline_costs, weights=None)
      :canonical: sammo.search.Optimizer.break_even

      .. autodoc2-docstring:: sammo.search.Optimizer.break_even

   .. py:method:: fit(dataset: sammo.data.DataTable)
      :canonical: sammo.search.Optimizer.fit

      .. autodoc2-docstring:: sammo.search.Optimizer.fit

   .. py:method:: fit_transform(dataset: sammo.data.DataTable) -> sammo.data.DataTable
      :canonical: sammo.search.Optimizer.fit_transform

      .. autodoc2-docstring:: sammo.search.Optimizer.fit_transform

   .. py:method:: score(dataset: sammo.data.DataTable) -> dict
      :canonical: sammo.search.Optimizer.score

      .. autodoc2-docstring:: sammo.search.Optimizer.score

   .. py:method:: transform(dataset: sammo.data.DataTable) -> sammo.data.DataTable
      :canonical: sammo.search.Optimizer.transform

      .. autodoc2-docstring:: sammo.search.Optimizer.transform

   .. py:property:: best
      :canonical: sammo.search.Optimizer.best
      :type: dict

      .. autodoc2-docstring:: sammo.search.Optimizer.best

   .. py:property:: best_score
      :canonical: sammo.search.Optimizer.best_score

      .. autodoc2-docstring:: sammo.search.Optimizer.best_score

   .. py:property:: best_prompt
      :canonical: sammo.search.Optimizer.best_prompt
      :type: sammo.components.Output

      .. autodoc2-docstring:: sammo.search.Optimizer.best_prompt

   .. py:property:: fit_costs
      :canonical: sammo.search.Optimizer.fit_costs

      .. autodoc2-docstring:: sammo.search.Optimizer.fit_costs

   .. py:method:: save(fname: str | pathlib.Path | None = None, **extra_info)
      :canonical: sammo.search.Optimizer.save

      .. autodoc2-docstring:: sammo.search.Optimizer.save

   .. py:method:: show_report()
      :canonical: sammo.search.Optimizer.show_report

      .. autodoc2-docstring:: sammo.search.Optimizer.show_report

   .. py:method:: evaluate(candidates: list[sammo.components.Output], runner: sammo.base.Runner, objective: collections.abc.Callable[[sammo.data.DataTable, sammo.data.DataTable], sammo.base.EvaluationScore], dataset: sammo.data.DataTable, colbar: sammo.compactbars.CompactProgressBars | None = None) -> list[dict]
      :canonical: sammo.search.Optimizer.evaluate
      :async:

      .. autodoc2-docstring:: sammo.search.Optimizer.evaluate

   .. py:method:: validate(dataset: sammo.data.DataTable, k_best=5)
      :canonical: sammo.search.Optimizer.validate

      .. autodoc2-docstring:: sammo.search.Optimizer.validate

.. py:class:: BeamSearch(runner: sammo.base.Runner, mutator: sammo.mutators.Mutator, objective: collections.abc.Callable[[sammo.data.DataTable, sammo.data.DataTable, bool], float], maximize: bool = True, beam_width: int = 4, depth: int = 6, mutations_per_beam: int = 8, n_initial_candidates: int = 1, add_previous: bool = False, priors: typing.Literal[uniform] | dict = 'uniform')
   :canonical: sammo.search.BeamSearch

   Bases: :py:obj:`sammo.search.Optimizer`

   .. autodoc2-docstring:: sammo.search.BeamSearch

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.search.BeamSearch.__init__

   .. py:attribute:: REPORT_COLUMNS
      :canonical: sammo.search.BeamSearch.REPORT_COLUMNS
      :value: ('iteration', 'action', 'objective', 'costs', 'parse_errors', 'prev_actions')

      .. autodoc2-docstring:: sammo.search.BeamSearch.REPORT_COLUMNS

   .. py:method:: log(depth, items)
      :canonical: sammo.search.BeamSearch.log

      .. autodoc2-docstring:: sammo.search.BeamSearch.log

   .. py:method:: afit_transform(dataset: sammo.data.DataTable) -> sammo.data.DataTable
      :canonical: sammo.search.BeamSearch.afit_transform
      :async:

      .. autodoc2-docstring:: sammo.search.BeamSearch.afit_transform

.. py:class:: RegularizedEvolution(runner: sammo.base.Runner, mutator: sammo.mutators.Mutator, objective: collections.abc.Callable[[sammo.data.DataTable, sammo.data.DataTable, bool], float], maximize: bool = True, beam_width: int = 4, depth: int = 6, mutations_per_beam: int = 8, n_initial_candidates: int = 1, add_previous: bool = False, priors: typing.Literal[uniform] | dict = 'uniform')
   :canonical: sammo.search.RegularizedEvolution

   Bases: :py:obj:`sammo.search.BeamSearch`

   .. autodoc2-docstring:: sammo.search.RegularizedEvolution

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.search.RegularizedEvolution.__init__

.. py:class:: SequentialSearch(runner: sammo.base.Runner, mutator: sammo.mutators.Mutator, objective: collections.abc.Callable[[sammo.data.DataTable, sammo.data.DataTable, bool], float], maximize: bool = True, depth: int = 25)
   :canonical: sammo.search.SequentialSearch

   Bases: :py:obj:`sammo.search.BeamSearch`

   .. autodoc2-docstring:: sammo.search.SequentialSearch

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.search.SequentialSearch.__init__

.. py:class:: EnumerativeSearch(runner: sammo.base.Runner, search_space: collections.abc.Callable[[], sammo.components.Output], objective: collections.abc.Callable[[sammo.data.DataTable, sammo.data.DataTable], sammo.base.EvaluationScore], maximize: bool = True, algorithm: typing.Literal[grid, random] = 'grid', max_candidates: int | None = None, n_evals_parallel: int = 2)
   :canonical: sammo.search.EnumerativeSearch

   Bases: :py:obj:`sammo.search.Optimizer`

   .. autodoc2-docstring:: sammo.search.EnumerativeSearch

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.search.EnumerativeSearch.__init__

   .. py:attribute:: REPORT_COLUMNS
      :canonical: sammo.search.EnumerativeSearch.REPORT_COLUMNS
      :value: ('iteration', 'action', 'objective', 'costs', 'parse_errors')

      .. autodoc2-docstring:: sammo.search.EnumerativeSearch.REPORT_COLUMNS

   .. py:method:: afit_transform(dataset: sammo.data.DataTable) -> sammo.data.DataTable
      :canonical: sammo.search.EnumerativeSearch.afit_transform
      :async:

      .. autodoc2-docstring:: sammo.search.EnumerativeSearch.afit_transform
