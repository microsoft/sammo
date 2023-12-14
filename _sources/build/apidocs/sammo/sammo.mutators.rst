:py:mod:`sammo.mutators`
========================

.. py:module:: sammo.mutators

.. autodoc2-docstring:: sammo.mutators
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`MutatedCandidate <sammo.mutators.MutatedCandidate>`
     - .. autodoc2-docstring:: sammo.mutators.MutatedCandidate
          :summary:
   * - :py:obj:`Mutator <sammo.mutators.Mutator>`
     -
   * - :py:obj:`SyntaxTreeMutator <sammo.mutators.SyntaxTreeMutator>`
     -
   * - :py:obj:`PruneSyntaxTree <sammo.mutators.PruneSyntaxTree>`
     -
   * - :py:obj:`ShortenSegment <sammo.mutators.ShortenSegment>`
     -
   * - :py:obj:`APO <sammo.mutators.APO>`
     -
   * - :py:obj:`APE <sammo.mutators.APE>`
     -
   * - :py:obj:`InduceInstructions <sammo.mutators.InduceInstructions>`
     -
   * - :py:obj:`SegmentToBulletPoints <sammo.mutators.SegmentToBulletPoints>`
     -
   * - :py:obj:`Paraphrase <sammo.mutators.Paraphrase>`
     -
   * - :py:obj:`ParaphraseStatic <sammo.mutators.ParaphraseStatic>`
     -
   * - :py:obj:`RemoveStopWordsFromSegment <sammo.mutators.RemoveStopWordsFromSegment>`
     -
   * - :py:obj:`DropParameter <sammo.mutators.DropParameter>`
     -
   * - :py:obj:`RepeatSegment <sammo.mutators.RepeatSegment>`
     -
   * - :py:obj:`DropExamples <sammo.mutators.DropExamples>`
     -
   * - :py:obj:`DropIntro <sammo.mutators.DropIntro>`
     -
   * - :py:obj:`ReplaceParameter <sammo.mutators.ReplaceParameter>`
     -
   * - :py:obj:`ChangeDataFormat <sammo.mutators.ChangeDataFormat>`
     -
   * - :py:obj:`ChangeSectionsFormat <sammo.mutators.ChangeSectionsFormat>`
     -
   * - :py:obj:`DecreaseInContextExamples <sammo.mutators.DecreaseInContextExamples>`
     -
   * - :py:obj:`BagOfMutators <sammo.mutators.BagOfMutators>`
     -
   * - :py:obj:`StopwordsCompressor <sammo.mutators.StopwordsCompressor>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`logger <sammo.mutators.logger>`
     - .. autodoc2-docstring:: sammo.mutators.logger
          :summary:

API
~~~

.. py:data:: logger
   :canonical: sammo.mutators.logger
   :value: None

   .. autodoc2-docstring:: sammo.mutators.logger

.. py:class:: MutatedCandidate(action, candidate, **kwargs)
   :canonical: sammo.mutators.MutatedCandidate

   .. autodoc2-docstring:: sammo.mutators.MutatedCandidate

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.mutators.MutatedCandidate.__init__

   .. py:method:: with_parent(parent)
      :canonical: sammo.mutators.MutatedCandidate.with_parent

      .. autodoc2-docstring:: sammo.mutators.MutatedCandidate.with_parent

.. py:class:: Mutator(starting_prompt: sammo.components.Output | typing.Callable | None = None, seed: int = 42, sample_for_init_candidates: bool = True)
   :canonical: sammo.mutators.Mutator

   Bases: :py:obj:`abc.ABC`

   .. py:method:: applicable(candidate: sammo.components.Output) -> bool
      :canonical: sammo.mutators.Mutator.applicable

      .. autodoc2-docstring:: sammo.mutators.Mutator.applicable

   .. py:property:: priors
      :canonical: sammo.mutators.Mutator.priors

      .. autodoc2-docstring:: sammo.mutators.Mutator.priors

   .. py:property:: objective
      :canonical: sammo.mutators.Mutator.objective

      .. autodoc2-docstring:: sammo.mutators.Mutator.objective

   .. py:method:: update_priors(priors: dict[dict])
      :canonical: sammo.mutators.Mutator.update_priors

      .. autodoc2-docstring:: sammo.mutators.Mutator.update_priors

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.Mutator.mutate
      :abstractmethod:
      :async:

      .. autodoc2-docstring:: sammo.mutators.Mutator.mutate

   .. py:method:: get_initial_candidates(runner: sammo.base.Runner | None, n_initial_candidates: int) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.Mutator.get_initial_candidates
      :async:

      .. autodoc2-docstring:: sammo.mutators.Mutator.get_initial_candidates

.. py:class:: SyntaxTreeMutator(path_descriptor: str | dict, starting_prompt: sammo.components.Output | typing.Callable, cache: collections.abc.MutableMapping | None = None)
   :canonical: sammo.mutators.SyntaxTreeMutator

   Bases: :py:obj:`sammo.mutators.Mutator`

   .. py:attribute:: syntax_parser
      :canonical: sammo.mutators.SyntaxTreeMutator.syntax_parser
      :value: None

      .. autodoc2-docstring:: sammo.mutators.SyntaxTreeMutator.syntax_parser

   .. py:method:: parse_document(raw_doc)
      :canonical: sammo.mutators.SyntaxTreeMutator.parse_document
      :staticmethod:

      .. autodoc2-docstring:: sammo.mutators.SyntaxTreeMutator.parse_document

   .. py:method:: split_into_phrases(sent)
      :canonical: sammo.mutators.SyntaxTreeMutator.split_into_phrases
      :classmethod:

      .. autodoc2-docstring:: sammo.mutators.SyntaxTreeMutator.split_into_phrases

   .. py:method:: get_phrases(raw_doc)
      :canonical: sammo.mutators.SyntaxTreeMutator.get_phrases
      :classmethod:

      .. autodoc2-docstring:: sammo.mutators.SyntaxTreeMutator.get_phrases

   .. py:method:: applicable(candidate: sammo.components.Output)
      :canonical: sammo.mutators.SyntaxTreeMutator.applicable

   .. py:method:: replace_all_elements(l, needle, replacement)
      :canonical: sammo.mutators.SyntaxTreeMutator.replace_all_elements
      :staticmethod:

      .. autodoc2-docstring:: sammo.mutators.SyntaxTreeMutator.replace_all_elements

   .. py:method:: delete_all_elements(l, needle)
      :canonical: sammo.mutators.SyntaxTreeMutator.delete_all_elements
      :staticmethod:

      .. autodoc2-docstring:: sammo.mutators.SyntaxTreeMutator.delete_all_elements

   .. py:method:: swap_all_elements(lst, x, y)
      :canonical: sammo.mutators.SyntaxTreeMutator.swap_all_elements
      :staticmethod:

      .. autodoc2-docstring:: sammo.mutators.SyntaxTreeMutator.swap_all_elements

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.SyntaxTreeMutator.mutate
      :async:

.. py:class:: PruneSyntaxTree(path_descriptor: str | dict, starting_prompt: sammo.components.Output | typing.Callable, prune_metric: typing.Callable, cache: collections.abc.MutableMapping | None = None)
   :canonical: sammo.mutators.PruneSyntaxTree

   Bases: :py:obj:`sammo.mutators.SyntaxTreeMutator`

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.PruneSyntaxTree.mutate
      :async:

.. py:class:: ShortenSegment(path_descriptor: str | dict, reduction_factor: float = 0.5)
   :canonical: sammo.mutators.ShortenSegment

   Bases: :py:obj:`sammo.mutators.Mutator`

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.ShortenSegment.mutate
      :async:

   .. py:method:: applicable(candidate: sammo.components.Output)
      :canonical: sammo.mutators.ShortenSegment.applicable

.. py:class:: APO(path_descriptor: str | dict, starting_prompt: sammo.components.Output | typing.Callable | None, num_rewrites=2, num_sampled_errors=4, num_gradients=4, steps_per_gradient=2, minibatch_size=None, seed=42)
   :canonical: sammo.mutators.APO

   Bases: :py:obj:`sammo.mutators.Mutator`

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.APO.mutate
      :async:

.. py:class:: APE(path_descriptor: str | dict, starting_prompt: sammo.components.Output | typing.Callable, d_incontext: sammo.data.DataTable, n_incontext_subsamples: int | None = None)
   :canonical: sammo.mutators.APE

   Bases: :py:obj:`sammo.mutators.ShortenSegment`

   .. py:attribute:: RESAMPLE
      :canonical: sammo.mutators.APE.RESAMPLE
      :value: None

      .. autodoc2-docstring:: sammo.mutators.APE.RESAMPLE

   .. py:method:: get_initial_candidates(runner: sammo.base.Runner, n_initial_candidates: int) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.APE.get_initial_candidates
      :async:

      .. autodoc2-docstring:: sammo.mutators.APE.get_initial_candidates

.. py:class:: InduceInstructions(path_descriptor: str | dict, d_incontext: sammo.data.DataTable)
   :canonical: sammo.mutators.InduceInstructions

   Bases: :py:obj:`sammo.mutators.APE`

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.InduceInstructions.mutate
      :async:

.. py:class:: SegmentToBulletPoints(path_descriptor: str | dict, reduction_factor: float = 0.5)
   :canonical: sammo.mutators.SegmentToBulletPoints

   Bases: :py:obj:`sammo.mutators.ShortenSegment`

.. py:class:: Paraphrase(path_descriptor: str | dict, reduction_factor: float = 0.5)
   :canonical: sammo.mutators.Paraphrase

   Bases: :py:obj:`sammo.mutators.ShortenSegment`

.. py:class:: ParaphraseStatic(path_descriptor: str | dict, static_content: str)
   :canonical: sammo.mutators.ParaphraseStatic

   Bases: :py:obj:`sammo.mutators.ShortenSegment`

.. py:class:: RemoveStopWordsFromSegment(path_descriptor: str | dict, stopwords_compressors: list)
   :canonical: sammo.mutators.RemoveStopWordsFromSegment

   Bases: :py:obj:`sammo.mutators.ShortenSegment`

.. py:class:: DropParameter(path_descriptor: str | dict)
   :canonical: sammo.mutators.DropParameter

   Bases: :py:obj:`sammo.mutators.Mutator`

   .. py:method:: applicable(candidate: sammo.components.Output)
      :canonical: sammo.mutators.DropParameter.applicable

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.DropParameter.mutate
      :async:

.. py:class:: RepeatSegment(path_descriptor: str | dict, after: str | dict)
   :canonical: sammo.mutators.RepeatSegment

   Bases: :py:obj:`sammo.mutators.DropParameter`

   .. py:method:: applicable(candidate: sammo.components.Output)
      :canonical: sammo.mutators.RepeatSegment.applicable

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.RepeatSegment.mutate
      :async:

.. py:class:: DropExamples(path_descriptor: str | dict)
   :canonical: sammo.mutators.DropExamples

   Bases: :py:obj:`sammo.mutators.DropParameter`

.. py:class:: DropIntro(path_descriptor: str | dict)
   :canonical: sammo.mutators.DropIntro

   Bases: :py:obj:`sammo.mutators.DropParameter`

.. py:class:: ReplaceParameter(path_descriptor: str | dict, choices: list)
   :canonical: sammo.mutators.ReplaceParameter

   Bases: :py:obj:`sammo.mutators.DropParameter`

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.ReplaceParameter.mutate
      :async:

.. py:class:: ChangeDataFormat(path_descriptor: str | dict, choices: list)
   :canonical: sammo.mutators.ChangeDataFormat

   Bases: :py:obj:`sammo.mutators.ReplaceParameter`

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state: int = 42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.ChangeDataFormat.mutate
      :async:

.. py:class:: ChangeSectionsFormat(path_descriptor: str | dict, choices: list)
   :canonical: sammo.mutators.ChangeSectionsFormat

   Bases: :py:obj:`sammo.mutators.ReplaceParameter`

.. py:class:: DecreaseInContextExamples(d_incontext, reduction_factor=0.8, min_examples=1)
   :canonical: sammo.mutators.DecreaseInContextExamples

   Bases: :py:obj:`sammo.mutators.Mutator`

   .. py:method:: applicable(candidate: sammo.components.Output)
      :canonical: sammo.mutators.DecreaseInContextExamples.applicable

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state=42) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.DecreaseInContextExamples.mutate
      :async:

.. py:class:: BagOfMutators(starting_prompt: sammo.components.Output | typing.Callable, *bag, seed: int = 42, sample_for_init_candidates: bool = True)
   :canonical: sammo.mutators.BagOfMutators

   Bases: :py:obj:`sammo.mutators.Mutator`

   .. py:method:: applicable(candidate: sammo.components.Output)
      :canonical: sammo.mutators.BagOfMutators.applicable

   .. py:method:: draw_beta_bernoulli(n_samples: int, success_failure_pairs: list[tuple[int]], priors=(2, 6), seed=42)
      :canonical: sammo.mutators.BagOfMutators.draw_beta_bernoulli
      :staticmethod:

      .. autodoc2-docstring:: sammo.mutators.BagOfMutators.draw_beta_bernoulli

   .. py:method:: draw_map_beta_bernoulli(n_samples: int, success_failure_pairs: list[tuple[int]], priors=(5, 5), seed=None)
      :canonical: sammo.mutators.BagOfMutators.draw_map_beta_bernoulli
      :staticmethod:

      .. autodoc2-docstring:: sammo.mutators.BagOfMutators.draw_map_beta_bernoulli

   .. py:method:: mutate(candidate: sammo.components.Output, data: sammo.data.DataTable, runner: sammo.base.Runner, n_mutations: int = 1, random_state=None) -> list[sammo.mutators.MutatedCandidate]
      :canonical: sammo.mutators.BagOfMutators.mutate
      :async:

.. py:class:: StopwordsCompressor(filter_stopwords: typing.Literal[reuters, spacy, none], remove_punctuation=False, remove_whitespace=False)
   :canonical: sammo.mutators.StopwordsCompressor

   Bases: :py:obj:`sammo.base.Component`

   .. py:attribute:: REUTERS_STOPLIST
      :canonical: sammo.mutators.StopwordsCompressor.REUTERS_STOPLIST
      :value: ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'it', 'it', 'its...

      .. autodoc2-docstring:: sammo.mutators.StopwordsCompressor.REUTERS_STOPLIST

   .. py:method:: compress(x: str)
      :canonical: sammo.mutators.StopwordsCompressor.compress

      .. autodoc2-docstring:: sammo.mutators.StopwordsCompressor.compress
