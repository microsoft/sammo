:py:mod:`sammo.search_op`
=========================

.. py:module:: sammo.search_op

.. autodoc2-docstring:: sammo.search_op
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`one_of <sammo.search_op.one_of>`
     - .. autodoc2-docstring:: sammo.search_op.one_of
          :summary:
   * - :py:obj:`many_of <sammo.search_op.many_of>`
     - .. autodoc2-docstring:: sammo.search_op.many_of
          :summary:
   * - :py:obj:`permutate <sammo.search_op.permutate>`
     - .. autodoc2-docstring:: sammo.search_op.permutate
          :summary:
   * - :py:obj:`optional <sammo.search_op.optional>`
     - .. autodoc2-docstring:: sammo.search_op.optional
          :summary:

API
~~~

.. py:function:: one_of(candidates: typing.Iterable, name: str | None = None) -> typing.Any
   :canonical: sammo.search_op.one_of

   .. autodoc2-docstring:: sammo.search_op.one_of

.. py:function:: many_of(num_choices: int, candidates: typing.Iterable, name: str | None = None) -> typing.Any
   :canonical: sammo.search_op.many_of

   .. autodoc2-docstring:: sammo.search_op.many_of

.. py:function:: permutate(candidates: typing.Iterable, name: str | None = None) -> typing.Any
   :canonical: sammo.search_op.permutate

   .. autodoc2-docstring:: sammo.search_op.permutate

.. py:function:: optional(candidate, name=None) -> typing.Any
   :canonical: sammo.search_op.optional

   .. autodoc2-docstring:: sammo.search_op.optional
