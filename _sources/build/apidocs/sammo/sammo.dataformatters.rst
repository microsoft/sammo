:py:mod:`sammo.dataformatters`
==============================

.. py:module:: sammo.dataformatters

.. autodoc2-docstring:: sammo.dataformatters
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DataFormatter <sammo.dataformatters.DataFormatter>`
     -
   * - :py:obj:`JSONDataFormatter <sammo.dataformatters.JSONDataFormatter>`
     -
   * - :py:obj:`XMLDataFormatter <sammo.dataformatters.XMLDataFormatter>`
     -
   * - :py:obj:`MultiLabelFormatter <sammo.dataformatters.MultiLabelFormatter>`
     -
   * - :py:obj:`QuestionAnswerFormatter <sammo.dataformatters.QuestionAnswerFormatter>`
     -

API
~~~

.. py:class:: DataFormatter(names: dict | None = None, flatten_1d_dicts: bool = True, include_ids: bool = True, orient: typing.Literal[item, kind] = 'item', all_labels=None)
   :canonical: sammo.dataformatters.DataFormatter

   Bases: :py:obj:`sammo.base.Component`

   .. py:attribute:: DEFAULT_NAMES
      :canonical: sammo.dataformatters.DataFormatter.DEFAULT_NAMES
      :value: None

      .. autodoc2-docstring:: sammo.dataformatters.DataFormatter.DEFAULT_NAMES

   .. py:method:: format_datatable(data, offset: int = 0)
      :canonical: sammo.dataformatters.DataFormatter.format_datatable

      .. autodoc2-docstring:: sammo.dataformatters.DataFormatter.format_datatable

   .. py:method:: format_single(attributes: dict = None, gold_label: dict = None, predicted_label: dict = None, x_id: int = 0) -> str
      :canonical: sammo.dataformatters.DataFormatter.format_single

      .. autodoc2-docstring:: sammo.dataformatters.DataFormatter.format_single

   .. py:method:: format_batch(attributes: list[dict], gold_label: list[dict] = None, predicted_label: list[dict] = None, offset: int = 0)
      :canonical: sammo.dataformatters.DataFormatter.format_batch

      .. autodoc2-docstring:: sammo.dataformatters.DataFormatter.format_batch

.. py:class:: JSONDataFormatter(newline_delimited=False, indent=None, **kwargs)
   :canonical: sammo.dataformatters.JSONDataFormatter

   Bases: :py:obj:`sammo.dataformatters.DataFormatter`

   .. py:method:: get_extractor(child, on_error='raise')
      :canonical: sammo.dataformatters.JSONDataFormatter.get_extractor

      .. autodoc2-docstring:: sammo.dataformatters.JSONDataFormatter.get_extractor

.. py:class:: XMLDataFormatter(names: dict | None = None, flatten_1d_dicts: bool = True, include_ids: bool = True, orient: typing.Literal[item, kind] = 'item', all_labels=None)
   :canonical: sammo.dataformatters.XMLDataFormatter

   Bases: :py:obj:`sammo.dataformatters.DataFormatter`

   .. py:method:: get_extractor(child, on_error='raise')
      :canonical: sammo.dataformatters.XMLDataFormatter.get_extractor

      .. autodoc2-docstring:: sammo.dataformatters.XMLDataFormatter.get_extractor

.. py:class:: MultiLabelFormatter(all_labels: list, **kwargs)
   :canonical: sammo.dataformatters.MultiLabelFormatter

   Bases: :py:obj:`sammo.dataformatters.DataFormatter`

   .. py:method:: get_extractor(child, on_error='raise')
      :canonical: sammo.dataformatters.MultiLabelFormatter.get_extractor

      .. autodoc2-docstring:: sammo.dataformatters.MultiLabelFormatter.get_extractor

.. py:class:: QuestionAnswerFormatter(all_labels: list, **kwargs)
   :canonical: sammo.dataformatters.QuestionAnswerFormatter

   Bases: :py:obj:`sammo.dataformatters.MultiLabelFormatter`

   .. py:method:: get_extractor(child, on_error='raise')
      :canonical: sammo.dataformatters.QuestionAnswerFormatter.get_extractor

      .. autodoc2-docstring:: sammo.dataformatters.QuestionAnswerFormatter.get_extractor
