:py:mod:`sammo.instructions`
============================

.. py:module:: sammo.instructions

.. autodoc2-docstring:: sammo.instructions
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Section <sammo.instructions.Section>`
     -
   * - :py:obj:`Paragraph <sammo.instructions.Paragraph>`
     -
   * - :py:obj:`MetaPrompt <sammo.instructions.MetaPrompt>`
     -
   * - :py:obj:`FewshotExamples <sammo.instructions.FewshotExamples>`
     -
   * - :py:obj:`InputData <sammo.instructions.InputData>`
     -

API
~~~

.. py:class:: Section(name, content, id=None)
   :canonical: sammo.instructions.Section

   Bases: :py:obj:`sammo.base.Component`

   .. py:method:: static_text(sep='\n')
      :canonical: sammo.instructions.Section.static_text

      .. autodoc2-docstring:: sammo.instructions.Section.static_text

   .. py:method:: set_static_text(text)
      :canonical: sammo.instructions.Section.set_static_text

      .. autodoc2-docstring:: sammo.instructions.Section.set_static_text

.. py:class:: Paragraph(content, id=None)
   :canonical: sammo.instructions.Paragraph

   Bases: :py:obj:`sammo.instructions.Section`

.. py:class:: MetaPrompt(structure: list[sammo.instructions.Paragraph | sammo.instructions.Section], render_as: typing.Literal[raw, json, xml, markdown, markdown-alt] = 'markdown', data_formatter: sammo.dataformatters.DataFormatter | None = None, name: str | None = None, seed: int = 0)
   :canonical: sammo.instructions.MetaPrompt

   Bases: :py:obj:`sammo.base.Component`

   .. py:method:: with_extractor(on_error: typing.Literal[raise, empty_result] = 'raise')
      :canonical: sammo.instructions.MetaPrompt.with_extractor

      .. autodoc2-docstring:: sammo.instructions.MetaPrompt.with_extractor

   .. py:method:: render_as_json(data, is_key=False)
      :canonical: sammo.instructions.MetaPrompt.render_as_json
      :classmethod:

      .. autodoc2-docstring:: sammo.instructions.MetaPrompt.render_as_json

   .. py:method:: render_as_markdown(data, alternative_headings=False, depth=0)
      :canonical: sammo.instructions.MetaPrompt.render_as_markdown
      :classmethod:

      .. autodoc2-docstring:: sammo.instructions.MetaPrompt.render_as_markdown

   .. py:method:: render_as_xml(data, depth=0, use_attr=True)
      :canonical: sammo.instructions.MetaPrompt.render_as_xml
      :classmethod:

      .. autodoc2-docstring:: sammo.instructions.MetaPrompt.render_as_xml

.. py:class:: FewshotExamples(data: sammo.data.DataTable, n_examples: int | None = None, name: str | None = None)
   :canonical: sammo.instructions.FewshotExamples

   Bases: :py:obj:`sammo.base.ScalarComponent`

.. py:class:: InputData(id_offset: int = 0, name: str | None = None)
   :canonical: sammo.instructions.InputData

   Bases: :py:obj:`sammo.base.ScalarComponent`
