:py:mod:`sammo.extractors`
==========================

.. py:module:: sammo.extractors

.. autodoc2-docstring:: sammo.extractors
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Extractor <sammo.extractors.Extractor>`
     - .. autodoc2-docstring:: sammo.extractors.Extractor
          :summary:
   * - :py:obj:`DefaultExtractor <sammo.extractors.DefaultExtractor>`
     - .. autodoc2-docstring:: sammo.extractors.DefaultExtractor
          :summary:
   * - :py:obj:`SplitLines <sammo.extractors.SplitLines>`
     - .. autodoc2-docstring:: sammo.extractors.SplitLines
          :summary:
   * - :py:obj:`StripWhitespace <sammo.extractors.StripWhitespace>`
     - .. autodoc2-docstring:: sammo.extractors.StripWhitespace
          :summary:
   * - :py:obj:`LambdaExtractor <sammo.extractors.LambdaExtractor>`
     - .. autodoc2-docstring:: sammo.extractors.LambdaExtractor
          :summary:
   * - :py:obj:`ParseJSON <sammo.extractors.ParseJSON>`
     - .. autodoc2-docstring:: sammo.extractors.ParseJSON
          :summary:
   * - :py:obj:`ExtractRegex <sammo.extractors.ExtractRegex>`
     - .. autodoc2-docstring:: sammo.extractors.ExtractRegex
          :summary:
   * - :py:obj:`MarkdownParser <sammo.extractors.MarkdownParser>`
     - .. autodoc2-docstring:: sammo.extractors.MarkdownParser
          :summary:
   * - :py:obj:`YAMLParser <sammo.extractors.YAMLParser>`
     - .. autodoc2-docstring:: sammo.extractors.YAMLParser
          :summary:
   * - :py:obj:`ParseXML <sammo.extractors.ParseXML>`
     - .. autodoc2-docstring:: sammo.extractors.ParseXML
          :summary:
   * - :py:obj:`JSONPath <sammo.extractors.JSONPath>`
     - .. autodoc2-docstring:: sammo.extractors.JSONPath
          :summary:
   * - :py:obj:`ToNum <sammo.extractors.ToNum>`
     - .. autodoc2-docstring:: sammo.extractors.ToNum
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`logger <sammo.extractors.logger>`
     - .. autodoc2-docstring:: sammo.extractors.logger
          :summary:

API
~~~

.. py:data:: logger
   :canonical: sammo.extractors.logger
   :value: None

   .. autodoc2-docstring:: sammo.extractors.logger

.. py:class:: Extractor(child: sammo.base.Component, on_error: typing.Literal[raise, empty_result] = 'raise', flatten=True)
   :canonical: sammo.extractors.Extractor

   Bases: :py:obj:`sammo.base.ListComponent`

   .. autodoc2-docstring:: sammo.extractors.Extractor

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.Extractor.__init__

.. py:class:: DefaultExtractor(child: sammo.base.Component, on_error: typing.Literal[raise, empty_result] = 'raise', flatten=True)
   :canonical: sammo.extractors.DefaultExtractor

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.DefaultExtractor

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.DefaultExtractor.__init__

.. py:class:: SplitLines(child: sammo.base.Component, on_error: typing.Literal[raise, empty_result] = 'raise', flatten=True)
   :canonical: sammo.extractors.SplitLines

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.SplitLines

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.SplitLines.__init__

.. py:class:: StripWhitespace(child: sammo.base.Component, on_error: typing.Literal[raise, empty_result] = 'raise', flatten=True)
   :canonical: sammo.extractors.StripWhitespace

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.StripWhitespace

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.StripWhitespace.__init__

.. py:class:: LambdaExtractor(child: sammo.base.Component, lambda_src_code: str, on_error: typing.Literal[raise, empty_result] = 'raise', flatten=True)
   :canonical: sammo.extractors.LambdaExtractor

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.LambdaExtractor

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.LambdaExtractor.__init__

.. py:class:: ParseJSON(child: sammo.base.Component, parse_fragments: typing.Literal[all, first, whole] = 'all', lowercase_fieldnames: bool = True, on_error='raise')
   :canonical: sammo.extractors.ParseJSON

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.ParseJSON

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.ParseJSON.__init__

   .. py:attribute:: OPENING
      :canonical: sammo.extractors.ParseJSON.OPENING
      :value: '[{'

      .. autodoc2-docstring:: sammo.extractors.ParseJSON.OPENING

   .. py:attribute:: CLOSING
      :canonical: sammo.extractors.ParseJSON.CLOSING
      :value: ']}'

      .. autodoc2-docstring:: sammo.extractors.ParseJSON.CLOSING

.. py:class:: ExtractRegex(child: sammo.base.Component, regex: str, max_matches: int | None = None, strip_whitespaces: bool = True)
   :canonical: sammo.extractors.ExtractRegex

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.ExtractRegex

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.ExtractRegex.__init__

   .. py:attribute:: INT_EOL
      :canonical: sammo.extractors.ExtractRegex.INT_EOL
      :value: '\\d+$'

      .. autodoc2-docstring:: sammo.extractors.ExtractRegex.INT_EOL

   .. py:attribute:: FRACTION_EOL
      :canonical: sammo.extractors.ExtractRegex.FRACTION_EOL
      :value: '[0-9]+/0*[1-9][0-9]*$'

      .. autodoc2-docstring:: sammo.extractors.ExtractRegex.FRACTION_EOL

   .. py:attribute:: PERCENTAGE_EOL
      :canonical: sammo.extractors.ExtractRegex.PERCENTAGE_EOL
      :value: '(\\.\\d+%$)|(\\d+.\\d+?%$)'

      .. autodoc2-docstring:: sammo.extractors.ExtractRegex.PERCENTAGE_EOL

   .. py:attribute:: LAST_TSV_COL
      :canonical: sammo.extractors.ExtractRegex.LAST_TSV_COL
      :value: '(?:\\t)([^\\t]*)$'

      .. autodoc2-docstring:: sammo.extractors.ExtractRegex.LAST_TSV_COL

.. py:class:: MarkdownParser(child: sammo.base.Component, on_error: typing.Literal[raise, empty_result] = 'raise', flatten=True)
   :canonical: sammo.extractors.MarkdownParser

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.MarkdownParser

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.MarkdownParser.__init__

.. py:class:: YAMLParser(child: sammo.base.Component, on_error: typing.Literal[raise, empty_result] = 'raise', flatten=True)
   :canonical: sammo.extractors.YAMLParser

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.YAMLParser

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.YAMLParser.__init__

.. py:class:: ParseXML(child: sammo.base.Component, parse_fragments: typing.Literal[all, first, none] = 'first', ignore_fragments_with_tags=tuple(), on_error: str = 'raise', use_attributes_marker=False, lowercase_fieldnames=False)
   :canonical: sammo.extractors.ParseXML

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.ParseXML

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.ParseXML.__init__

   .. py:attribute:: XML_TAGS
      :canonical: sammo.extractors.ParseXML.XML_TAGS
      :value: None

      .. autodoc2-docstring:: sammo.extractors.ParseXML.XML_TAGS

.. py:class:: JSONPath(child: sammo.extractors.Extractor, path: str, on_error='raise', max_results=None, flatten_lists=True)
   :canonical: sammo.extractors.JSONPath

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.JSONPath

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.JSONPath.__init__

.. py:class:: ToNum(child: sammo.base.Component, on_error: typing.Literal[raise, empty_result] = 'raise', dtype: typing.Literal[fraction, int, float] = 'float', factor: float = 1, offset: float = 0)
   :canonical: sammo.extractors.ToNum

   Bases: :py:obj:`sammo.extractors.Extractor`

   .. autodoc2-docstring:: sammo.extractors.ToNum

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.extractors.ToNum.__init__

   .. py:property:: factor
      :canonical: sammo.extractors.ToNum.factor

      .. autodoc2-docstring:: sammo.extractors.ToNum.factor

   .. py:property:: offset
      :canonical: sammo.extractors.ToNum.offset

      .. autodoc2-docstring:: sammo.extractors.ToNum.offset
