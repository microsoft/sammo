:py:mod:`sammo.data`
====================

.. py:module:: sammo.data

.. autodoc2-docstring:: sammo.data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DataTable <sammo.data.DataTable>`
     -

API
~~~

.. py:class:: DataTable(inputs: list, outputs: list | None = None, constants: dict | None = None, seed=42)
   :canonical: sammo.data.DataTable

   Bases: :py:obj:`pyglove.JSONConvertible`

   .. py:property:: inputs
      :canonical: sammo.data.DataTable.inputs

      .. autodoc2-docstring:: sammo.data.DataTable.inputs

   .. py:property:: outputs
      :canonical: sammo.data.DataTable.outputs

      .. autodoc2-docstring:: sammo.data.DataTable.outputs

   .. py:property:: constants
      :canonical: sammo.data.DataTable.constants
      :type: dict | None

      .. autodoc2-docstring:: sammo.data.DataTable.constants

   .. py:method:: to_json(**kwargs)
      :canonical: sammo.data.DataTable.to_json

      .. autodoc2-docstring:: sammo.data.DataTable.to_json

   .. py:method:: persistent_hash()
      :canonical: sammo.data.DataTable.persistent_hash

      .. autodoc2-docstring:: sammo.data.DataTable.persistent_hash

   .. py:method:: from_json(json_value, **kwargs)
      :canonical: sammo.data.DataTable.from_json
      :classmethod:

   .. py:method:: from_pandas(df: pandas.DataFrame, output_fields: list[str] | str = 'output', input_fields: list[str] | str | None = None, constants: dict | None = None, seed=42)
      :canonical: sammo.data.DataTable.from_pandas
      :classmethod:

      .. autodoc2-docstring:: sammo.data.DataTable.from_pandas

   .. py:method:: from_records(records: list[dict], output_fields: list[str] | str = 'output', input_fields: list[str] | str | None = None, **kwargs)
      :canonical: sammo.data.DataTable.from_records
      :classmethod:

      .. autodoc2-docstring:: sammo.data.DataTable.from_records

   .. py:method:: to_records(only_values=True)
      :canonical: sammo.data.DataTable.to_records

      .. autodoc2-docstring:: sammo.data.DataTable.to_records

   .. py:method:: to_string(max_rows: int = 10, max_col_width: int = 60, max_cell_length: int = 500)
      :canonical: sammo.data.DataTable.to_string

      .. autodoc2-docstring:: sammo.data.DataTable.to_string

   .. py:method:: sample(k: int, seed: int | None = None) -> typing_extensions.Self
      :canonical: sammo.data.DataTable.sample

      .. autodoc2-docstring:: sammo.data.DataTable.sample

   .. py:method:: shuffle(seed: int | None = None) -> typing_extensions.Self
      :canonical: sammo.data.DataTable.shuffle

      .. autodoc2-docstring:: sammo.data.DataTable.shuffle

   .. py:method:: random_split(*sizes: int, seed=None) -> tuple
      :canonical: sammo.data.DataTable.random_split

      .. autodoc2-docstring:: sammo.data.DataTable.random_split

   .. py:method:: copy() -> typing_extensions.Self
      :canonical: sammo.data.DataTable.copy

      .. autodoc2-docstring:: sammo.data.DataTable.copy

   .. py:method:: get_minibatch_iterator(minibatch_size)
      :canonical: sammo.data.DataTable.get_minibatch_iterator

      .. autodoc2-docstring:: sammo.data.DataTable.get_minibatch_iterator
