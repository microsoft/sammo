:py:mod:`sammo.store`
=====================

.. py:module:: sammo.store

.. autodoc2-docstring:: sammo.store
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PersistentDict <sammo.store.PersistentDict>`
     - .. autodoc2-docstring:: sammo.store.PersistentDict
          :summary:
   * - :py:obj:`InMemoryDict <sammo.store.InMemoryDict>`
     - .. autodoc2-docstring:: sammo.store.InMemoryDict
          :summary:

API
~~~

.. py:class:: PersistentDict(filename: os.PathLike | str, project_keys: typing.Callable = None)
   :canonical: sammo.store.PersistentDict

   Bases: :py:obj:`collections.abc.MutableMapping`

   .. autodoc2-docstring:: sammo.store.PersistentDict

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.store.PersistentDict.__init__

   .. py:method:: vacuum() -> None
      :canonical: sammo.store.PersistentDict.vacuum

      .. autodoc2-docstring:: sammo.store.PersistentDict.vacuum

   .. py:method:: to_json(**kwargs)
      :canonical: sammo.store.PersistentDict.to_json

      .. autodoc2-docstring:: sammo.store.PersistentDict.to_json

   .. py:method:: from_json(json_value, **kwargs)
      :canonical: sammo.store.PersistentDict.from_json
      :classmethod:

      .. autodoc2-docstring:: sammo.store.PersistentDict.from_json

.. py:class:: InMemoryDict(project_keys: typing.Callable = None)
   :canonical: sammo.store.InMemoryDict

   Bases: :py:obj:`sammo.store.PersistentDict`

   .. autodoc2-docstring:: sammo.store.InMemoryDict

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.store.InMemoryDict.__init__

   .. py:method:: persist(filename: os.PathLike | str)
      :canonical: sammo.store.InMemoryDict.persist

      .. autodoc2-docstring:: sammo.store.InMemoryDict.persist
