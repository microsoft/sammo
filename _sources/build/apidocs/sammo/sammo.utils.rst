:py:mod:`sammo.utils`
=====================

.. py:module:: sammo.utils

.. autodoc2-docstring:: sammo.utils
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CodeTimer <sammo.utils.CodeTimer>`
     - .. autodoc2-docstring:: sammo.utils.CodeTimer
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`sync <sammo.utils.sync>`
     - .. autodoc2-docstring:: sammo.utils.sync
          :summary:
   * - :py:obj:`serialize_json <sammo.utils.serialize_json>`
     - .. autodoc2-docstring:: sammo.utils.serialize_json
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`MAIN_PATH <sammo.utils.MAIN_PATH>`
     - .. autodoc2-docstring:: sammo.utils.MAIN_PATH
          :summary:
   * - :py:obj:`MAIN_NAME <sammo.utils.MAIN_NAME>`
     - .. autodoc2-docstring:: sammo.utils.MAIN_NAME
          :summary:
   * - :py:obj:`DEFAULT_SAVE_PATH <sammo.utils.DEFAULT_SAVE_PATH>`
     - .. autodoc2-docstring:: sammo.utils.DEFAULT_SAVE_PATH
          :summary:

API
~~~

.. py:class:: CodeTimer()
   :canonical: sammo.utils.CodeTimer

   .. autodoc2-docstring:: sammo.utils.CodeTimer

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.utils.CodeTimer.__init__

   .. py:property:: interval
      :canonical: sammo.utils.CodeTimer.interval
      :type: float

      .. autodoc2-docstring:: sammo.utils.CodeTimer.interval

.. py:data:: MAIN_PATH
   :canonical: sammo.utils.MAIN_PATH
   :value: None

   .. autodoc2-docstring:: sammo.utils.MAIN_PATH

.. py:data:: MAIN_NAME
   :canonical: sammo.utils.MAIN_NAME
   :value: None

   .. autodoc2-docstring:: sammo.utils.MAIN_NAME

.. py:data:: DEFAULT_SAVE_PATH
   :canonical: sammo.utils.DEFAULT_SAVE_PATH
   :value: None

   .. autodoc2-docstring:: sammo.utils.DEFAULT_SAVE_PATH

.. py:function:: sync(f: collections.abc.Coroutine)
   :canonical: sammo.utils.sync

   .. autodoc2-docstring:: sammo.utils.sync

.. py:function:: serialize_json(key) -> bytes
   :canonical: sammo.utils.serialize_json

   .. autodoc2-docstring:: sammo.utils.serialize_json
