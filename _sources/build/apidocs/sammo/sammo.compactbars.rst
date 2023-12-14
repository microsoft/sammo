:py:mod:`sammo.compactbars`
===========================

.. py:module:: sammo.compactbars

.. autodoc2-docstring:: sammo.compactbars
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CompactProgressBars <sammo.compactbars.CompactProgressBars>`
     - .. autodoc2-docstring:: sammo.compactbars.CompactProgressBars
          :summary:
   * - :py:obj:`SubProgressBar <sammo.compactbars.SubProgressBar>`
     - .. autodoc2-docstring:: sammo.compactbars.SubProgressBar
          :summary:

API
~~~

.. py:class:: CompactProgressBars(width: int | None = None, refresh_interval: float = 1 / 50)
   :canonical: sammo.compactbars.CompactProgressBars

   .. autodoc2-docstring:: sammo.compactbars.CompactProgressBars

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.compactbars.CompactProgressBars.__init__

   .. py:method:: get(id: str, total: int | None = None, position: int | None = None, display_name: str | None = None, **kwargs) -> sammo.compactbars.SubProgressBar
      :canonical: sammo.compactbars.CompactProgressBars.get

      .. autodoc2-docstring:: sammo.compactbars.CompactProgressBars.get

   .. py:method:: finalize() -> None
      :canonical: sammo.compactbars.CompactProgressBars.finalize

      .. autodoc2-docstring:: sammo.compactbars.CompactProgressBars.finalize

.. py:class:: SubProgressBar(total: int, parent: sammo.compactbars.CompactProgressBars, moving_avg_size: int = 10, width: int = 100, prefix: str = '', show_rate: bool = True, show_time: bool = True, ascii: str = 'auto')
   :canonical: sammo.compactbars.SubProgressBar

   .. autodoc2-docstring:: sammo.compactbars.SubProgressBar

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.__init__

   .. py:attribute:: phases
      :canonical: sammo.compactbars.SubProgressBar.phases
      :value: None

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.phases

   .. py:property:: total
      :canonical: sammo.compactbars.SubProgressBar.total

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.total

   .. py:property:: done
      :canonical: sammo.compactbars.SubProgressBar.done

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.done

   .. py:property:: elapsed_long
      :canonical: sammo.compactbars.SubProgressBar.elapsed_long

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.elapsed_long

   .. py:property:: elapsed
      :canonical: sammo.compactbars.SubProgressBar.elapsed

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.elapsed

   .. py:property:: phase
      :canonical: sammo.compactbars.SubProgressBar.phase

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.phase

   .. py:property:: barwidth
      :canonical: sammo.compactbars.SubProgressBar.barwidth

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.barwidth

   .. py:property:: rate
      :canonical: sammo.compactbars.SubProgressBar.rate

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.rate

   .. py:property:: eta
      :canonical: sammo.compactbars.SubProgressBar.eta

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.eta

   .. py:method:: update(*args, **kwargs)
      :canonical: sammo.compactbars.SubProgressBar.update

      .. autodoc2-docstring:: sammo.compactbars.SubProgressBar.update
