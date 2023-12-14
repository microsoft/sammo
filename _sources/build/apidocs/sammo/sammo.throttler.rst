:py:mod:`sammo.throttler`
=========================

.. py:module:: sammo.throttler

.. autodoc2-docstring:: sammo.throttler
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Throttler <sammo.throttler.Throttler>`
     - .. autodoc2-docstring:: sammo.throttler.Throttler
          :summary:
   * - :py:obj:`AtMost <sammo.throttler.AtMost>`
     - .. autodoc2-docstring:: sammo.throttler.AtMost
          :summary:

API
~~~

.. py:class:: Throttler(limits: list[sammo.throttler.AtMost], sleep_interval: float = 0.01, impute_pending_costs: bool = True, n_cost_samples: int = 10, rejection_window: int | float = 0.5)
   :canonical: sammo.throttler.Throttler

   .. autodoc2-docstring:: sammo.throttler.Throttler

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.throttler.Throttler.__init__

   .. py:attribute:: DEBUG_INTERVAL_SECONDS
      :canonical: sammo.throttler.Throttler.DEBUG_INTERVAL_SECONDS
      :value: 3

      .. autodoc2-docstring:: sammo.throttler.Throttler.DEBUG_INTERVAL_SECONDS

   .. py:method:: sleep(delay: float)
      :canonical: sammo.throttler.Throttler.sleep
      :async:
      :staticmethod:

      .. autodoc2-docstring:: sammo.throttler.Throttler.sleep

   .. py:method:: update_job_stats(job: sammo.throttler.Job, cost: float | int = 0, failed: bool = False) -> None
      :canonical: sammo.throttler.Throttler.update_job_stats

      .. autodoc2-docstring:: sammo.throttler.Throttler.update_job_stats

   .. py:method:: wait_in_line(priority: int = 0) -> sammo.throttler.Job
      :canonical: sammo.throttler.Throttler.wait_in_line
      :async:

      .. autodoc2-docstring:: sammo.throttler.Throttler.wait_in_line

.. py:class:: AtMost
   :canonical: sammo.throttler.AtMost

   .. autodoc2-docstring:: sammo.throttler.AtMost

   .. py:attribute:: value
      :canonical: sammo.throttler.AtMost.value
      :type: float | int
      :value: None

      .. autodoc2-docstring:: sammo.throttler.AtMost.value

   .. py:attribute:: type
      :canonical: sammo.throttler.AtMost.type
      :type: typing.Literal[calls, running, failed, rejected]
      :value: None

      .. autodoc2-docstring:: sammo.throttler.AtMost.type

   .. py:attribute:: period
      :canonical: sammo.throttler.AtMost.period
      :type: float | int
      :value: 1

      .. autodoc2-docstring:: sammo.throttler.AtMost.period

   .. py:attribute:: pause_for
      :canonical: sammo.throttler.AtMost.pause_for
      :type: float | int
      :value: 0

      .. autodoc2-docstring:: sammo.throttler.AtMost.pause_for
