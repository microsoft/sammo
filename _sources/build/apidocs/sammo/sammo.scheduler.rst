:py:mod:`sammo.scheduler`
=========================

.. py:module:: sammo.scheduler

.. autodoc2-docstring:: sammo.scheduler
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ComputeNode <sammo.scheduler.ComputeNode>`
     - .. autodoc2-docstring:: sammo.scheduler.ComputeNode
          :summary:
   * - :py:obj:`Scheduler <sammo.scheduler.Scheduler>`
     - .. autodoc2-docstring:: sammo.scheduler.Scheduler
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`logger <sammo.scheduler.logger>`
     - .. autodoc2-docstring:: sammo.scheduler.logger
          :summary:
   * - :py:obj:`TEMPLATE <sammo.scheduler.TEMPLATE>`
     - .. autodoc2-docstring:: sammo.scheduler.TEMPLATE
          :summary:

API
~~~

.. py:data:: logger
   :canonical: sammo.scheduler.logger
   :value: None

   .. autodoc2-docstring:: sammo.scheduler.logger

.. py:data:: TEMPLATE
   :canonical: sammo.scheduler.TEMPLATE
   :value: <Multiline-String>

   .. autodoc2-docstring:: sammo.scheduler.TEMPLATE

.. py:class:: ComputeNode(job, local_cache, priority)
   :canonical: sammo.scheduler.ComputeNode

   .. autodoc2-docstring:: sammo.scheduler.ComputeNode

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.scheduler.ComputeNode.__init__

.. py:class:: Scheduler(runner, jobs, base_priority=0)
   :canonical: sammo.scheduler.Scheduler

   .. autodoc2-docstring:: sammo.scheduler.Scheduler

   .. rubric:: Initialization

   .. autodoc2-docstring:: sammo.scheduler.Scheduler.__init__

   .. py:method:: plot(open_in_browser=False)
      :canonical: sammo.scheduler.Scheduler.plot

      .. autodoc2-docstring:: sammo.scheduler.Scheduler.plot

   .. py:method:: display()
      :canonical: sammo.scheduler.Scheduler.display

      .. autodoc2-docstring:: sammo.scheduler.Scheduler.display

   .. py:method:: run_node(node)
      :canonical: sammo.scheduler.Scheduler.run_node
      :async:

      .. autodoc2-docstring:: sammo.scheduler.Scheduler.run_node

   .. py:method:: arun()
      :canonical: sammo.scheduler.Scheduler.arun
      :async:

      .. autodoc2-docstring:: sammo.scheduler.Scheduler.arun

   .. py:method:: run()
      :canonical: sammo.scheduler.Scheduler.run

      .. autodoc2-docstring:: sammo.scheduler.Scheduler.run
