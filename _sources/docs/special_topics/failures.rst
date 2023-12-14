Managing failures
=================

There are two main types of failures that can happen during the execution of a metaprompt -- network failures
and processing failures.
`SAMMO` by default will retry the network request a few times before giving up,
whereas processing failures are raised immediately.

If failures of any type are not raised but caught, the metaprompt will continue to run and will instead return
:class:`sammo.base.EmptyResult` objects.

Network request failures
------------------------
The two most common network request failures are timeouts and rejected requests (mostly due to rate limiting).
To customize how these are handled, you can specify the following parameters when
creating a :class:`sammo.runners.Runner` instance.

* ``timeout``: The timeout for the network request. Defaults to 60 seconds.
* ``retry_on``: List of exceptions that should be retried on. Defaults to common network exceptions.
* ``max_timeout_retries``: The maximum number of times to retry a network request in case of a timeout. Defaults to 1.

Processing failures
-------------------
Processing failures typically occur when the LLM output cannot be parsed correctly or contains the wrong number of rows
for a minibatch. By default, these failures are raised in order to have the user make an explicit decision on how
exceptions should handled.

To manage exception, you can specify the following parameter when creating certain :class:`sammo.base.Component` instances:

* ``on_error``: Choose between ``raise`` (default) or ``empty_result`` (other options might be available).
  If ``empty_result`` is chosen, the component will return
  an :class:`sammo.base.EmptyResult` object instead of raising an exception.

