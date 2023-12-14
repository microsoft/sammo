Rate Limiting
=============

Many APIs have rate limits, often in terms of number of requests within a certain time period or a total cost.

To honor these limits, the :class:`sammo.runners.Runner` class allows you to specify rate limits that are applied to all
requests made by the runner.

You have three options to specify rate limits (in increasing order of flexibility):

1. Specify a number for the ``rate_limit`` parameter. This will enforce a requests per second limit equal to that number.
2. Specify a list of :class:`sammo.throttler.AtMost` objects. These ``AtMost`` objects are applied in an logical AND
   fashion, so that the runner will not make more than the specified number of requests per second, and will not make
   more than the specified number of requests per minute, etc.
3. Pass an instance of :class:`sammo.throttler.Throttler` (or a subclass of it). This allows you to fine-tune some
   settings, e.g., how costs are calculated. See the documentation for :class:`sammo.throttler.Throttler` for more
   information.

Rate Limit Types
----------------

When specifying limits, there are
type: Literal["calls", "cost", "running", "failed", "rejected"]

The rate limits are specified as a list of :class:`sammo.runners.RateLimit` objects.