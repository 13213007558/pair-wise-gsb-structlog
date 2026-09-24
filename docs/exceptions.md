# Exceptions

While you should use a proper crash reporter like [Sentry](https://sentry.io) in production, *structlog* has helpers for formatting exceptions for humans and machines.

All *structlog*'s exception features center around passing an `exc_info` key-value pair in the event dict.
There are three possible behaviors depending on its value:

1. If the value is a tuple, render it as if it was returned by {func}`sys.exc_info`.
2. If the value is an Exception, render it.
3. If the value is true but no tuple, call {func}`sys.exc_info` and render that.

If there is no `exc_info` key or false, the event dict is not touched.
This behavior is analog to the one of the stdlib's logging.


## Transformations

*structlog* comes with {class}`structlog.processors.ExceptionRenderer` that deduces and removes the `exc_info` key as outlined above, calls a user-supplied function with the synthesized `exc_info`, and stores its return value in the `exception` key.
The most common use-cases are already covered by the following processors:

{func}`structlog.processors.format_exc_info`

: Formats it to a flat string like the standard library would on the console.

{obj}`structlog.processors.dict_tracebacks`

: Uses {class}`structlog.tracebacks.ExceptionDictTransformer` to give you a structured and JSON-serializable `exception` key.


### Customizing Dict Tracebacks

{class}`structlog.tracebacks.ExceptionDictTransformer` can be customized with a number of parameters.
If these prove insufficient, see {ref}`Custom Filtering of Dict Tracebacks <custom-dict-tracebacks>`.

#### Budgeting the size of the whole exception tree

*max_frames* applies to each stack separately, so with long *cause*/*context* chains or wide `ExceptionGroup`s, the resulting JSON can still grow large enough to be dropped by your log aggregator.
To bound the *whole* exception tree, {class}`~structlog.tracebacks.ExceptionDictTransformer` offers three budgets that are off by default and can be enabled independently:

- *max_total_frames* limits the total number of frames across all stacks.
- *max_branches* limits the number of rendered exception branches (stacks).
- *locals_max_total_bytes* limits the total number of bytes (UTF-8) spent on rendered local variables.

Budgets are spent in a fixed order: a stack, then its exception group children in order, then its cause/context chain.
Every rendered branch keeps its overview (exception type, value, notes, and position in the hierarchy), and within each stack the shallowest and innermost frames are kept while frames are removed from the middle out.
Omitted parts leave structured counts behind: elided frames are reported in place, omitted locals in the frame's `locals_omitted` key, and omitted branches as a `{"truncated": {"stacks": N, "frames": M}}` placeholder.
Since budgets are enforced during extraction, no `repr()` is computed for locals that are omitted.


## Console rendering

Our {doc}`console-output`'s {class}`structlog.dev.ConsoleRenderer` takes an *exception_formatter* argument that allows for customizing the output of exceptions.

{func}`structlog.dev.plain_traceback`

: Is the default if [Rich] is not installed.
  As the name suggests, it renders a plain traceback.

{class}`structlog.dev.RichTracebackFormatter`

: Uses [Rich] to render a colorful traceback.
  It's a class because it allows for customizing the output by passing arguments to Rich.
: It's the default if Rich is installed.

:::{seealso}
{doc}`console-output` for more information on *structlog*'s console features.
:::

:::{deprecated} 26.1.0
Support for [*better-exceptions*] is deprecated and will be removed in a future release.
Use [Rich] instead.
:::

[*better-exceptions*]: https://github.com/qix-/better-exceptions
[Rich]: https://github.com/Textualize/rich
