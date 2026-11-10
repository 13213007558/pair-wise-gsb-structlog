(protected-fields)=

# Protected Fields

```{testsetup}
import structlog
```

```{testcleanup}
import structlog
structlog.reset_defaults()
```

Sometimes a field in your log context carries security-relevant meaning -- for example an `actor_id` that your authentication middleware binds at the beginning of a request.
If business code later logs an event with attacker-controlled data expanded into keyword arguments, the payload can silently override that field:

```python
log = log.bind(actor_id="user-123")  # bound by auth middleware
# ... later, in a request handler:
log.info("profile updated", **request.json())  # may contain "actor_id"!
```

By the time the event dict reaches your processors, it is impossible to tell where a value came from.

The {mod}`structlog.protected` module closes this gap: it lets integration code declare field *names* as protected.
Once declared, a bound value for such a field can not be overridden by event keyword arguments -- no matter whether the value was bound via {meth}`structlog.BoundLoggerBase.bind`, {meth}`~structlog.BoundLoggerBase.new`, or {func}`structlog.contextvars.bind_contextvars`.

```python
import structlog

structlog.protected.protect_keys("actor_id")

log = structlog.get_logger().bind(actor_id="user-123")
log.info("profile updated", actor_id="admin", bio="hi")
```

The bound `actor_id` survives, and the conflicting payload value is *not* silently dropped -- it is moved aside:

```python
{"event": "profile updated", "actor_id": "user-123", "bio": "hi",
 "protected_conflicts": {"actor_id": "admin"}}
```

:::{important}
This is **not** tamper-proof log storage.
Protection happens where event keyword arguments are merged with bound context; trusted processors still run afterwards and can modify any key of the final event -- including protected fields and the conflicts key.
The mechanism protects field *provenance* at merge time, nothing more.
:::


## Conflict Handling

{func}`structlog.protected.configure_protection` controls what happens when an event keyword argument conflicts with a bound protected value:

- `on_conflict="move"` (default): the bound value wins and the conflicting event keyword argument is collected in a dictionary under the *conflict_key* (default: `protected_conflicts`).
- `on_conflict="raise"`: a {class}`structlog.exceptions.ProtectedKeyError` is raised and nothing is logged.

Conflicting values are never silently discarded in either mode.

If an event keyword argument is *equal* to the bound value, it is not treated as a conflict.
If a protected field is declared but not bound at log time, event keyword arguments for it pass through normally.


## Semantics

The mechanism is deliberately small and consistent across all of *structlog*'s context-carrying APIs:

- **Binding**: values bound via {meth}`structlog.BoundLoggerBase.bind`, {meth}`~structlog.BoundLoggerBase.new`, {func}`structlog.contextvars.bind_contextvars`, or {func}`structlog.contextvars.bound_contextvars` are all protected equally.
  Binding is trusted code: re-binding a protected key replaces the trusted value.
- **Unbinding**: {meth}`~structlog.BoundLoggerBase.unbind`, {meth}`~structlog.BoundLoggerBase.try_unbind`, and {func}`structlog.contextvars.unbind_contextvars` remove the trusted value; afterwards event keyword arguments may set the field again.
  {meth}`~structlog.BoundLoggerBase.new` clears the whole context, including protected values.
- **Context variables**: in {func}`structlog.contextvars.merge_contextvars`, protected context-local values take precedence over values already present in the event dict (whether from bound context or event keyword arguments); displaced values are handled according to the conflict configuration.
- **Bound loggers**: enforcement lives in {meth}`structlog.BoundLoggerBase._process_event`, so it applies uniformly to the generic `structlog.BoundLogger`, {class}`structlog.stdlib.BoundLogger`, {class}`structlog.stdlib.AsyncBoundLogger` (async log calls included), and custom wrappers.
- **Defaults**: without {func}`~structlog.protected.protect_keys`, nothing changes; {func}`structlog.reset_defaults` removes all declarations and restores default conflict handling.

Because protection travels with the execution context, concurrent requests each keep their own bound identity -- even when logging asynchronously:

```python
import structlog
from structlog.contextvars import bound_contextvars

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        # ... your processors ...
    ],
)
structlog.protected.protect_keys("actor_id")

log = structlog.get_logger()

async def handle_request(user_id, payload):
    with bound_contextvars(actor_id=user_id):
        # payload can not spoof actor_id, even from a thread pool:
        await log.ainfo("request", **payload)
```
