"""Bound cooperative async-generator cleanup without spawning cleanup tasks."""

import asyncio
from types import coroutine


@coroutine
def _drive_close(generator, primary_error):
    """Forward close resumptions, but never let a finalizer suppress cancellation.

    Using ``yield from`` here would forward cancellation straight into the
    finalizer, which could suppress it and defeat the cleanup timeout. Instead,
    interrupt its native close iterator once with GeneratorExit and stop driving
    it. The iterator protocol works on Python 3.12 too, where aclose().close()
    alone does not interrupt the underlying generator.
    """
    close = generator.aclose()
    try:
        pending = close.send(None)
        while True:
            try:
                value = yield pending
            except asyncio.CancelledError as error:
                # A cancelled awaited Future is not necessarily cancellation of
                # this execution Task. Let the finalizer handle that normal
                # await outcome, while retaining the outer timeout protection.
                if not asyncio.current_task().cancelling():
                    pending = close.throw(error)
                    continue
                try:
                    close.throw(GeneratorExit)
                except (StopIteration, StopAsyncIteration, GeneratorExit):
                    pass
                else:
                    primary_error.add_note(
                        "Cleanup unresolved: the rejected async-generator "
                        "ignored GeneratorExit while its close was interrupted; "
                        "no background cleanup was scheduled."
                    )
                raise
            except BaseException as error:
                pending = close.throw(error)
            else:
                pending = close.send(value)
    except StopIteration:
        return


async def close_asyncgen_best_effort(generator, primary_error, timeout):
    """Close within a cooperative timeout, or diagnose incomplete cleanup.

    No child Task is created: a timeout or external cancellation stops this
    close driver before returning. Like other asyncio timeouts, the budget
    cannot preempt synchronous user code that does not yield to the event loop.
    Ordinary cleanup exceptions and external BaseExceptions remain the caller's
    responsibility; only this scope's timeout is converted to a diagnostic.
    """
    budget = asyncio.timeout(timeout)
    try:
        async with budget:
            await _drive_close(generator, primary_error)
    except TimeoutError:
        if not budget.expired():
            raise
    finally:
        if budget.expired():
            primary_error.add_note(
                "Cleanup unresolved: the rejected async-generator did not finish "
                "closing within the bounded cleanup timeout; its finalizer was "
                "interrupted and may be incomplete. No background cleanup was scheduled."
            )
