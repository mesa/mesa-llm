"""Best-effort async-generator closure without dependent cancellation waits."""

import asyncio
import contextlib
import inspect


def close_asyncgen_without_loop(generator, primary_error):
    """Close inert generators without starting arbitrary asynchronous finalizers.

    A started finalizer can create work whose loop shutdown waits indefinitely.
    Leave it to the owning asynchronous context rather than creating a new loop.
    Created and closed generators can be closed without executing user code.
    """
    if inspect.getasyncgenstate(generator) not in {
        inspect.AGEN_CREATED,
        inspect.AGEN_CLOSED,
    }:
        primary_error.add_note(
            "Cleanup unresolved: the rejected async-generator was left untouched; "
            "synchronous cleanup cannot safely run a started finalizer in a new "
            "event loop with bounded shutdown. Close it from its owning context."
        )
        return
    with contextlib.suppress(StopIteration):
        generator.aclose().send(None)


def _interrupt_close(close, primary_error, cancellation):
    """Interrupt once while keeping ordinary secondary failures diagnostic."""
    try:
        close.throw(GeneratorExit)
    except (StopIteration, StopAsyncIteration, GeneratorExit):
        return
    except Exception as error:
        try:
            detail = str(error)
        except BaseException:
            detail = "<unprintable cleanup exception>"
        note = (
            "Cleanup of the rejected async-generator failed during interruption "
            f"with {type(error).__name__}: {detail}"
        )
    else:
        note = (
            "Cleanup unresolved: the rejected async-generator ignored GeneratorExit "
            "while its close was interrupted; no helper Task was scheduled."
        )
    # The timeout's caller receives primary_error; an external canceller receives
    # cancellation. Neither outcome may be replaced by an ordinary close failure.
    primary_error.add_note(note)
    cancellation.add_note(note)


async def _drive_close(generator, primary_error):
    """Resume close steps without making cancellation depend on awaited work."""
    close = generator.aclose()
    try:
        pending = close.send(None)
        while True:
            try:
                if pending is None:
                    await asyncio.sleep(0)
                else:
                    if not isinstance(pending, asyncio.Future):
                        raise RuntimeError(
                            "Async-generator cleanup yielded a non-Future."
                        )
                    if pending.get_loop() is not asyncio.get_running_loop():
                        raise RuntimeError(
                            "Async-generator cleanup awaited a foreign loop."
                        )
                    # Consuming the native await's yield replaces Task.__step at
                    # this boundary. Reset its protocol flag as Task would do.
                    pending._asyncio_future_blocking = False
                    # wait() uses a separate waiter and removes its callback on
                    # cancellation. Never await or cancel the child directly:
                    # its cancellation/finally may itself wait indefinitely.
                    await asyncio.wait({pending})
            except asyncio.CancelledError as cancellation:
                _interrupt_close(close, primary_error, cancellation)
                raise
            except BaseException as error:
                pending = close.throw(error)
            else:
                # A cancelled awaited object is a normal await outcome, not
                # cancellation of the execution Task. Deliver it to the finalizer.
                try:
                    value = None if pending is None else pending.result()
                except BaseException as error:
                    pending = close.throw(error)
                else:
                    pending = close.send(value)
    except StopIteration:
        return


async def close_asyncgen_best_effort(generator, primary_error, timeout):
    """Close within a cooperative budget or report unresolved cleanup.

    No helper Task is created. Timeout and external cancellation stop this
    driver without cancelling or draining awaited work; that work remains its
    owner's responsibility. Synchronous user code that never yields cannot be
    preempted. Ordinary finalizer failures retain the caller's exception policy.
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
                "interrupted and may be incomplete. Awaited work was not cancelled "
                "or drained; no helper Task was scheduled."
            )
