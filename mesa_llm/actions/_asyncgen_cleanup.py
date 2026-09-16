"""Best-effort closure with separate finalizer and caller cancellation scopes."""

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
            "while its close was interrupted; its driver will not resume it."
        )
    primary_error.add_note(note)
    if cancellation is not None:
        cancellation.add_note(note)


async def _drive_close(generator, primary_error, stop):
    """Deliver local cancellation normally; an independent signal stops closure.

    The driver never directly awaits user work. Its local timeout scopes may
    cancel this Task without affecting the action-execution Task. The caller's
    stop signal is not cancellation and cannot be swallowed by those scopes.
    Return failures as values so BaseExceptions remain under caller control,
    rather than escaping from a separate Task into the event loop.
    """
    try:
        if stop.done():
            return None
        close = generator.aclose()
        pending = close.send(None)
        while True:
            if stop.done():
                _interrupt_close(close, primary_error, stop.result())
                return None
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
                    # Replace Task.__step's consumption of the native await yield.
                    pending._asyncio_future_blocking = False
                    while True:
                        try:
                            await asyncio.wait(
                                {pending, stop}, return_when=asyncio.FIRST_COMPLETED
                            )
                        except asyncio.CancelledError as error:
                            if stop.done() or pending.done():
                                raise
                            # Cancellation originating in the finalizer keeps
                            # native child-cancellation semantics. Waiting for
                            # that child is still interruptible by the caller's
                            # independent stop signal.
                            message = error.args[0] if error.args else None
                            if not pending.cancel(msg=message):
                                raise
                        else:
                            break
            except BaseException as error:
                if stop.done():
                    _interrupt_close(close, primary_error, stop.result())
                    return None
                # In particular, a finalizer-local timeout must receive its own
                # CancelledError so __aexit__ can convert it to TimeoutError.
                pending = close.throw(error)
            else:
                if stop.done():
                    _interrupt_close(close, primary_error, stop.result())
                    return None
                try:
                    value = None if pending is None else pending.result()
                except BaseException as error:
                    pending = close.throw(error)
                else:
                    pending = close.send(value)
    except StopIteration:
        return None
    except BaseException as error:
        return error


async def close_asyncgen_best_effort(generator, primary_error, timeout):
    """Supervise closure within budget and finish its driver before returning.

    Finalizer timeouts use a separate Task, while the caller's deadline waits
    independently of user work. On caller timeout or cancellation, signal the
    driver to interrupt once without cancelling or draining user-owned children. The
    driver is joined before rejection/cancellation returns, not left running.
    The two Tasks share the caller's context so context-variable tokens remain
    valid. Synchronous user code that never yields cannot be preempted.
    """
    loop = asyncio.get_running_loop()
    caller = asyncio.current_task()
    stop = loop.create_future()
    coroutine = _drive_close(generator, primary_error, stop)
    try:
        # Explicit non-eager start avoids re-entering the shared Context while
        # the caller is still running, including with an eager loop factory.
        driver = asyncio.Task(
            coroutine, loop=loop, context=caller.get_context(), eager_start=False
        )
    except BaseException:
        coroutine.close()
        raise
    cancellation = None
    note_start = len(getattr(primary_error, "__notes__", ()))
    try:
        done, _ = await asyncio.wait({driver}, timeout=timeout)
        if driver in done:
            failure = driver.result()
            if failure is not None:
                raise failure
            return
        primary_error.add_note(
            "Cleanup unresolved: the rejected async-generator did not finish "
            "closing within the bounded cleanup timeout; its finalizer was "
            "interrupted and may be incomplete. The framework did not cancel "
            "or drain awaited work."
        )
    except asyncio.CancelledError as error:
        cancellation = error
    finally:
        # The caller does not cancel the driver or its awaited child. The stop
        # signal wakes the driver's independent waiter even if the child has an
        # indefinitely waiting cancellation finalizer.
        if not stop.done():
            stop.set_result(cancellation)
        while not driver.done():
            try:
                await asyncio.shield(driver)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if cancellation is not None:
            for note in getattr(primary_error, "__notes__", ())[note_start:]:
                if note not in getattr(cancellation, "__notes__", ()):
                    cancellation.add_note(note)
            raise cancellation
    failure = driver.result()
    if failure is not None:
        raise failure
