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
        _note_interruption_failure(primary_error, cancellation, error)
        return
    else:
        note = (
            "Cleanup unresolved: the rejected async-generator ignored GeneratorExit "
            "while its close was interrupted; its driver will not resume it."
        )
    primary_error.add_note(note)
    if cancellation is not None:
        cancellation.add_note(note)


def _note_interruption_failure(primary_error, cancellation, error):
    try:
        detail = str(error)
    except BaseException:
        detail = "<unprintable cleanup exception>"
    note = (
        "Cleanup of the rejected async-generator failed during interruption "
        f"with {type(error).__name__}: {detail}"
    )
    primary_error.add_note(note)
    if cancellation is not None:
        cancellation.add_note(note)


class _CloseControl:
    """Relay caller cancellation without confusing it with the cleanup deadline."""

    def __init__(self, loop, caller):
        self.loop = loop
        self.caller = caller
        self.baseline = caller.cancelling()
        self.stop = loop.create_future()
        self.wakeup = loop.create_future()
        self.request = None
        self.cancellation = None
        self.request_count = 0
        self.withdrawals = 0
        self.interrupted_work = []

    def resume(self, operation, *args):
        before = self.caller.cancelling()
        try:
            return operation(*args)
        finally:
            remaining = self.caller.cancelling()
            self.withdrawals += max(0, before - remaining)
            self.resolve()

    def consumed_delivery(self):
        # A scope can unwind in the driver before the caller receives the
        # cancellation it already withdrew. Do not inject that stale delivery
        # into the next scope (or resurrect it after the generator has closed).
        if (
            self.cancellation is None
            and self.withdrawals
            and self.caller.cancelling() <= self.baseline
        ):
            self.withdrawals = 0
            return True
        return False

    def relay(self, cancellation):
        if self.cancellation is None:
            self.cancellation = cancellation
            self.request_count = self.caller.cancelling()
        if self.request is None:
            self.request = cancellation
        if not self.wakeup.done():
            self.wakeup.set_result(None)

    def take_request(self, pending):
        # Caller cancellation may be external. Do not cancel user-owned work
        # to discover its source; diagnose it if native unwinding leaves it live.
        if isinstance(pending, asyncio.Future) and type(pending) is not asyncio.Future:
            self.interrupted_work.append(pending)
        cancellation = self.request
        self.request = None
        self.wakeup = self.loop.create_future()
        return cancellation

    def resolve(self):
        # Inspect consumption only AFTER resuming the generator's native scope,
        # not as a guess about the origin of a newly delivered cancellation.
        # Do not clear/uncancel requests ourselves, and do not lose an additional
        # external request when a timeout removes only its own cancellation.
        remaining = self.caller.cancelling()
        if remaining < self.request_count and remaining <= self.baseline:
            self.cancellation = None
            self.request = None
            self.request_count = 0
            self.withdrawals = 0
            self.wakeup = self.loop.create_future()


async def _drive_close(generator, primary_error, control):
    """Deliver local cancellation normally; an independent signal stops closure.

    The driver never directly awaits user work. Its local timeout scopes may
    cancel this Task without affecting the action-execution Task. The caller's
    stop signal is not cancellation and cannot be swallowed by those scopes.
    Return failures as values so BaseExceptions remain under caller control,
    rather than escaping from a separate Task into the event loop.
    """
    stop = control.stop
    try:
        if stop.done() and control.request is None:
            return None
        close = generator.aclose()
        pending = control.resume(close.send, None)
        while True:
            control.resolve()
            if control.request is not None:
                pending = control.resume(close.throw, control.take_request(pending))
                continue
            if stop.done():
                control.resume(_interrupt_close, close, primary_error, stop.result())
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
                                {pending, stop, control.wakeup},
                                return_when=asyncio.FIRST_COMPLETED,
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
                if control.request is not None:
                    pending = control.resume(close.throw, control.take_request(pending))
                    continue
                if stop.done():
                    control.resume(
                        _interrupt_close, close, primary_error, stop.result()
                    )
                    return None
                # In particular, a finalizer-local timeout must receive its own
                # CancelledError so __aexit__ can convert it to TimeoutError.
                pending = control.resume(close.throw, error)
            else:
                control.resolve()
                if control.request is not None:
                    pending = control.resume(close.throw, control.take_request(pending))
                    continue
                if stop.done():
                    control.resume(
                        _interrupt_close, close, primary_error, stop.result()
                    )
                    return None
                try:
                    value = None if pending is None else pending.result()
                except BaseException as error:
                    pending = control.resume(close.throw, error)
                else:
                    pending = control.resume(close.send, value)
    except StopIteration:
        return None
    except BaseException as error:
        control.resolve()
        if control.cancellation is not None and isinstance(error, Exception):
            _note_interruption_failure(primary_error, control.cancellation, error)
            return None
        return error
    finally:
        control.resolve()


async def close_asyncgen_best_effort(generator, primary_error, timeout):
    """Supervise closure while preserving scopes entered before the first yield.

    Finalizer-local scopes run in the driver. Cancellation of the caller is
    relayed into the generator first, so a scope entered while priming it can
    perform its native conversion and withdraw its own request. Unconsumed
    caller cancellation remains primary. The framework deadline is a separate
    signal, never a Task cancellation; it does not restart after a local timeout.
    The driver is joined at every exit and shares the caller's Context.
    Cooperative deadlines cannot preempt synchronous user code that never yields.
    """
    loop = asyncio.get_running_loop()
    caller = asyncio.current_task()
    control = _CloseControl(loop, caller)
    deadline = None if timeout is None else loop.time() + timeout
    coroutine = _drive_close(generator, primary_error, control)
    try:
        driver = asyncio.Task(
            coroutine, loop=loop, context=caller.get_context(), eager_start=False
        )
    except BaseException:
        coroutine.close()
        raise
    cancellation = None
    note_start = len(getattr(primary_error, "__notes__", ()))
    try:
        while True:
            remaining = None if deadline is None else max(0, deadline - loop.time())
            try:
                done, _ = await asyncio.wait({driver}, timeout=remaining)
            except asyncio.CancelledError as error:
                if control.consumed_delivery():
                    continue
                if driver.done():
                    cancellation = error
                    break
                control.relay(error)
                continue
            if driver in done:
                break
            primary_error.add_note(
                "Cleanup unresolved: the rejected async-generator did not finish "
                "closing within the bounded cleanup timeout; its finalizer was "
                "interrupted and may be incomplete. The framework did not cancel "
                "or drain awaited work."
            )
            break
    finally:
        if not control.stop.done():
            control.stop.set_result(None)
        while not driver.done():
            try:
                await asyncio.shield(driver)
            except asyncio.CancelledError as error:
                if control.consumed_delivery():
                    continue
                if driver.done():
                    if cancellation is None:
                        cancellation = error
                else:
                    control.relay(error)
        if any(not pending.done() for pending in control.interrupted_work):
            primary_error.add_note(
                "Cleanup unresolved: caller-scope cancellation interrupted an "
                "awaited Task or composite Future which remains live; the "
                "framework did not cancel or drain that user-owned work."
            )
        cancellation = control.cancellation or cancellation
        if cancellation is not None:
            for note in getattr(primary_error, "__notes__", ())[note_start:]:
                if note not in getattr(cancellation, "__notes__", ()):
                    cancellation.add_note(note)
            raise cancellation
    failure = driver.result()
    if failure is not None:
        raise failure
