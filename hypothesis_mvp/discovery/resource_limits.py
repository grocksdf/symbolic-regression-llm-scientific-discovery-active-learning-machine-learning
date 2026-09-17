"""Hard wall-time isolation and pre-transport provider quotas.

The process runner supports internal engines only (no external engine process).
It caps wall time, not RAM or CPU instructions. Provider quota is shared across
all runtimes and threads within one isolated stage, including retries/cycles.
"""
from contextlib import contextmanager
import math
import multiprocessing as mp
import threading
import time
import traceback
from pathlib import Path


class ResourceLimitExceeded(RuntimeError):
    pass


class StageBudget:
    def __init__(self, seconds, provider_attempts):
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                or type(provider_attempts) is not int or provider_attempts < 0):
            raise ValueError("invalid hard stage budget")
        self.deadline = time.monotonic() + seconds
        self.provider_limit = provider_attempts
        self.provider_used = 0
        self.denied = False
        self.lock = threading.Lock()

    def before_provider(self):
        with self.lock:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self.provider_used >= self.provider_limit:
                self.denied = True
                raise ResourceLimitExceeded("provider quota or deadline exhausted before transport")
            self.provider_used += 1
            return remaining


_active = None
_context_lock = threading.Lock()


def before_provider_transport():
    return None if _active is None else _active.before_provider()


@contextmanager
def stage_budget(seconds, provider_attempts):
    global _active
    budget = StageBudget(seconds, provider_attempts)
    if not _context_lock.acquire(blocking=False):
        raise ValueError("concurrent stage budget contexts are unsupported")
    _active = budget
    try:
        yield budget
        if budget.denied:
            raise ResourceLimitExceeded("resource denial blocks stage completion; no silent fallback")
    finally:
        _active = None
        _context_lock.release()


def _worker(connection, function, args, kwargs, seconds, provider_attempts):
    try:
        with stage_budget(seconds, provider_attempts) as budget:
            result = function(*args, **kwargs)
        connection.send(("succeeded", result, budget.provider_used))
    except BaseException as error:
        # Keep errors non-secret: provider errors can contain endpoint bodies.
        diagnostic = {"error_type": type(error).__name__,
            "frames": [{"file": Path(frame.filename).name, "line": frame.lineno,
                        "function": frame.name} for frame in traceback.extract_tb(error.__traceback__)]}
        if isinstance(error, ResourceLimitExceeded):
            diagnostic["message"] = str(error)
        connection.send(("failed", diagnostic, None))
    finally:
        connection.close()


def run_bounded(function, args=(), kwargs=None, *, seconds, provider_attempts):
    """Spawn one internal-only stage; terminate and join on timeout/interruption.

    Parent timer includes process start/import and result publication. No restart
    or retry is performed. All persistent experiment markers belong to caller.
    """
    StageBudget(seconds, provider_attempts)  # validate before process creation
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker,
        args=(sender, function, args, kwargs or {}, seconds, provider_attempts), daemon=True)
    deadline = time.monotonic() + seconds
    started = time.monotonic()
    next_progress = started + 15.
    received = threading.Event()
    outcome = []
    reader = None
    def receive():
        try:
            outcome.append(receiver.recv())
        except (EOFError, OSError):
            outcome.append(("failed", "MissingStageResult", None))
        finally:
            received.set()
    try:
        process.start()
        sender.close()
        # A Pipe can become readable before a large frame finishes arriving.
        # Keep recv off the supervising thread so partial writes cannot defeat
        # the wall timer. Killing the owned child closes its sender on timeout.
        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        while not received.wait(min(5., max(0., deadline - time.monotonic()))):
            now = time.monotonic()
            if now >= deadline:
                raise ResourceLimitExceeded("hard stage wall-time ceiling reached")
            if now >= next_progress:
                print(f"isolated stage running: elapsed={now - started:.1f}s ceiling={seconds}s", flush=True)
                next_progress = now + 15.
        status, result, used = outcome[0]
        process.join(max(0., deadline - time.monotonic()))
        if process.is_alive() or time.monotonic() > deadline:
            raise ResourceLimitExceeded("stage exceeded deadline while finalizing")
        if process.exitcode != 0 or status != "succeeded":
            raise RuntimeError(f"isolated stage failed: {result}")
        return result, {"provider_transports_used": used, "hard_wall_time_seconds": seconds}
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(5.)
            if process.is_alive():
                process.kill()
                process.join(5.)
            process.close()
        receiver.close()
        sender.close()
        if reader is not None:
            reader.join(5.)
