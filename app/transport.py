"""Bounded worker protocol, concurrent pipe draining and cancellation escalation."""
import json
import math
import queue
import signal
import subprocess
import threading
import time


class WorkerError(ValueError):
    """Admission, protocol, process-exit or worker-deadline failure."""


def _decode_event(line):
    def nonfinite(value):
        raise WorkerError('Non-finite number in worker protocol: '+value)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise WorkerError('Duplicate key in worker protocol: '+key)
            result[key] = value
        return result

    try:
        # json.loads(bytes) accepts UTF-16/32, incompatible with UTF-8 framing.
        event = json.loads(line.decode('utf-8'), parse_constant=nonfinite,
                           object_pairs_hook=unique_object)
        # Overflowed exponents (1e999) bypass parse_constant. Encoding also
        # rejects unpaired surrogate escapes before consumers/UI see them.
        json.dumps(event, allow_nan=False, ensure_ascii=False).encode('utf-8')
    except WorkerError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise WorkerError('Malformed worker JSON: '+str(exc)) from exc
    if not isinstance(event, dict) or set(event) != {'kind', 'payload'}:
        raise WorkerError('Malformed worker event.')
    return event


def run_json_worker(command, request, on_progress=None, cancel=None, *, cwd=None,
                    on_process=None, timeout=None, grace_seconds=5, kill_seconds=3,
                    max_line_bytes=256*1024**2, stderr_limit=64*1024):
    """Launch without a shell; drain both pipes while writing the request.

    Validate/serialize admission arguments before launch. Deliver progress in
    order, require exactly one final result and exit status zero, retain only a
    bounded stderr tail. An optional wall deadline covers the launched worker;
    cancellation sends TERM then KILL. Every post-launch exception reaps the
    direct child, including a callback failure. Callbacks are trusted, synchronous
    and must not block. This is not isolation against hostile grandchildren.
    """
    for value in (max_line_bytes, stderr_limit):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WorkerError('Protocol byte limits must be positive integers.')
    for value in (grace_seconds, kill_seconds) + (() if timeout is None else (timeout,)):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0):
            raise WorkerError('Worker deadlines must be finite positive numbers.')
    try:
        request_bytes = json.dumps(request, allow_nan=False, ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise WorkerError('Worker input failed: '+str(exc)) from exc

    events = queue.Queue(maxsize=32)
    finished = threading.Event()
    stderr = bytearray()
    stderr_lock = threading.Lock()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def enqueue(kind, value=None):
        while not finished.is_set():
            try:
                events.put((kind, value), timeout=.05)
                return
            except queue.Full:
                pass

    def read_stdout():
        try:
            while True:
                line = process.stdout.readline(max_line_bytes+1)
                if not line:
                    break
                if len(line) > max_line_bytes:
                    raise WorkerError('Worker protocol line exceeds its byte limit.')
                if not line.endswith(b'\n'):
                    raise WorkerError('Worker protocol ended with a truncated line.')
                enqueue('line', line)
        except Exception as exc:
            enqueue('failure', str(exc))
        finally:
            enqueue('stdout_end')

    def read_stderr():
        try:
            while True:
                chunk = process.stderr.read1(8192)
                if not chunk:
                    break
                with stderr_lock:
                    stderr.extend(chunk)
                    if len(stderr) > stderr_limit:
                        del stderr[:-stderr_limit]
        except (OSError, ValueError):
            pass

    def write_request():
        try:
            process.stdin.write(request_bytes)
            process.stdin.close()
        except Exception as exc:
            enqueue('failure', 'Worker input failed: '+str(exc))
        finally:
            # A valid stdout result cannot mask a late broken-input error.
            enqueue('input_end')

    threads = [threading.Thread(target=target, daemon=True,
                                name='vtw-worker-'+name)
               for target, name in ((read_stdout, 'stdout'),
                                    (read_stderr, 'stderr'), (write_request, 'input'))]
    active_threads = []
    termination = None
    stdout_end = input_end = False
    result = None
    try:
        for thread in threads:
            thread.start()
            active_threads.append(thread)
        if on_process:
            on_process(process)
        while not (stdout_end and input_end and process.poll() is not None and events.empty()):
            now = time.monotonic()
            if timeout is not None and now-started > timeout:
                raise WorkerError('Worker execution deadline exceeded.')
            if cancel is not None and cancel.is_set() and termination is None:
                termination = now
                process.send_signal(signal.SIGTERM)
            if termination is not None and now-termination > grace_seconds:
                process.kill()
            if termination is not None and now-termination > grace_seconds+kill_seconds:
                raise WorkerError('Cancelled worker did not finish within its termination deadline.')
            try:
                kind, payload = events.get(timeout=.05)
            except queue.Empty:
                continue
            if kind == 'failure':
                raise WorkerError(payload)
            if kind == 'stdout_end':
                stdout_end = True
                continue
            if kind == 'input_end':
                input_end = True
                continue
            event = _decode_event(payload)
            if result is not None:
                raise WorkerError('Worker emitted an event after its result.')
            if event['kind'] == 'progress':
                if not isinstance(event['payload'], dict):
                    raise WorkerError('Malformed progress payload.')
                if on_progress:
                    on_progress(event['payload'])
            elif event['kind'] == 'result':
                if not isinstance(event['payload'], dict):
                    raise WorkerError('Malformed result payload.')
                result = event['payload']
            elif event['kind'] == 'error':
                raise WorkerError(str(event['payload']))
            else:
                raise WorkerError('Unknown worker event kind.')
        threads[1].join(timeout=kill_seconds)
        with stderr_lock:
            tail = bytes(stderr).decode('utf-8', errors='replace')[-1000:]
        if process.returncode != 0:
            raise WorkerError('Worker exited with status '+str(process.returncode)+': '+tail)
        if result is None:
            raise WorkerError('Worker stopped without a result: '+tail)
        return result
    finally:
        finished.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=kill_seconds)
        join_deadline = time.monotonic()+kill_seconds
        for thread in active_threads:
            thread.join(timeout=max(0, join_deadline-time.monotonic()))
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
