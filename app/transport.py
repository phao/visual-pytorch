"""Bounded worker protocol, concurrent pipe draining and cancellation escalation."""
import json
import queue
import signal
import subprocess
import threading
import time


class WorkerError(ValueError):
    pass


def run_json_worker(command, request, on_progress=None, cancel=None, *, cwd=None,
                    on_process=None, timeout=None, grace_seconds=5, kill_seconds=3,
                    max_line_bytes=256*1024**2, stderr_limit=64*1024):
    """No shell. Drain both streams from launch, including while writing stdin.

    Progress is incremental. Stderr retains only a bounded tail. Cancellation
    sends TERM then KILL; after a protocol failure cleanup is also bounded.
    This is a process transport, not a protection against hostile grandchildren.
    """
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (max_line_bytes,stderr_limit)):
        raise WorkerError('Protocol byte limits must be positive integers.')
    if grace_seconds<=0 or kill_seconds<=0 or timeout is not None and timeout<=0:
        raise WorkerError('Worker deadlines must be positive.')
    events=queue.Queue(maxsize=32)
    finished=threading.Event()
    stderr=bytearray()
    stderr_lock=threading.Lock()
    process=subprocess.Popen(command,cwd=cwd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    if on_process:on_process(process)
    def enqueue(kind,value=None):
        while not finished.is_set():
            try:events.put((kind,value),timeout=.05);return
            except queue.Full:pass
    def read_stdout():
        try:
            while True:
                line=process.stdout.readline(max_line_bytes+1)
                if not line:break
                if len(line)>max_line_bytes:raise WorkerError('Worker protocol line exceeds its byte limit.')
                if not line.endswith(b'\n'):raise WorkerError('Worker protocol ended with a truncated line.')
                enqueue('line',line)
        except Exception as exc:enqueue('failure',str(exc))
        finally:enqueue('stdout_end')
    def read_stderr():
        try:
            while True:
                chunk=process.stderr.read1(8192)
                if not chunk:break
                with stderr_lock:
                    stderr.extend(chunk)
                    if len(stderr)>stderr_limit:del stderr[:-stderr_limit]
        except (OSError,ValueError):pass
    def write_request():
        try:
            payload=json.dumps(request,allow_nan=False).encode('utf-8')
            process.stdin.write(payload);process.stdin.close()
        except Exception as exc:enqueue('failure','Worker input failed: '+str(exc))
    readers=[threading.Thread(target=f,daemon=True) for f in (read_stdout,read_stderr,write_request)]
    for thread in readers:thread.start()
    started=time.monotonic();termination=None;stdout_end=False;result=None
    try:
        while not stdout_end or process.poll() is None:
            now=time.monotonic()
            if timeout is not None and now-started>timeout:raise WorkerError('Worker execution deadline exceeded.')
            if cancel is not None and cancel.is_set() and termination is None:
                termination=now
                if process.poll() is None:process.send_signal(signal.SIGTERM)
            if termination is not None and now-termination>grace_seconds and process.poll() is None:
                process.kill()
            if termination is not None and now-termination>grace_seconds+kill_seconds:
                raise WorkerError('Cancelled worker did not finish within its termination deadline.')
            try:kind,payload=events.get(timeout=.05)
            except queue.Empty:continue
            if kind=='failure':raise WorkerError(payload)
            if kind=='stdout_end':stdout_end=True;continue
            def nonfinite(value):raise WorkerError('Non-finite number in worker protocol: '+value)
            event=json.loads(payload,parse_constant=nonfinite)
            json.dumps(event,allow_nan=False)  # Reject overflowed JSON exponents too.
            if not isinstance(event,dict) or set(event)!={'kind','payload'}:
                raise WorkerError('Malformed worker event.')
            if result is not None:raise WorkerError('Worker emitted an event after its result.')
            if event['kind']=='progress':
                if not isinstance(event['payload'],dict):raise WorkerError('Malformed progress payload.')
                if on_progress:on_progress(event['payload'])
            elif event['kind']=='result':
                if not isinstance(event['payload'],dict):raise WorkerError('Malformed result payload.')
                result=event['payload']
            elif event['kind']=='error':raise WorkerError(str(event['payload']))
            else:raise WorkerError('Unknown worker event kind.')
        readers[1].join(timeout=kill_seconds)
        tail=bytes(stderr).decode('utf-8',errors='replace')
        if process.returncode!=0:raise WorkerError('Worker exited with status '+str(process.returncode)+': '+tail[-1000:])
        if result is None:raise WorkerError('Worker stopped without a result: '+tail[-1000:])
        return result
    finally:
        finished.set()
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=kill_seconds)
        for thread in readers:thread.join(timeout=kill_seconds)
        for stream in (process.stdin,process.stdout,process.stderr):
            stream.close()
