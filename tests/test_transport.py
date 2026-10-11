"""Real private child processes, bounded fixtures, no Qt/Torch dependency."""
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from app.transport import WorkerError, run_json_worker


class TransportTests(unittest.TestCase):
    def run_child(self, source, request=None, **options):
        processes = []
        hook = options.pop('on_process', None)

        def launched(process):
            processes.append(process)
            if hook:
                hook(process)

        options.setdefault('timeout', 5)
        options.setdefault('grace_seconds', .2)
        options.setdefault('kill_seconds', .5)
        try:
            return run_json_worker([sys.executable, '-u', '-c', source],
                                   {} if request is None else request,
                                   on_process=launched, **options)
        finally:
            # Test ownership safety even when the old implementation leaks.
            leaked = [p for p in processes if p.poll() is None]
            for process in leaked:
                process.kill()
                process.wait(timeout=2)
            for process in processes:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if not stream.closed:
                        stream.close()
            self.assertFalse(leaked, 'Transport left a private worker alive.')

    def raw_child(self, data, **options):
        return self.run_child('import sys; sys.stdin.buffer.read(); '
                              'sys.stdout.buffer.write('+repr(data)+'); '
                              'sys.stdout.buffer.flush()', **options)

    def test_pipe_backpressure_before_large_request_is_read(self):
        source = '''
import json, sys
sys.stderr.buffer.write(b'E' * (2 * 1024 * 1024)); sys.stderr.buffer.flush()
for i in range(400):
    print(json.dumps({'kind':'progress','payload':{'i':i,'text':'p'*2048}}), flush=True)
request=json.load(sys.stdin)
print(json.dumps({'kind':'result','payload':{'length':len(request['text'])}}), flush=True)
'''
        events = []
        result = self.run_child(source, {'text': 'x' * (1024*1024)},
                                on_progress=lambda p: events.append(p['i']),
                                max_line_bytes=8192, stderr_limit=2048)
        self.assertEqual(events, list(range(400)))
        self.assertEqual(result, {'length': 1024*1024})

    def test_progress_is_delivered_while_worker_is_alive(self):
        processes, live = [], []
        result = self.run_child('''
import json, sys, time
json.load(sys.stdin)
print('{"kind":"progress","payload":{"step":1}}', flush=True)
time.sleep(.15)
print('{"kind":"result","payload":{"ok":true}}', flush=True)
''', on_process=processes.append,
            on_progress=lambda p: live.append((p['step'], processes[0].poll())))
        self.assertEqual(result, {'ok': True})
        self.assertEqual(live, [(1, None)])

    def test_large_finite_result_and_exact_line_boundary(self):
        line = json.dumps({'kind':'result','payload':{'values':list(range(25000))}}).encode()+b'\n'
        source = ('import json,sys;sys.stdin.read();'
                  'print(json.dumps({"kind":"result","payload":{"values":list(range(25000))}}),flush=True)')
        self.assertEqual(len(self.run_child(source, max_line_bytes=len(line))['values']), 25000)
        with self.assertRaisesRegex(WorkerError, 'byte limit'):
            self.run_child(source, max_line_bytes=len(line)-1)

    def test_missing_truncated_and_oversized_result(self):
        for data, error in [(b'', 'without a result'),
                            (b'{"kind":"result","payload":{}}', 'truncated'),
                            (b'x'*257+b'\n', 'byte limit')]:
            with self.subTest(data=data[:40]), self.assertRaisesRegex(WorkerError, error):
                self.raw_child(data, max_line_bytes=256)

    def test_malformed_json_and_event_shapes_are_worker_errors(self):
        bad = [b'not-json\n', b'\xff\n', b'[]\n', b'null\n',
               b'{"kind":"unknown","payload":{}}\n',
               b'{"kind":"progress","payload":[]}\n',
               b'{"kind":"result","payload":0}\n',
               b'{"kind":"result","payload":{},"extra":1}\n']
        for data in bad:
            with self.subTest(data=data), self.assertRaises(WorkerError):
                self.raw_child(data)

    def test_duplicate_keys_are_rejected_in_envelope_and_nested_payload(self):
        bad = [b'{"kind":"error","kind":"result","payload":{}}\n',
               b'{"kind":"result","payload":{},"payload":{"ok":true}}\n',
               b'{"kind":"result","payload":{"x":1,"x":2}}\n']
        for data in bad:
            with self.subTest(data=data), self.assertRaisesRegex(WorkerError, 'Duplicate'):
                self.raw_child(data)

    def test_nonfinite_numbers_are_rejected_at_any_depth(self):
        for value in ('NaN', 'Infinity', '-Infinity', '1e999', '-1e999'):
            with self.subTest(value=value), self.assertRaises(WorkerError):
                self.raw_child(('{"kind":"result","payload":{"nested":['+value+']}}\n').encode())

    def test_utf8_only_not_utf16_auto_detection(self):
        line = '{"kind":"result","payload":{}}\n'.encode('utf-16')
        with self.assertRaises(WorkerError):
            self.raw_child(line)

    def test_unicode_utf8_payload(self):
        value = {'text':'Σ → λ, ação'}
        line = json.dumps({'kind':'result','payload':value}, ensure_ascii=False).encode()+b'\n'
        self.assertEqual(self.raw_child(line), value)

    def test_result_must_be_last_and_unique(self):
        result = b'{"kind":"result","payload":{}}\n'
        for event in (result, b'{"kind":"progress","payload":{}}\n',
                      b'{"kind":"error","payload":"late"}\n'):
            with self.subTest(event=event), self.assertRaisesRegex(WorkerError, 'after its result'):
                self.raw_child(result+event)

    def test_nonzero_exit_cannot_be_success_even_with_result(self):
        with self.assertRaisesRegex(WorkerError, 'status 17'):
            self.run_child('import sys; sys.stdin.read(); '
                           'print(\'{"kind":"result","payload":{}}\',flush=True); sys.exit(17)')

    def test_stderr_flood_retains_bounded_final_diagnostic(self):
        with self.assertRaises(WorkerError) as raised:
            self.run_child('import sys; sys.stdin.read(); '
                           'sys.stderr.write("old-marker"+"e"*2000000+"tail-marker"); '
                           'sys.stderr.flush(); sys.exit(4)', stderr_limit=64)
        self.assertIn('tail-marker', str(raised.exception))
        self.assertNotIn('old-marker', str(raised.exception))
        self.assertLess(len(str(raised.exception)), 200)

    def test_launch_callback_exception_reaps_child(self):
        def callback(process):
            raise RuntimeError('launch hook failure')
        with self.assertRaisesRegex(RuntimeError, 'launch hook failure'):
            self.run_child('import time; time.sleep(20)', on_process=callback)

    def test_progress_callback_exception_reaps_child(self):
        def callback(payload):
            raise RuntimeError('progress hook failure')
        with self.assertRaisesRegex(RuntimeError, 'progress hook failure'):
            self.run_child('import sys,time;sys.stdin.read();'
                           'print(\'{"kind":"progress","payload":{}}\',flush=True);'
                           'time.sleep(20)', on_progress=callback)

    def test_deadline_stops_silent_worker_or_worker_after_result(self):
        for prefix in ('', 'print(\'{"kind":"result","payload":{}}\',flush=True);'):
            started = time.monotonic()
            with self.subTest(prefix=prefix), self.assertRaisesRegex(WorkerError, 'deadline'):
                self.run_child('import sys,time;sys.stdin.read();'+prefix+'time.sleep(20)', timeout=.15)
            self.assertLess(time.monotonic()-started, 2)

    @unittest.skipUnless(os.name=='posix', 'POSIX SIGTERM handler fixture')
    def test_cooperative_cancel_can_return_partial_checkpoint(self):
        cancel = threading.Event()
        result = self.run_child('''
import signal,sys,time
def stop(*args):
    print('{"kind":"result","payload":{"cancelled":true,"updates":3}}',flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
sys.stdin.read()
print('{"kind":"progress","payload":{"ready":true}}',flush=True)
while True: time.sleep(.01)
''', cancel=cancel, on_progress=lambda p: cancel.set())
        self.assertEqual(result, {'cancelled': True, 'updates': 3})

    @unittest.skipUnless(os.name=='posix', 'POSIX signal escalation fixture')
    def test_cancel_escalates_when_worker_ignores_term(self):
        cancel = threading.Event()
        processes = []
        started = time.monotonic()
        with self.assertRaises(WorkerError):
            self.run_child('''
import signal,sys,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
sys.stdin.read()
print('{"kind":"progress","payload":{"ready":true}}',flush=True)
while True: time.sleep(.01)
''', cancel=cancel, on_progress=lambda p: cancel.set(), on_process=processes.append)
        self.assertEqual(processes[0].returncode, -signal.SIGKILL)
        self.assertLess(time.monotonic()-started, 2)

    def test_explicit_cooperative_cancel_hook_runs_once_and_can_preserve_result(self):
        cancel = threading.Event(); calls = []
        with tempfile.TemporaryDirectory() as folder:
            marker = os.path.join(folder, 'stop')
            def stop(process):
                calls.append(process.pid)
                with open(marker, 'x', encoding='utf-8') as stream: stream.write('stop')
            result = self.run_child('''
import json,sys,time,os
r=json.load(sys.stdin)
print('{"kind":"progress","payload":{"ready":true}}',flush=True)
while not os.path.exists(r['marker']): time.sleep(.005)
print('{"kind":"result","payload":{"stopped":true}}',flush=True)
''', {'marker':marker}, cancel=cancel, on_progress=lambda p:cancel.set(), on_cancel=stop)
        self.assertEqual(result, {'stopped':True}); self.assertEqual(len(calls),1)

    def test_cancel_hook_failure_reaps_worker_and_does_not_fabricate_result(self):
        cancel = threading.Event()
        def fail(process): raise RuntimeError('private-stop-hook-failure')
        with self.assertRaisesRegex(RuntimeError, 'private-stop-hook-failure'):
            self.run_child("import sys,time;sys.stdin.read();print('{\"kind\":\"progress\",\"payload\":{}}',flush=True);time.sleep(20)",
                           cancel=cancel,on_progress=lambda p:cancel.set(),on_cancel=fail)

    def test_input_failure_is_not_ignored(self):
        with self.assertRaisesRegex(WorkerError, 'input failed'):
            self.run_child('import os,time;os.close(0);time.sleep(20)',
                           {'data':'a'*1024*1024})

    def test_invalid_limits_and_request_fail_before_launch(self):
        cases = [{'timeout':float('nan')}, {'timeout':float('inf')}, {'timeout':True},
                 {'grace_seconds':float('nan')}, {'kill_seconds':float('inf')},
                 {'timeout':0}, {'grace_seconds':-1}, {'kill_seconds':False},
                 {'max_line_bytes':False}, {'max_line_bytes':0}, {'stderr_limit':1.5}]
        for options in cases:
            with self.subTest(options=options), patch('app.transport.subprocess.Popen') as launch:
                launch.side_effect = AssertionError('Worker launched with invalid admission arguments')
                with self.assertRaises(WorkerError):
                    run_json_worker([sys.executable,'-c','pass'], {}, **options)
                launch.assert_not_called()
        for request in ({'x':float('nan')}, {'x':object()}, {'x':'\ud800'}):
            with self.subTest(request=request), patch('app.transport.subprocess.Popen') as launch:
                launch.side_effect = AssertionError('Worker launched with unserializable request')
                with self.assertRaises(WorkerError):
                    run_json_worker([sys.executable,'-c','pass'], request)
                launch.assert_not_called()

    def test_pipe_threads_finish_and_streams_close(self):
        baseline = set(threading.enumerate())
        processes = []
        self.raw_child(b'{"kind":"result","payload":{}}\n', on_process=processes.append)
        self.assertEqual(set(threading.enumerate()), baseline)
        self.assertTrue(all(s.closed for s in (processes[0].stdin,processes[0].stdout,processes[0].stderr)))

    def test_early_result_cannot_mask_broken_request_delivery(self):
        with self.assertRaisesRegex(WorkerError, 'input failed'):
            self.run_child('import os;os.close(0);'
                           'print(\'{"kind":"result","payload":{"ok":true}}\',flush=True)',
                           {'data':'x'*1024*1024})

    def test_deadline_also_covers_blocked_request_write_and_partial_line(self):
        for prefix in ('', 'sys.stdout.write(\'{"kind":"result"\');sys.stdout.flush();'):
            started = time.monotonic()
            with self.subTest(prefix=prefix), self.assertRaisesRegex(WorkerError, 'deadline'):
                self.run_child('import sys,time;'+prefix+'time.sleep(20)',
                               {'data':'x'*1024*1024}, timeout=.15)
            self.assertLess(time.monotonic()-started, 2)

    def test_error_event_is_surfaced_and_worker_reaped(self):
        with self.assertRaisesRegex(WorkerError, 'private-worker-error'):
            self.run_child('import sys,time;sys.stdin.read();'
                           'print(\'{"kind":"error","payload":"private-worker-error"}\',flush=True);'
                           'time.sleep(20)')

    def test_surrogate_escape_blank_line_and_bom_are_rejected(self):
        for line in (b'{"kind":"result","payload":{"x":"\\ud800"}}\n',
                     b'\n', b'\xef\xbb\xbf{"kind":"result","payload":{}}\n'):
            with self.subTest(line=line), self.assertRaises(WorkerError):
                self.raw_child(line)

    def test_cwd_with_spaces_and_request_shell_symbols_are_literal(self):
        with tempfile.TemporaryDirectory(prefix='vtw worker ') as directory:
            text = '; touch no-such-marker | $(false) `false`'
            result = self.run_child('import json,sys,os;request=json.load(sys.stdin);'
                                    'print(json.dumps({"kind":"result","payload":'
                                    '{"cwd":os.getcwd(),"text":request["text"]}}),flush=True)',
                                    {'text':text}, cwd=directory)
            self.assertEqual(result, {'cwd':directory,'text':text})
            self.assertEqual(os.listdir(directory), [])

    def test_reader_thread_start_failure_reaps_child(self):
        start = threading.Thread.start
        def fail_stderr(thread):
            if thread.name == 'vtw-worker-stderr':
                raise RuntimeError('private-thread-start-failure')
            return start(thread)
        baseline = set(threading.enumerate())
        with patch('app.transport.threading.Thread.start', new=fail_stderr):
            with self.assertRaisesRegex(RuntimeError, 'private-thread-start-failure'):
                # Hook cannot run before this error, so capture Popen directly.
                from app.transport import subprocess
                launch = subprocess.Popen
                processes = []
                def launched(*args, **kwargs):
                    process = launch(*args, **kwargs)
                    processes.append(process)
                    return process
                with patch('app.transport.subprocess.Popen', side_effect=launched):
                    try:
                        run_json_worker([sys.executable,'-c','import time;time.sleep(20)'],
                                        {}, grace_seconds=.2,kill_seconds=.5,timeout=2)
                    finally:
                        leaked = [p for p in processes if p.poll() is None]
                        for p in leaked:
                            p.kill();p.wait(timeout=2)
                        self.assertFalse(leaked)
        self.assertEqual(set(threading.enumerate()), baseline)


if __name__ == '__main__':
    unittest.main()
