#!/usr/bin/env python3
"""Run the Python unit tests in parallel: one test module per process, workers pinned to CPUs.

The suite's tests are independent files that mostly start workers, fit small networks or run
notebook kernels, so a plain `unittest discover` is dominated by waiting on one core. This runs
each tests/test_*.py module as its own `python -m unittest <module>` (same module names and
PYTHONPATH as discover), a few at a time, each worker pinned to its own CPUs with one thread per
numeric library, so total CPU use stays bounded. Linux only (sched_setaffinity).

  python tools/run_tests_parallel.py                 # whole suite, 4 workers x 2 CPUs
  python tools/run_tests_parallel.py test_core test_mlp_block
  python tools/run_tests_parallel.py --python /path/to/torch/python --max-cpus 4

Notes: tests that need the Qt desktop libraries are skipped under a Torch-only interpreter, as with
discover; run them with the Qt interpreter (see docs/development.md). Per-module logs and timings
(used to start the slowest modules first) live in the Git-ignored .scratch/.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.scratch'
TIMINGS = SCRATCH / 'test-timings.json'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def one_cpu_per_core(limit):
    """Allowed logical CPUs, one per physical core (sibling hyperthreads share a worker's budget)."""
    allowed = sorted(os.sched_getaffinity(0))
    seen, picked = set(), []
    for cpu in allowed:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            key = ((topology / 'physical_package_id').read_text().strip(), (topology / 'core_id').read_text().strip())
        except OSError:
            key = ('0', str(cpu))
        if key not in seen:
            seen.add(key)
            picked.append(cpu)
    return (picked or allowed)[:limit]


def cpu_groups(cpus, per_worker, workers):
    count = min(workers, len(cpus) // per_worker) or 1
    return [cpus[i * per_worker:(i + 1) * per_worker] or cpus[:per_worker] for i in range(count)]


def parse_summary(output, returncode):
    """(tests, skipped, passed) from a unittest run's output and exit status."""
    ran = re.findall(r'^Ran (\d+) tests? in', output, re.M)
    skipped = re.search(r'skipped=(\d+)', output)
    return (int(ran[-1]) if ran else 0, int(skipped.group(1)) if skipped else 0, returncode == 0 and bool(ran))


def modules(root, pattern, wanted):
    found = sorted(p.stem for p in (root / 'tests').glob(pattern))
    if not wanted:
        return found
    unknown = [w for w in wanted if w not in found]
    if unknown:
        raise SystemExit('Unknown test modules: ' + ', '.join(unknown))
    return list(wanted)


def run_module(name, python, cpus, slot, timeout, logs):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'tests')]))
    env.update({variable: '1' for variable in THREAD_VARIABLES})
    tmp = SCRATCH / f'test-tmp-{slot}'
    tmp.mkdir(parents=True, exist_ok=True)
    env['TMPDIR'] = str(tmp)
    started = time.monotonic()
    try:
        done = subprocess.run([python, '-m', 'unittest', name], cwd=ROOT, env=env, text=True, errors='replace',
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                              preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)))
        output, code = done.stdout, done.returncode
    except subprocess.TimeoutExpired as expired:
        output, code = (expired.stdout or '') + f'\nTIMEOUT after {timeout}s', 124
        if isinstance(output, bytes):
            output = output.decode('utf-8', 'replace')
    seconds = time.monotonic() - started
    (logs / f'{name}.log').write_text(output)
    tests, skipped, passed = parse_summary(output, code)
    return {'module': name, 'seconds': seconds, 'tests': tests, 'skipped': skipped, 'passed': passed, 'output': output}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modules', nargs='*', help='test module names without .py (default: all)')
    parser.add_argument('--python', default=sys.executable, help='interpreter for the tests (default: this one)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpus-per-worker', type=int, default=2)
    parser.add_argument('--max-cpus', type=int, default=8, help='total logical CPUs used across workers (default 8)')
    parser.add_argument('--cpus', help='explicit CPU list, e.g. 0,2,4,5,6,7,8,9 (overrides detection)')
    parser.add_argument('--pattern', default='test_*.py')
    parser.add_argument('--timeout', type=int, default=1800, help='seconds allowed per module')
    args = parser.parse_args(argv)
    if min(args.workers, args.cpus_per_worker, args.max_cpus) < 1:
        parser.error('workers, cpus-per-worker and max-cpus must be positive')

    cpus = [int(c) for c in args.cpus.split(',')] if args.cpus else one_cpu_per_core(args.max_cpus)
    groups = cpu_groups(cpus, args.cpus_per_worker, args.workers)
    names = modules(ROOT, args.pattern, args.modules)
    try:
        timings = json.loads(TIMINGS.read_text())
    except (OSError, ValueError):
        timings = {}
    # Unknown or slow modules first, so the long ones overlap with the many short ones.
    names.sort(key=lambda n: -timings.get(n, float('inf')))
    logs = SCRATCH / 'test-runs'
    logs.mkdir(parents=True, exist_ok=True)
    slots = queue.Queue()
    for index in range(len(groups)):
        slots.put(index)
    print(f'{len(names)} modules, {len(groups)} workers on CPUs {" | ".join(",".join(map(str, g)) for g in groups)}',
          flush=True)

    def work(name):
        slot = slots.get()
        try:
            return run_module(name, args.python, groups[slot], slot, args.timeout, logs)
        finally:
            slots.put(slot)

    wall = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures = [pool.submit(work, name) for name in names]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{done:3}/{len(names)}] {'PASS' if result['passed'] else 'FAIL'} {result['module']} "
                  f"{result['seconds']:.1f}s ({result['tests']} tests)", flush=True)
    elapsed = time.monotonic() - wall

    timings.update({r['module']: r['seconds'] for r in results})
    TIMINGS.write_text(json.dumps(timings, indent=1, sort_keys=True))
    failed = sorted((r for r in results if not r['passed']), key=lambda r: r['module'])
    for r in failed:
        print(f"\n===== {r['module']} (log: {logs / (r['module'] + '.log')})")
        print('\n'.join(r['output'].rstrip().splitlines()[-40:]))
    total = sum(r['tests'] for r in results)
    busy = sum(r['seconds'] for r in results)
    print(f"\n{total} tests in {len(results)} modules, {sum(r['skipped'] for r in results)} skipped, "
          f"{len(failed)} module(s) failed; wall {elapsed:.0f}s (sequential sum {busy:.0f}s, "
          f"{busy / elapsed:.1f}x)." if elapsed else '')
    print('slowest: ' + ', '.join(f"{r['module']} {r['seconds']:.0f}s" for r in sorted(results, key=lambda r: -r['seconds'])[:6]))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
