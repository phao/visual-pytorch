# Local worker transport contract

This specifies `app.transport.run_json_worker`, used by the Qt `TrainingJob`.
It is a local, single-direct-child process transport, not a detached job service
or a sandbox against malicious code. It does not implement roadmap Stage E.

## Request and admission

The caller supplies an argv sequence, not a shell command. `cwd` is passed
literally. One finite JSON request is serialized as UTF-8 before launch, then
written to stdin; stdin closes at the end. Invalid serialization fails before
creating a process. No credentials or user workspaces are used in transport tests.

`max_line_bytes` and `stderr_limit` must be positive integers, excluding booleans.
`timeout`, when supplied, and `grace_seconds`/`kill_seconds` must be finite positive
numbers, excluding booleans. The elapsed deadline starts immediately before
launch. JSON serialization precedes it. Launch itself and synchronous callbacks
are trusted and are not preempted by the loop deadline; callbacks must not block.
The Qt caller currently supplies no total wall deadline: long authorized runs
remain possible, and Stop initiates the termination policy.

## Stream framing and success

Stdout is UTF-8 JSON Lines. Every line must end in LF and decode to an object with
exactly `kind` and `payload`. Accepted event kinds:

- `progress`: object payload, delivered incrementally and in order;
- `result`: object payload, exactly once and last;
- `error`: diagnostic payload, terminates the request with `WorkerError`.

Unknown kinds, wrong envelope/payload shapes, invalid JSON/UTF-8, blank lines,
BOM/UTF-16 auto-detection, duplicate object keys at any depth, non-finite numbers
(including overflowing exponents), and unpaired Unicode surrogate escapes are
rejected. A result does not prove success until stdout/input complete and the
child exits with status zero. Trailing events, missing results, nonzero exit and
failed request delivery cannot be masked by an earlier valid-looking result.
Numerical training-result semantics are validated separately, not by this generic
transport envelope.

Three threads service stdout, stderr and stdin concurrently from launch. This
avoids the pipe dependency cycle when a child fills output before consuming a
large request. Stdout uses a bounded queue (32 entries) and a per-line byte limit;
stderr keeps a bounded trailing byte window rather than the entire diagnostic
stream. Defaults are 256 MiB per stdout line and 64 KiB stderr tail; an error
message includes at most the final 1,000 decoded stderr characters. This is a
finite buffering contract, not a hard RAM sandbox: queue entries, JSON decoding,
request/result copies and model allocations also consume memory. Run admission
uses the separate memory policy.

## Termination and ownership

Cancellation sends SIGTERM; after `grace_seconds` (default 5 s), SIGKILL follows
if needed. A cooperative worker may return a partial result with exit zero.
Cancellation has an additional `kill_seconds` deadline (default 3 s). Protocol,
callback, thread-start and execution-deadline failures also run cleanup:
terminate, bounded wait, kill, reap the direct child, join pipe threads and close
streams. Callback exceptions retain their original type; protocol/admission,
deadline and unsuccessful-exit errors use `WorkerError`.

This contract assumes the direct worker owns its pipes and does not create
grandchildren that inherit and retain them. Such descendants can outlive the
worker and prevent stream closure; process-group/container supervision belongs
to a future execution backend. No claim is made that pipe backpressure explains
the user's earlier workstation freeze.

## Evidence

`tests/test_transport.py` adds 26 focused tests using bounded private children and
no Torch/Qt dependency. Fixtures include 1 MiB stdin, 2 MiB stderr and 400 queued
progress events before stdin consumption; a large finite result at the exact
line boundary; cooperative TERM and ignored-TERM escalation; blocked input and
partial output deadlines; early-result/input-failure races; malformed protocols;
callback/startup cleanup; thread/stream closure; and literal cwd/shell-symbol data.
Tests reap even a leaking implementation's children before reporting the defect.
One initial large-result fixture exceeded the OS argv limit; it was corrected to
generate its result inside the child, not treated as an application failure.

After repair, the full 145-test Python suite and 54-test JavaScript suite pass.
The real PyQt training signal path with the separate Torch interpreter also
passes a bounded linear fit (81 progress points, held-out MSE about 1.19e-12).
These checks do not establish full Stage A checkpoint/binding/GUI correctness.

The design follows Python's documented pipe backpressure and cleanup warnings:
[subprocess documentation](https://docs.python.org/3/library/subprocess.html).
Strict JSON handling is intentional because Python otherwise accepts repeated
object names and non-standard NaN/Infinity:
[JSON documentation](https://docs.python.org/3/library/json.html).
