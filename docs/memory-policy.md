# Constrained execution: RAM admission policy, estimator version 2

This specification matches app/memory.py and app/core.py. It governs admission
to the current float32 CPU prototype, not graph validity or file format grammar.
Editing, validating, saving and exporting a large valid architecture do not
allocate its parameter tensors and remain permitted.

## Observation and decision

Each preflight observes Linux /proc/meminfo MemAvailable, in bytes, excluding
swap. If a unified cgroup-v2 membership is observable, cap that observation by
max(0, memory.max − memory.current) for every accessible ancestor with a finite
limit. Hidden ancestors cannot be inspected. Unknown MemAvailable or an
unreadable known limit returns unknown RAM and blocks constrained execution.
This is currently a Linux observer, not a portable cross-platform RAM API.

The definition follows the [kernel proc documentation](https://www.kernel.org/doc/html/v6.16/filesystems/proc.html)
and [cgroup-v2 memory interface](https://docs.kernel.org/admin-guide/cgroup-v2.html).

Let R be observed available bytes and q the user percentage. Budget is
floor(R × q / 100). Default q is 10.0. Config validation permits finite numeric
0 < q ≤ 100; the native control permits 0.01–100 with two decimal places.
Options → Run safety persists memory_limit_percent in ~/.tensorloon/config.json.
The per-workspace constrained_mode boolean is stored in .vtw metadata.

Constrained admission succeeds iff estimate ≤ budget. Above budget or unknown
RAM blocks; unconstrained override leaves violations visible but admits a
semantically valid run. Disable/re-enable never shrinks a model or deletes data.
Host and isolated worker capture the same mode/percentage; the worker rechecks
before importing Torch, and training checks again before tensor allocation.
Changing safety policy during training is prohibited. Validate refreshes the UI
observation; the displayed observation is not a reservation of future memory.

## Whole-network statistics

P: number of learned parameters, including bias and all independent reference
occurrences. Shared-position Linear uses one weight matrix, not one per position.
I/O: canonical input/prediction sample element counts, excluding runtime B.
S: sum of the element counts of each topological node's outputs per sample,
including expanded referenced internals once per occurrence and their outer
boundary outputs. Router aliases/views and boundary identities are counted too;
this deliberately overcounts storage rather than assuming safe buffer reuse.
N: dataset rows; training batch B = max(1, floor(0.8 N)) in the implementation.
K: sum over expanded Conv2d occurrences of C_in × kH × kW × H_out × W_out.
This counts a full-channel unfolding workspace per sample, not a dense global
convolution matrix. Grouped kernels still reserve full-channel scratch.

These symbolic statistics are computed without constructing tensors. Input
width alone is not the criterion: a tiny input with huge hidden layers may be
blocked, and thousands of inputs with a small model may fit.

## Exact estimated byte components

One MiB is 1048576 bytes. Sum all applicable components:

| Operation | Component | Bytes |
| --- | --- | --- |
| Model construction | Runtime headroom | 64 MiB |
| Model construction | Parameter construction | 8 P |
| Model construction | Forward activations | 8 S |
| Model construction | Convolution workspace | 8 K |
| Synthetic data | Runtime headroom | 8 MiB |
| Synthetic data | Dataset and copies | 128 N (I + O) |
| Synthetic data | Affine teacher | 128 (I O + O) |
| Synthetic data | Nonlinear or spatial teacher allowance, instead of affine teacher | 128 (4 I + 5 O + 4) |
| Training | Runtime headroom | 256 MiB |
| Training | Parameters, optimizer and result evidence | 128 P |
| Training | Activations and backward | 16 B S |
| Training | Convolution workspace | 32 B K |
| Training | Dataset and copies | 128 N (I + O) |
| Training | Compact affine report, only matrix_affine graphs | 128 (I O + O) |

Auto generation selects the spatial teacher for eligible CNNs, otherwise affine
iff contract.affine. The generator passes its resolved kind to preflight.
Explicit teacher kind overrides that choice. Data admission estimates data/teacher allocations, not a concurrent
training job. Model admission estimates construction and one sample. Training
admission includes the whole network, batch, data, optimizer and result copies.

The 128 P coefficient includes 24 bytes/parameter for float32 weights, gradients,
Adam moments/scratch and 104 bytes/parameter for Python lists, JSON and host/worker
evidence copies. Adam memory behavior motivates accounting beyond weights alone;
see [PyTorch Adam](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html).
Coefficients are engineering allowances, not measurements promised by PyTorch.
Loss samples are decimated by the training loop (roughly 80 points), not stored
for every epoch; epochs therefore are not a separate fixed memory cap.

Report fields: version=2, operation=model/data/training, batch, parameter_bytes=4P,
components, estimated_bytes, available_bytes, budget_bytes and limit_percent.
Safety report also supplies constrained, allowed and violations with rule
memory or memory_unavailable. The UI shows the total estimate, observed RAM and
budget; compiler/data/training entry points repeat admission checks.

## Limits and non-guarantees

There is no former 64-feature, 100000-parameter, fixed-row, epoch or work-product
execution cap. Structural graph/transport limits remain, independently of mode.
One CPU worker thread and Stop remain; this policy does not estimate or limit
CPU time. A very long fitting run can still monopolize that worker until stopped.

The estimate is conservative by construction but is NOT an OOM guarantee,
allocator quota or defense against hostile workspace input. Runtime libraries,
allocator fragmentation, archived workspace evidence, concurrent processes and
RAM changes can exceed allowances. Imported data/JSON and the in-memory SQLite
workspace already occupy memory when available RAM is observed. Exported Python
executed elsewhere does not enforce this admission policy. Conv scratch
coefficients reserve space for a conventional unfolded implementation and
backward scratch; actual Torch backend algorithms can allocate differently.
Revisit and measure the estimator before adding new operator families, GPUs or
another optimizer.
