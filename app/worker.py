"""Isolated PyTorch worker; stdin request and stdout JSON Lines events."""
import json
import signal
import sys
import threading

from .core import train_model
from .memory import DEFAULT_MEMORY_LIMIT_PERCENT


def main():
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())

    def send(kind, payload):
        print(json.dumps({"kind": kind, "payload": payload}, allow_nan=False), flush=True)

    try:
        request = json.load(sys.stdin)
        if request.get('kind') == 'task-v1':
            from .datasets import _keys
            from .task_training import train_task
            _keys(request, ('kind', 'bundle'), ('constrained', 'memory_limit_percent'), 'Task worker request')
            result = train_task(request['bundle'], progress=lambda point: send('progress', point), cancel=cancel,
                                constrained=request.get('constrained', True),
                                memory_limit_percent=request.get('memory_limit_percent', DEFAULT_MEMORY_LIMIT_PERCENT))
            send('result', result)
            return 0
        if 'kind' in request:
            raise ValueError('Unsupported worker request kind.')
        # Reject unsafe jobs before importing Torch or allocating model/data tensors.
        from .core import preflight
        doc = request["models"][request["id"]]
        data = doc.get("dataset")
        constrained = request.get("constrained", True)
        percentage = request.get('memory_limit_percent', DEFAULT_MEMORY_LIMIT_PERCENT)
        preflight(request["models"], request["id"], constrained,
                  samples=len(data["x"]) if isinstance(data, dict) and isinstance(data.get("x"), list) else None,
                  epochs=doc.get("training", {}).get("epochs", 400),
                  memory_limit_percent=percentage)
        import torch
        # Small CPU prototypes do not benefit from taking every desktop core.
        torch.set_num_threads(1)
        result = train_model(request["models"], request["id"],
                             progress=lambda point: send("progress", point), cancel=cancel,
                             constrained=constrained, memory_limit_percent=percentage)
        send("result", result)
    except Exception as exc:
        send("error", str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
