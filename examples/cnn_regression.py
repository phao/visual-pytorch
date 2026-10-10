"""Create a reproducible CNN workspace; optional bounded one-thread training.

Run from repository root: python -m examples.cnn_regression /path/to/new.vtw
Add --train to train, store the inference checkpoint and historical run.
Never overwrites an existing workspace. No Qt import.
"""
import argparse
import json
from pathlib import Path
from app.core import Workspace, default_cnn, generate_dataset, train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    path = Path(args.workspace).expanduser().resolve()
    if path.suffix.lower() != '.vtw':
        parser.error('Choose a new .vtw filename.')
    if path.exists():
        parser.error('Refusing to overwrite an existing file.')
    workspace = Workspace()
    try:
        doc = default_cnn(); mid = doc['id']; models = {mid: doc}
        doc['dataset'] = generate_dataset(models, mid, samples=128, kind='spatial')
        workspace.put(doc)
        # Reserve the new target before an optional fit; leave recoverable input
        # if training is interrupted. Save checks races under its sidecar lock.
        workspace.save(path, require_new=True)
        if args.train:
            import torch
            torch.set_num_threads(1)
            snapshot = workspace.models()
            result = train_model(snapshot, mid)
            trained = workspace.models()[mid]; trained['result'] = result; workspace.put(trained)
            workspace.record_run(mid, result, snapshot); workspace.save(path)
        print(json.dumps({'path': str(path), 'id': mid, 'trained': args.train,
                          'mse': workspace.models()[mid].get('result', {}).get('mse') if args.train else None}))
    finally:
        workspace.db.close()


if __name__ == '__main__':
    main()
