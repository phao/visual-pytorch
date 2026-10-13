"""Replay the actual GUI-frozen LoRA candidate through CLI in a private copy."""
import json
import subprocess
import sys
from pathlib import Path
from app.core import Workspace


def main(argv=None):
    source, gui_output, output = map(Path, sys.argv[1:] if argv is None else argv)
    output = output.resolve(); output.mkdir(exist_ok=False)
    snapshot = json.loads((gui_output / 'worker-snapshot.json').read_text())
    gui = json.loads((gui_output / 'worker-result.json').read_text())
    original = source.read_bytes(); w = Workspace.load(source)
    try: w.save(output / 'cli.vtw', require_new=True)
    finally: w.db.close()
    coordinates = output / 'coordinates.json'
    with coordinates.open('x') as stream: json.dump(snapshot['coordinates'], stream)
    root = Path(__file__).resolve().parents[1]
    ref = snapshot['experiment']
    command = [sys.executable, str(root / 'vpw_cli.py'), str(output / 'cli.vtw'), 'run-candidate',
        '--experiment', ref['id'], ref['revision'], '--coordinates', str(coordinates),
        '--run-id', 'cli-lora-acceptance', '--name', 'Explicit GUI-frozen LoRA CLI parity',
        '--torch-python', sys.executable, '--timeout', '90']
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    with (output / 'cli-stdout.json').open('x') as stream: stream.write(process.stdout)
    with (output / 'cli-stderr.txt').open('x') as stream: stream.write(process.stderr)
    assert process.returncode == 0, process.stderr + process.stdout[:1000]
    answer = json.loads(process.stdout); assert answer['ok'] and answer['value']['persisted']
    record = answer['value']['record']; actual = record['attempts'][0]
    assert actual['status'] == 'completed' and actual['result']['counts']['completed_updates'] == 200
    assert record['selection']['experiment'] == ref
    assert record['selection']['coordinates'] == snapshot['coordinates']
    assert record['resolved']['manifest'] == snapshot['manifest']
    assert actual['result']['checkpoint_id'] == gui['result']['checkpoint_id']
    assert actual['result']['checkpoint']['parameter_policy_receipt'] == gui['result']['checkpoint']['parameter_policy_receipt']
    assert actual['result']['final_metrics'] == gui['result']['final_metrics']
    assert source.read_bytes() == original
    assert 'torch' not in sys.modules
    print(json.dumps({'status': 'PASS', 'actual_cli_updates': 200,
        'exact_gui_selection_manifest_checkpoint_policy_metrics': True,
        'original_workspace_untouched': True, 'host_torch': 'torch' in sys.modules}), flush=True)


if __name__ == '__main__': main()
