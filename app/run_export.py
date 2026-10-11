"""Exact saved-attempt inference source, generated without numerical execution.

Checkpoint validation checks saved preprocessing through existing contracts.
Generation does not import Torch or run the emitted module.
"""
import hashlib
from pathlib import Path
from .datasets import _integer, _require
from .experiment_objects import candidate_bundle, reference
from .task_execution import generate_task_predictor


def predictor_source(workspace, ref, *, attempt=None, memory_limit_percent=10.):
    reference(ref, 'run')
    record = workspace.get_spec('run', ref['id'], ref['revision'])
    number = len(record['attempts']) if attempt is None else attempt
    _require(_integer(number, 1, len(record['attempts'])),
             'Predictor needs a retained attempt in this exact run revision.')
    selected = record['attempts'][number - 1]; result = selected['result']
    _require(result is not None, 'This retained attempt has no inference checkpoint. Select another attempt explicitly.')
    expref = record['selection']['experiment']
    experiment = workspace.get_spec('experiment', expref['id'], expref['revision'])
    bundle = candidate_bundle(experiment, record['selection']['coordinates'], workspace.get_spec,
                              saved_manifest=record['resolved']['manifest'])
    provenance = {'version': 1, 'reference': dict(ref), 'attempt': number,
                  'status': selected['status'], 'checkpoint_id': result['checkpoint_id'],
                  'completed_updates': result['counts']['completed_updates'],
                  'prediction_semantics': 'stable-output-ids; classification-logits; declared-transformed-target-space',
                  'optimizer_resume': False}
    code = generate_task_predictor(bundle, result['checkpoint'],
        constrained=workspace.constrained_mode, memory_limit_percent=memory_limit_percent)
    code += '\n# Exact retained attempt, not the mutable drawing or latest run head.\n'
    code += 'saved_run_provenance = ' + repr(provenance) + '\n'
    return {'version': 1, 'kind': 'visual-pytorch-saved-predictor-source',
            'scope': 'generate-source-only; no-training-or-inference',
            'provenance': provenance, 'code': code,
            'code_sha256': hashlib.sha256(code.encode('utf-8')).hexdigest()}


def write_new_predictor(source, output):
    """Exclusive UTF-8 creation: never overwrite or follow a final symlink."""
    code = source['code']
    _require(isinstance(code, str) and hashlib.sha256(code.encode('utf-8')).hexdigest() == source['code_sha256'],
             'Predictor source bytes no longer match the generated source hash. Nothing exported.')
    path = Path(output).expanduser().absolute()
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(code)
    return {'version': 1, 'path': str(path), 'provenance': source['provenance'],
            'code_sha256': source['code_sha256'], 'status': 'predictor-source-exported; not-executed'}
