"""Pure exact-revision editor queries and isolated execution capture."""
import copy
from .core import GraphError
from .evidence import digest
from .experiment_objects import candidate, dependencies, reference, report_data
from .datasets import _identifier


def index(workspace):
    rows = workspace.list_specs(all_revisions=True)
    return sorted(rows, key=lambda r: (r['kind'], r['name'].casefold(), r['id'], r['revision']))


def capture(workspace, experiment_ref, coordinates, run_id, name, *, retry=False):
    """Capture only selected dependencies, never an unrelated dataset collection."""
    reference(experiment_ref, 'experiment')
    if not _identifier(run_id) or not _identifier(name) or type(retry) is not bool:
        raise GraphError('Run ID/name and boolean retry must be explicit and valid.')
    experiment = workspace.get_spec('experiment', experiment_ref['id'], experiment_ref['revision'])
    old = [r for r in workspace.list_specs('run', True) if r['id'] == run_id]
    if old and not retry:
        raise GraphError('Run ID already exists; use explicit retry.')
    if retry and not old:
        raise GraphError('Retry requires an existing run, not a new ID.')
    records = [workspace.get_spec('run', run_id, r['revision']) for r in old]
    previous = max(records, key=lambda r: len(r['attempts'])) if records else None
    if previous and digest(previous['selection']['experiment']) != digest(experiment_ref):
        raise GraphError('Retry cannot change its experiment revision.')
    if previous and digest(previous['selection']['coordinates']) != digest(coordinates):
        raise GraphError('Retry cannot change its selected coordinates.')
    manifest = previous['resolved']['manifest'] if previous and previous['resolved'] else None
    selected = candidate(experiment, coordinates, workspace.get_spec, saved_manifest=manifest)
    documents = []; seen = set()
    def visit(ref):
        key = tuple(ref[k] for k in ('kind', 'id', 'revision'))
        if key in seen: return
        seen.add(key); doc = workspace.get_spec(*key)
        for dep in dependencies(ref['kind'], doc): visit(dep)
        documents.append({'kind': ref['kind'], 'document': doc})
    visit(experiment_ref)
    # Longest immutable history is sufficient; no mutable head substitution.
    if previous:
        visit({'kind': 'run', 'id': run_id, 'revision': previous['revision']})
    return copy.deepcopy({'experiment': experiment_ref, 'coordinates': coordinates,
        'run_id': run_id, 'name': name, 'retry': retry, 'documents': documents,
        'manifest': selected['manifest'] if selected['status'] == 'compatible' else None,
        'status': selected['status'], 'reason': selected['reason'],
        'origin': {'workspace_id': workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0],
                   'path': workspace.path, 'disk_revision': workspace.disk_revision},
        'constrained': workspace.constrained_mode})


def result_report(workspace, run_refs, *, sid, name, question):
    """Create a reading recipe; a report is never an implicit training request."""
    return workspace.put_spec('report', {'version': 1, 'id': sid, 'name': name,
        'question': question, 'mode': 'result-report', 'runs': run_refs,
        'experiment': None, 'sections': ['question', 'models', 'datasets', 'protocol',
            'training', 'evaluation', 'failures', 'environment', 'checkpoints'],
        'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})


def read_report(workspace, ref):
    reference(ref, 'report')
    return report_data(workspace.get_spec('report', ref['id'], ref['revision']), workspace.get_spec)
