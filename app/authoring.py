"""Optimistic scientific drafts; no fitting, execution or alternate persistence.

Drafts are private editor state, not vtw objects. Admission uses the same closed
version-1 grammar and immutable registry as CLI imports. A historical base is
explicit; its current browsing head is captured separately, never substituted.
"""
import copy
import uuid

from .core import GraphError, validate_model
from .datasets import _identifier, _keys, _require, validate_task
from .experiment_objects import reference, validate_dependencies, preview_experiment
from .revisions import frozen_document

KINDS = ('dataset', 'experiment', 'family', 'task')


def workspace_id(workspace):
    return workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]


def head(workspace, kind, sid):
    row = workspace.db.execute('SELECT revision FROM spec_heads WHERE kind=? AND id=?', (kind, sid)).fetchone()
    return row[0] if row else None


def drawing_seed(workspace, root, admit=None):
    """Read only the selected concrete reference closure, never dataset payloads.

    Check source-size estimates before retrieving/parsing each document when the
    owned session supplies admission. SQL/native parser costs remain separate.
    """
    from .revisions import model_document, _parse
    _require(_identifier(root), 'Family seed needs an existing architecture ID.')
    models = {}; source_bytes = 0
    def visit(mid):
        nonlocal source_bytes
        if mid in models: return
        size = workspace.db.execute('SELECT length(CAST(document AS BLOB)) FROM architectures WHERE id=?', (mid,)).fetchone()
        _require(size is not None, 'Family seed refers to a missing architecture.')
        source_bytes += size[0]
        if admit is not None: admit(source_bytes)
        row = workspace.db.execute('SELECT document FROM architectures WHERE id=?', (mid,)).fetchone()
        parsed = _parse(row[0])
        models[mid] = {k: parsed[k] for k in ('id', 'name', 'graph')}
        for node in models[mid]['graph']['nodes']:
            if node['type'] in {'Module', 'Recurrent'}: visit(node['ref'])
    visit(root)
    return model_document(models, root)


def begin(workspace, kind, *, base=None, sid=None, model=None, seed_admit=None):
    """Read-only draft, including deliberately incomplete new-object templates."""
    _require(kind in KINDS, 'Choose a dataset, task, experiment or architecture family draft.')
    _require(model is None or (kind == 'family' and base is None),
             'A drawing seed applies only to a new family, not an exact historical base.')
    if base is not None:
        reference(base, kind)
        _require(sid is None, 'Editing uses its exact base ID, not a replacement ID.')
        document = workspace.get_spec(kind, base['id'], base['revision'])
        document.pop('revision')  # Verified immutable bytes become an explicit draft.
        sid = base['id']
    else:
        sid = uuid.uuid4().hex if sid is None else sid
        _require(_identifier(sid), 'Draft requires a stable object ID.')
        _require(not workspace.db.execute('SELECT 1 FROM spec_revisions WHERE kind=? AND id=?', (kind, sid)).fetchone(),
                 'Object ID is already retained. Edit an exact revision instead.')
        document = {'version': 1, 'id': sid, 'name': 'New ' + kind}
        if kind == 'dataset':
            document.update(storage={'kind': 'inline-records', 'version': 1}, fields=[], records=[])
        elif kind == 'task':
            document.update(inputs=[], heads=[])
        elif kind == 'family':
            document.update(root='', models={}, symbols=[], assignments=[], repeats=[])
            if model is not None:
                concrete = drawing_seed(workspace, model, seed_admit)
                document['version'] = 6 if concrete['version'] == 3 else concrete['version']
                if document['version'] == 6: document['runtime_policy'] = None
                document.update(root=concrete['root'], models=concrete['models'])
        else:
            document.update(question='', models=[], data_tasks=[], splits=[], initialization_seeds=[],
                            protocols=[], selection={'kind': 'crossed', 'pairs': [], 'exclusions': []})
    return copy.deepcopy({'version': 1, 'kind': kind, 'id': sid,
        'workspace_id': workspace_id(workspace), 'base': base,
        'expected_head': head(workspace, kind, sid), 'document': document})


def check(workspace, draft):
    """Validate without SQL writes, tensor allocation, RNG use or transform fit."""
    _keys(draft, ('version', 'kind', 'id', 'workspace_id', 'base', 'expected_head', 'document'), context='Authoring draft')
    _require(type(draft['version']) is int and draft['version'] == 1, 'Authoring draft requires version 1.')
    kind, sid = draft['kind'], draft['id']
    _require(isinstance(kind, str) and kind in KINDS and _identifier(sid), 'Invalid authoring kind or object ID.')
    _require(draft['workspace_id'] == workspace_id(workspace), 'Draft belongs to another workspace. Reopen the editor.')
    expected = draft['expected_head']
    if expected is not None:
        reference({'kind': kind, 'id': sid, 'revision': expected}, kind)
    _require(expected == head(workspace, kind, sid), 'Browsing head changed while editing. Reopen the exact revision; your draft has not been saved.')
    if draft['base'] is None:
        _require(expected is None and not workspace.db.execute('SELECT 1 FROM spec_revisions WHERE kind=? AND id=?', (kind, sid)).fetchone(),
                 'New-object ID is already retained. Reopen an exact revision instead.')
    else:
        reference(draft['base'], kind)
        _require(draft['base']['id'] == sid, 'Draft and base identities disagree.')
        workspace.get_spec(kind, sid, draft['base']['revision'])
    document = draft['document']
    _require(isinstance(document, dict) and document.get('id') == sid and 'revision' not in document,
             'Draft document must keep its stable ID and cannot supply a revision hash.')
    frozen = frozen_document(kind, document)
    validate_dependencies(kind, frozen, workspace.get_spec)
    return {'kind': kind, 'id': sid, 'revision': frozen['revision']}


def save(workspace, draft):
    """GUI-thread admission after optimistic check; uses existing undo wrapper."""
    result = check(workspace, draft)
    if head(workspace, draft['kind'], draft['id']) == result['revision']:
        return result  # Exact unchanged head: no SQL write, dirty flag or history.
    # No event-loop yield between check and put. SQLite is the private working
    # copy on the GUI thread; external file conflicts still require explicit Save.
    actual = workspace.put_spec(draft['kind'], draft['document'])
    if actual != result:
        raise GraphError('Authoring content identity changed during admission.')
    return actual


def task_context(workspace, model, dataset):
    """Exact interface/field suggestions, never records, tensors or learned state."""
    reference(model, 'model'); reference(dataset, 'dataset')
    architecture = workspace.get_spec('model', model['id'], model['revision'])
    observations = workspace.get_spec('dataset', dataset['id'], dataset['revision'])
    contract = validate_model(architecture['models'], architecture['root'])
    return copy.deepcopy({'version': 1, 'model': model, 'dataset': dataset,
        'input_ports': contract['input_groups'], 'output_ports': contract['output_groups'],
        'fields': observations['fields'], 'record_count': len(observations['records'])})


def task_compatibility(workspace, draft, model, dataset):
    """Intrinsic + exact interface checks only, not a numerical/data/RAM audit."""
    prospective = check(workspace, draft)
    _require(draft['kind'] == 'task', 'Task compatibility requires a task draft.')
    reference(model, 'model'); reference(dataset, 'dataset')
    architecture = workspace.get_spec('model', model['id'], model['revision'])
    observations = workspace.get_spec('dataset', dataset['id'], dataset['revision'])
    contract = validate_model(architecture['models'], architecture['root'])
    try:
        validate_task(draft['document'], observations, contract)
        status, reason = 'compatible', None
    except GraphError as error:
        status, reason = 'incompatible', str(error)
    return copy.deepcopy({'version': 1, 'task': prospective, 'model': model, 'dataset': dataset,
                          'status': status, 'reason': reason,
                          'scope': 'interface-only-no-fitting-or-execution'})


def experiment_preview(workspace, draft, *, offset=0, limit=25):
    """Bounded train-free prospective selection; never admit a draft or a run."""
    check(workspace, draft)
    _require(draft['kind'] == 'experiment', 'Experiment preview requires an experiment draft.')
    _require(type(limit) is int and 1 <= limit <= 25, 'Draft preview limit must be an integer in 1..25.')
    frozen = frozen_document('experiment', draft['document'])
    page = preview_experiment(frozen, workspace.get_spec, offset=offset, limit=limit)
    # Cartesian counts can exceed JavaScript's exact integer range. Preserve
    # their decimal representation separately; coordinates remain bounded pages.
    return {'version': 1, 'preview': page,
            'total_combinations_exact': str(page['total_combinations'])}
