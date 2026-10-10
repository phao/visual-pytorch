"""Optimistic scientific drafts; no fitting, execution or alternate persistence.

Drafts are private editor state, not vtw objects. Admission uses the same closed
version-1 grammar and immutable registry as CLI imports. A historical base is
explicit; its current browsing head is captured separately, never substituted.
"""
import copy
import uuid

from .core import GraphError
from .datasets import _identifier, _keys, _require
from .experiment_objects import reference, validate_dependencies
from .revisions import frozen_document

KINDS = ('dataset', 'experiment', 'task')


def workspace_id(workspace):
    return workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]


def head(workspace, kind, sid):
    row = workspace.db.execute('SELECT revision FROM spec_heads WHERE kind=? AND id=?', (kind, sid)).fetchone()
    return row[0] if row else None


def begin(workspace, kind, *, base=None, sid=None):
    """Read-only draft, including deliberately incomplete new-object templates."""
    _require(kind in KINDS, 'Only datasets, tasks and experiments have authoring drafts.')
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
