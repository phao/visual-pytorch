"""Read-only edit planning and optimistic, atomic working-copy admission.

The same ordered executor is used by preview, CLI and Qt admission. A preview
never saves, fits, reads user settings or touches the source history. The cloned
SQLite state is published only after all edits and checks succeed and the source
still has its exact snapshot identity. File saving retains Workspace.save's own
external revision check; a preview is not permission to bypass a later conflict.
"""
import copy
import hashlib
import json

from .core import GraphError, Workspace, validate_model
from .discovery import parameter_contract, port_contract

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
MAX_EDITS = 256


def _budget(workspace, limit):
    if type(limit) is not int or not 1 <= limit <= 1024 * 1024 * 1024:
        raise GraphError('Edit preview max_bytes must be an exact integer in [1,1073741824].')
    pages=workspace.db.execute('PRAGMA page_count').fetchone()[0]
    size=workspace.db.execute('PRAGMA page_size').fetchone()[0]
    if pages*size > limit:
        raise GraphError('Edit preview exceeds its SQLite snapshot budget before copying.')


def _interfaces(models):
    result={}; findings=[]
    for mid in models:
        try:
            contract=validate_model(models,mid)
            result[mid]={'ports':port_contract(models,mid),'parameters':contract['parameters'],
                         'parameter_policy':parameter_contract(models,mid)}
        except (GraphError,ValueError) as error:
            result[mid]=None
            findings.append({'code':'GRAPH_INVALID','path':['architectures',mid],
                'message':str(error),'repair':'Inspect the named architecture and its referenced blocks.'})
    return result,findings


def _heads(workspace):
    return [{'kind':kind,'id':sid,'revision':revision} for kind,sid,revision in
            workspace.db.execute('SELECT kind,id,revision FROM spec_heads ORDER BY kind,id')]


def _effects(before,after,old,new):
    effects=[]
    for mid in list(before)+[key for key in after if key not in before]:
        if before.get(mid)==after.get(mid):continue
        prior=before.get(mid);current=after.get(mid)
        prior_nodes={n['id']:n for n in prior['graph']['nodes']} if prior else {}
        current_nodes={n['id']:n for n in current['graph']['nodes']} if current else {}
        changed=[]
        for nid in prior_nodes.keys() & current_nodes.keys():
            fields=[field for field in sorted(prior_nodes[nid].keys()|current_nodes[nid].keys())
                    if prior_nodes[nid].get(field)!=current_nodes[nid].get(field)]
            if fields:changed.append({'id':nid,'fields':fields})
        ports_before=(old.get(mid) or {}).get('ports')
        ports_after=(new.get(mid) or {}).get('ports')
        effects.append({'id':mid,'operation':'created' if prior is None else 'deleted' if current is None else 'changed',
            'nodes_added':[key for key in current_nodes if key not in prior_nodes],
            'nodes_removed':[key for key in prior_nodes if key not in current_nodes],
            'nodes_changed':sorted(changed,key=lambda item:item['id']),
            'before':old.get(mid),'after':new.get(mid),
            'port_contract_changed':ports_before!=ports_after,
            'dataset_binding_before':copy.deepcopy(prior.get('dataset_binding')) if prior else None,
            'dataset_binding_after':copy.deepcopy(current.get('dataset_binding')) if current else None,
            'result_invalidated':bool(prior and prior.get('result') is not None and (not current or current.get('result') is None)),
            'dataset_changed':(prior or {}).get('dataset')!=(current or {}).get('dataset'),
            'references_before':[n['ref'] for n in prior_nodes.values() if 'ref' in n],
            'references_after':[n['ref'] for n in current_nodes.values() if 'ref' in n]})
    return effects


def _prepare(workspace,edits, *, allow_invalid=False,max_bytes=DEFAULT_MAX_BYTES,executor=None):
    if type(allow_invalid) is not bool:raise GraphError('allow_invalid must be a boolean.')
    if not isinstance(edits,list) or len(edits)>MAX_EDITS or any(not isinstance(e,dict) for e in edits):
        raise GraphError('Edits must be an ordered JSON array of at most256 objects.')
    _budget(workspace,max_bytes)
    if len(json.dumps(edits,allow_nan=False).encode('utf8'))>max_bytes:
        raise GraphError('Edit declarations exceed the preview byte budget.')
    snapshot=workspace.db.serialize();token=hashlib.sha256(snapshot).hexdigest()
    before=workspace.models();old,existing=_interfaces(before);old_heads=_heads(workspace)
    old_metadata=dict(workspace.db.execute('SELECT key,value FROM metadata'))
    clone=Workspace()
    try:
        clone.db.deserialize(snapshot);clone.db.execute('PRAGMA foreign_keys=ON')
        clone.path=workspace.path;clone.disk_revision=workspace.disk_revision
        clone.disk_stamp=workspace.disk_stamp;clone.panel_scope=workspace.panel_scope
        if executor is None:
            from .edit_batch import apply_edits as executor
        applied=[];edit_failure=[]
        for index,edit in enumerate(edits):
            try:applied.extend(executor(clone,[copy.deepcopy(edit)]))
            except Exception as error:
                edit_failure=[{'code':'EDIT_REFUSED','path':['edits',index],
                    'architecture':edit.get('id'),'node':edit.get('node') if isinstance(edit.get('node'),str) else None,
                    'operation':edit.get('op'),'message':str(error),
                    'repair':'Correct this edit against the current IDs, fields, dependencies and optimistic tokens.'}]
                break
        _budget(clone,max_bytes)
        after=clone.models();new,invalid=_interfaces(after)
        previous={(item['path'][-1],item['message']) for item in existing}
        for item in invalid:item['origin']='existing' if (item['path'][-1],item['message']) in previous else 'introduced'
        integrity=clone.db.execute('PRAGMA foreign_key_check').fetchall()
        if integrity:
            edit_failure.append({'code':'REFERENCE_INTEGRITY','path':['workspace'],
                                 'message':'Proposed state has invalid SQLite references.','repair':'Repair dependency order/targets.'})
        ready=not edit_failure and (not invalid or allow_invalid)
        new_metadata=dict(clone.db.execute('SELECT key,value FROM metadata'))
        report={'version':1,'kind':'visual-pytorch-edit-plan','status':'draft' if ready and invalid else 'ready' if ready else 'refused',
            'admissible':ready,'snapshot_sha256':token,'disk_revision':workspace.disk_revision,
            'ordered_edits':copy.deepcopy(edits),'applied_preview_count':len(applied),
            'effects':_effects(before,after,old,new),'existing_findings':existing,
            'findings':edit_failure+invalid,'heads_before':old_heads,'heads_after':_heads(clone),
            'metadata_changes':[{'key':key,'before':old_metadata.get(key),'after':new_metadata.get(key)}
                for key in sorted(old_metadata.keys()|new_metadata.keys()) if old_metadata.get(key)!=new_metadata.get(key)],
            'allow_invalid':allow_invalid,'scope':'preview only; no disk, history, configuration or numerical execution',
            'partial_effects':bool(edit_failure),'results':applied}
        return clone,report
    except BaseException:
        clone.db.close()
        if clone.history.directory:clone.history.directory.cleanup()
        raise


def preview(workspace,edits, **options):
    clone,report=_prepare(workspace,edits,**options)
    try:return report
    finally:
        clone.db.close()
        if clone.history.directory:clone.history.directory.cleanup()


def admit(workspace,edits, *, expected_snapshot=None,**options):
    """Publish one fully checked in-memory batch; caller owns history and Save."""
    clone,report=_prepare(workspace,edits,**options)
    try:
        if not report['admissible']:
            raise GraphError('Edit batch refused: '+report['findings'][0]['message'])
        current=hashlib.sha256(workspace.db.serialize()).hexdigest()
        if current!=report['snapshot_sha256'] or expected_snapshot is not None and current!=expected_snapshot:
            raise GraphError('Edit preview is stale; inspect the current workspace and preview again.')
        report['changed']=clone.db.serialize()!=workspace.db.serialize()
        if report['changed']:
            clone.db.backup(workspace.db)
            workspace.dirty=True
        return report
    finally:
        clone.db.close()
        if clone.history.directory:clone.history.directory.cleanup()
