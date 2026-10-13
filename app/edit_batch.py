"""Shared ordered edit executor; publication is owned by edit_planning/Save."""
from .core import default_model,check_document

def apply_edits(workspace, edits):
    """One atomic file replacement for the entire ordered edit batch."""
    if not isinstance(edits, list):
        raise ValueError('Edits must be a JSON array.')
    results = []
    for edit in edits:
        kind = edit['op']
        if kind == 'specialize-family':
            from app.family_records import retain
            from app.datasets import _keys
            _keys(edit, ('op',), ('family', 'family_ref', 'bindings', 'id', 'name', 'materialize'), 'Specialize-family edit')
            retained = retain(workspace, family=edit.get('family'), family_ref=edit.get('family_ref'),
                              bindings=edit.get('bindings'), sid=edit.get('id'), name=edit.get('name'),
                              materialize=edit.get('materialize', True))
            results.append({'op': kind, **retained})
            continue
        elif kind == 'admit-run':
            from app.experiment_objects import admit_run
            from app.datasets import _keys
            _keys(edit, ('op', 'document'), ('select',), 'Admit-run edit')
            ref = admit_run(workspace, edit['document'], select=edit.get('select', True))
            results.append({'op': kind, 'reference': ref})
            continue
        elif kind == 'derive-lora-merge':
            from app.parameter_derivation import admit
            from app.datasets import _keys
            _keys(edit, ('op', 'sources'), ('alpha', 'id', 'name'), 'Derive-LoRA-merge edit')
            results.append({'op': kind, **admit(workspace, edit['sources'], alpha=edit.get('alpha', 4.),
                model_id=edit.get('id', 'lora-merged'), name=edit.get('name', 'LoRA merged inference · explicitly derived'))})
            continue
        elif kind == 'bind-parameter':
            from app.parameter_binding import admit
            from app.datasets import _keys
            _keys(edit, ('op', 'token', 'selector', 'source'), ('binding_id', 'name'), 'Bind-parameter edit')
            results.append({'op': kind, **admit(workspace, edit['token'], edit['selector'], edit['source'],
                binding_id=edit.get('binding_id'), name=edit.get('name'))})
            continue
        elif kind == 'put-spec':
            ref = workspace.put_spec(edit['kind'], edit['document'], edit.get('select', True))
            results.append({'op': kind, 'reference': ref})
            continue
        elif kind == 'select-spec':
            ref = edit['reference']
            workspace.select_spec(ref['kind'], ref['id'], ref['revision'])
            results.append({'op': kind, 'reference': ref})
            continue
        elif kind == 'archive-spec':
            workspace.archive_spec(edit['kind'], edit['id'])
            results.append({'op': kind, 'kind': edit['kind'], 'id': edit['id']})
            continue
        elif kind == 'freeze-model':
            results.append({'op': kind, 'reference': workspace.freeze_model(edit['id'])})
            continue
        elif kind in {'add-lstm','edit-lstm'}:
            from app.lstm_authoring import build, build_edit
            if set(edit) != {'op','token','options'}:
                raise ValueError('LSTM edit requires an exact token and explicit options.')
            document,nid = (build_edit if kind=='edit-lstm' else build)(workspace.models(),edit['token'],edit['options'])
            if document != workspace.models()[document['id']]:
                workspace.put(document)
            results.append({'op':kind,'id':document['id'],'node':nid})
            continue
        elif kind == 'edit-recurrent':
            from app.recurrent_reconfiguration import build_edit
            if set(edit) != {'op', 'token', 'options'}:
                raise ValueError('Edit-recurrent requires an exact token and explicit options.')
            document, nid = build_edit(workspace.models(), edit['token'], edit['options'])
            if document != workspace.models()[document['id']]:
                workspace.put(document)
            results.append({'op': kind, 'id': document['id'], 'node': nid})
            continue
        elif kind == 'add-recurrent':
            from app.recurrent_authoring import build
            if set(edit) != {'op', 'token', 'options'}:
                raise ValueError('Add-recurrent requires an exact token and explicit options.')
            document, nid = build(workspace.models(), edit['token'], edit['options'])
            workspace.put(document)
            results.append({'op': kind, 'id': document['id'], 'node': nid})
            continue
        elif kind == 'create':
            doc = edit.get('document') or default_model(edit.get('name', 'New architecture'), edit.get('inputs', 2), edit.get('outputs', 1))
            if doc['id'] in workspace.models():
                raise ValueError('Architecture ID already exists; use replace explicitly.')
        elif kind == 'mode':
            workspace.set_constrained_mode(edit['enabled'])
            results.append({'op': kind, 'enabled': workspace.constrained_mode})
            continue
        elif kind == 'layout':
            workspace.set_layout_preferences(edit['value'])
            results.append({'op': kind, 'value': workspace.layout_preferences})
            continue
        elif kind == 'panel-layout':
            workspace.set_panel_layout(edit['value'])
            results.append({'op': kind, 'value': workspace.panel_layout})
            continue
        elif kind == 'delete':
            workspace.delete(edit['id'])
            results.append({'op': kind, 'id': edit['id']})
            continue
        else:
            doc = workspace.models()[edit['id']]
            if kind == 'replace':
                doc = edit['document']
                if doc['id'] != edit['id']:
                    raise ValueError('Replacement must preserve the architecture ID.')
            elif kind == 'rename':
                doc['name'] = edit['name']
            elif kind == 'move':
                node = next(n for n in doc['graph']['nodes'] if n['id'] == edit['node'])
                node.update(x=edit['x'], y=edit['y'])
            elif kind == 'resize':
                node = next(n for n in doc['graph']['nodes'] if n['id'] == edit['node'])
                node.update(width=edit['width'], height=edit['height'])
            elif kind == 'rebind-dataset':
                from app.evidence import binding_for
                from app.core import validate_dataset
                if doc.get('dataset') is None:raise ValueError('There is no dataset to rebind.')
                contract=validate_model(workspace.models(),doc['id'])
                validate_dataset(doc['dataset'],contract['inputs'],contract['outputs'])
                from app.tensor_data import check_adapter
                check_adapter(doc['dataset'], contract)
                doc['dataset_binding']=binding_for(contract)
                doc['result']=None
            elif kind == 'set-parameter-policy':
                from app.datasets import _keys
                from app.policy_graph import declaration, plan
                _keys(edit, ('op', 'id', 'policy'), context='Set-parameter-policy edit')
                if edit['policy'] is None: doc['graph'].pop('parameter_policy', None)
                else: doc['graph']['parameter_policy'] = declaration(edit['policy'])
                prospective = workspace.models(); prospective[edit['id']] = doc
                plan(prospective, edit['id'])
            elif kind == 'set-node':
                node = next(n for n in doc['graph']['nodes'] if n['id'] == edit['node'])
                if set(edit['values']) & {'id', 'type'}:
                    raise ValueError('set-node cannot change ID/type; use replace.')
                node.update(edit['values'])
                for field in edit.get('remove_fields', []):
                    if field in {'id','type'}:
                        raise ValueError('Cannot remove ID/type.')
                    node.pop(field, None)
            elif kind == 'add-node':
                doc['graph']['nodes'].append(edit['node'])
            elif kind == 'remove-node':
                nid = edit['node']
                if not any(n['id'] == nid for n in doc['graph']['nodes']):
                    raise ValueError('Node does not exist.')
                doc['graph']['nodes'] = [n for n in doc['graph']['nodes'] if n['id'] != nid]
                doc['graph']['edges'] = [e for e in doc['graph']['edges'] if e['from'] != nid and e['to'] != nid]
                for n in doc['graph']['nodes']:
                    if n['type'] == 'Concat':
                        n['sources'] = [s for s in n['sources'] if s != nid and not s.startswith(nid+'@')]
            elif kind == 'connect':
                doc['graph']['edges'].append(edit['edge'])
            elif kind == 'disconnect':
                if edit['edge'] not in doc['graph']['edges']:
                    raise ValueError('Wire does not exist.')
                doc['graph']['edges'].remove(edit['edge'])
            else:
                raise ValueError('Unknown edit operation: '+kind)
        check_document(doc)
        workspace.put(doc)
        results.append({'op': kind, 'id': doc['id']})
    return results

