"""Agent/user workspace CLI. JSON output; imports no Qt or PyTorch."""
import argparse
import json
import sys
from pathlib import Path

from app.core import Workspace, default_model, generate_code, generate_predictor, validate_model, check_document


def read_json(path):
    return json.load(sys.stdin) if path == '-' else json.loads(Path(path).read_text())


def apply_edits(workspace, edits):
    """One atomic file replacement for the entire ordered edit batch."""
    if not isinstance(edits, list):
        raise ValueError('Edits must be a JSON array.')
    results = []
    for edit in edits:
        kind = edit['op']
        if kind == 'create':
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', help='Path to a .vtw workspace')
    parser.add_argument('--expect-revision', help='Refuse if the file no longer has this SHA-256 revision')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init', help='Create a new empty workspace; never overwrite an existing file')
    sub.add_parser('inspect', help='SQLite schema, architecture list, policy and revision')
    for command in ('show','validate','export','export-trained'):
        p = sub.add_parser(command)
        p.add_argument('id')
    apply = sub.add_parser('apply', help='Atomic ordered JSON edit batch')
    apply.add_argument('edits', help='JSON file or - for stdin')
    apply.add_argument('--allow-invalid', action='store_true', help='Save semantically invalid drafts, still structurally checked')
    args = parser.parse_args(argv)
    workspace = None
    try:
        path = Path(args.workspace).expanduser().resolve()
        if path.suffix.lower() != '.vtw':
            raise ValueError('Workspace path must end in .vtw.')
        if args.command == 'init':
            if path.exists():
                raise ValueError('Refusing to overwrite an existing workspace.')
            workspace = Workspace()
            # Save checks creation races under its persistent sidecar lock.
            workspace.save(path, require_new=True)
            value = {'path': str(path), 'revision': workspace.disk_revision}
        else:
            workspace = Workspace.load(path)
            if args.expect_revision and workspace.disk_revision != args.expect_revision:
                raise ValueError('Revision conflict. Inspect again before applying edits.')
            if args.command == 'inspect':
                value = {'path': str(path), 'revision': workspace.disk_revision,
                         'format_version': workspace.db.execute('PRAGMA user_version').fetchone()[0],
                         'constrained_mode': workspace.constrained_mode,
                         'layout_preferences': workspace.layout_preferences,
                         'architectures': [{'id':d['id'],'name':d['name'],'nodes':len(d['graph']['nodes'])} for d in sorted(workspace.models().values(),key=lambda d:(d['name'].casefold(),d['id']))],
                         'schema': [{'name':name,'sql':sql} for name,sql in workspace.db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")]}
            elif args.command == 'show':
                value = workspace.models()[args.id]
            elif args.command == 'validate':
                value = validate_model(workspace.models(), args.id)
            elif args.command == 'export':
                value = {'code': generate_code(workspace.models(), args.id)}
            elif args.command == 'export-trained':
                value={'code':generate_predictor(workspace.models(),args.id,workspace.models()[args.id].get('result'))}
            elif args.command == 'apply':
                results = apply_edits(workspace, read_json(args.edits))
                if not args.allow_invalid:
                    models = workspace.models()
                    for mid in models:
                        validate_model(models, mid)
                workspace.save(path)
                value = {'edits': results, 'revision': workspace.disk_revision,
                         'draft_allowed': args.allow_invalid}
        print(json.dumps({'ok':True,'value':value}, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({'ok':False,'error':str(error)}))
        return 1
    finally:
        if workspace:
            workspace.db.close()


if __name__ == '__main__':
    raise SystemExit(main())
