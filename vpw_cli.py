"""Agent/user workspace CLI. JSON output; imports no Qt or PyTorch."""
import argparse
import json
import sys
from pathlib import Path

from app.core import Workspace, default_model, generate_code, generate_predictor, validate_model, check_document


def read_json(path):
    return json.load(sys.stdin) if path == '-' else json.loads(Path(path).read_text())


def ram_percentage(override):
    """Read the user preference without saving it; malformed config fails closed."""
    if override is not None:
        return override
    from app.config import Config
    config = Config()
    if config.error:
        raise ValueError('Invalid user configuration; repair it or explicitly provide --memory-limit-percent.')
    return config.values['memory_limit_percent']


def apply_edits(workspace, edits):
    """One atomic file replacement for the entire ordered edit batch."""
    if not isinstance(edits, list):
        raise ValueError('Edits must be a JSON array.')
    results = []
    for edit in edits:
        kind = edit['op']
        if kind == 'put-spec':
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
        elif kind == 'create':
            doc = edit.get('document') or default_model(edit.get('name', 'New architecture'), edit.get('inputs', 2), edit.get('outputs', 1))
            if doc['id'] in workspace.models():
                raise ValueError('Architecture ID already exists; use replace explicitly.')
        elif kind == 'mode':
            workspace.set_constrained_mode(edit['enabled'])
            results.append({'op': kind, 'enabled': workspace.constrained_mode})
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
    specs = sub.add_parser('spec-list', help='Independent object browsing heads or all retained revisions')
    specs.add_argument('--kind', choices=('dataset', 'model', 'task'))
    specs.add_argument('--all-revisions', action='store_true')
    show_spec = sub.add_parser('spec-show', help='Read an exact independent revision or the current browsing head')
    show_spec.add_argument('kind', choices=('dataset', 'model', 'task'))
    show_spec.add_argument('id')
    show_spec.add_argument('--revision')
    preview = sub.add_parser('preview-dataset', help='Bounded typed dataset diagnostics; does not change the workspace')
    preview.add_argument('dataset', help='DatasetSpec JSON file')
    preview.add_argument('--limit', type=int, default=20)
    prepare = sub.add_parser('prepare-task', help='Resolve stable task bindings, save split IDs in output and fit train-only transforms; no training or workspace edits')
    prepare.add_argument('id', help='Architecture ID to bind')
    prepare.add_argument('--dataset', required=True, help='DatasetSpec JSON file')
    prepare.add_argument('--task', required=True, help='TaskSpec JSON file')
    splits = prepare.add_mutually_exclusive_group(required=True)
    splits.add_argument('--split', help='Explicit Split protocol JSON file')
    splits.add_argument('--manifest', help='Saved SplitManifest JSON file for replay')
    prepare.add_argument('--memory-limit-percent', type=float, help='Override the user-level constrained RAM percentage for preparation')
    for command in ('prepare-stored', 'train-stored'):
        stored = sub.add_parser(command, help='Use EXACT stored model/dataset/task revisions, never implicit heads; no workspace edits')
        for kind in ('model', 'dataset', 'task'):
            stored.add_argument('--' + kind, nargs=2, metavar=('ID', 'REVISION'), required=True)
        splits = stored.add_mutually_exclusive_group(required=True)
        splits.add_argument('--split', help='Split protocol JSON file')
        splits.add_argument('--manifest', help='Saved SplitManifest JSON file')
        stored.add_argument('--memory-limit-percent', type=float)
        if command == 'train-stored':
            stored.add_argument('--protocol', required=True, help='Explicit TrainingProtocol v1 JSON file')
            stored.add_argument('--torch-python', default=sys.executable, help='Python for isolated numerical worker')
    trained_task = sub.add_parser('export-task-trained', help='Validate saved task result and emit standalone fitted predictor source; no training')
    trained_task.add_argument('result', help='Task-training-result JSON file (the value object from train-stored)')
    trained_task.add_argument('--memory-limit-percent', type=float)
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
                         'architectures': [{'id':d['id'],'name':d['name'],'nodes':len(d['graph']['nodes'])} for d in sorted(workspace.models().values(),key=lambda d:(d['name'].casefold(),d['id']))],
                         'specifications': workspace.list_specs(),
                         'schema': [{'name':name,'sql':sql} for name,sql in workspace.db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")]}
            elif args.command == 'show':
                value = workspace.models()[args.id]
            elif args.command == 'spec-list':
                value = workspace.list_specs(args.kind, args.all_revisions)
            elif args.command == 'spec-show':
                value = workspace.get_spec(args.kind, args.id, args.revision)
            elif args.command == 'validate':
                value = validate_model(workspace.models(), args.id)
            elif args.command == 'export':
                value = {'code': generate_code(workspace.models(), args.id)}
            elif args.command == 'export-trained':
                value={'code':generate_predictor(workspace.models(),args.id,workspace.models()[args.id].get('result'))}
            elif args.command == 'export-task-trained':
                from app.task_execution import generate_task_predictor
                result = read_json(args.result)
                value = {'code': generate_task_predictor(result['bundle'], result['checkpoint'],
                         constrained=workspace.constrained_mode, memory_limit_percent=ram_percentage(args.memory_limit_percent))}
            elif args.command == 'preview-dataset':
                from app.datasets import preview
                value = preview(read_json(args.dataset), args.limit)
            elif args.command in ('prepare-task', 'prepare-stored', 'train-stored'):
                from app.datasets import make_split, prepare
                percentage = ram_percentage(args.memory_limit_percent)
                if args.command in ('prepare-stored', 'train-stored'):
                    model = workspace.get_spec('model', *args.model)
                    dataset = workspace.get_spec('dataset', *args.dataset)
                    task = workspace.get_spec('task', *args.task)
                    contract = validate_model(model['models'], model['root'])
                else:
                    dataset = read_json(args.dataset)
                    task = read_json(args.task)
                    contract = validate_model(workspace.models(), args.id)
                manifest = read_json(args.manifest) if args.manifest else make_split(dataset, read_json(args.split))
                if args.command == 'train-stored':
                    from app.task_execution import task_bundle, resolve_run
                    from app.transport import run_json_worker
                    bundle = task_bundle(model, dataset, task, manifest, read_json(args.protocol))
                    resolve_run(bundle, constrained=workspace.constrained_mode, memory_limit_percent=percentage)
                    value = run_json_worker([args.torch_python, '-u', '-m', 'app.worker'],
                        {'kind': 'task-v1', 'bundle': bundle, 'constrained': workspace.constrained_mode,
                         'memory_limit_percent': percentage}, cwd=str(Path(__file__).resolve().parent))
                else:
                    value = prepare(dataset, task, contract, manifest,
                                    constrained=workspace.constrained_mode, memory_limit_percent=percentage)
                if args.command == 'prepare-stored':
                    value['references'] = {kind: {'kind': kind, 'id': values[0], 'revision': values[1]}
                                           for kind, values in (('model', args.model), ('dataset', args.dataset), ('task', args.task))}
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
