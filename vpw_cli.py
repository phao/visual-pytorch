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
    for command in ('show','validate','parameter-counts','export','export-trained'):
        p = sub.add_parser(command)
        p.add_argument('id')
    specs = sub.add_parser('spec-list', help='Independent object browsing heads or all retained revisions')
    from app.revisions import KINDS
    specs.add_argument('--kind', choices=KINDS)
    specs.add_argument('--all-revisions', action='store_true')
    show_spec = sub.add_parser('spec-show', help='Read an exact independent revision or the current browsing head')
    show_spec.add_argument('kind', choices=KINDS)
    show_spec.add_argument('id')
    show_spec.add_argument('--revision')
    browse = sub.add_parser('spec-page', help='Bounded alphabetic scientific metadata page; no documents or execution')
    browse.add_argument('kind', choices=KINDS)
    browse.add_argument('--query', default='')
    browse.add_argument('--offset', type=int, default=0)
    browse.add_argument('--limit', type=int, default=25)
    browse.add_argument('--heads-only', action='store_true')
    browse.add_argument('--pin', nargs=2, metavar=('ID', 'REVISION'), help='Keep an exact selection independently of this page/search')
    browse.add_argument('--allow-missing-pin', action='store_true', help='Report a missing exact pin explicitly; never choose a replacement')
    for command in ('spec-summary', 'spec-inspect'):
        p = sub.add_parser(command, help='Bounded read of an exact scientific revision; never an implicit head')
        p.add_argument('kind', choices=KINDS); p.add_argument('id'); p.add_argument('revision')
        if command == 'spec-inspect':
            p.add_argument('--path', default='[]', help='JSON array of stored member/array ordinals')
            p.add_argument('--part', choices=('value', 'key'), default='value')
            p.add_argument('--offset', type=int, default=0); p.add_argument('--limit', type=int, default=25)
    context=sub.add_parser('task-context-page',help='Bounded exact model-port/dataset-field suggestions; no execution')
    context.add_argument('collection',choices=('fields','input_ports','output_ports'))
    context.add_argument('--model',nargs=2,required=True,metavar=('ID','REVISION'))
    context.add_argument('--dataset',nargs=2,required=True,metavar=('ID','REVISION'))
    context.add_argument('--query',default='');context.add_argument('--offset',type=int,default=0)
    context.add_argument('--limit',type=int,default=25);context.add_argument('--pin')
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
    trained_run = sub.add_parser('export-run-trained', help='Generate exact saved-attempt inference source; never train or execute it')
    trained_run.add_argument('id'); trained_run.add_argument('revision')
    trained_run.add_argument('--attempt', type=int, help='Retained attempt; default latest of this exact revision')
    trained_run.add_argument('--memory-limit-percent', type=float)
    selections = sub.add_parser('preview-experiment', help='Read-only bounded exact-revision selection preview; no fitting/training')
    selections.add_argument('id'); selections.add_argument('revision')
    selections.add_argument('--offset', type=int, default=0); selections.add_argument('--limit', type=int, default=100)
    page = sub.add_parser('preview-experiment-page', help='Byte-bounded exact candidate projection; no manifests, fitting or training')
    page.add_argument('id'); page.add_argument('revision')
    page.add_argument('--offset', type=int, default=0); page.add_argument('--limit', type=int, default=25)
    reporting = sub.add_parser('report-data', help='Read exact saved report recipe/evidence; does not train or execute notebooks')
    reporting.add_argument('id'); reporting.add_argument('revision')
    for command,maximum in (('run-evidence',25),('report-evidence',10)):
        evidence = sub.add_parser(command, help='Read paged exact saved evidence; never train or execute notebooks')
        evidence.add_argument('id'); evidence.add_argument('revision')
        evidence.add_argument('--offset', type=int, default=0)
        evidence.add_argument('--limit', type=int, default=25 if command=='run-evidence' else 5,
                              help='Response row limit in 1..'+str(maximum))
        if command=='run-evidence':
            evidence.add_argument('--attempt', type=int, help='Retained attempt number; default latest of this exact revision')
    curve = sub.add_parser('run-curve', help='Read exact saved-attempt head curves; no fitting, inference or training')
    curve.add_argument('id'); curve.add_argument('revision')
    curve.add_argument('--attempt', type=int); curve.add_argument('--head')
    curve.add_argument('--offset', type=int, default=0); curve.add_argument('--limit', type=int, default=100)
    curve.add_argument('--head-offset', type=int, default=0); curve.add_argument('--head-limit', type=int, default=25)
    notebook = sub.add_parser('generate-report', help='Export a NEW portable notebook bundle; does not train or execute cells')
    notebook.add_argument('id'); notebook.add_argument('revision'); notebook.add_argument('--output', required=True)
    notebook.add_argument('--offset', type=int, default=0); notebook.add_argument('--limit', type=int, default=100)
    notebook.add_argument('--allow-partial', action='store_true'); notebook.add_argument('--memory-limit-percent', type=float)
    notebook.add_argument('--worker-timeout', type=float)
    execution = sub.add_parser('execute-report', help='Explicitly execute a verified notebook bundle using a real Jupyter kernel')
    execution.add_argument('bundle'); execution.add_argument('--kernel-python', default=sys.executable)
    execution.add_argument('--cell-timeout', type=int, default=600)
    run = sub.add_parser('run-candidate', help='Explicit frozen selection execution and terminal RunRecord save; never silently retry')
    run.add_argument('--experiment', nargs=2, metavar=('ID', 'REVISION'), required=True)
    run.add_argument('--coordinates', required=True, help='Coordinates JSON from selection preview')
    run.add_argument('--run-id', required=True); run.add_argument('--name', required=True)
    run.add_argument('--retry', action='store_true'); run.add_argument('--torch-python', default=sys.executable)
    run.add_argument('--memory-limit-percent', type=float); run.add_argument('--timeout', type=float)
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
                         'panel_layout': workspace.panel_layout,
                         'architectures': [{'id':d['id'],'name':d['name'],'nodes':len(d['graph']['nodes'])} for d in sorted(workspace.models().values(),key=lambda d:(d['name'].casefold(),d['id']))],
                         'specifications': workspace.list_specs(),
                         'schema': [{'name':name,'sql':sql} for name,sql in workspace.db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")]}
            elif args.command == 'show':
                value = workspace.models()[args.id]
            elif args.command == 'parameter-counts':
                from app.presentation import block_parameter_counts
                models=workspace.models();model=models[args.id];contract=validate_model(models,args.id)
                interfaces={n['id']:validate_model(models,n['ref']) for n in model['graph']['nodes'] if n['type']=='Module'}
                value={'id':args.id,'counts':block_parameter_counts(model,contract,interfaces)}
            elif args.command == 'spec-list':
                value = workspace.list_specs(args.kind, args.all_revisions)
            elif args.command == 'spec-show':
                value = workspace.get_spec(args.kind, args.id, args.revision)
            elif args.command in ('spec-page', 'spec-summary', 'spec-inspect'):
                from app import science_browser
                if args.command == 'spec-page':
                    value = science_browser.index_page(workspace, args.kind, offset=args.offset,
                        limit=args.limit, query=args.query, heads_only=args.heads_only,
                        preferred=None if args.pin is None else {'kind':args.kind,'id':args.pin[0],'revision':args.pin[1]},
                        allow_missing_pin=args.allow_missing_pin)
                else:
                    ref = {'kind': args.kind, 'id': args.id, 'revision': args.revision}
                    value = science_browser.summary(workspace, ref) if args.command == 'spec-summary' else \
                        science_browser.inspect_page(workspace, ref, path=json.loads(args.path),
                            part=args.part, offset=args.offset, limit=args.limit)
            elif args.command == 'task-context-page':
                from app import task_context
                value=task_context.page(workspace,{'kind':'model','id':args.model[0],'revision':args.model[1]},
                    {'kind':'dataset','id':args.dataset[0],'revision':args.dataset[1]},args.collection,
                    query=args.query,offset=args.offset,limit=args.limit,pin=args.pin)
            elif args.command == 'preview-experiment':
                from app.experiment_objects import preview_experiment
                value = preview_experiment(workspace.get_spec('experiment', args.id, args.revision), workspace.get_spec,
                                           offset=args.offset, limit=args.limit)
            elif args.command == 'preview-experiment-page':
                from app.candidate_views import stored
                value = stored(workspace, {'kind':'experiment','id':args.id,'revision':args.revision},
                               offset=args.offset, limit=args.limit)
            elif args.command == 'report-data':
                from app.experiment_objects import report_data
                value = report_data(workspace.get_spec('report', args.id, args.revision), workspace.get_spec)
            elif args.command in ('run-evidence', 'report-evidence'):
                from app.run_views import run_view,report_view
                ref={'kind':'run' if args.command=='run-evidence' else 'report','id':args.id,'revision':args.revision}
                options={'offset':args.offset,'limit':args.limit}
                if args.command=='run-evidence':options['attempt']=args.attempt
                value=(run_view if args.command=='run-evidence' else report_view)(workspace,ref,**options)
            elif args.command == 'run-curve':
                from app.run_curves import curve_view
                value = curve_view(workspace, {'kind':'run','id':args.id,'revision':args.revision},
                    attempt=args.attempt, head=args.head, offset=args.offset, limit=args.limit,
                    head_offset=args.head_offset, head_limit=args.head_limit)
            elif args.command == 'generate-report':
                from app.notebooks import create_bundle
                value = create_bundle(workspace, {'kind': 'report', 'id': args.id, 'revision': args.revision}, args.output,
                    offset=args.offset, limit=args.limit, allow_partial=args.allow_partial,
                    memory_limit_percent=ram_percentage(args.memory_limit_percent), worker_timeout=args.worker_timeout)
            elif args.command == 'execute-report':
                from app.notebooks import execute_bundle
                value = execute_bundle(args.bundle, python=args.kernel_python, timeout=args.cell_timeout)
            elif args.command == 'run-candidate':
                from app.experiment_objects import execute_candidate
                value = execute_candidate(workspace, {'kind': 'experiment', 'id': args.experiment[0], 'revision': args.experiment[1]},
                    read_json(args.coordinates), args.run_id, args.name, python=args.torch_python, retry=args.retry,
                    memory_limit_percent=ram_percentage(args.memory_limit_percent), timeout=args.timeout)
                try:
                    workspace.save(path)
                    value.update(persisted=True, workspace_revision=workspace.disk_revision)
                except Exception as error:
                    # Do not lose actual training evidence or overwrite concurrent
                    # user edits. The JSON artifact is recoverable for a later put.
                    value.update(persisted=False, save_error=str(error))
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
            elif args.command == 'export-run-trained':
                from app.run_export import predictor_source
                value = predictor_source(workspace, {'kind': 'run', 'id': args.id, 'revision': args.revision},
                    attempt=args.attempt, memory_limit_percent=ram_percentage(args.memory_limit_percent))
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
