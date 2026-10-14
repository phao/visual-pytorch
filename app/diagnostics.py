"""Read-only application/workspace doctor; never a CPW diagnostic or fit."""
import importlib.metadata
import sqlite3
import sys
from .core import FORMAT,VERSION,SUPPORTED_VERSIONS,GraphError,validate_model


def doctor(workspace):
    integrity=[row[0] for row in workspace.db.execute('PRAGMA integrity_check')]
    foreign=[list(row) for row in workspace.db.execute('PRAGMA foreign_key_check')]
    issues=[];models=workspace.models();architectures=[]
    for mid,document in models.items():
        try:
            contract=validate_model(models,mid)
            architectures.append({'id':mid,'name':document['name'],'status':'valid',
                'parameters':contract['parameters'],'inputs':contract['input_groups'],'outputs':contract['output_groups']})
        except GraphError as error:
            for problem in error.located() or [{'node':None,'message':str(error)}]:
                path=['architectures',mid]+(['nodes',problem['node']] if problem['node'] else [])
                issues.append({'code':'GRAPH_INVALID','path':path,'message':problem['message']})
            architectures.append({'id':mid,'name':document['name'],'status':'invalid'})
    if integrity!=['ok']:issues.append({'code':'SQLITE_INTEGRITY','path':['workspace'],'message':integrity})
    if foreign:issues.append({'code':'SQLITE_REFERENCES','path':['workspace'],'message':foreign})
    packages={}
    for name in ('torch','numpy','PyQt6','PyQt6-WebEngine','nbclient','nbformat','ipykernel'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return {'version':1,'kind':'visual-pytorch-doctor','status':'issues' if issues else 'ok',
        'format':FORMAT,'working_container_version':workspace.db.execute('PRAGMA user_version').fetchone()[0],
        'current_writer_version':VERSION,'reader_versions':sorted(SUPPORTED_VERSIONS),
        'disk_revision':workspace.disk_revision,'sqlite_integrity':integrity,'foreign_key_issues':foreign,
        'architectures':architectures,'issues':issues,
        'retained_counts':dict(workspace.db.execute('SELECT kind,count(*) FROM spec_revisions GROUP BY kind')),
        'environment':{'python':sys.version.split()[0],'sqlite':sqlite3.sqlite_version,'package_metadata':packages,
            'scope':'package metadata in this interpreter; not successful imports or training/notebook/GPU execution'},
        'scope':'read loaded working copy; canonical loader checks file/schema/immutable revisions; no Save, config, history or training'}
