"""Trusted notebook runner process, not a training worker or an execution sandbox."""
import json
import sys
from pathlib import Path

from .notebook_execution import validate_selection, inspect_bundle
from .notebooks import execute_bundle, publish_bundle


def main():
    def emit(kind, value):
        print(json.dumps({'kind': kind, 'payload': value}, allow_nan=False), flush=True)
    try:
        req = json.load(sys.stdin)
        if req['operation'] == 'generate':
            if set(req) != {'operation', 'capture', 'directory'}:
                raise ValueError('Invalid notebook generation request.')
            generated = publish_bundle(req['capture'], req['directory'])
            emit('result', {'generated': generated, 'selection': inspect_bundle(req['directory'])})
            return 0
        if req['operation'] == 'inspect':
            emit('result', {'selection': inspect_bundle(req['directory'])})
            return 0
        if req['operation'] != 'execute':
            raise ValueError('Unknown notebook runner operation.')
        validate_selection(req['selection'])
        class Stop:
            def is_set(self):
                return Path(req['stop']).exists()
        result = execute_bundle(req['selection']['path'], python=sys.executable, timeout=req['timeout'],
            expected=req['selection'], cancel=Stop(), progress=lambda value: emit('progress', value))
        emit('result', result)
        return 0
    except Exception as error:
        emit('error', str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
