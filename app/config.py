"""User-level display and RAM-policy preferences, separate from workspace semantics."""
import json
import math
import os
import tempfile
from pathlib import Path
from .memory import DEFAULT_MEMORY_LIMIT_PERCENT, memory_percent
from .themes import resolve_theme

DEFAULTS = {'ui_zoom': 1.15, 'font_size': 14, 'memory_limit_percent': DEFAULT_MEMORY_LIMIT_PERCENT, 'theme':'dark','spacing':4,'box_spacing':35}


class Config:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else Path.home() / '.tensorloon' / 'config.json'
        self.values = dict(DEFAULTS)
        self.error = None
        if self.path.exists():
            try:
                values = json.loads(self.path.read_text())
                # Read older appearance choices without rewriting the file on open.
                if isinstance(values, dict):
                    if values.get('spacing') == 'compact':values['spacing'] = 4
                    elif values.get('spacing') == 'spaced':values['spacing'] = 10
                self.validate(values)
                self.values.update(values)
            except (ValueError, OSError, TypeError) as error:
                self.error = str(error)

    @staticmethod
    def validate(values):
        if not isinstance(values, dict):
            raise ValueError('Configuration must be a JSON object.')
        zoom, font = values.get('ui_zoom', DEFAULTS['ui_zoom']), values.get('font_size', DEFAULTS['font_size'])
        if isinstance(zoom, bool) or not isinstance(zoom, (int, float)) or not math.isfinite(zoom) or not .5 <= zoom <= 3:
            raise ValueError('UI zoom must be between 0.5 and 3.0.')
        if isinstance(font, bool) or not isinstance(font, int) or not 12 <= font <= 28:
            raise ValueError('Font size must be an integer from 12 to 28 pixels.')
        memory_percent(values.get('memory_limit_percent', DEFAULTS['memory_limit_percent']))
        resolve_theme(values.get('theme', DEFAULTS['theme']))
        spacing=values.get('spacing',DEFAULTS['spacing'])
        if isinstance(spacing,bool) or not isinstance(spacing,int) or not 1<=spacing<=10:
            raise ValueError('Menu spacing must be an integer from 1 to 10.')
        box=values.get('box_spacing',DEFAULTS['box_spacing'])
        if isinstance(box,bool) or not isinstance(box,int) or not 1<=box<=100:
            raise ValueError('NN box spacing must be an integer from 1 to 100.')

    def save(self, **changes):
        values = {**self.values, **changes}
        self.validate(values)
        # Do not silently overwrite a malformed file a user may want to repair.
        if self.error:
            raise ValueError('Repair the invalid configuration first: ' + str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.config-', suffix='.json', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(values, stream, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.values = values
