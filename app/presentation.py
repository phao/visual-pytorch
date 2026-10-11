"""Pure workspace presentation contracts; independent of per-user appearance."""
import copy
import math

PANEL_IDS = ('architectures-panel', 'experiment-dock', 'inspector-panel',
             'layout-panel', 'library-panel')


def validate_panel_layout(value):
    """Closed portable CSS-pixel preset, not arbitrary CSS or graph geometry."""
    if value is None:
        return None
    def keys(obj, expected):
        if not isinstance(obj, dict) or set(obj) != set(expected):
            raise ValueError('Panel arrangement has missing or unsupported fields.')
    def number(v, low, high):
        if type(v) not in (int, float) or not low <= v <= high or not math.isfinite(v):
            raise ValueError('Panel arrangement coordinates/sizes must be finite numbers in range.')
    keys(value, ('version', 'pack', 'panels', 'order'))
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Panel arrangement requires version 1.')
    keys(value['pack'], ('x', 'y', 'collapsed'))
    for key in ('x', 'y'):
        number(value['pack'][key], -1000000, 1000000)
    if type(value['pack']['collapsed']) is not bool:
        raise ValueError('Panel pack collapsed must be a boolean.')
    keys(value['panels'], PANEL_IDS)
    for panel in value['panels'].values():
        keys(panel, ('x', 'y', 'width', 'height', 'visible', 'expanded'))
        for key in ('x', 'y'):
            number(panel[key], -1000000, 1000000)
        for key in ('width', 'height'):
            number(panel[key], 1, 100000)
        for key in ('visible', 'expanded'):
            if type(panel[key]) is not bool:
                raise ValueError('Panel visibility/expansion must be booleans.')
        if panel['expanded'] and not panel['visible']:
            raise ValueError('A hidden panel cannot be expanded.')
        if value['pack']['collapsed'] and panel['visible']:
            raise ValueError('A collapsed menu pack cannot contain visible panels.')
    order = value['order']
    if (not isinstance(order, list) or len(order) != len(PANEL_IDS) or
            any(type(item) is not str for item in order) or set(order) != set(PANEL_IDS)):
        raise ValueError('Panel order must contain each supported panel exactly once.')
    return copy.deepcopy(value)


LAYOUT_DEFAULTS = {'version': 1,
    'collapsed': {'style': 'layered', 'direction': 'LR', 'layer_gap': 48,
                  'block_gap': 16, 'rearrange': True, 'fit': True},
    'expanded': {'style': 'layered', 'direction': 'LR', 'layer_gap': 80,
                 'block_gap': 32, 'rearrange': True, 'fit': True}}


def validate_layout(value):
    if (not isinstance(value, dict) or set(value) != {'version', 'collapsed', 'expanded'} or
            type(value['version']) is not int or value['version'] != 1):
        raise ValueError('Layout preferences require version 1 and exactly two profiles.')
    for mode in ('collapsed', 'expanded'):
        profile = value[mode]
        if (not isinstance(profile, dict) or
                set(profile) != {'style', 'direction', 'layer_gap', 'block_gap', 'rearrange', 'fit'} or
                profile['style'] != 'layered' or profile['direction'] not in ('LR', 'TB')):
            raise ValueError('Choose a Layered layout and supported direction for each block state.')
        for key in ('layer_gap', 'block_gap'):
            if type(profile[key]) is not int or not 8 <= profile[key] <= 1000:
                raise ValueError('Layout gaps must be integers from 8 to 1000 model units.')
        for key in ('rearrange', 'fit'):
            if type(profile[key]) is not bool:
                raise ValueError('Layout toggles must be booleans.')
    return copy.deepcopy(value)
