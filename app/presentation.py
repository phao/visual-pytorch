"""Pure workspace presentation contracts; independent of per-user appearance."""
import copy


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
