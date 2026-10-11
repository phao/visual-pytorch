"""Versioned draft admission estimates, not allocator or lazy-storage limits."""
import math

from . import memory
from .datasets import _require

HEADROOM = 8 * memory.MIB


def footprint(value):
    """Conservative JSON-shaped Python storage count without encoding/copying it."""
    size = 0; stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            size += 128 + 64 * len(item)
            stack.extend(item.keys()); stack.extend(item.values())
        elif isinstance(item, list):
            size += 64 + 8 * len(item); stack.extend(item)
        elif isinstance(item, str): size += 64 + 4 * len(item)
        elif type(item) in (int, float):
            _require(type(item) is int or math.isfinite(item), 'Draft values must remain finite JSON.')
            size += 64
        else:
            _require(item is None or type(item) is bool, 'Draft values must remain JSON-shaped.')
            size += 32
    return size


def estimate(document=None, *, source_bytes=0, staging_bytes=0):
    # Source JSON may consist of many tiny scalars/containers. Six live Python
    # object/copy/encoding equivalents are allowed after parsing. The SQL-size
    # and staged JSON multipliers are coarser admission estimates, not measured
    # peaks or dependency/history/file-size bounds.
    return HEADROOM + (6 * footprint(document) if document is not None else 64 * source_bytes) + 64 * staging_bytes


def admit(workspace, estimated, percentage, *, available_fn=None):
    percentage = memory.memory_percent(percentage)
    if not workspace.constrained_mode:
        return {'version':1, 'constrained':False, 'estimated_bytes':estimated, 'budget_bytes':None}
    available = (available_fn or memory.available_memory)()
    _require(type(available) is int and available >= 0,
             'Cannot observe available RAM for constrained draft admission. Inspect/close the draft or explicitly disable constrained mode.')
    budget = int(available * percentage / 100.)
    _require(estimated <= budget,
             'Draft operation exceeds the constrained available-RAM estimate. Inspect/close it, reduce its size or explicitly disable constrained mode.')
    return {'version':1, 'constrained':True, 'estimated_bytes':estimated, 'budget_bytes':budget}
