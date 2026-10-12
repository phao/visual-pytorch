"""Allocation-free concrete-graph standalone runtime-temporal-v2 export."""
from .temporal_export import _temporal_source


def temporal_source(models, mid, runtime_inputs, *, constrained=True, memory_limit_percent=10.):
    return _temporal_source(models, mid, runtime_inputs, constrained=constrained,
                            memory_limit_percent=memory_limit_percent, version=2)
