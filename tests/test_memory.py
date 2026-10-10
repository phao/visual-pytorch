"""Whole-network RAM policy tests: symbolic sizes only, never large allocations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core import default_model, default_mlp, safety_report, preflight, GraphError, validate_model
from app.memory import available_memory, estimate_memory


class MemoryTests(unittest.TestCase):
    def test_tiny_input_huge_hidden_layer_is_blocked(self):
        d = default_mlp(inputs=1, outputs=1, hidden=[100000,100000])
        with patch('app.core.available_memory', return_value=16*1024**3):
            r = safety_report({d['id']:d}, d['id'])
        self.assertFalse(r['allowed'])
        self.assertGreater(r['memory']['parameter_bytes'], 40_000_000_000)
        self.assertEqual([v['rule'] for v in r['violations']], ['memory'])

    def test_large_input_small_model_allowed_and_exact_boundary_inclusive(self):
        d = default_model(inputs=4096)
        models = {d['id']:d}
        estimate = safety_report(models,d['id'])['memory']['estimated_bytes']
        with patch('app.core.available_memory', return_value=10*estimate):
            self.assertTrue(preflight(models,d['id'])['allowed'])
        with patch('app.core.available_memory', return_value=10*estimate-10):
            with self.assertRaisesRegex(GraphError, 'Constrained mode'):preflight(models,d['id'])

    def test_higher_dimensional_activations_batch_and_training_states_count(self):
        d=default_model(inputs=1,outputs=2);d['graph']['nodes'][0]['shape']=[1024,1024,1]
        c=validate_model({d['id']:d},d['id'])
        self.assertEqual(c['parameters'],4)
        self.assertGreater(c['activation_elements'],4*1024**2)
        a=estimate_memory(c,samples=2,training=True)
        b=estimate_memory(c,samples=100,training=True)
        self.assertGreater(b['components']['activations_backward'],a['components']['activations_backward'])
        self.assertEqual(b['components']['parameters_optimizer_evidence'],128*4)
        with patch('app.core.available_memory',return_value=8*1024**3):
            self.assertFalse(safety_report({d['id']:d},d['id'],samples=100,epochs=1)['allowed'])

    def test_missing_ram_fails_closed_but_does_not_make_model_invalid(self):
        d=default_model();models={d['id']:d}
        with patch('app.core.available_memory',return_value=None):
            self.assertFalse(safety_report(models,d['id'])['allowed'])
            self.assertTrue(safety_report(models,d['id'],False)['allowed'])
        self.assertEqual(validate_model(models,d['id'])['parameters'],3)

    def test_memavailable_and_cgroup_ancestor_headroom_no_swap(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);mem=root/'meminfo';membership=root/'membership';cg=root/'cg';child=cg/'child'
            child.mkdir(parents=True)
            mem.write_text('MemFree: 1 kB\nMemAvailable: 4096 kB\nSwapFree: 99999999 kB\n')
            membership.write_text('0::/child\n')
            (child/'memory.max').write_text('max');(cg/'memory.max').write_text('1000000')
            (cg/'memory.current').write_text('600000')
            self.assertEqual(available_memory(mem,membership,cg),400000)
            (child/'memory.max').write_text('100000');(child/'memory.current').write_text('200000')
            self.assertEqual(available_memory(mem,membership,cg),0)
            (child/'memory.current').write_text('bad')
            self.assertIsNone(available_memory(mem,membership,cg))
            mem.write_text('MemFree: 1 kB\n')
            self.assertIsNone(available_memory(mem,membership,cg))


if __name__ == '__main__':unittest.main()
