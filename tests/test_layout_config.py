import copy
import tempfile
import unittest
from pathlib import Path
from app.config import Config,DEFAULTS,validate_layout


class LayoutConfigTests(unittest.TestCase):
    def test_profiles_are_independent_and_saved_without_workspace_or_display_changes(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'config.json';c=Config(path);other=Config(Path(td)/'other.json')
            c.values['layout']['collapsed']['layer_gap']=73
            self.assertEqual(other.values['layout']['collapsed']['layer_gap'],48)
            self.assertEqual(DEFAULTS['layout']['collapsed']['layer_gap'],48)
            value=copy.deepcopy(c.values['layout']);value['expanded'].update(direction='TB',fit=False,rearrange=False)
            c.save(layout=value,future_key='retained')
            loaded=Config(path);self.assertIsNone(loaded.error)
            self.assertEqual(loaded.values['layout'],value)
            for key in ['theme','box_spacing','spacing','font_size','ui_zoom']:
                self.assertEqual(loaded.values[key],DEFAULTS[key])
            self.assertEqual(loaded.values['future_key'],'retained')

    def test_invalid_preferences_are_rejected_without_replacing_the_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'config.json';c=Config(path);c.save();before=path.read_bytes()
            for mode in ('collapsed','expanded'):
                for key,values in {'layer_gap':[0,7,1001,True,1.5,'48',None],
                                   'block_gap':[0,1001,False], 'fit':[1,'yes',None],
                                   'rearrange':[0,'no'], 'style':['unknown',None],
                                   'direction':['RL','BT',None]}.items():
                    for value in values:
                        bad=copy.deepcopy(DEFAULTS['layout']);bad[mode][key]=value
                        with self.subTest(mode=mode,key=key,value=value):
                            with self.assertRaises(ValueError):c.save(layout=bad)
                            self.assertEqual(path.read_bytes(),before)
            for version in (0,2,True,1.0,'1',None):
                bad=copy.deepcopy(DEFAULTS['layout']);bad['version']=version
                with self.assertRaises(ValueError):validate_layout(bad)
