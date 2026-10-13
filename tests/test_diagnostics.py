"""Actual CLI doctor preserves files and reports invalid drafts without fitting."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import vpw_cli
from app.core import Workspace
from app.neural_msvar import controlled_drawing
from app.diagnostics import doctor


class DiagnosticsTests(unittest.TestCase):
    def test_current_controlled_model_and_actual_cli_doctor_are_readonly(self):
        workspace=Workspace();self.addCleanup(workspace.db.close);workspace.put(controlled_drawing())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.vtw';workspace.save(path)
            before=workspace.db.serialize();disk=path.read_bytes()
            result=doctor(workspace);self.assertEqual(result['status'],'ok')
            self.assertEqual(result['architectures'][0]['parameters'],1315)
            self.assertEqual(workspace.db.serialize(),before);self.assertFalse(workspace.dirty)
            out=io.StringIO()
            with redirect_stdout(out):code=vpw_cli.main([str(path),'doctor'])
            self.assertEqual(code,0);self.assertTrue(json.loads(out.getvalue())['ok']);self.assertEqual(path.read_bytes(),disk)

    def test_invalid_draft_is_located_and_cannot_be_called_executable(self):
        workspace=Workspace();self.addCleanup(workspace.db.close)
        doc=controlled_drawing();doc['graph']['edges']=[];workspace.put(doc)
        before=workspace.db.serialize();result=doctor(workspace)
        self.assertEqual(result['status'],'issues');self.assertEqual(result['issues'][0]['path'],['architectures',doc['id']])
        self.assertEqual(workspace.db.serialize(),before)

    def test_fresh_no_site_doctor_never_imports_numeric_or_Qt_packages(self):
        root=Path(__file__).parents[1]
        script='from app.core import Workspace; from app.diagnostics import doctor; import sys; w=Workspace(); assert doctor(w)["status"]=="ok"; w.db.close(); assert not any(n.split(".")[0] in ("torch","numpy","PyQt6") for n in sys.modules)'
        result=subprocess.run([sys.executable,'-S','-B','-c',script],cwd=root,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
