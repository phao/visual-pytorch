import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run


class LauncherTests(unittest.TestCase):
    def test_launcher_preserves_relative_workspace_path_and_cwd(self):
        caller_directory = Path.cwd()
        with patch.dict(os.environ, {"PYTHONPATH": "previous-search-path"}), \
                patch.object(run.sys, "argv", ["run.py", "experiment.vtw"]), \
                patch.object(run.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                patch.object(run.os, "execv", side_effect=SystemExit) as launch:
            with self.assertRaises(SystemExit):
                run.main()
            self.assertEqual(Path.cwd(), caller_directory)
            self.assertEqual(launch.call_args.args[1][-1], "experiment.vtw")
            self.assertEqual(os.environ["PYTHONPATH"],
                             str(Path(run.__file__).resolve().parent) + os.pathsep + "previous-search-path")
